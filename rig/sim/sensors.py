import math
import queue
import threading
import time

import numpy as np

from rig.views.bev import Bev
from rig.views.compose import robot_views, role_of
from rig.views.schedule import Schedule

ROBOTS = "/World/Obstacles/Robots"
EYE = (0.30, 0.0, 0.32)
LOOK = (6.0, 0.0, -0.5)
APERTURE = 20.955
SETTLE = 2


class SensorRig:
    def __init__(self, stage, obstacles, leak=None, size=(256, 192), rate=2.0, focal=10.5, mode="RaytracedLighting"):
        self.stage, self.obstacles, self.leak = stage, obstacles, leak
        self.size, self.focal, self.mode = tuple(size), focal, mode
        self.schedule = Schedule(rate)
        self.cams: dict[str, dict] = {}
        self.bevs: dict[str, Bev] = {}
        self.force: dict[str, list] = {}
        self.pending = None
        self.seq = 0
        self.jobs = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.out: dict[str, dict] = {}
        self.main_ms = self.compose_ms = 0.0
        self.captures = 0
        threading.Thread(target=self._work, daemon=True).start()

    def _mount(self, rid: str) -> dict:
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom

        from rig.sim.orbit import look_at_rows

        path = f"{ROBOTS}/{rid}/head_cam"
        cam = UsdGeom.Camera.Define(self.stage, path)
        cam.CreateFocalLengthAttr(self.focal)
        cam.CreateHorizontalApertureAttr(APERTURE)
        cam.CreateVerticalApertureAttr(APERTURE * self.size[1] / self.size[0])
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 400.0))
        cam.AddTransformOp().Set(Gf.Matrix4d(look_at_rows(EYE, LOOK)))
        rp = rep.create.render_product(path, self.size)
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(rp)
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth.attach(rp)
        rp_path = getattr(rp, "path", None)
        prim = self.stage.GetPrimAtPath(rp_path) if rp_path else None
        if self.mode and prim is not None and prim.IsValid():
            prim.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set(self.mode)
        texture = getattr(rp, "hydra_texture", None)
        c = {"path": path, "cam": cam, "rp": rp, "rgb": rgb, "depth": depth, "texture": texture,
             "role": role_of(len(self.cams))}
        self._enable(c, False)
        print(f"rig sim: sensor cam {path} {self.size[0]}x{self.size[1]} role={c['role']} rp={rp_path}", flush=True)
        return c

    def _enable(self, c: dict, on: bool) -> None:
        tex = c["texture"]
        if tex is not None and hasattr(tex, "set_updates_enabled"):
            tex.set_updates_enabled(bool(on))

    def _header(self, c: dict) -> dict:
        from pxr import Usd

        from rig.camera import from_usd

        m = c["cam"].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return from_usd([list(m.GetRow(i)) for i in range(4)], self.focal, APERTURE, *self.size)

    def _pose(self, rid: str):
        r = self.obstacles.robots[rid]
        x, y, z = r.get("pose", (0.0, 0.0, 0.0))
        yaw = math.radians(float(r["ops"][1].Get() or 0.0))
        return (x, y, z, yaw)

    def event(self, ev: dict) -> bool:
        if ev.get("name") != "force":
            return False
        a = ev.get("args") or {}
        rid = str(a.get("robot") or next(iter(self.cams), "R1"))
        self.force.setdefault(rid, []).append(float(a.get("force_n", 0.0)))
        del self.force[rid][:-120]
        return True

    def tick(self, t: float) -> None:
        t0 = time.perf_counter()
        for rid in list(self.obstacles.robots):
            if rid not in self.cams:
                self.cams[rid] = self._mount(rid)
        if self.pending is not None:
            p = self.pending
            p["left"] -= 1
            if p["left"] <= 0:
                c = self.cams[p["rid"]]
                rgb, depth = np.asarray(c["rgb"].get_data()), np.asarray(c["depth"].get_data())
                self._enable(c, False)
                self.pending = None
                if rgb.ndim == 3 and rgb.size and depth.size:
                    field = self.leak.plume.field() if self.leak is not None and c["role"] == "gas" else None
                    self.seq += 1
                    self.captures += 1
                    try:
                        self.jobs.put_nowait((p["rid"], c["role"], rgb[..., :3].copy(), depth.copy(), p["cam"], p["pose"],
                                              p["path"], field, list(self.force.get(p["rid"], [])) or None, self.seq))
                    except queue.Full:
                        pass
        else:
            rid = self.schedule.due(list(self.cams), t)
            if rid is not None:
                c = self.cams[rid]
                self._enable(c, True)
                path = (self.obstacles.paths.get(rid) or {}).get("path")
                self.pending = {"rid": rid, "left": SETTLE, "cam": self._header(c), "pose": self._pose(rid), "path": path}
        self.main_ms = 0.9 * self.main_ms + 0.1 * (time.perf_counter() - t0) * 1e3

    def _work(self) -> None:
        while True:
            rid, role, rgb, depth, cam, pose, path, field, force, seq = self.jobs.get()
            t0 = time.perf_counter()
            try:
                v = robot_views(rid, role, rgb, depth, cam, pose, path=path, field=field,
                                bev=self.bevs.setdefault(rid, Bev()), force=force, seq=seq)
            except Exception as exc:
                print(f"rig sim: views {rid} failed: {exc!r}", flush=True)
                continue
            self.compose_ms = 0.8 * self.compose_ms + 0.2 * (time.perf_counter() - t0) * 1e3
            with self.lock:
                self.out[rid] = v

    def take(self) -> dict | None:
        with self.lock:
            out, self.out = self.out, {}
        return out or None

    def summary(self) -> str:
        return f"views cams={len(self.cams)} caps={self.captures} main_ms={self.main_ms:.2f} compose_ms={self.compose_ms:.0f}"

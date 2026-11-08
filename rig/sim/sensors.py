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
SETTLE = 3


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
        self.on = True
        self.rp = self.rp_prim = self.texture = None
        self.read_ms = 0.0
        threading.Thread(target=self._work, daemon=True).start()

    def _mount(self, rid: str) -> dict:
        from pxr import Gf, UsdGeom

        from rig.sim.orbit import look_at_rows

        path = f"{ROBOTS}/{rid}/head_cam"
        cam = UsdGeom.Camera.Define(self.stage, path)
        cam.CreateFocalLengthAttr(self.focal)
        cam.CreateHorizontalApertureAttr(APERTURE)
        cam.CreateVerticalApertureAttr(APERTURE * self.size[1] / self.size[0])
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 400.0))
        cam.AddTransformOp().Set(Gf.Matrix4d(look_at_rows(EYE, LOOK)))
        if self.rp is None:
            self._product(path)
        c = {"path": path, "cam": cam, "role": role_of(len(self.cams))}
        print(f"rig sim: sensor cam {path} {self.size[0]}x{self.size[1]} role={c['role']}", flush=True)
        return c

    def _product(self, path: str) -> None:
        import omni.replicator.core as rep
        from pxr import Sdf

        self.rp = rep.create.render_product(path, self.size)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach(self.rp)
        self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth.attach(self.rp)
        self.texture = getattr(self.rp, "hydra_texture", None)
        self.rp_prim = self.stage.GetPrimAtPath(getattr(self.rp, "path", ""))
        if self.mode and self.rp_prim.IsValid():
            with self._session():
                self.rp_prim.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set(self.mode)
        print(f"rig sim: sensor render product {getattr(self.rp, 'path', None)} valid={self.rp_prim.IsValid()} mode={self.mode}", flush=True)

    def _session(self):
        from pxr import Usd

        return Usd.EditContext(self.stage, self.stage.GetSessionLayer())

    def _aim(self, path: str) -> None:
        if self.rp_prim is not None and self.rp_prim.IsValid():
            with self._session():
                self.rp_prim.GetRelationship("camera").SetTargets([path])

    def _enable(self, on: bool) -> None:
        if self.texture is not None and hasattr(self.texture, "set_updates_enabled"):
            self.texture.set_updates_enabled(bool(on))

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
        a = ev.get("args") or {}
        if ev.get("name") == "views":
            self.on = bool(a.get("on", True))
            self._enable(self.on)
            print(f"rig sim: views {'on' if self.on else 'off'}", flush=True)
            return True
        if ev.get("name") != "force":
            return False
        rid = str(a.get("robot") or next(iter(self.cams), "R1"))
        self.force.setdefault(rid, []).append(float(a.get("force_n", 0.0)))
        del self.force[rid][:-120]
        return True

    def tick(self, t: float) -> None:
        t0 = time.perf_counter()
        for rid in list(self.obstacles.robots):
            if rid not in self.cams:
                self.cams[rid] = self._mount(rid)
        if not self.on:
            return
        if self.pending is not None:
            p = self.pending
            p["left"] -= 1
            if p["left"] <= 0:
                c = self.cams[p["rid"]]
                t1 = time.perf_counter()
                rgb, depth = np.asarray(self.rgb.get_data()), np.asarray(self.depth.get_data())
                self.read_ms = 0.8 * self.read_ms + 0.2 * (time.perf_counter() - t1) * 1e3
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
                self._aim(c["path"])
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
        return f"views cams={len(self.cams)} caps={self.captures} main_ms={self.main_ms:.2f} read_ms={self.read_ms:.1f} compose_ms={self.compose_ms:.0f}"

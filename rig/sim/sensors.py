import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np

from rig.views.bev import Bev
from rig.views.compose import depth_views, robot_views, role_of
from rig.views.raycast import depth_from_hits, fan, ray_grid
from rig.views.schedule import Demand, Guard

ROBOTS = "/World/Obstacles/Robots"
MOUNTS = {"rust": ((0.35, 0.0, 1.05), (2.0, 0.0, 0.55)), "gas": ((0.3, 0.0, 0.95), (6.0, 0.0, 0.4)),
          "inspection": ((0.32, 0.0, 0.45), (6.0, 0.0, -0.5))}
LIGHT = 5000.0
NEAR = 0.6
APERTURE = 20.955
BURST = 3
DEPTH_SIZE = (96, 72)
MESH_HALF = (30.0, 30.0, 8.0)
SKIP = ("/World/Obstacles", "/World/Plume", "/Render", "/World/Cinematic/Ocean", "/World/Environment")
OUTPUT = Path(__file__).resolve().parents[2] / "sim" / "output"


def map_targets(leak=None) -> dict:
    out = {"rust": [], "gas": [], "inspection": []}
    try:
        dmg = json.loads((OUTPUT / "rig_merged_meta.json").read_text())["world"]["damage"]["position_m"]
        out["rust"].append((tuple(dmg), "rust target (job map)"))
    except (OSError, KeyError, ValueError):
        pass
    try:
        for f in json.loads((OUTPUT / "leak_flanges_meta.json").read_text())["flanges"]:
            out["gas"].append((tuple(f["center_m"]), f["id"]))
    except (OSError, KeyError, ValueError):
        pass
    if leak is not None:
        out["gas"].append((tuple(float(c) for c in leak.plume.source), f"leak {leak.args.leak}"))
    return out


def stage_meshes(stage, center, half=MESH_HALF):
    from pxr import Usd, UsdGeom

    lo, hi = np.asarray(center) - half, np.asarray(center) + half
    verts, tris, n = [], [], 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if str(prim.GetPath()).startswith(SKIP) or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        m = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), np.float64)
        p = np.asarray(pts, np.float64) @ m[:3, :3] + m[3, :3]
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        if not inside.any():
            continue
        t = fan(counts, idx)
        t = t[inside[t].any(1)]
        if not len(t):
            continue
        verts.append(p)
        tris.append(t + n)
        n += len(p)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(verts), np.concatenate(tris)


class SensorRig:
    def __init__(self, stage, obstacles, leak=None, size=(256, 192), rate=4.0, focal=10.5, mode="", keepalive=45.0):
        self.stage, self.obstacles, self.leak = stage, obstacles, leak
        self.size, self.focal, self.mode = tuple(size), focal, mode
        self.demand = Demand(rate, keepalive)
        self.guard = Guard()
        self.cams: dict[str, dict] = {}
        self.bevs: dict[str, Bev] = {}
        self.force: dict[str, list] = {}
        self.product = None
        self.caster = None
        self.burst = False
        self.armed = 0
        self.seq = 0
        self.jobs = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.out: dict[str, dict] = {}
        self.main_ms = self.compose_ms = self.read_ms = self.cast_ms = 0.0
        self.captures = self.stale = 0
        self.targets = map_targets(leak)
        threading.Thread(target=self._work, daemon=True).start()

    def _mount(self, rid: str) -> dict:
        from pxr import Gf, UsdGeom, UsdLux

        from rig.sim.orbit import look_at_rows

        order = list(self.obstacles.robots)
        role = role_of(order.index(rid) if rid in order else len(self.cams))
        path = f"{ROBOTS}/{rid}/head_cam"
        cam = UsdGeom.Camera.Define(self.stage, path)
        cam.CreateFocalLengthAttr(self.focal)
        cam.CreateHorizontalApertureAttr(APERTURE)
        cam.CreateVerticalApertureAttr(APERTURE * self.size[1] / self.size[0])
        far = 400.0 + len(self.cams)
        cam.CreateClippingRangeAttr(Gf.Vec2f(NEAR, far))
        cam.AddTransformOp().Set(Gf.Matrix4d(look_at_rows(*MOUNTS[role])))
        lamp = UsdLux.SphereLight.Define(self.stage, f"{path}/work_light")
        lamp.CreateRadiusAttr(0.04)
        lamp.CreateIntensityAttr(LIGHT)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))
        shaping = UsdLux.ShapingAPI.Apply(lamp.GetPrim())
        shaping.CreateShapingConeAngleAttr(38.0)
        shaping.CreateShapingConeSoftnessAttr(0.4)
        lamp.GetPrim().SetActive(False)
        print(f"rig sim: sensor cam {path} role={role}", flush=True)
        return {"path": path, "cam": cam, "role": role, "far": far, "lamp": f"{path}/work_light"}

    def _product(self, path: str) -> dict:
        import omni.replicator.core as rep

        rp = rep.create.render_product(path, self.size)
        out = {"rp": rp, "params": None, "prim": self.stage.GetPrimAtPath(getattr(rp, "path", ""))}
        for key, name in (("rgb", "rgb"), ("depth", "distance_to_image_plane"), ("params", "camera_params")):
            try:
                ann = rep.AnnotatorRegistry.get_annotator(name)
                ann.attach(rp)
                out[key] = ann
            except Exception as exc:
                print(f"rig sim: annotator {name} unavailable {exc!r}", flush=True)
        out["texture"] = getattr(rp, "hydra_texture", None)
        print(f"rig sim: sensor render product {getattr(rp, 'path', None)} {self.size[0]}x{self.size[1]} for {path}", flush=True)
        return out

    def _destroy(self) -> None:
        if self.product is None:
            return
        for key in ("rgb", "depth", "params"):
            ann = self.product.get(key)
            if ann is not None:
                try:
                    ann.detach()
                except Exception:
                    pass
        destroy = getattr(self.product["rp"], "destroy", None)
        if destroy is not None:
            destroy()
        else:
            self._live(False)
        self.product = None

    def _session(self):
        from pxr import Usd

        return Usd.EditContext(self.stage, self.stage.GetSessionLayer())

    def _live(self, on: bool) -> None:
        tex = self.product and self.product["texture"]
        if tex is not None and hasattr(tex, "set_updates_enabled"):
            tex.set_updates_enabled(bool(on))

    def _lamp(self, rid, on: bool) -> None:
        c = self.cams.get(rid)
        if c is not None:
            prim = self.stage.GetPrimAtPath(c["lamp"])
            if prim.IsValid():
                prim.SetActive(on)

    def _start(self, key: str, prev) -> None:
        rid, mode = key.split("/")
        if rid not in self.cams:
            self.cams[rid] = self._mount(rid)
        prid, pmode = prev.split("/") if prev else (None, None)
        if pmode == "cam" and mode != "cam":
            self._live(False)
        self._lamp(prid, False)
        if mode == "depth":
            self._mesh(rid)
            print(f"rig sim: views on {rid} depth (ray cast)", flush=True)
            return
        self._lamp(rid, True)
        path = self.cams[rid]["path"]
        if self.product is None:
            self.product = self._product(path)
        else:
            prim = self.product["prim"]
            if prim is not None and prim.IsValid():
                with self._session():
                    prim.GetRelationship("camera").SetTargets([path])
        self._live(not self.burst)
        self.armed = 0
        print(f"rig sim: views on {rid} cam burst={self.burst} {self.size[0]}x{self.size[1]}", flush=True)

    def _stop(self, prev) -> None:
        self._live(False)
        if prev:
            self._lamp(prev.split("/")[0], False)
        self.armed = 0
        print(f"rig sim: views off {prev}", flush=True)

    def _mesh(self, rid: str) -> None:
        x, y, z, _ = self._pose(rid)
        c = np.array([x, y, z])
        if self.caster is not None and np.all(np.abs(c - self.caster[1]) < np.asarray(MESH_HALF) * 0.5):
            return
        from rig.views.raycast import WarpCaster

        t0 = time.perf_counter()
        verts, tris = stage_meshes(self.stage, c)
        if not len(tris):
            print("rig sim: ray-cast mesh empty", flush=True)
            self.caster = None
            return
        self.caster = (WarpCaster(verts, tris), c)
        print(f"rig sim: ray-cast mesh {len(tris)} tris around {np.round(c, 1).tolist()} in {(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)

    def _fresh(self, c: dict) -> bool:
        params = self.product["params"]
        if params is None:
            return True
        nf = (params.get_data() or {}).get("cameraNearFar")
        return nf is not None and len(nf) > 1 and abs(float(nf[1]) - c["far"]) < 0.5

    def _header(self, c: dict, size=None) -> dict:
        from pxr import Usd

        from rig.camera import from_usd

        m = c["cam"].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return from_usd([list(m.GetRow(i)) for i in range(4)], self.focal, APERTURE, *(size or self.size))

    def _pose(self, rid: str):
        r = self.obstacles.robots[rid]
        x, y, z = r.get("pose", (0.0, 0.0, 0.0))
        yaw = math.radians(float(r["ops"][1].Get() or 0.0))
        return (x, y, z, yaw)

    def event(self, ev: dict) -> bool:
        a = ev.get("args") or {}
        name = ev.get("name")
        if name == "views":
            rid = a.get("robot")
            rid = str(rid) if rid else None
            if rid and rid not in self.obstacles.robots:
                print(f"rig sim: views for unknown robot {rid}", flush=True)
                return True
            key = f"{rid}/{'depth' if a.get('kind') == 'depth' else 'cam'}" if rid else None
            prev = self.demand.active
            on = bool(a.get("on", True))
            if not on and rid and prev and prev.split("/")[0] == rid:
                key = prev
            action = self.demand.request(key, on, time.monotonic())
            if action in ("start", "switch"):
                self._start(key, prev)
            elif action == "stop":
                self._stop(prev)
            return True
        if name != "force":
            return False
        rid = str(a.get("robot") or (self.demand.active or "R1/").split("/")[0])
        self.force.setdefault(rid, []).append(float(a.get("force_n", 0.0)))
        del self.force[rid][:-120]
        return True

    def _escalate(self, level: int) -> None:
        if level == 1:
            self.burst = True
            self.demand.period *= 2
            self._live(False)
            self.armed = 0
            print(f"rig sim: views guard state_hz={self.guard.hz:.1f} < {self.guard.threshold:.0f}: burst capture at "
                  f"{1 / self.demand.period:.1f} Hz", flush=True)
        elif level == 2:
            self.size = (max(64, self.size[0] // 2), max(48, self.size[1] // 2))
            key = self.demand.active
            self._destroy()
            if key and key.endswith("/cam"):
                self._start(key, None)
            print(f"rig sim: views guard state_hz={self.guard.hz:.1f}: head cam size -> {self.size[0]}x{self.size[1]}", flush=True)

    def tick(self, t: float) -> None:
        now = time.monotonic()
        t0 = time.perf_counter()
        prev = self.demand.active
        if self.demand.expire(now):
            self._stop(prev)
        key = self.demand.active
        level = self.guard.tick(now, bool(key and key.endswith("/cam")))
        if level:
            self._escalate(level)
        if key is None:
            return
        rid, mode = key.split("/")
        if rid not in self.obstacles.robots:
            return
        if mode == "depth":
            if self.demand.due(now):
                self.demand.captured(now)
                self._cast(rid)
        elif self.product is not None:
            self._capture(rid, now)
        self.main_ms = 0.9 * self.main_ms + 0.1 * (time.perf_counter() - t0) * 1e3

    def _capture(self, rid: str, now: float) -> None:
        c = self.cams[rid]
        if self.burst:
            if self.armed == 0:
                if not self.demand.due(now):
                    return
                self._live(True)
                self.armed = BURST
                return
            self.armed -= 1
            if self.armed > 0:
                return
        elif not self.demand.due(now):
            return
        if not self._fresh(c):
            self.stale += 1
            if self.burst:
                self.armed = 1
            return
        self.demand.captured(now)
        t1 = time.perf_counter()
        rgb, depth = np.asarray(self.product["rgb"].get_data()), np.asarray(self.product["depth"].get_data())
        if self.burst:
            self._live(False)
        self.read_ms = 0.8 * self.read_ms + 0.2 * (time.perf_counter() - t1) * 1e3
        if rgb.ndim == 3 and rgb.size and depth.size:
            field = self.leak.plume.field() if self.leak is not None and c["role"] == "gas" else None
            self._enqueue({"kind": "cam", "rid": rid, "role": c["role"], "rgb": rgb[..., :3].copy(), "depth": depth.copy(),
                           "cam": self._header(c), "field": field})

    def _cast(self, rid: str) -> None:
        if rid not in self.cams:
            self.cams[rid] = self._mount(rid)
        self._mesh(rid)
        if self.caster is None:
            return
        t1 = time.perf_counter()
        c = self.cams[rid]
        cam = self._header(c, DEPTH_SIZE)
        o, d = ray_grid(cam, *DEPTH_SIZE)
        hits = self.caster[0].cast(o, d.reshape(-1, 3), 60.0)
        depth = depth_from_hits(hits, d, cam).astype(np.float32)
        self.cast_ms = 0.8 * self.cast_ms + 0.2 * (time.perf_counter() - t1) * 1e3
        self._enqueue({"kind": "depth", "rid": rid, "role": c["role"], "depth": depth, "cam": cam})

    def _enqueue(self, job: dict) -> None:
        rid = job["rid"]
        self.seq += 1
        self.captures += 1
        targets = list(self.targets.get(job["role"], []))
        if job["role"] == "inspection":
            targets += [(tuple(p["xyz"]), oid) for oid, p in self.obstacles.props.items()]
        job.update(seq=self.seq, pose=self._pose(rid), path=(self.obstacles.paths.get(rid) or {}).get("path"),
                   force=list(self.force.get(rid, [])) or None, targets=targets)
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            pass

    def _work(self) -> None:
        while True:
            j = self.jobs.get()
            t0 = time.perf_counter()
            bev = self.bevs.setdefault(j["rid"], Bev())
            try:
                if j["kind"] == "depth":
                    v = depth_views(j["rid"], j["role"], j["depth"], j["cam"], j["pose"], path=j["path"], bev=bev, seq=j["seq"])
                else:
                    v = robot_views(j["rid"], j["role"], j["rgb"], j["depth"], j["cam"], j["pose"], path=j["path"],
                                    field=j["field"], bev=bev, force=j["force"], seq=j["seq"], targets=j["targets"])
            except Exception as exc:
                print(f"rig sim: views {j['rid']} failed: {exc!r}", flush=True)
                continue
            self.compose_ms = 0.8 * self.compose_ms + 0.2 * (time.perf_counter() - t0) * 1e3
            with self.lock:
                self.out[j["rid"]] = {**self.out.get(j["rid"], {}), **v}

    def take(self) -> dict | None:
        with self.lock:
            out, self.out = self.out, {}
        return out or None

    def summary(self) -> str:
        return (f"views active={self.demand.active} caps={self.captures} stale={self.stale} guard={self.guard.level} "
                f"hz={self.guard.hz:.1f} main_ms={self.main_ms:.2f} read_ms={self.read_ms:.1f} cast_ms={self.cast_ms:.1f} "
                f"compose_ms={self.compose_ms:.0f}")

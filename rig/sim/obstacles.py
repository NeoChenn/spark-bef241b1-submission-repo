import os
import math
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2] / "sim"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import carb.settings
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from rig.sim.obstacle_math import (DROP_S, DRUM_R, dashes, drop_offset, drum_pose, flame_offset, ghost_alpha, lerp,
                                   puddle_edge, seg_frame, segments, weather_params)
from rig.sim.gait import ARM, LEGS, PARK, STAND, GaitLoop, Quantiser, Walker, smoothstep
from rig.sim.taskmotion import Rollout, TaskMotion
from rig.sim import weld as W
from rig.sim.zoom import ease
from scene_builder import ASSETS, GENERATED, OUTPUT, SceneBuilder, rel

ROOT = "/World/Obstacles"
CORAL = (1.1, 0.27, 0.11)
GHOST = (0.32, 0.35, 0.4)
CLOSED = (1.4, 0.05, 0.03)
ROBOT_GLOW = [(0.6, 2.2, 9.0), (0.8, 7.0, 2.0), (9.0, 6.0, 0.8), (7.0, 1.2, 6.0)]
BEACON_W = 15000.0
ROBOT_USD = OUTPUT / "robot" / "go2_ur3_lift" / "go2_ur3_lift.usda"
ROBOT_Z = 0.45
LIFT = 0.1
SKIES = {False: GENERATED / "sky_night.hdr", True: GENERATED / "sky_storm.hdr"}
RAMP_S = 3.0
REVEAL_S = 1.2
CAN_SIZE = {"small": (0.085, 0.24), "large": (0.15, 0.42)}
CAN_PAINT = {"red": (0.62, 0.05, 0.03), "blue": (0.04, 0.16, 0.55), "yellow": (0.78, 0.55, 0.04), "green": (0.06, 0.35, 0.1)}
WEATHER_KEYS = ("fog_color", "fog_height", "wave_height", "dome", "wind", "sun", "fog_dist")
GRIP_ROBOTS = ("R3",)
SODA = (0.045, 0.18)
EE_LINK = ("Geometry/base/lift_housing/lift_carriage/ur3_base_link/ur3_shoulder_link/ur3_upper_arm_link/ur3_forearm_link/"
           "ur3_wrist_1_link/ur3_wrist_2_link/ur3_wrist_3_link/ur3_ee_link")
TOTE = (-0.27, 0.0, 0.062, 0.26, 0.22, 0.09, 0.008)
CUP = ((0.032, 0.02, 0.01, "grey"), (0.012, 0.172, 0.086, "grey"), (0.016, 0.022, 0.183, "black"), (0.022, 0.006, 0.197, "black"))


def _emissive(stage, name, color, intensity=1.0):
    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * intensity for c in color)))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat, sh


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _truncate(path, frac):
    segs = segments(path)
    total = sum(math.dist(a, b) for a, b in segs)
    left = total * frac
    out = [path[0]]
    for a, b in segs:
        L = math.dist(a, b)
        if left >= L:
            out.append(b)
            left -= L
        else:
            out.append(lerp(tuple(a), tuple(b), left / L))
            break
    return out


OVERLAYS = os.environ.get("RIG_OVERLAYS", "0") == "1"
RUST_INPUTS = (("/World/RepairZone/Looks/rust_coarse_01/hero_mdl", "amount"),
               ("/World/RepairZone/Looks/green_metal_rust/hero_mdl", "halo"),
               ("/World/RepairZone/Looks/green_metal_rust/hero_mdl", "streak_amount"))
STATIC_DOG = "/World/Robots/scrubber_dog"
RUST_SLEW = 0.25
VAPOUR_N, VAPOUR_LIFE, VAPOUR_MS, VAPOUR_JET = 90, 2.4, 0.8, 1.3
VAPOUR_ALPHA = (0.12, 0.08, 0.05, 0.025)
ARC_W, POOL_W, ARC_R, POOL_R = 3.0e7, 2.5e6, 0.004, 0.008
BEAD_GLOW, CORE_GLOW = 60.0, 900.0
SPARK_LOOK = (((1.0, 0.85, 0.55), 220.0, 1.3), ((1.0, 0.5, 0.12), 90.0, 1.0), ((0.9, 0.16, 0.03), 35.0, 0.7))
HALO = ((0.02, 0.3, 120.0), (0.045, 0.12, 40.0))


class ObstacleScene:
    def __init__(self, stage, wave=None, radius=0.06, robot_model=True):
        self.stage, self.wave, self.radius, self.robot_model = stage, wave if wave is not None else {}, radius, robot_model
        self.b = SceneBuilder(stage)
        self._roots()
        self.now = 0.0
        self.props: dict[str, dict] = {}
        self.paths: dict[str, dict] = {}
        self.robots: dict[str, dict] = {}
        self.closed_key = None
        self.weather_key = None
        self.look = None
        self.w_from = self.w_to = weather_params("calm")
        self.w_t0 = -1e9
        self.flames = []
        for k in range(8):
            prim = stage.GetPrimAtPath(f"/World/Cinematic/Flare/flame_{k}")
            if not prim.IsValid():
                break
            op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
            self.flames.append((op, op.Get()))
        self.tip_z = min((p[2] for _, p in self.flames), default=0.0)
        self.n = 0
        self.gait = GaitLoop.load() if robot_model and ROBOT_USD.exists() else None
        self.selected = None
        self.motion = {"gait": self.gait.source} if self.gait else {}
        self.clips = TaskMotion.load().clips if self.gait else {}
        self.rollout = Rollout.load() if self.gait else None
        self.anim_ms = 0.0
        self.rust, self.rust_q, self.rust_attrs, self.dog = 1.0, Quantiser(0.005), None, None
        self.leak, self.vap, self.vap_on, self.vap_t = None, None, False, -1e9
        self.hidden = []
        self.weld_clip = W.WeldClip.load() if self.gait else None
        self.wfx = None

    def _roots(self):
        UsdGeom.Scope.Define(self.stage, ROOT)
        for g in ("Props", "Paths", "Closed", "Robots"):
            UsdGeom.Scope.Define(self.stage, f"{ROOT}/{g}")
        self.coral, _ = _emissive(self.stage, "obst_coral", CORAL)
        self.red, _ = _emissive(self.stage, "obst_closed", CLOSED)

    def republish(self):
        for p in (ROOT, "/World/Looks/obst_coral", "/World/Looks/obst_closed"):
            self.stage.RemovePrim(p)
        self.props, self.paths, self.robots, self.closed_key, self.vap, self.wfx = {}, {}, {}, None, None, None
        self.b.materials = {k: v for k, v in self.b.materials.items() if not k.startswith("obst_")}
        self._roots()

    def _id(self, s):
        self.n += 1
        return f"{s}_{self.n}"

    def sync(self, state: dict) -> None:
        hz = state.get("hazards") or {}
        w = hz.get("weather")
        if w:
            self.set_weather(w.get("level", "calm"), w.get("wind_dir"), w.get("gust"))
        if "leak" in state:
            self.leak = state["leak"]
            on = bool(self.leak and self.leak.get("on"))
            if on != self.vap_on:
                self.vap_on, self.vap_t = on, self.now
        obs = {o["id"]: o for o in hz.get("obstacles", [])}
        for oid in [k for k in self.props if k not in obs]:
            self.stage.RemovePrim(self.props.pop(oid)["path"])
        for oid, o in obs.items():
            if oid not in self.props:
                self.props[oid] = self._spawn(o)
            elif "rollout" in o:
                self.props[oid]["o"] = o
            elif "mv" in self.props[oid]:
                pr = self.props[oid]
                pr["mv"] = {"p": tuple(o["xyz"]), "v": tuple(o.get("vel") or (0.0, 0.0)) if o.get("rolling") else (0.0, 0.0),
                            "t": self.now, "from": pr.get("at", tuple(o["xyz"]))}
        drums = {o["id"] for o in hz.get("obstacles", []) if o.get("kind") == "barrel"}
        key = tuple((tuple(c["a"]), tuple(c["b"])) for c in hz.get("closed", []) if c.get("why") not in drums)
        if OVERLAYS and key != self.closed_key:
            self.closed_key = key
            self.stage.RemovePrim(f"{ROOT}/Closed")
            UsdGeom.Scope.Define(self.stage, f"{ROOT}/Closed")
            for a, b in key:
                for p, q in dashes(a, b, 0.7, 0.5):
                    self._cyl(f"{ROOT}/Closed/{self._id('d')}", p, q, self.radius * 1.6, self.red, LIFT + 0.05)
        for k, r in enumerate(state.get("robots", [])):
            self._robot(k, r)
            if OVERLAYS:
                self._path(k, r)

    def _cyl(self, path, a, b, radius, mat, lift=LIFT):
        a, b = (a[0], a[1], a[2] + lift), (b[0], b[1], b[2] + lift)
        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr("X")
        c.CreateRadiusAttr(radius)
        ops = UsdGeom.Xformable(c)
        ops.AddTranslateOp()
        ops.AddRotateZOp()
        ops.AddRotateYOp()
        _bind(c.GetPrim(), mat)
        self._place(c, a, b)
        return c

    def _place(self, c, a, b):
        mid, L, yaw, pitch = seg_frame(a, b)
        c.GetHeightAttr().Set(max(L, 1e-3))
        t, rz, ry = UsdGeom.Xformable(c).GetOrderedXformOps()
        t.Set(Gf.Vec3d(*mid))
        rz.Set(yaw)
        ry.Set(pitch)

    def _robot(self, k, r):
        rid, p = r["id"], r["pose"]
        if rid not in self.robots:
            root = f"{ROOT}/Robots/{rid}"
            x = UsdGeom.Xform.Define(self.stage, root)
            ops = UsdGeom.Xformable(x)
            ops.AddTranslateOp()
            ops.AddRotateZOp()
            if self.robot_model and ROBOT_USD.exists():
                body = UsdGeom.Xform.Define(self.stage, f"{root}/body")
                body.GetPrim().GetReferences().AddReference(rel(ROBOT_USD))
                vs = body.GetPrim().GetVariantSets()
                if vs.HasVariantSet("Physics"):
                    vs.GetVariantSet("Physics").SetVariantSelection("none")
            if OVERLAYS:
                light = UsdLux.SphereLight.Define(self.stage, f"{root}/beacon")
                light.CreateRadiusAttr(0.15)
                light.CreateIntensityAttr(BEACON_W)
                light.CreateColorAttr(Gf.Vec3f(*(c / max(ROBOT_GLOW[k % len(ROBOT_GLOW)]) for c in ROBOT_GLOW[k % len(ROBOT_GLOW)])))
                UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.2))
            self.robots[rid] = {"ops": ops.GetOrderedXformOps()}
            if rid in GRIP_ROBOTS and self.robot_model and ROBOT_USD.exists():
                self._grip(root)
            if self.gait is not None:
                from rig.sim.joints import RobotJoints

                j = RobotJoints(self.stage, f"{root}/body")
                j.set(LEGS + ARM, [STAND[n] for n in LEGS] + [PARK[n] for n in ARM])
                self.robots[rid].update(joints=j, walker=Walker(self.gait, (p["x"], p["y"], p["z"], p.get("yaw", 0.0)), self.now),
                                        arm=TaskMotion(self.clips), q=Quantiser(), qy=Quantiser(0.05))
        rb = self.robots[rid]
        rb["r"] = r
        self._weld_sync(rid, rb, r)
        if "walker" in rb:
            rb["walker"].target((p["x"], p["y"], p["z"], p.get("yaw", 0.0)), self.now)
            return
        t, rz = rb["ops"]
        t.Set(Gf.Vec3d(p["x"], p["y"], p["z"] + ROBOT_Z))
        rz.Set(math.degrees(p.get("yaw", 0.0)))
        rb["pose"] = (p["x"], p["y"], p["z"])

    def select(self, rid) -> None:
        self.selected = rid

    def _animate(self, t: float) -> None:
        t0 = time.perf_counter()
        motion = {"gait": self.gait.source} if self.gait else {}
        with Sdf.ChangeBlock():
            for rid, rb in self.robots.items():
                if "walker" not in rb:
                    continue
                (x, y, z, yaw), q, bob = rb["walker"].step(t)
                top, rz = rb["ops"]
                if rb["q"].changed("pos", (x, y, z + bob)):
                    top.Set(Gf.Vec3d(round(x, 3), round(y, 3), round(z + self.gait.zmean + bob, 3)))
                if rb["qy"].changed("yaw", math.degrees(yaw)):
                    rz.Set(round(math.degrees(yaw), 2))
                rb["pose"] = (x, y, z)
                r = rb.get("r") or {}
                ro = r.get("rollout") if r.get("job_state") == "ROLLOUT" and self.rollout is not None else None
                if ro is None:
                    rb["ro_t0"] = None
                else:
                    if rb.get("ro_t0") is None:
                        rb["ro_t0"] = t
                    rb["ro_k"] = min(ro[0] + (t - rb["ro_t0"]) * self.rollout.hz, ro[1])
                    q, aq, _, _ = self.rollout.at(rb["ro_k"])
                    top.Set(Gf.Vec3d(round(x, 3), round(y, 3), round(z + self.rollout.base_z, 3)))
                    rb["q"].last.pop("pos", None)
                    rb["arm"].out = aq.copy()
                    rb["joints"].set(ARM, aq)
                rb["root"] = (x, y, z + (self.rollout.base_z if ro else self.gait.zmean + bob), yaw)
                rb["joints"].set(LEGS, q)
                if ro is not None:
                    continue
                wd = rb.get("weld")
                if wd is not None and wd["arm"]:
                    c = t - wd["t0"]
                    aq = self.weld_clip.arm(c - W.LEAD_S)
                    if wd["from"] is not None:
                        aq = wd["from"] + (aq - wd["from"]) * smoothstep(c / W.LEAD_S)
                    rb["arm"].out, rb["arm"].key = aq, "weld"
                    rb["joints"].set(ARM, aq)
                    motion[rid] = self.weld_clip.source
                    continue
                arm = rb["arm"]
                live = rid == self.selected or str(r.get("task") or "").endswith(":demo")
                rb["joints"].set(ARM, arm.step(rb.get("r") if live else None, t))
                if live and arm.source:
                    motion[rid] = arm.source
        self.motion = motion
        self._rust(t)
        self.anim_ms += ((time.perf_counter() - t0) * 1e3 - self.anim_ms) * 0.05

    def _rust(self, t: float) -> None:
        want = [float(rb["r"]["rust"]) for rb in self.robots.values() if (rb.get("r") or {}).get("rust") is not None]
        target = min(want) if want else 1.0
        dt = max(t - getattr(self, "rust_t", t), 0.0)
        self.rust_t = t
        self.rust = target if target >= self.rust else max(target, self.rust - RUST_SLEW * dt)
        if self.rust_attrs is None:
            self.rust_attrs = []
            for path, name in RUST_INPUTS:
                prim = self.stage.OverridePrim(path)
                attr = prim.GetAttribute(f"inputs:{name}")
                base = attr.Get() if attr and attr.Get() is not None else 1.0
                self.rust_attrs.append((attr or prim.CreateAttribute(f"inputs:{name}", Sdf.ValueTypeNames.Float), float(base)))
            dog = self.stage.GetPrimAtPath(STATIC_DOG)
            self.dog = UsdGeom.Imageable(dog) if dog.IsValid() else None
        if self.rust_q.changed("rust", self.rust):
            for attr, base in self.rust_attrs:
                attr.Set(round(base * self.rust, 4))
        if self.dog is not None and self.rust_q.changed("dog", float(bool(want))):
            self.dog.MakeInvisible() if want else self.dog.MakeVisible()

    def _path(self, k, r):
        rid = r["id"]
        cur = self.paths.setdefault(rid, {"key": None, "prev": None, "lead": None, "reveal": -1e9, "ghost_t": -1e9})
        path = [tuple(p) for p in r.get("path") or []]
        prev = [tuple(p) for p in r.get("prev_path") or []] or None
        key = tuple(path[1:])
        root = f"{ROOT}/Paths/{rid}"
        if prev != cur["prev"]:
            cur["prev"] = prev
            self.stage.RemovePrim(f"{root}/ghost")
            if prev:
                mat, sh = _emissive(self.stage, f"obst_ghost_{rid}", GHOST)
                cur["ghost_sh"], cur["ghost_t"] = sh, self.now
                UsdGeom.Scope.Define(self.stage, f"{root}/ghost")
                for i, (a, b) in enumerate(segments(prev)):
                    for j, (p, q) in enumerate(dashes(a, b, 0.5, 0.35)):
                        self._cyl(f"{root}/ghost/g{i}_{j}", p, q, self.radius * 0.8, mat, LIFT + 0.02)
        if key != cur["key"]:
            changed = cur["key"] is not None and prev is not None
            cur["key"], cur["path"] = key, path
            cur["reveal"] = self.now if changed else -1e9
            self._draw_new(rid, path, 1.0 if not changed else 0.0)
        elif path:
            cur["path"] = path
            if self.now - cur["reveal"] >= REVEAL_S and cur["lead"] is not None and len(path) > 1:
                self._place(cur["lead"], (path[0][0], path[0][1], path[0][2] + LIFT), (path[1][0], path[1][1], path[1][2] + LIFT))

    def _draw_new(self, rid, path, frac):
        root = f"{ROOT}/Paths/{rid}"
        cur = self.paths[rid]
        self.stage.RemovePrim(f"{root}/new")
        cur["lead"] = None
        if len(path) < 2:
            return
        UsdGeom.Scope.Define(self.stage, f"{root}/new")
        pts = _truncate(path, ease(frac)) if frac < 1.0 else path
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            if math.dist(a, b) > 1e-3:
                c = self._cyl(f"{root}/new/s{i}", a, b, self.radius, self.coral)
                if i == 0:
                    cur["lead"] = c

    def _vapour(self, t: float) -> None:
        lk = self.leak
        if lk is None and self.vap is None:
            return
        from pxr import Vt

        from sim.plume.isaac import PARK as OFF, omnipbr

        if self.vap is None:
            inst = UsdGeom.PointInstancer.Define(self.stage, f"{ROOT}/Vapour")
            protos = []
            for k, a in enumerate(VAPOUR_ALPHA):
                sp = UsdGeom.Sphere.Define(self.stage, f"{ROOT}/Vapour/puff{k}")
                sp.CreateRadiusAttr(1.0)
                _bind(sp.GetPrim(), omnipbr(self.stage, f"/World/Looks/obst_vapour{k}", (0.78, 0.8, 0.82), a, 1.0))
                protos.append(sp.GetPath())
            inst.CreatePrototypesRel().SetTargets(protos)
            self.vap = inst
        src = lk["xyz"] if lk else (0.0, 0.0, OFF[2])
        wx, wy = (lk.get("wind") or (0.0, 0.0)) if lk else (0.0, 0.0)
        pos, scl, idx = [], [], []
        for i in range(VAPOUR_N):
            age = (t + i * VAPOUR_LIFE / VAPOUR_N) % VAPOUR_LIFE
            born = t - age
            live = lk is not None and (born >= self.vap_t if self.vap_on else born < self.vap_t)
            j, k = 0.12 * age, VAPOUR_MS * age + VAPOUR_JET * (1.0 - math.exp(-2.5 * age))
            p = (src[0] + wx * k + j * math.sin(1.7 * i + 0.9 * age),
                 src[1] + wy * k + j * math.cos(2.3 * i + 0.7 * age), src[2] + 0.1 * age + 0.04 * j * math.sin(3.1 * i))
            pos.append(p if live else OFF)
            r = 0.05 + 0.19 * age
            scl.append((r, r, r * 0.8))
            idx.append(min(int(age / VAPOUR_LIFE * len(VAPOUR_ALPHA)), len(VAPOUR_ALPHA) - 1))
        self.vap.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pos]))
        self.vap.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in scl]))
        self.vap.GetProtoIndicesAttr().Set(Vt.IntArray(idx))

    def _weld_sync(self, rid, rb, r) -> None:
        w = r.get("weld") if self.weld_clip is not None and "arm" in rb else None
        cur = rb.get("weld")
        if w is None:
            if cur is not None:
                rb["weld"] = None
                self._torch(rid, rb, False)
            return
        c = float(w["c"])
        if cur is None or cur["id"] != w["id"]:
            cur = rb["weld"] = {"id": w["id"], "t0": self.now - c, "from": rb["arm"].out.copy() if c < W.LEAD_S else None}
            self._torch(rid, rb, True)
        elif abs(self.now - cur["t0"] - c) > 0.5:
            cur["t0"] = self.now - c
        cur.update(at=tuple(float(v) for v in w["at"]), arm=bool(w.get("arm", True)))

    def _torch(self, rid, rb, on: bool) -> None:
        ee = f"{ROOT}/Robots/{rid}/body/{EE_LINK}"
        if "torch" not in rb:
            rb["torch"] = None
            if not self.stage.GetPrimAtPath(ee).IsValid():
                return
            dark = self._mat("obst_torch", (0.08, 0.08, 0.09), 0.5, 0.6)
            copper = self._mat("obst_torch_tip", (0.72, 0.42, 0.2), 0.35, 1.0)
            UsdGeom.Xform.Define(self.stage, f"{ee}/torch")
            for k, (rr, a, b, m) in enumerate(((0.017, 0.0, 0.075, dark), (0.011, 0.075, 0.108, copper),
                                               (0.0035, 0.108, 0.117, copper))):
                c = UsdGeom.Cylinder.Define(self.stage, f"{ee}/torch/part_{k}")
                c.CreateAxisAttr("X")
                c.CreateRadiusAttr(rr)
                c.CreateHeightAttr(b - a)
                UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d((a + b) / 2, 0, 0))
                _bind(c.GetPrim(), m)
            brush = self.stage.GetPrimAtPath(f"{ee}/scrubber_link")
            rb["torch"] = (UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{ee}/torch")),
                           UsdGeom.Imageable(brush) if brush.IsValid() and rid not in GRIP_ROBOTS else None)
        if rb["torch"] is None:
            return
        torch, brush = rb["torch"]
        torch.MakeVisible() if on else torch.MakeInvisible()
        if brush is not None:
            brush.MakeInvisible() if on else brush.MakeVisible()

    def _weld_prims(self) -> dict:
        from pxr import Vt

        root = f"{ROOT}/Weld"
        UsdGeom.Xform.Define(self.stage, root)
        mats = [_emissive(self.stage, "obst_weld_crack", (0.0, 0.0, 0.0))[0]]
        mats += [_emissive(self.stage, f"obst_weld_hot{k}", rgb, e * BEAD_GLOW)[0] for k, (rgb, e) in enumerate(W.HOT)]
        mats += [self._mat("obst_weld_cool", W.COOLED, 0.28, 1.0), mats[0]]
        from sim.plume.isaac import omnipbr

        sparks = [_emissive(self.stage, f"obst_spark{k}", rgb, e)[0] for k, (rgb, e, _) in enumerate(SPARK_LOOK)]
        sparks += [_emissive(self.stage, "obst_arc_core", W.ARC_RGB, CORE_GLOW)[0]]
        sparks += [omnipbr(self.stage, f"/World/Looks/obst_arc_halo{k}", W.ARC_RGB, a, e) for k, (_, a, e) in enumerate(HALO)]
        radii = {"bead": [1.0] * (len(mats) - 1) + [1e-5], "sparks": [r for _, _, r in SPARK_LOOK] + [1.0] * 3}
        out = {"key": None, "on": False, "idx": None, "sp": False}
        for name, ms in (("bead", mats), ("sparks", sparks)):
            inst = UsdGeom.PointInstancer.Define(self.stage, f"{root}/{name}")
            protos = []
            for k, m in enumerate(ms):
                sp = UsdGeom.Sphere.Define(self.stage, f"{root}/{name}/p{k}")
                sp.CreateRadiusAttr(radii[name][k])
                _bind(sp.GetPrim(), m)
                protos.append(sp.GetPath())
            inst.CreatePrototypesRel().SetTargets(protos)
            out[name] = inst
        sc = W.spark_scales()
        out["sparks"].GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(x)) for x in np.r_[sc, 0.007, HALO[0][0], HALO[1][0]]]))
        lights = []
        for name, r, rgb in (("arc", ARC_R, W.ARC_RGB), ("pool", POOL_R, W.POOL_RGB)):
            lt = UsdLux.SphereLight.Define(self.stage, f"{root}/{name}")
            lt.CreateRadiusAttr(r)
            lt.CreateColorAttr(Gf.Vec3f(*rgb))
            lt.CreateIntensityAttr(0.0)
            lights.append((lt.GetIntensityAttr(), UsdGeom.Xformable(lt).AddTranslateOp()))
        out["lights"] = lights
        self.wfx = out
        return out

    def _weld(self, t: float) -> None:
        from pxr import Vt

        rb = next((rb for rb in self.robots.values() if rb.get("weld")), None)
        fx, cl = self.wfx, self.weld_clip
        v3 = lambda a: Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in a])
        if rb is None:
            if fx is not None and fx["on"]:
                for name in ("bead", "sparks"):
                    n = len(fx[name].GetPositionsAttr().Get() or [])
                    fx[name].GetPositionsAttr().Set(v3([W.OFF] * n))
                    fx[name].GetProtoIndicesAttr().Set(Vt.IntArray([0] * n))
                for attr, _ in fx["lights"]:
                    attr.Set(0.0)
                fx.update(on=False, key=None, idx=None, sp=False)
            return
        wd = rb["weld"]
        fx = fx or self._weld_prims()
        at = wd["at"]
        if fx["key"] != (wd["id"], at):
            r = cl.bead_w / 2
            bead = W.to_world(cl.bead - cl.normals(cl.bead) * (0.55 * r)[:, None], at)
            fx["bead"].GetPositionsAttr().Set(v3(np.concatenate([W.to_world(cl.crack - cl.normals(cl.crack) * 0.0006, at), bead])))
            fx["bead"].GetScalesAttr().Set(v3([(W.CRACK_R,) * 3] * len(cl.crack) + [(x, x, x) for x in r]))
            fx.update(key=(wd["id"], at), idx=None, on=True)
        c = t - wd["t0"] - W.LEAD_S
        shown, bins = W.bead_state(cl, c)
        idx = np.concatenate([np.zeros(len(cl.crack), int), np.where(shown, 1 + bins, len(W.HOT) + 2)])
        if fx["idx"] is None or not np.array_equal(idx, fx["idx"]):
            fx["idx"] = idx
            fx["bead"].GetProtoIndicesAttr().Set(Vt.IntArray(idx.tolist()))
        arc = bool(cl.arc(c))
        pos, ages, alive = W.sparks(cl, c, 0.005)
        if alive.any() or arc or fx["sp"]:
            fx["sp"] = bool(alive.any() or arc)
            tip = cl.tip_at(c)
            core = tip - cl.normals(tip[None])[0] * 0.006
            fl = W.arc_flicker(c)
            pts = np.concatenate([np.where(alive[:, None], W.to_world(pos, at), W.OFF),
                                  np.repeat(W.to_world(core[None], at) if arc else [W.OFF], 3, axis=0)])
            fx["sparks"].GetPositionsAttr().Set(v3(np.round(pts, 3)))
            fx["sparks"].GetProtoIndicesAttr().Set(Vt.IntArray(np.r_[ages, 3, 4, 5].astype(int).tolist()))
            p = W.to_world(core, at)
            for (attr, op), w in zip(fx["lights"], (ARC_W * fl, POOL_W * (0.85 + 0.15 * fl))):
                op.Set(Gf.Vec3d(*map(float, p)))
                attr.Set(round(float(w), -2) if arc else 0.0)

    def hide_prims(self, paths) -> None:
        for p in self.hidden:
            if p.IsValid():
                UsdGeom.Imageable(p).MakeVisible()
        self.hidden = [p for p in (self.stage.GetPrimAtPath(str(k)) for k in paths or ()) if p.IsValid() and p.IsA(UsdGeom.Imageable)]
        for p in self.hidden:
            UsdGeom.Imageable(p).MakeInvisible()
        print(f"rig sim: hid {[str(p.GetPath()) for p in self.hidden]}", flush=True)

    def tick(self, t: float) -> None:
        self.now = t
        self._animate(t)
        self._vapour(t)
        self._weld(t)
        for o in self.props.values():
            age = t - o["t0"]
            if "o" in o:
                self._held(o, age)
                continue
            if "mv" in o:
                self._roll(o, t, age)
                continue
            if age <= DROP_S + 0.1:
                o["op"].Set(Gf.Vec3d(o["xyz"][0], o["xyz"][1], o["xyz"][2] + drop_offset(age)))
        for rid, cur in self.paths.items():
            age = t - cur["reveal"]
            if 0 <= age < REVEAL_S + 0.2 and cur.get("path"):
                self._draw_new(rid, cur["path"], min(age / REVEAL_S, 1.0))
            if cur.get("prev"):
                a = ghost_alpha(t - cur["ghost_t"])
                cur["ghost_sh"].GetInput("emissiveColor").Set(Gf.Vec3f(*(c * a for c in GHOST)))
        s = ease((t - self.w_t0) / RAMP_S)
        w = {k: lerp(self.w_from[k], self.w_to[k], s) for k in WEATHER_KEYS}
        self.wave["height"] = w["wave_height"]
        look = (tuple(round(c, 3) for c in w["fog_color"]), round(w["fog_height"], 4), round(w["dome"], -1),
                round(w["sun"]), round(w["fog_dist"], 4))
        if look != self.look:
            st = carb.settings.get_settings()
            if self.look is None:
                st.set_bool("/rtx/fog/enabled", True)
                st.set_bool("/rtx/fog/fogZup/enabled", True)
                st.set_float("/rtx/fog/fogHeightFalloff", 0.12)
                st.set_float("/rtx/fog/fogStartHeight", 0.0)
            self.look = look
            st.set_float_array("/rtx/fog/fogColor", list(look[0]))
            st.set_float("/rtx/fog/fogHeightDensity", look[1])
            st.set_float("/rtx/fog/fogDistanceDensity", look[4])
            dome = UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/Environment/sky"))
            if dome:
                dome.GetIntensityAttr().Set(look[2])
            sun = UsdLux.DistantLight(self.stage.GetPrimAtPath("/World/Environment/sun"))
            if sun:
                sun.GetIntensityAttr().Set(float(look[3]))
        wd = self.w_to["wind_dir"]
        sway = 1.0 + 0.25 * math.sin(t * 3.1) * min(w["wind"], 1.0)
        for op, rest in self.flames:
            dx, dy, dz = flame_offset((rest[2] - self.tip_z + 2.0) * sway, wd, w["wind"])
            op.Set(Gf.Vec3d(rest[0] + dx, rest[1] + dy, rest[2] + dz))

    def set_weather(self, level="calm", wind_dir=None, gust=None) -> None:
        key = (level, wind_dir, gust)
        if key == self.weather_key:
            return
        self.weather_key = key
        cur = {k: lerp(self.w_from[k], self.w_to[k], ease((self.now - self.w_t0) / RAMP_S))
               for k in WEATHER_KEYS}
        self.w_from = {**self.w_to, **cur}
        self.w_to = weather_params(level, wind_dir, gust)
        self.w_t0 = self.now
        dome = UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/Environment/sky"))
        if dome:
            dome.GetTextureFileAttr().Set(rel(SKIES[self.w_to["storm_sky"]]))

    def _roll(self, o, t, age) -> None:
        at, yaw, spin = drum_pose(o["mv"], t, o.get("at"), o["yaw"], o["spin"])
        o["at"], o["yaw"], o["spin"] = at, yaw, spin
        o["op"].Set(Gf.Vec3d(at[0], at[1], at[2] + drop_offset(age)))
        o["rz"].Set(yaw)
        o["spin_op"].Set(spin)

    def _grip(self, root):
        lx, ly, h, w = TOTE[3:]
        UsdGeom.Xform.Define(self.stage, f"{root}/tote").AddTranslateOp().Set(Gf.Vec3d(*TOTE[:3]))
        yellow = self._mat("obst_tote", (0.85, 0.62, 0.08), 0.55)
        for k, (size, at) in enumerate((((lx, ly, w), (0, 0, w / 2)), ((w, ly, h), (lx / 2 - w / 2, 0, h / 2)),
                                        ((w, ly, h), (-lx / 2 + w / 2, 0, h / 2)), ((lx, w, h), (0, ly / 2 - w / 2, h / 2)),
                                        ((lx, w, h), (0, -ly / 2 + w / 2, h / 2)))):
            c = UsdGeom.Cube.Define(self.stage, f"{root}/tote/wall_{k}")
            c.CreateSizeAttr(1.0)
            ops = UsdGeom.Xformable(c)
            ops.AddTranslateOp().Set(Gf.Vec3d(*at))
            ops.AddScaleOp().Set(Gf.Vec3f(*size))
            _bind(c.GetPrim(), yellow)
        ee = f"{root}/body/{EE_LINK}"
        brush = self.stage.GetPrimAtPath(f"{ee}/scrubber_link")
        if brush.IsValid():
            UsdGeom.Imageable(brush).MakeInvisible()
        mats = {"grey": self._mat("obst_cup_grey", (0.45, 0.45, 0.47), 0.4, 0.6), "black": self._mat("obst_cup", (0.03, 0.03, 0.035), 0.7)}
        for k, (r, L, cx, m) in enumerate(CUP):
            c = UsdGeom.Cylinder.Define(self.stage, f"{ee}/grip_{k}")
            c.CreateAxisAttr("X")
            c.CreateRadiusAttr(r)
            c.CreateHeightAttr(L)
            UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(cx, 0, 0))
            _bind(c.GetPrim(), mats[m])

    def _held(self, pr, age) -> None:
        o, ro = pr["o"], self.rollout
        f0, f1 = o["frames"]
        rb = self.robots.get(o["rollout"]) or {}
        if o.get("phase") == "ground" or "root" not in rb:
            x, y, z, yaw = o["anchor"]
            k, lift = f0, drop_offset(age)
        else:
            x, y, z, yaw = rb["root"]
            k, lift = (f1 if o.get("phase") == "tote" else rb.get("ro_k", f0)), -ro.base_z
        _, _, p, q = ro.at(k)
        lift += SODA[0] - ro.radius
        rot = Gf.Matrix4d().SetRotate(Gf.Quatd(*q)) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
        at = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw))).Transform(Gf.Vec3d(p[0], p[1], p[2] + lift))
        pr["op"].Set(rot.SetTranslateOnly(Gf.Vec3d(x, y, z) + at))

    def _spawn(self, o) -> dict:
        path = f"{ROOT}/Props/{o['id']}"
        if "rollout" in o and self.rollout is not None:
            return self._spawn_held(path, o)
        x = UsdGeom.Xform.Define(self.stage, path)
        op = UsdGeom.Xformable(x).AddTranslateOp()
        flat = o["kind"] in ("puddle", "spill")
        op.Set(Gf.Vec3d(o["xyz"][0], o["xyz"][1], o["xyz"][2] + (0.0 if flat else drop_offset(0.0))))
        rz = UsdGeom.Xformable(x).AddRotateZOp()
        rz.Set(float(o.get("yaw", 0.0)) if o["kind"] == "barrel" else float(sum(map(ord, o["id"])) * 37 % 60 - 30))
        if o["kind"] == "barrel":
            xyz = tuple(o["xyz"])
            return {"path": path, "op": op, "rz": rz, "t0": self.now, "xyz": o["xyz"], "yaw": float(o.get("yaw", 0.0)),
                    "spin": 0.0, "spin_op": self._build_barrel(path, o), "at": xyz,
                    "mv": {"p": xyz, "v": tuple(o.get("vel") or (0.0, 0.0)), "t": self.now, "from": xyz}}
        getattr(self, f"_build_{o['kind']}", self._build_debris)(path, o)
        return {"path": path, "op": op, "t0": self.now - (10.0 if flat else 0.0), "xyz": o["xyz"]}

    def _spawn_held(self, path, o) -> dict:
        UsdGeom.Xform.Define(self.stage, path)
        op = UsdGeom.Xformable(UsdGeom.Xform.Define(self.stage, f"{path}/can")).AddTransformOp()
        r, h = SODA
        paint = self._mat("obst_soda_red", (0.9, 0.035, 0.02), 0.3, 0.3)
        white = self._mat("obst_soda_white", (0.92, 0.92, 0.9), 0.35)
        tin = self._mat("obst_can_tin", (0.7, 0.7, 0.68), 0.25, 0.95)
        if o.get("lamp"):
            lamp = UsdLux.SphereLight.Define(self.stage, f"{path}/lamp")
            lamp.CreateRadiusAttr(0.4)
            lamp.CreateIntensityAttr(float(o["lamp"][3]))
            lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.82))
            UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(*o["lamp"][:3]))
        for name, rr, hh, z, mat in (("body", r, h, 0.0, paint), ("band", r * 1.01, h * 0.3, 0.0, white),
                                     ("rim_0", r * 1.04, 0.01, -h / 2 + 0.005, tin), ("rim_1", r * 1.04, 0.01, h / 2 - 0.005, tin)):
            c = UsdGeom.Cylinder.Define(self.stage, f"{path}/can/{name}")
            c.CreateAxisAttr("Z")
            c.CreateRadiusAttr(rr)
            c.CreateHeightAttr(hh)
            UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
            _bind(c.GetPrim(), mat)
        pr = {"path": path, "op": op, "t0": self.now, "xyz": o["xyz"], "o": o}
        self._held(pr, 0.0)
        return pr

    def _liquid(self, name, color, opacity, ior=1.33):
        if name in self.b.materials:
            return self.b.materials[name]
        mat, sh = self.b._surface(name)
        for k, t, v in (("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
                        ("roughness", Sdf.ValueTypeNames.Float, 0.015), ("metallic", Sdf.ValueTypeNames.Float, 0.0),
                        ("ior", Sdf.ValueTypeNames.Float, ior), ("clearcoat", Sdf.ValueTypeNames.Float, 1.0),
                        ("clearcoatRoughness", Sdf.ValueTypeNames.Float, 0.0), ("opacity", Sdf.ValueTypeNames.Float, opacity)):
            sh.CreateInput(k, t).Set(v)
        return mat

    def _blob(self, path, name, r, seed, z, mat, n=48):
        pts, nrm, uvs, idx = [(0.0, 0.0, z)], [(0, 0, 1)], [(0.5, 0.5)], []
        for k in range(n):
            a = 2 * math.pi * k / n
            rr = r * puddle_edge(a, seed)
            pts.append((rr * math.cos(a), rr * math.sin(a), z))
            nrm.append((0, 0, 1))
            uvs.append((0.5 + rr * math.cos(a) / (2 * r), 0.5 + rr * math.sin(a) / (2 * r)))
            idx.extend((0, k + 1, (k + 1) % n + 1))
        return self.b._mesh(self.b.unique(path, name), pts, nrm, uvs, [3] * n, idx, mat, None)

    def _build_puddle(self, path, o):
        import photoreal

        photoreal.make_puddle(self.stage, f"{path}/water", (0.0, 0.0, 0.0), o.get("radius", 0.8) * 1.1,
                              seed=sum(map(ord, o["id"])), mdl_path=rel(photoreal.PUDDLE_MDL),
                              rain=float(self.w_to.get("rain", 0.0)))

    def _build_can(self, path, o):
        r, h = CAN_SIZE.get(o.get("size", "small"), CAN_SIZE["small"])
        c = o.get("colour", "red")
        paint = self._mat(f"obst_can_{c}", CAN_PAINT.get(c, CAN_PAINT["red"]), 0.38, 0.35)
        tin = self._mat("obst_can_tin", (0.7, 0.7, 0.68), 0.25, 0.95)
        body = UsdGeom.Cylinder.Define(self.stage, f"{path}/body")
        body.CreateAxisAttr("Z")
        body.CreateRadiusAttr(r)
        body.CreateHeightAttr(h)
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0, 0, h / 2))
        _bind(body.GetPrim(), paint)
        for k, z in enumerate((0.006, h - 0.006)):
            rim = UsdGeom.Cylinder.Define(self.stage, f"{path}/rim_{k}")
            rim.CreateAxisAttr("Z")
            rim.CreateRadiusAttr(r * 1.04)
            rim.CreateHeightAttr(0.012)
            UsdGeom.Xformable(rim).AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
            _bind(rim.GetPrim(), tin)
        self.b.tube(path, "handle", (-r * 0.6, 0, h + 0.012), (r * 0.6, 0, h + 0.012), 0.008, tin, segments=8)

    def _build_barrel(self, path, o):
        paint = self._mat("obst_drum_paint", (0.2, 0.035, 0.025), 0.62, 0.35)
        rim = self._tex("rusty_metal_02", self._mat("obst_drum_rim", (0.16, 0.15, 0.14), 0.6, 0.8))
        lid = self._mat("obst_drum_lid", (0.17, 0.035, 0.025), 0.7, 0.3)
        band = self._mat("obst_drum_band", (0.42, 0.38, 0.3), 0.75, 0.05)
        spin = UsdGeom.Xform.Define(self.stage, f"{path}/spin")
        ops = UsdGeom.Xformable(spin)
        ops.AddTranslateOp().Set(Gf.Vec3d(0, 0, DRUM_R))
        sop = ops.AddRotateYOp()
        sop.Set(0.0)
        drum = UsdGeom.Xform.Define(self.stage, f"{path}/spin/drum")
        UsdGeom.Xformable(drum).AddRotateZOp().Set(90.0)
        d, h = f"{path}/spin/drum", 0.44
        self.b.tube(d, "shell", (-h, 0, 0), (h, 0, 0), DRUM_R - 0.006, paint, segments=40)
        for x in (-0.29, 0.29):
            self.b.tube(d, "hoop", (x - 0.018, 0, 0), (x + 0.018, 0, 0), DRUM_R + 0.008, rim, segments=40)
        for x in (-h, h):
            self.b.tube(d, "chime", (x - 0.02, 0, 0), (x + 0.004, 0, 0), DRUM_R + 0.004, rim, segments=40)
        self.b.tube(d, "band", (-0.08, 0, 0), (0.08, 0, 0), DRUM_R - 0.002, band, segments=40)
        self.b.disc(d, "lid", (-h + 0.012, 0, 0), -1, DRUM_R - 0.01, lid)
        self.b.disc(d, "lid", (h - 0.012, 0, 0), 1, DRUM_R - 0.01, lid)
        self.b.tube(d, "bung", (h - 0.012, 0.16, 0.0), (h + 0.01, 0.16, 0.0), 0.03, rim, segments=12)
        return sop

    def _mat(self, name, color, rough=0.6, metal=0.0):
        return self.b.plain(name, color, rough, metal)

    def _tex(self, name, fallback):
        return self.b.textured(name) if (ASSETS / "textures" / name).exists() else fallback

    def _asset(self, path, asset_id, pos, rz=0.0, scale=1.0):
        if (ASSETS / "models" / asset_id).exists():
            h = self.b.asset(path, asset_id, pos, rotate_z=rz, scale=scale)
            for prim in Usd.PrimRange(h.GetPrim()):
                for api in ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"):
                    if prim.HasAPI(api):
                        prim.RemoveAPI(api)
            return h

    def _build_crate(self, path, o):
        wood = self._mat("obst_wood", (0.42, 0.27, 0.14), 0.85)
        dark = self._mat("obst_wood_dark", (0.24, 0.15, 0.08), 0.9)
        for i, y in enumerate((-0.45, 0.0, 0.45)):
            self.b.box(path, "pallet_board", (0, y, 0.12), (1.2, 0.16, 0.04), wood, collide=None)
        for x in (-0.5, 0.0, 0.5):
            self.b.box(path, "pallet_block", (x, 0, 0.05), (0.12, 1.1, 0.1), dark, collide=None)
        self.b.box(path, "crate", (0, 0, 0.6), (1.1, 1.0, 0.9), wood, collide=None)
        for z in (0.25, 0.95):
            self.b.box(path, "slat", (0, 0, z), (1.14, 1.04, 0.1), dark, collide=None)
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.b.box(path, "corner", (sx * 0.54, sy * 0.49, 0.6), (0.08, 0.08, 0.92), dark, collide=None)
        self.b.box(path, "crate_top", (0.1, -0.05, 1.32), (0.8, 0.7, 0.55), wood, collide=None)

    def _build_spill(self, path, o):
        r = o.get("radius", 2.0) * 0.8
        self._blob(path, "slick", r, sum(map(ord, o["id"])), 0.008, self._liquid("obst_oil", (0.012, 0.008, 0.004), 1.0))
        orange = self._mat("obst_orange", (0.95, 0.35, 0.03), 0.45)
        white = self._mat("obst_white", (0.9, 0.9, 0.88), 0.5)
        for k, ang in enumerate((0, 120, 240)):
            a = math.radians(ang)
            px, py = (r + 0.3) * math.cos(a), (r + 0.3) * 0.75 * math.sin(a)
            self.b.tube(path, "cone", (px, py, 0.0), (px, py, 0.7), 0.12, orange, segments=12)
            self.b.tube(path, "band", (px, py, 0.4), (px, py, 0.5), 0.125, white, segments=12)
        self._asset(path, "WetFloorSign_01", (0.0, -(r + 0.4) * 0.75, 0.0), 0.0)

    def _build_debris(self, path, o):
        steel = self._tex("rusty_metal_02", self._mat("obst_steel", (0.35, 0.33, 0.3), 0.6, 0.8))
        for k, (x, y, L, ang) in enumerate(((0, 0, 1.6, 20), (0.3, -0.4, 1.2, -50), (-0.5, 0.3, 0.9, 75))):
            a = math.radians(ang)
            self.b.tube(path, "pipe", (x - L / 2 * math.cos(a), y - L / 2 * math.sin(a), 0.1 + 0.08 * k),
                        (x + L / 2 * math.cos(a), y + L / 2 * math.sin(a), 0.1 + 0.08 * k), 0.08, steel, segments=14)
        self.b.box(path, "plate", (0.2, 0.3, 0.05), (0.9, 0.6, 0.04), steel, collide=None)

    def _build_scaffold(self, path, o):
        steel = self._mat("obst_galv", (0.62, 0.64, 0.66), 0.35, 0.9)
        plank = self._mat("obst_plank", (0.55, 0.42, 0.22), 0.8)
        h, s = 3.2, 0.75
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.b.tube(path, "leg", (sx * s, sy * s, 0), (sx * s, sy * s, h), 0.03, steel, segments=10)
        for z in (0.1, 1.6, 3.1):
            for (ax, ay), (bx, by) in (((-s, -s), (s, -s)), ((s, -s), (s, s)), ((s, s), (-s, s)), ((-s, s), (-s, -s))):
                self.b.tube(path, "ledger", (ax, ay, z), (bx, by, z), 0.025, steel, segments=8)
        self.b.tube(path, "brace", (-s, -s, 0.1), (s, -s, 1.6), 0.02, steel, segments=8)
        self.b.tube(path, "brace", (s, s, 1.6), (-s, s, 3.1), 0.02, steel, segments=8)
        self.b.box(path, "deck", (0, 0, 1.62), (1.5, 1.5, 0.05), plank, collide=None)
        self.b.box(path, "deck", (0, 0, 3.12), (1.5, 1.5, 0.05), plank, collide=None)

    def _build_hose(self, path, o):
        rubber = self._mat("obst_hose", (0.85, 0.62, 0.05), 0.55)
        for ring, (rad, z) in enumerate(((0.45, 0.05), (0.38, 0.13), (0.45, 0.21), (0.31, 0.29))):
            n = 16
            for k in range(n):
                a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
                self.b.tube(path, "coil", (rad * math.cos(a0), rad * math.sin(a0), z),
                            (rad * math.cos(a1), rad * math.sin(a1), z), 0.04, rubber, segments=8)
        self.b.tube(path, "tail", (0.45, 0.0, 0.05), (1.6, 0.4, 0.04), 0.04, rubber, segments=8)
        self.b.tube(path, "tail", (1.6, 0.4, 0.04), (2.3, -0.3, 0.04), 0.04, rubber, segments=8)

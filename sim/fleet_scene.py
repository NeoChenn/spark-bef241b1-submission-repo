"""Compose the full rig (plus the cinematic sky/ocean layer) with the robot fleet, job markers and gas plumes.

Import inside Isaac Sim after SimulationApp has started. The fleet lives on the session's root layer under
/World/Fleet, so sim/output/full_rig.usda and cinematic.usda are never modified. Used by fleet_replay.py;
`python.bat sim/fleet_scene.py` renders a few fixed preview stills for checking placement.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "sim", Path(os.environ.get("ISAAC_DEPS", Path.home() / ".isaac-deps"))):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Render fleet placement previews")
    _parser.add_argument("--timeline", default="")
    _parser.add_argument("--width", type=int, default=960)
    _parser.add_argument("--height", type=int, default=540)
    _parser.add_argument("--subframes", type=int, default=8)
    _parser.add_argument("--preset", default="golden")
    _parser.add_argument("--only", default="")
    ARGS = _parser.parse_args()
    from isaacsim import SimulationApp

    APP = SimulationApp({"headless": True, "width": ARGS.width, "height": ARGS.height})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import fleet_models as fm  # noqa: E402
from fleet_replay import (OUT_DIR, TOP_DECK, Timeline, active_jobs, deck_height, detector_states, fixed_detectors,
                          gas_plumes, look_at, state_color)
from scene_builder import OUTPUT  # noqa: E402

FULL_RIG = OUTPUT / "full_rig.usda"
CINEMATIC = OUTPUT / "cinematic.usda"
ROOT = "/World/Fleet"
CAMERA_PATH = f"{ROOT}/camera"
MAX_MARKERS = 6
MAX_PLUMES = 2
PUFFS = 10
BULB_CLEARANCE = 9.0
WIND = np.array([0.8, -0.45, 0.0])          # gas drifts downwind, rising slowly


def ring_mesh(stage, path, inner: float, outer: float, segments: int = 64):
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    pts = [(outer * math.cos(x), outer * math.sin(x), 0.0) for x in a] + \
          [(inner * math.cos(x), inner * math.sin(x), 0.0) for x in a]
    faces, counts = [], []
    for k in range(segments):
        n = (k + 1) % segments
        faces += [k, n, segments + n, segments + k]
        counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(pts)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


class Marker:
    """Job marker: a thin glowing ring on the deck at the target and a faint vertical beam above it."""

    def __init__(self, stage, index: int):
        self.path = f"{ROOT}/Markers/marker_{index}"
        self.xform = UsdGeom.Xform.Define(stage, self.path)
        xf = UsdGeom.Xformable(self.xform)
        self.translate = xf.AddTranslateOp()
        self.scale = xf.AddScaleOp()
        self.ring_mat = fm.material(stage, f"marker_ring_{index}", (0.2, 0.8, 1.0), 0.4, emissive=(0.2, 0.6, 0.9))
        self.beam_mat = fm.translucent(stage, f"marker_beam_{index}", (0.2, 0.8, 1.0), (0.2, 0.8, 1.0), 400.0, 0.18)
        fm.bind(ring_mesh(stage, f"{self.path}/ring", 0.94, 1.0).GetPrim(), self.ring_mat)
        fm.bind(ring_mesh(stage, f"{self.path}/ring_inner", 0.5, 0.52).GetPrim(), self.ring_mat)
        fm.cylinder(stage, f"{self.path}/beam", (0, 0, 7.5), 0.035, 15.0, self.beam_mat)
        self.visible = True
        self.hide()
        self._color = None

    def hide(self) -> None:
        if self.visible:
            UsdGeom.Imageable(self.xform).MakeInvisible()
            self.visible = False

    def show(self, target: np.ndarray, color, t: float, flying_target: bool) -> None:
        z = deck_height(target[2]) + 0.03 if not flying_target else target[2]
        self.translate.Set(Gf.Vec3d(float(target[0]), float(target[1]), float(z)))
        pulse = 1.0 + 0.08 * math.sin(t * 3.0)
        self.scale.Set(Gf.Vec3f(pulse, pulse, 1.0))
        if color != self._color:
            c = np.array(color, float) / 255.0
            fm.set_emissive(self.ring_mat[1], c * 0.9)
            self.ring_mat[1].GetInput("diffuseColor").Set(Gf.Vec3f(*c))
            fm.set_emissive(self.beam_mat[1], c)
            self.beam_mat[1].GetInput("diffuse_color_constant").Set(Gf.Vec3f(*c))
            self._color = color
        if not self.visible:
            UsdGeom.Imageable(self.xform).MakeVisible()
            self.visible = True


class Plume:
    """Translucent gas cloud: puffs that leave the leak point, rise, drift downwind, swell and fade."""

    def __init__(self, stage, index: int):
        self.path = f"{ROOT}/Gas/plume_{index}"
        UsdGeom.Xform.Define(stage, self.path)
        self.puffs = []
        for k in range(PUFFS):
            mat = fm.translucent(stage, f"gas_{index}_{k}", (0.7, 0.9, 0.5), (0.75, 1.0, 0.45), 600.0, 0.0)
            s = UsdGeom.Sphere.Define(stage, f"{self.path}/puff_{k}")
            s.CreateRadiusAttr(1.0)
            xf = UsdGeom.Xformable(s)
            fm.bind(s.GetPrim(), mat)
            self.puffs.append((s, xf.AddTranslateOp(), xf.AddScaleOp(), mat[1]))
        self.visible = True
        self.hide()

    def hide(self) -> None:
        if self.visible:
            UsdGeom.Imageable(UsdGeom.Xform.Get(self.puffs[0][0].GetPrim().GetStage(), self.path)).MakeInvisible()
            self.visible = False

    def show(self, origin: np.ndarray, strength: float, t: float) -> None:
        for k, (s, translate, scale, shader) in enumerate(self.puffs):
            phase = (t * 0.22 + k / PUFFS) % 1.0
            drift = WIND * 5.0 * phase + np.array([0.0, 0.0, 2.6 * phase ** 0.8])
            wobble = np.array([math.sin(k * 2.3 + t), math.cos(k * 1.7 + t * 0.8), 0.0]) * 0.25 * phase
            r = 0.18 + 1.3 * phase
            translate.Set(Gf.Vec3d(*map(float, origin + drift + wobble)))
            scale.Set(Gf.Vec3f(r, r, r * 0.8))
            alpha = strength * 0.55 * min(phase * 6.0, 1.0) * (1.0 - phase) ** 1.5
            fm.set_opacity(shader, alpha)
        if not self.visible:
            UsdGeom.Imageable(UsdGeom.Xform.Get(self.puffs[0][0].GetPrim().GetStage(), self.path)).MakeVisible()
            self.visible = True


class FleetScene:
    def __init__(self, stage, timeline: Timeline, preset: Optional[str]):
        self.stage, self.timeline, self.preset = stage, timeline, preset
        UsdGeom.Scope.Define(stage, ROOT)
        UsdGeom.Scope.Define(stage, f"{ROOT}/Markers")
        UsdGeom.Scope.Define(stage, f"{ROOT}/Gas")
        self.robots = {rid: fm.build_robot(stage, rid, tr.kind, k, ROOT)
                       for k, (rid, tr) in enumerate(timeline.tracks.items())}
        UsdGeom.Scope.Define(stage, f"{ROOT}/Detectors")
        self.detectors = {d["id"]: fm.FixedDetector(stage, d["id"], d["position"], d["deck_z"], d["type"], ROOT)
                          for d in fixed_detectors()}
        self.markers = [Marker(stage, k) for k in range(MAX_MARKERS)]
        self.plumes = [Plume(stage, k) for k in range(MAX_PLUMES)]
        self.camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 60000.0))
        self.camera.CreateFocalLengthAttr(28.0)
        self.camera.CreateHorizontalApertureAttr(20.955)
        self.camera.CreateVerticalApertureAttr(20.955 * 9 / 16)
        self.cam_op = UsdGeom.Xformable(self.camera).AddTransformOp()
        self.ocean = None
        # Deck lamp bulbs (cinematic layer) that end up right in front of a follow camera are hidden, light kept.
        self.bulbs = []
        lamps = stage.GetPrimAtPath("/World/Cinematic/Lamps")
        if lamps.IsValid():
            for lamp in lamps.GetChildren():
                bulb = lamp.GetChild("bulb")
                if bulb.IsValid():
                    p = UsdGeom.Xformable(lamp).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                    self.bulbs.append([UsdGeom.Imageable(bulb), np.array(p), True])

    # ---------------------------------------------------------------------------------- loading
    @classmethod
    def open(cls, app, timeline: Timeline, preset: Optional[str] = "golden", ocean: bool = True) -> "FleetScene":
        import carb.settings
        import omni.usd

        path = CINEMATIC if preset and CINEMATIC.exists() else FULL_RIG
        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(path)):
            raise SystemExit(f"could not open {path}")
        for _ in range(900):
            app.update()
            if ctx.get_stage_loading_status()[2] == 0:
                break
        stage = ctx.get_stage()
        for _ in range(30):
            app.update()
        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        if path == CINEMATIC:
            import cinematic
            cinematic.apply_preset(stage, preset)
        stage.SetEditTarget(stage.GetSessionLayer())     # nothing we add is ever saved into the rig files
        scene = cls(stage, timeline, preset if path == CINEMATIC else None)
        if path == CINEMATIC and ocean:
            import ocean as ocean_mod
            rest, distance, _, _ = ocean_mod.make_grid()
            scene.ocean = (rest, distance)
        for _ in range(20):
            app.update()
        return scene

    # ----------------------------------------------------------------------------------- update
    def update(self, t: float, assets: dict) -> None:
        for rid, robot in self.robots.items():
            robot.update(self.timeline.tracks[rid], t)
        jobs = active_jobs(self.timeline, t, assets)
        for k, marker in enumerate(self.markers):
            if k < len(jobs):
                j = jobs[k]
                kind = self.timeline.tracks[j["robot"]].kind
                flying = kind == "A" and j["target"][2] > TOP_DECK + 3.0
                marker.show(j["target"], state_color(j["state"]), t, flying)
            else:
                marker.hide()
        alarms = detector_states(self.timeline, t)
        for did, det in self.detectors.items():
            det.set_state(alarms.get(did), t)
        plumes = gas_plumes(self.timeline, t)
        for k, plume in enumerate(self.plumes):
            if k < len(plumes):
                plume.show(plumes[k]["position"], plumes[k]["strength"], t)
            else:
                plume.hide()
        if self.preset:
            import cinematic
            cinematic.animate_effects(self.stage, t)
            if self.ocean is not None:
                cinematic.animate_ocean(self.stage, *self.ocean, t, cinematic.PRESETS[self.preset]["wave_height"])

    def set_camera(self, eye, target, focal: float) -> None:
        self.camera.GetFocalLengthAttr().Set(float(focal))
        self.cam_op.Set(fm.matrix(look_at(eye, target)))
        eye = np.asarray(eye, float)
        for bulb in self.bulbs:
            show = np.linalg.norm(bulb[1] - eye) > BULB_CLEARANCE
            if show != bulb[2]:
                bulb[0].MakeVisible() if show else bulb[0].MakeInvisible()
                bulb[2] = show

    def tag_point(self, rid: str, t: float) -> np.ndarray:
        return self.robots[rid].tag_point()


# ===================================================================== preview stills (placement)
PREVIEWS = {
    # name: (time, eye, target, focal)
    "deck_overview": (25.0, (-45.0, -30.0, 75.0), (-8.0, 12.0, 43.0), 22.0),
    "q_gauge_row": (16.0, (-14.0, 13.5, 46.0), (-18.0, 20.8, 42.8), 28.0),
    "q2_valve": (38.0, (-11.5, 15.0, 45.5), (-15.5, 20.8, 43.0), 30.0),
    "crawler": (20.0, (21.0, -4.0, 44.3), (18.5, -9.0, 42.6), 35.0),
    "drone_leak": (24.0, (-4.0, 36.0, 47.0), (-14.0, 25.0, 44.5), 26.0),
    "helideck": (0.5, (-8.0, -38.0, 60.0), (-22.0, -22.0, 51.5), 30.0),
}


def probe(stage, points) -> None:
    """Print the solid geometry above each (x, y): what a robot there stands on and flies under."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    hits = {p: [] for p in points}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if prim.GetPath().pathString.startswith(ROOT) or not prim.IsA(UsdGeom.Gprim):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = box.GetMin(), box.GetMax()
        for (x, y) in points:
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and hi[2] > 40.0 and (hi[0] - lo[0]) < 60:
                hits[(x, y)].append((round(lo[2], 2), round(hi[2], 2), prim.GetPath().pathString[-48:]))
    for p, h in hits.items():
        print(f"PROBE {p}: " + "; ".join(f"{a}-{b} {n}" for a, b, n in sorted(h)[:14]), flush=True)


def main() -> None:
    args, app = ARGS, APP
    import omni.kit.app
    from PIL import Image
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep

    import fleet_replay as fr
    path = Path(args.timeline) if args.timeline else fr.DEFAULT_TIMELINE
    if not path.exists():
        fr.write_fake_timeline(path)
    timeline = fr.load_timeline(path)
    assets = fr.rig_assets()
    scene = FleetScene.open(app, timeline, None if args.preset == "none" else args.preset, ocean=True)
    stage = scene.stage
    heli = stage.GetPrimAtPath("/World/Details/helideck_circle")
    if heli.IsValid():
        box = UsdGeom.Imageable(heli).ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
        print(f"HELIDECK_CIRCLE min={box.GetMin()} max={box.GetMax()}", flush=True)
    product = rep.create.render_product(CAMERA_PATH, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe(stage, [(-14.0, 21.5), (-14.0, 24.3), (-14.0, 28.0), (-17.5, 20.5), (18.0, -9.0)])
    names = [n for n in PREVIEWS if not args.only or n in args.only.split(",")]
    for name in names:
        t, eye, target, focal = PREVIEWS[name]
        scene.update(t, assets)
        scene.set_camera(np.array(eye), np.array(target), focal)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        rgb = np.asarray(annotator.get_data())[..., :3].astype(np.uint8)
        cam = look_at(eye, target)
        tags = {}
        for rid in timeline.tracks:
            px = fr.project([scene.tag_point(rid, t)], cam, focal, args.width, args.height)[0]
            if px is not None:
                tags[rid] = px
        rgb = fr.draw_fleet_hud(rgb, timeline, t, fr.Shot(t, t + 1, "follow"), tags)
        out = OUT_DIR / f"preview_{name}.png"
        Image.fromarray(rgb).save(out)
        print(f"PREVIEW {out} t={t}", flush=True)
        for rid, robot in scene.robots.items():
            print(f"   {rid} at {np.round(robot.last_position, 2)}", flush=True)
    rep.orchestrator.wait_until_complete()
    app.close()


if __name__ == "__main__":
    main()
    os._exit(0)

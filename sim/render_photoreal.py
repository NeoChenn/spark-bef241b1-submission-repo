from __future__ import annotations

import argparse
import json
import os
import sys
import time as clock

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="night")
parser.add_argument("--scene", default="merged")
parser.add_argument("--modes", default="before,after")
parser.add_argument("--quality", default="offline")
parser.add_argument("--renderer", default="PathTracing")
parser.add_argument("--spp", type=int, default=256)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--views", default="wide,deck_mid,repair,waterline")
parser.add_argument("--prefix", default="pt")
parser.add_argument("--out", default="/rig/runs/photoreal")
parser.add_argument("--fps-seconds", type=float, default=0.0)
parser.add_argument("--variants", default="")
args = parser.parse_args()
modes = [m for m in args.modes.split(",") if m]
os.environ["RIG_PHOTOREAL"] = "1" if modes[0] == "after" else "0"
os.environ["RIG_QUALITY"] = args.quality

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})

import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

sys.path[:0] = ["/rig", "/rig/sim"]
import cinematic
import ocean
from rig.sim.zoom import ZoomPath

VIEWS = {
    "wide": ((270.0, -200.0, 110.0), (4.0, -4.0, 30.0), 28.0),
    "deck_mid": ((25.25, 3.25, 47.0), (18.25, -5.75, 43.5), 22.0),
    "waterline": ((75.0, -105.0, 4.0), (2.0, -8.0, 24.0), 18.0),
    "topdeck": ((30.0, -36.0, 52.0), (-6.0, 4.0, 44.0), 20.0),
    "laydown": ((31.0, -9.6, 44.6), (17.5, -12.0, 43.1), 22.0),
}

settings = carb.settings.get_settings()
base = {"/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False, "/rtx/post/lensFlares/enabled": False}
if args.renderer == "PathTracing":
    base.update({"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 16, "/rtx/pathtracing/totalSpp": args.spp,
                 "/rtx/pathtracing/maxBounces": 8, "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 8,
                 "/rtx/pathtracing/optixDenoiser/enabled": True})
else:
    base.update({"/rtx/rtpt/spp": 1, "/rtx/post/dlss/execMode": 2})
for key, value in base.items():
    settings.set(key, value)

scene = cinematic.SCENE_PATH.with_name("rig_merged.usda") if args.scene == "merged" else cinematic.SCENE_PATH
context = omni.usd.get_context()
context.open_stage(str(scene))


def settle(limit=900):
    for _ in range(limit):
        simulation_app.update()
        if context.get_stage_loading_status()[2] == 0:
            break


settle()
stage = context.get_stage()
path = "/World/Cinematic/Cameras/photoreal"
camera = UsdGeom.Camera.Define(stage, path)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60000))
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateFocalLengthAttr(28.0)
op = UsdGeom.Xformable(camera).AddTransformOp()
op.Set(Gf.Matrix4d(1))
rest, distance, _, _ = ocean.make_grid()
height = cinematic.PRESETS[args.preset]["wave_height"]
meta = json.loads(scene.with_name("rig_merged_meta.json").read_text())
zoom = ZoomPath(wide_target=tuple(meta["zoom"]["wide_target"]), close_target=tuple(meta["zoom"]["close_target"]))
VIEWS["repair"] = zoom.at(zoom.orbit_s + zoom.descend_s + 1.0)
VIEWS["approach"] = zoom.at(zoom.orbit_s + 0.62 * zoom.descend_s)
product = rep.create.render_product(path, (args.width, args.height))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach(product)
omni.timeline.get_timeline_interface().play()


def look(eye, target, focal):
    camera.GetFocalLengthAttr().Set(float(focal))
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse())


def shoot(tag, name):
    start = clock.monotonic()
    if args.renderer == "PathTracing":
        rep.orchestrator.step(rt_subframes=max(args.spp // 16, 1) + 2, pause_timeline=False)
    else:
        for _ in range(40):
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
    data = np.asarray(annotator.get_data())
    out = f"{args.out}/{tag}_{name}.png"
    Image.fromarray(np.ascontiguousarray(data[:, :, :3], dtype=np.uint8)).save(out)
    print(f"VIEW {tag} {name} mean={data[:, :, :3].mean():.1f} s={clock.monotonic() - start:.1f}", flush=True)


def measure(tag):
    start = clock.monotonic()
    fresh, last, frames = 0, b"", 0
    while clock.monotonic() - start < args.fps_seconds:
        t = clock.monotonic() - start
        cinematic.animate_ocean(stage, rest, distance, t, height)
        cinematic.animate_effects(stage, t)
        look(*zoom.at(t * 2.0))
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)
        data = np.asarray(annotator.get_data())
        sample = data[::24, ::24, :3].tobytes() if data.ndim == 3 and data.size else b""
        fresh += bool(sample) and sample != last
        last = sample or last
        frames += 1
    elapsed = clock.monotonic() - start
    print(f"FPS {tag} fresh={fresh / elapsed:.2f} steps={frames / elapsed:.2f} quality={args.quality}", flush=True)


for mode in modes:
    os.environ["RIG_PHOTOREAL"] = "1" if mode == "after" else "0"
    t0 = clock.monotonic()
    cinematic.apply_preset(stage, args.preset)
    settings.set("/rtx/post/lensFlares/enabled", False)
    settle(300)
    for _ in range(30):
        simulation_app.update()
    cinematic.animate_ocean(stage, rest, distance, 3.0, height)
    cinematic.animate_effects(stage, 3.0)
    tag = f"{args.prefix}_{mode}"
    print(f"MODE {tag} preset={args.preset} quality={args.quality} apply_s={clock.monotonic() - t0:.1f}", flush=True)
    for name in (v for v in args.views.split(",") if v):
        look(*VIEWS[name])
        shoot(tag, name)
    if args.fps_seconds > 0:
        measure(tag)


def set_inputs(name, value):
    from pxr import Sdf

    import photoreal

    for shader in photoreal.rig_shaders(stage):
        if shader.GetSourceAssetSubIdentifier("mdl") == "rig_weathered":
            shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)


VARIANTS = {
    "full": lambda: set_inputs("lite", False),
    "lite": lambda: set_inputs("lite", True),
    "nocoat": lambda: set_inputs("wet_coat", False),
    "nodress": lambda: stage.GetPrimAtPath("/World/Photoreal").SetActive(False),
    "coat": lambda: set_inputs("wet_coat", True),
    "off": lambda: stage.GetRootLayer().subLayerPaths.remove("photoreal.usda"),
    "on": lambda: stage.GetRootLayer().subLayerPaths.insert(0, "photoreal.usda"),
}
for variant in (v for v in args.variants.split(",") if v):
    VARIANTS[variant]()
    settle(300)
    for _ in range(60):
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
    measure(f"{args.prefix}_{variant}")
os._exit(0)

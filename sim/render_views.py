from __future__ import annotations

import argparse
import os
import time as clock

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="night")
parser.add_argument("--out", default="/rig/runs/modules")
parser.add_argument("--times", default="0,6,12,16,20,23")
parser.add_argument("--settle", type=int, default=40)
parser.add_argument("--fps-seconds", type=float, default=12.0)
parser.add_argument("--tag", default="v")
parser.add_argument("--extras", default="lower,middle,topdeck,west")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": 1280, "height": 720})

import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

import cinematic
import ocean

EXTRA = {
    "lower": ((48.0, -52.0, 24.0), (12.0, -14.0, 26.0), 20.0),
    "middle": ((58.0, -30.0, 38.0), (20.0, 0.0, 35.0), 20.0),
    "topdeck": ((30.0, -36.0, 52.0), (-6.0, 4.0, 44.0), 20.0),
    "west": ((-60.0, 10.0, 34.0), (-18.0, -4.0, 32.0), 22.0),
}

settings = carb.settings.get_settings()
for key, value in {
    "/rtx/post/motionblur/enabled": False,
    "/rtx/post/dof/enabled": False,
    "/rtx/post/lensFlares/enabled": False,
    "/rtx/rtpt/spp": 1,
    "/rtx/rtpt/maxBounces": 6,
    "/rtx/rtpt/maxSpecularAndTransmissionBounces": 6,
    "/rtx/post/dlss/execMode": 2,
}.items():
    settings.set(key, value)

context = omni.usd.get_context()
context.open_stage(str(cinematic.SCENE_PATH))
for _ in range(900):
    simulation_app.update()
    if context.get_stage_loading_status()[2] == 0:
        break
stage = context.get_stage()
path = "/World/Cinematic/Cameras/stream"
camera = UsdGeom.Camera.Define(stage, path)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.5, 60000))
camera.CreateFocalLengthAttr(28.0)
op = UsdGeom.Xformable(camera).AddTransformOp()
op.Set(Gf.Matrix4d(1))
cinematic.apply_preset(stage, args.preset)
settings.set("/rtx/post/lensFlares/enabled", False)
rest, distance, _, _ = ocean.make_grid()
height = cinematic.PRESETS[args.preset]["wave_height"]
product = rep.create.render_product(path, (1280, 720))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach(product)
omni.timeline.get_timeline_interface().play()
for _ in range(30):
    simulation_app.update()


def look(eye, target, focal):
    camera.GetFocalLengthAttr().Set(focal)
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())


def shoot(name):
    for _ in range(args.settle):
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
    data = np.asarray(annotator.get_data())
    Image.fromarray(np.ascontiguousarray(data[:, :, :3], dtype=np.uint8)).save(f"{args.out}/{args.tag}_{name}.jpg", quality=90)
    print(f"VIEW {name} mean={data[:, :, :3].mean():.1f}", flush=True)


for t in (float(v) for v in args.times.split(",") if v):
    cinematic.animate_ocean(stage, rest, distance, t, height)
    cinematic.animate_effects(stage, t)
    cinematic.set_camera(stage, path, t)
    shoot(f"seq{t:04.1f}")
for name in (v for v in args.extras.split(",") if v):
    look(*EXTRA[name])
    shoot(name)

start = clock.monotonic()
fresh, last = 0, b""
while clock.monotonic() - start < args.fps_seconds:
    t = clock.monotonic() - start
    cinematic.animate_ocean(stage, rest, distance, t, height)
    cinematic.animate_effects(stage, t)
    cinematic.set_camera(stage, path, t % cinematic.SEQUENCE_SECONDS)
    rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)
    data = np.asarray(annotator.get_data())
    sample = data[::24, ::24, :3].tobytes() if data.ndim == 3 and data.size else b""
    fresh += bool(sample) and sample != last
    last = sample or last
print(f"FPS {fresh / (clock.monotonic() - start):.1f}", flush=True)
os._exit(0)

"""Render the cinematic camera sequence to an MP4.

    C:\\isaacsim\\python.bat sim\\render_sequence.py --preset night

Run sim/build_cinematic.py first. Writes sim/output/cinematic_<preset>.mp4.
"""

from __future__ import annotations

import argparse
import time as clock

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--preset", default="night", help="night, storm or golden")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--samples", type=int, default=8, help="render subframes per video frame")
parser.add_argument("--max-frames", type=int, default=0, help="stop early, for quick tests")
parser.add_argument("--stills", default="", help="folder for sample frames, used to check the shots")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np
import omni.kit.app
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

import cinematic
import ocean
from scene_builder import OUTPUT

CAMERA_PATH = "/World/Cinematic/Cameras/sequence"

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.replicator.core", True)
manager.set_extension_enabled_immediate("omni.videoencoding", True)
import omni.replicator.core as rep
from video_encoding import get_video_encoding_interface

context = omni.usd.get_context()
context.open_stage(str(cinematic.SCENE_PATH))
for _ in range(600):
    simulation_app.update()
    if context.get_stage_loading_status()[2] == 0:
        break
stage = context.get_stage()
for _ in range(60):
    simulation_app.update()

camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.5, 60000))
camera.CreateFocalLengthAttr(28.0)
UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(1))

cinematic.apply_preset(stage, args.preset)
wave_height = cinematic.PRESETS[args.preset]["wave_height"]
rest, distance, _, _ = ocean.make_grid()
for _ in range(40):
    simulation_app.update()

product = rep.create.render_product(CAMERA_PATH, (args.width, args.height))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach(product)

out = OUTPUT / f"cinematic_{args.preset}.mp4"
frames = int(cinematic.SEQUENCE_SECONDS * args.fps)
if args.max_frames:
    frames = min(frames, args.max_frames)
encoder = get_video_encoding_interface()
if not encoder.start_encoding(str(out), float(args.fps), frames, True):
    raise RuntimeError(f"Could not start video encoding for {out}")
sample_every = max(frames // 8, 1)
written = 0

started = clock.time()
for frame in range(frames):
    now = frame / args.fps
    cinematic.set_camera(stage, CAMERA_PATH, now)
    cinematic.animate_ocean(stage, rest, distance, now, wave_height)
    cinematic.animate_effects(stage, now)
    rep.orchestrator.step(rt_subframes=args.samples, delta_time=0.0)
    data = np.asarray(annotator.get_data())
    if data.size:
        rgba = data.astype(np.uint8)
        if encoder.encode_next_frame_from_buffer(rgba, args.width, args.height):
            written += 1
        if args.stills and frame % sample_every == 0:
            Image.fromarray(rgba[:, :, :3]).save(f"{args.stills}/frame_{frame:04d}.png")
    if frame % 24 == 0:
        print(f"FRAME {frame}/{frames} elapsed={clock.time() - started:.0f}s", flush=True)

encoder.finalize_encoding()
if not out.exists() or out.stat().st_size == 0 or written == 0:
    raise RuntimeError(f"No video was written to {out}")
print(f"VIDEO_WRITTEN {out} frames={written}/{frames} seconds={clock.time() - started:.0f}", flush=True)
rep.orchestrator.wait_until_complete()
simulation_app.close()

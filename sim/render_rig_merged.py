import argparse
import os
import sys
from pathlib import Path

sys.path[:0] = ["/rig", "/rig/sim"]

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="night")
ap.add_argument("--out", default="/out")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--subframes", type=int, default=24)
ap.add_argument("--video-subframes", type=int, default=8)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--video-start", type=float, default=1.0)
ap.add_argument("--video-seconds", type=float, default=10.0)
ap.add_argument("--stills", default="wide=t:0,approach=t:8.1,closeup=t:11,deck_mid=cam:/World/Cameras/deck_mid,hero=cam:/World/Cameras/hero,repair_closeup=cam:/World/Cameras/repair_closeup,zone_wide=cam:/World/Cameras/zone_wide")
ap.add_argument("--no-video", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": args.width, "height": args.height})

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image

from rig.sim.main import load_cinematic

settings = carb.settings.get_settings()
for k, v in {"/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False, "/rtx/rtpt/maxBounces": 6,
             "/rtx/rtpt/maxSpecularAndTransmissionBounces": 6}.items():
    settings.set(k, v)
ns = argparse.Namespace(merged=True, merged_hold=8.0, merged_start=0.0)
ctx = omni.usd.get_context()
stream, update = load_cinematic(app, ctx, args.preset, ns)
for _ in range(60):
    app.update()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def grab(cam, t, subframes, warm=2):
    update(t)
    rp = grab.products.get(cam)
    if rp is None:
        rp = rep.create.render_product(cam, (args.width, args.height))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        grab.products[cam] = (rp, ann)
        warm = 4
    rp, ann = grab.products[cam]
    for _ in range(warm):
        rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)
    return np.asarray(ann.get_data())[..., :3].astype(np.uint8)


grab.products = {}
for item in args.stills.split(","):
    name, spec = item.split("=")
    kind, value = spec.split(":", 1)
    cam, t = (stream, float(value)) if kind == "t" else (value, 11.0)
    rgb = grab(cam, t, args.subframes)
    Image.fromarray(rgb).save(out / f"merged_{args.preset}_{name}.png")
    print(f"STILL {name} mean={rgb.mean():.1f}", flush=True)

if not args.no_video:
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    n = int(args.video_seconds * args.fps)
    for i in range(n):
        t = args.video_start + i / args.fps
        rgb = grab(stream, t, args.video_subframes, warm=1)
        Image.fromarray(rgb).save(frames / f"{i:05d}.jpg", quality=92)
        if i % 24 == 0:
            print(f"FRAME {i}/{n} t={t:.2f} mean={rgb.mean():.1f}", flush=True)
print("DONE", flush=True)
os._exit(0)

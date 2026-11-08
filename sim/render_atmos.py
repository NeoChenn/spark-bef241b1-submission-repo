import argparse
import os
import sys
import time
from pathlib import Path

sys.path[:0] = ["/rig", "/rig/sim"]

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="night")
ap.add_argument("--tier", default="offline", choices=["none", "live", "offline"])
ap.add_argument("--weather", default="")
ap.add_argument("--wind-dir", type=float, default=None)
ap.add_argument("--gust", type=float, default=None)
ap.add_argument("--lens", action="store_true")
ap.add_argument("--out", default="/rig/runs/volume")
ap.add_argument("--tag", default="")
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--height", type=int, default=0)
ap.add_argument("--subframes", type=int, default=0)
ap.add_argument("--stills", default="wide,deck,waterline,flare")
ap.add_argument("--warm", type=float, default=4.0)
ap.add_argument("--video-seconds", type=float, default=0.0)
ap.add_argument("--video-subframes", type=int, default=0)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--fps-test", type=float, default=0.0)
ap.add_argument("--set", action="append", default=[])
args = ap.parse_args()

import atmos

tier = atmos.TIERS["live" if args.tier == "none" else args.tier]
W, H = args.width or tier["width"], args.height or tier["height"]
SUB = args.subframes or tier["subframes"]

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": tier["renderer"], "width": W, "height": H})

import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

from rig.camera import from_usd
from rig.sim.cli import parse_setting
from rig.sim.main import load_cinematic

CAMS = {
    "wide": ((110.0, -235.0, 9.0), (4.0, -6.0, 36.0), 28.0),
    "deck": ((48.0, -46.0, 58.0), (8.0, 0.0, 44.0), 24.0),
    "waterline": ((75.0, -105.0, 4.0), (2.0, -8.0, 24.0), 18.0),
    "flare": ((-22.0, -62.0, 58.0), (-56.0, 8.0, 76.0), 32.0),
}
DOLLY = (((165.0, -215.0, 26.0), (-8.0, -2.0, 42.0), 24.0), ((92.0, -128.0, 12.0), (-14.0, 0.0, 46.0), 22.0))

settings = carb.settings.get_settings()
base = {"/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False, "/rtx/rtpt/maxBounces": 6,
        "/rtx/rtpt/maxSpecularAndTransmissionBounces": 6, "/rtx/post/dlss/execMode": 2,
        "/rtx/rtpt/spp": tier["spp"]}
for k, v in base.items():
    settings.set(k, v)
ctx = omni.usd.get_context()
stream, cine_update = load_cinematic(app, ctx, args.preset, argparse.Namespace(merged=False))
stage = ctx.get_stage()
cam = UsdGeom.Camera.Define(stage, "/World/Atmos/Cam")
cam.CreateClippingRangeAttr(Gf.Vec2f(0.3, 60000))
cam_op = cam.AddTransformOp()
cam_path = str(cam.GetPath())


def aim(eye, target, focal):
    cam.GetFocalLengthAttr().Set(focal)
    cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())


aim(*CAMS["wide"])
atm = None
if args.tier != "none":
    from atmos_fx import Atmosphere

    atm = Atmosphere(stage, args.preset, args.tier, None, cam_path=cam_path)
    sched = atmos.parse_schedule(args.weather)
    if sched:
        atm.set_weather(sched[0][1], args.wind_dir, args.gust)
    print(f"ATMOS tier={args.tier} weather={atm.w}", flush=True)
for item in args.set:
    k, v = parse_setting(item)
    settings.set(k, v)
omni.timeline.get_timeline_interface().play()
for _ in range(20):
    app.update()
rp = rep.create.render_product(cam_path, (W, H))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tag = args.tag or f"{args.preset}_{args.tier}"


def header():
    m = UsdGeom.Camera(stage.GetPrimAtPath(cam_path)).ComputeLocalToWorldTransform(0)
    return from_usd([list(m.GetRow(i)) for i in range(4)], cam.GetFocalLengthAttr().Get(),
                    cam.GetHorizontalApertureAttr().Get(), W, H)


def tick(t):
    cine_update(t)
    if atm is not None:
        atm.update(t)


def frame(t, subframes):
    tick(t)
    rep.orchestrator.step(rt_subframes=subframes, delta_time=1.0 / args.fps, pause_timeline=False)
    rgb = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
    return atm.post(rgb, header(), t, args.lens) if atm is not None else rgb


for name in filter(None, args.stills.split(",")):
    aim(*CAMS[name])
    t = 6.0
    for i in range(int(args.warm * args.fps)):
        tick(t - args.warm + i / args.fps)
        rep.orchestrator.step(rt_subframes=1, delta_time=1.0 / args.fps, pause_timeline=False)
    rgb = frame(t, SUB)
    Image.fromarray(rgb).save(out / f"{tag}_{name}.png")
    print(f"STILL {name} mean={rgb.mean():.1f}", flush=True)

if args.fps_test:
    aim(*CAMS["wide"])
    t0, n, t = time.monotonic(), 0, 0.0
    while time.monotonic() - t0 < args.fps_test:
        t = time.monotonic() - t0
        tick(t)
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        rgb = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
        if atm is not None:
            atm.post(rgb, header(), t, args.lens)
        n += 1
    print(f"FPS tier={args.tier} {W}x{H} frames={n} fps={n / (time.monotonic() - t0):.2f}", flush=True)

if args.video_seconds:
    frames = out / f"frames_{tag}"
    frames.mkdir(exist_ok=True)
    n = int(args.video_seconds * args.fps)
    (e0, g0, f0), (e1, g1, f1) = DOLLY
    t_start = time.monotonic()
    for i in range(-int(args.warm * args.fps), n):
        u = max(0.0, i) / max(1, n - 1)
        u = u * u * (3 - 2 * u)
        lerp = lambda a, b: tuple(x + (y - x) * u for x, y in zip(a, b))
        aim(lerp(e0, e1), lerp(g0, g1), f0 + (f1 - f0) * u)
        t = 4.0 + i / args.fps
        if i < 0:
            tick(t)
            rep.orchestrator.step(rt_subframes=1, delta_time=1.0 / args.fps, pause_timeline=False)
            continue
        rgb = frame(t, args.video_subframes or SUB)
        Image.fromarray(rgb).save(frames / f"{i:05d}.jpg", quality=95)
        if i % 24 == 0:
            print(f"FRAME {i}/{n} mean={rgb.mean():.1f} s/frame={(time.monotonic() - t_start) / (i + 1):.2f}", flush=True)
print("DONE", flush=True)
os._exit(0)

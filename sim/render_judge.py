import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path[:0] = ["/rig", "/rig/sim"]

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="night")
ap.add_argument("--quality", default="offline")
ap.add_argument("--atmos", default="none", choices=["none", "live", "offline"])
ap.add_argument("--renderer", default="RealTimePathTracing")
ap.add_argument("--spp", type=int, default=2)
ap.add_argument("--subframes", type=int, default=32)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--views", default="wide,deck_mid,repair,waterline,flare")
ap.add_argument("--out", default="/rig/runs/judge")
ap.add_argument("--tag", default="judge")
ap.add_argument("--t", type=float, default=6.0)
ap.add_argument("--times", default="")
ap.add_argument("--warm", type=int, default=24)
ap.add_argument("--lens", action="store_true")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--spawn-puddles", action="store_true")
ap.add_argument("--rain", type=float, default=None)
ap.add_argument("--look", default="offline")
ap.add_argument("--weather", default="")
ap.add_argument("--weather-at", type=float, default=0.0)
ap.add_argument("--set-time", default="")
ap.add_argument("--zoom-at", default="")
ap.add_argument("--raw", action="store_true")
args = ap.parse_args()
os.environ["RIG_QUALITY"] = args.quality

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})

import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom

from rig.camera import from_usd
from rig.sim.cli import parse_setting
from rig.sim.main import load_cinematic
from rig.sim.zoom import ZoomPath
import look

VIEWS = {
    "wide": ((110.0, -235.0, 9.0), (4.0, -6.0, 36.0), 28.0),
    "deck_mid": ((25.25, 3.25, 47.0), (18.25, -5.75, 43.5), 22.0),
    "waterline": ((75.0, -105.0, 4.0), (2.0, -8.0, 24.0), 18.0),
    "flare": ((-22.0, -62.0, 58.0), (-56.0, 8.0, 76.0), 32.0),
    "orbit": ((270.0, -200.0, 110.0), (4.0, -4.0, 30.0), 28.0),
    "top": ((60.0, -120.0, 260.0), (-20.0, 0.0, 40.0), 24.0),
}

settings = carb.settings.get_settings()
pt = "/rtx/pathtracing" if args.renderer == "PathTracing" else "/rtx/rtpt"
base = {"/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False, f"{pt}/spp": args.spp,
        f"{pt}/maxBounces": 6, f"{pt}/maxSpecularAndTransmissionBounces": 6}
if args.renderer == "PathTracing":
    base.update({"/rtx/pathtracing/totalSpp": args.spp * args.subframes, "/rtx/pathtracing/optixDenoiser/enabled": True})
else:
    base["/rtx/post/dlss/execMode"] = 2
for k, v in base.items():
    settings.set(k, v)
ctx = omni.usd.get_context()
_, cine_update = load_cinematic(app, ctx, args.preset, argparse.Namespace(merged=True, merged_hold=0, merged_start=0.0))
stage = ctx.get_stage()
meta = json.loads((Path("/rig/sim/output") / "rig_merged_meta.json").read_text())
zoom = ZoomPath(wide_target=tuple(meta["zoom"]["wide_target"]), close_target=tuple(meta["zoom"]["close_target"]))
VIEWS["repair"] = zoom.at(zoom.orbit_s + zoom.descend_s + 1.0)
VIEWS["approach"] = zoom.at(zoom.orbit_s + 0.62 * zoom.descend_s)
for k, f in enumerate((0.0, 0.25, 0.45, 0.65, 0.85)):
    VIEWS[f"zoom{k}"] = zoom.at(zoom.orbit_s + f * zoom.descend_s)
VIEWS.update({v: zoom.at(float(v[5:])) for v in args.views.split(",") if v.startswith("zoom@")})
for zt in filter(None, args.zoom_at.split(",")):
    VIEWS[f"z{zt}"] = zoom.at(float(zt))
cam = UsdGeom.Camera.Define(stage, "/World/Judge/Cam")
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60000))
cam.CreateHorizontalApertureAttr(20.955)
cam_op = cam.AddTransformOp()
cam_path = str(cam.GetPath())


def aim(eye, target, focal):
    cam.GetFocalLengthAttr().Set(float(focal))
    cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse())


if args.spawn_puddles:
    import photoreal

    zt = meta["zone_to_world"]["translation"]
    UsdGeom.Scope.Define(stage, "/World/JudgePuddles")
    for k, (x, y, r) in enumerate(((-2.2, 1.0, 0.55), (-1.1, 1.7, 0.45), (0.5, 1.9, 0.5))):
        photoreal.make_puddle(stage, f"/World/JudgePuddles/p{k}", (zt[0] + x, zt[1] + y, zt[2]), r, seed=40 + k)
if args.rain is not None:
    import photoreal

    photoreal.set_rain(stage, args.rain)
atm = None
if args.atmos != "none":
    from atmos_fx import Atmosphere

    atm = Atmosphere(stage, args.preset, args.atmos, None, cam_path=cam_path)
overrides = dict(parse_setting(item) for item in args.set)
looks = args.look.split(",")
import cinematic
import shafts

tier = next((lk for lk in ("offline", "live") if lk in looks), "none")
beams = (shafts.specs(cinematic.AREA_LIGHTS, cinematic.SEA_FLOODS, cinematic.DECK_FLOODS, tier) if cinematic.PRESETS[args.preset]["lights"] > 0 else [])
keys = {k for lk in looks for k in look.rtx_settings(lk)}
neutral = {k: settings.get(k) if settings.get(k) is not None else (False if k.endswith("enabled") else None) for k in keys}


def use_look(lk, ev, eye, target, fstop):
    for k, v in {**neutral, **look.rtx_settings(lk, ev), **overrides}.items():
        if v is not None:
            settings.set(k, v)
    lens = look.lens(float(np.linalg.norm(np.subtract(target, eye))), fstop if lk == "offline" else 0.0)
    for k, v in lens.items():
        cam.GetPrim().CreateAttribute(k, Sdf.ValueTypeNames.Float).Set(v)


use_look(looks[0], 0.0, (0, 0, 0), (0, 0, 1), 0.0)
omni.timeline.get_timeline_interface().play()
for _ in range(20):
    app.update()
rp = rep.create.render_product(cam_path, (args.width, args.height))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
dist_ann = None
if beams or tier != "none":
    dist_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    dist_ann.attach(rp)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def header():
    m = cam.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return from_usd([list(m.GetRow(i)) for i in range(4)], cam.GetFocalLengthAttr().Get(),
                    cam.GetHorizontalApertureAttr().Get(), args.width, args.height)


wx = None
if args.weather:
    import weather as weather_mod

    wx = weather_mod.Controller(stage, getattr(cine_update, "wave", None), args.preset, atm)
    wx_args = weather_mod.parse_default(args.weather)


def tick(t):
    if wx is not None:
        if not wx.args and t >= args.weather_at:
            wx.now = args.weather_at
            wx.set(**wx_args)
            if atm is not None:
                atm.set_weather(level=weather_mod.atmos_kind(args.weather))
        wx.tick(t)
    cine_update(t)
    if atm is not None:
        atm.update(t)


if args.set_time:
    t_switch = time.monotonic()
    cinematic.set_time(stage, args.set_time, atm)
    print(f"SET_TIME {args.set_time} {time.monotonic() - t_switch:.2f}s", flush=True)
times = [float(x) for x in args.times.split(",") if x] or [args.t]
for t0 in times:
    tsuf = f"_t{t0:g}" if len(times) > 1 else ""
    for name in filter(None, args.views.split(",")):
        start = time.monotonic()
        eye, target, focal = VIEWS[name]
        vl = look.view_look(name, float(np.linalg.norm(np.subtract(target, eye))), atm is not None, cinematic.CURRENT["time"])
        use_look(looks[0], vl["ev"], eye, target, vl["fstop"])
        aim(*VIEWS[name])
        for i in range(args.warm):
            tick(t0 - args.warm / 24.0 + i / 24.0)
            aim(*VIEWS[name])
            rep.orchestrator.step(rt_subframes=1, delta_time=1.0 / 24.0, pause_timeline=False)
        tick(t0)
        aim(*VIEWS[name])
        for n, lk in enumerate(looks):
            use_look(lk, vl["ev"], eye, target, vl["fstop"])
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0 if n else 1.0 / 24.0, pause_timeline=False)
            rgb = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
            if atm is not None:
                rgb = atm.post(rgb, header(), t0, args.lens)
            if lk != "none":
                depth = np.asarray(dist_ann.get_data()) if dist_ann is not None else None
                if args.raw:
                    Image.fromarray(np.ascontiguousarray(rgb)).save(out / f"{args.tag}{tsuf}_{name}_raw.png")
                    np.save(out / f"{args.tag}{tsuf}_{name}_depth.npy", np.asarray(depth, np.float16))
                    (out / f"{args.tag}{tsuf}_{name}_cam.json").write_text(json.dumps(header()))
                rgb = shafts.add(rgb, shafts.layer(header(), beams, t0, depth=depth))
                now = cinematic.CURRENT["time"]
                rgb = look.inscatter(rgb, depth, header(), now, look.sun_direction(cinematic.PRESETS[now]["sun_rotation"]),
                                     0.45 if args.atmos == "offline" else 1.0, tier=lk,
                                     mist=look.mist_level(settings.get("/rtx/fog/fogDistanceDensity")))
            rgb = look.grade(rgb, lk, t0, cinematic.CURRENT["time"])
            suffix = f"_{lk}" if len(looks) > 1 else ""
            Image.fromarray(np.ascontiguousarray(rgb)).save(out / f"{args.tag}{tsuf}_{name}{suffix}.png")
            lum = rgb.mean(-1)
            print(f"VIEW {name}{suffix} mean={rgb.mean():.1f} clip={(lum > 250).mean():.4f} dark={(lum < 10).mean():.3f} "
                  f"s={time.monotonic() - start:.1f}", flush=True)
print("DONE", flush=True)
os._exit(0)

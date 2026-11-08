import argparse
import json
import os
import time
from pathlib import Path

from rig.sim.zoom import ease, interp

OBLIQUE = ((-6.0, -34.0, 88.0), (-7.0, 6.0, 42.5), 24.0)
WEST = ((-74.0, -16.0, 46.0), (-8.0, 5.0, 35.5), 24.0)
CAM_KEYS = [(0.0, *OBLIQUE), (5.6, (-8.0, -28.0, 80.0), (-9.0, 7.0, 42.5), 24.0),
            (7.0, (-62.0, -14.0, 96.0), (-12.0, 4.0, 38.0), 24.0), (8.4, *WEST),
            (18.0, (-70.0, -8.0, 47.0), (-6.0, 6.0, 35.0), 24.0)]
SIM_KEYS = [(0.0, 0.0), (2.2, 6.5), (5.6, 30.0), (8.6, 55.0), (18.0, 125.0)]


def sim_time(tv: float, keys=SIM_KEYS) -> float:
    return interp(tv, [k[0] for k in keys], [k[1] for k in keys])


def camera(tv: float, keys=CAM_KEYS):
    ts = [k[0] for k in keys]
    i = max(0, min(len(keys) - 2, sum(1 for t in ts if t <= tv) - 1))
    u = ease((tv - ts[i]) / (ts[i + 1] - ts[i]))
    lerp = lambda a, b: tuple(x + (y - x) * u for x, y in zip(a, b))
    return lerp(keys[i][1], keys[i + 1][1]), lerp(keys[i][2], keys[i + 1][2]), interp(u, [0, 1], [keys[i][3], keys[i + 1][3]])


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fleet-log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=18.0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--sim-t0", type=float, default=0.0)
    p.add_argument("--sim-t1", type=float, default=-1.0)
    p.add_argument("--frames", default="")
    p.add_argument("--from-frame", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--subframes", type=int, default=6)
    p.add_argument("--path-radius", type=float, default=0.16)
    p.add_argument("--preset", default="night")
    p.add_argument("--cam", default="")
    return p.parse_args(argv)


def main():
    args = parse()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": args.width, "height": args.height})
    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom

    from rig.sim.main import load_cinematic, load_fleet_log, make_obstacles

    st = carb.settings.get_settings()
    for k, v in {"/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False, "/rtx/rtpt/maxBounces": 6,
                 "/rtx/post/dlss/execMode": 2}.items():
        st.set(k, v)
    ns = argparse.Namespace(merged=True, merged_hold=8.0, merged_start=0.0, fleet_log="", fleet_speed=1.0,
                            path_radius=args.path_radius)
    ctx = omni.usd.get_context()
    cam_path, cine = load_cinematic(app, ctx, args.preset, ns)
    stage = ctx.get_stage()
    scene, _ = make_obstacles(stage, cine, ns)
    times, ticks = load_fleet_log(args.fleet_log)
    from rig.sim.obstacle_math import tick_index

    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    op = UsdGeom.Xformable(cam).GetOrderedXformOps()[0]
    keys = CAM_KEYS
    if args.cam:
        e, t, f = json.loads(args.cam)
        keys = [(0.0, tuple(e), tuple(t), f), (1e6, tuple(e), tuple(t), f)]
    rp = rep.create.render_product(cam_path, (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    omni.timeline.get_timeline_interface().play()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(round(args.seconds * args.fps))
    only = {int(x) for x in args.frames.split(",") if x} if args.frames else None
    if args.from_frame:
        only = set(range(args.from_frame, n))
    index, last = [], -1

    def place(tv):
        eye, target, focal = camera(tv, keys)
        cam.GetFocalLengthAttr().Set(focal)
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())

    for _ in range(20):
        cine(0.0)
        place(0.0)
        app.update()
    for k in range(n):
        tv = k / args.fps
        ts = sim_time(tv) if args.sim_t1 < 0 else args.sim_t0 + (args.sim_t1 - args.sim_t0) * tv / args.seconds
        i = tick_index(times, ts)
        if i != last:
            last = i
            scene.sync(ticks[i])
        cine(tv)
        scene.tick(tv)
        place(tv)
        if only is not None and k not in only:
            continue
        t0 = time.monotonic()
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False)
        rgb = np.ascontiguousarray(np.asarray(ann.get_data())[..., :3], dtype=np.uint8)
        Image.fromarray(rgb).save(out / f"{k:05d}.png")
        index.append({"frame": k, "tv": round(tv, 3), "sim_t": ticks[i]["t"]})
        if only is not None or k % 24 == 0:
            print(f"frame {k}/{n} tv={tv:.1f} sim_t={ticks[i]['t']} mean={rgb.mean():.1f} "
                  f"{time.monotonic() - t0:.1f}s", flush=True)
    (out / "index.json").write_text(json.dumps(index))
    print(f"wrote {len(index)} frames to {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

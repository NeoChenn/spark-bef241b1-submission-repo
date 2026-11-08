import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--preset", default="night")
    a.add_argument("--flange", default="F5")
    a.add_argument("--hole-mm", type=float, default=3.0)
    a.add_argument("--bar", type=float, default=20.0)
    a.add_argument("--wind", type=float, default=5.0)
    a.add_argument("--heading", type=float, default=90.0)
    a.add_argument("--storm", type=float, default=0.0)
    a.add_argument("--seconds", type=float, default=12.0)
    a.add_argument("--fps", type=float, default=24.0)
    a.add_argument("--every", type=int, default=1)
    a.add_argument("--width", type=int, default=1280)
    a.add_argument("--height", type=int, default=720)
    a.add_argument("--subframes", type=int, default=2)
    a.add_argument("--main", default="16,7,3", help="distance,height,ahead of the main view")
    a.add_argument("--side", type=float, default=-1.0)
    a.add_argument("--pip-side", type=float, default=-1.0)
    a.add_argument("--size", type=float, default=1.7)
    a.add_argument("--opacity", type=float, default=1.0)
    a.add_argument("--max-points", type=int, default=8000)
    a.add_argument("--out", default="/rig/runs/plume/video")
    return a.parse_args(argv)


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

    from rig.sim.main import load_cinematic
    from sim.plume.isaac import PlumeVisual, analysis_pose, make_camera
    from sim.plume.pip import compose
    from sim.plume.solver import Plume, Wind

    s = carb.settings.get_settings()
    for k, v in {"/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False, "/rtx/rtpt/spp": 1,
                 "/rtx/rtpt/maxBounces": 6, "/rtx/raytracing/fractionalCutoutOpacity": True}.items():
        s.set(k, v)
    ctx = omni.usd.get_context()
    _, cine = load_cinematic(app, ctx, args.preset)
    stage = ctx.get_stage()
    t0 = time.perf_counter()
    plume = Plume.for_flange(args.flange, args.hole_mm, args.bar, wind=Wind(args.wind, args.heading, args.storm), stage=stage)
    print(f"plume ready {time.perf_counter() - t0:.1f}s solid={plume.solid_np.mean():.3f} device={plume.device}", flush=True)
    vis = PlumeVisual(stage, plume, max_points=args.max_points, size=args.size, opacity=args.opacity)
    dist, hgt, ahead = (float(v) for v in args.main.split(","))
    main_cam, _ = make_camera(stage, *analysis_pose(plume, args.side, dist, hgt, ahead), path="/World/Plume/MainCamera", focal=24.0)
    pip_cam, _ = make_camera(stage, *analysis_pose(plume, args.pip_side), focal=18.0)
    rp_main = rep.create.render_product(main_cam, (args.width, args.height))
    rp_pip = rep.create.render_product(pip_cam, (960, 540))
    a_main = rep.AnnotatorRegistry.get_annotator("rgb")
    a_main.attach(rp_main)
    a_pip = rep.AnnotatorRegistry.get_annotator("rgb")
    a_pip.attach(rp_pip)
    omni.timeline.get_timeline_interface().play()
    for _ in range(30):
        app.update()
    out = Path(args.out)
    for sub in ("main", "pip", "comp"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dt = 1.0 / args.fps
    n = int(args.seconds * args.fps)
    log, sim_ms, ren_ms, vis_ms = [], [], [], []
    for f in range(n):
        t1 = time.perf_counter()
        st = plume.step(dt)
        tv = time.perf_counter()
        vis.update()
        t2 = time.perf_counter()
        vis_ms.append((t2 - tv) * 1e3)
        cine(plume.t)
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False, wait_for_render=True)
        t3 = time.perf_counter()
        sim_ms.append((t2 - t1) * 1e3)
        ren_ms.append((t3 - t2) * 1e3)
        st.update(points=vis.count)
        log.append(st)
        if f % args.every:
            continue
        main_rgb = np.asarray(a_main.get_data())[..., :3].astype(np.uint8)
        pip = compose(np.asarray(a_pip.get_data()), st, plume.flow_rate(), plume.wind.to_json(plume.t), f"GAS PLUME  ·  {args.flange}")
        Image.fromarray(main_rgb).save(out / "main" / f"{f:04d}.jpg", quality=92)
        pip.save(out / "pip" / f"{f:04d}.jpg", quality=92)
        comp = Image.fromarray(main_rgb).convert("RGB")
        inset = pip.resize((round(args.width * 0.36), round(args.width * 0.36 * 9 / 16)))
        comp.paste(inset, (args.width - inset.width - 24, args.height - inset.height - 24))
        comp.save(out / "comp" / f"{f:04d}.jpg", quality=92)
        if f % 24 == 0:
            print(f"frame {f}/{n} t={plume.t:.1f} pts={vis.count} vol10={st['vol_10lel_m3']:.2f} max={st['max_lel_pct']:.0f}% "
                  f"sim={np.median(sim_ms[-24:]):.1f}ms vis={np.median(vis_ms[-24:]):.1f}ms render={np.median(ren_ms[-24:]):.1f}ms", flush=True)
    (out / "log.json").write_text(json.dumps({"args": vars(args), "flow": plume.flow_rate(), "stats": log,
                                             "sim_ms_median": float(np.median(sim_ms)), "render_ms_median": float(np.median(ren_ms)), "vis_ms_median": float(np.median(vis_ms))}))
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

import argparse
import io
import os
import time

STATS_EVERY_S = 5.0
WARMUP_UPDATES = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless Isaac Sim camera stream for the rig test world")
    p.add_argument("--sim", default="127.0.0.1:9000", help="host:port of the web relay's frame ingest")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--renderer", default="RealTimePathTracing")
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 = forever")
    p.add_argument("--snapshot", default="", help="save the first non-black frame as a PNG here")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    from rig.sim import CAM_H, CAM_W

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": CAM_W, "height": CAM_H})

    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image

    from rig.sim.scene import CAMERA_PATH, build_scene, spin
    from rig.sim.sender import FrameSender

    settings = carb.settings.get_settings()
    for key in ("/rtx/post/motionblur/enabled", "/rtx/post/dof/enabled", "/rtx/post/lensFlares/enabled"):
        settings.set(key, False)

    scene = build_scene(omni.usd.get_context().get_stage())
    rp = rep.create.render_product(CAMERA_PATH, (CAM_W, CAM_H))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    omni.timeline.get_timeline_interface().play()
    for _ in range(WARMUP_UPDATES):
        app.update()

    host, port = args.sim.rsplit(":", 1)
    sender = FrameSender(host, int(port)).start()
    period = 1.0 / args.fps
    start = next_tick = time.monotonic()
    stats_at = start + STATS_EVERY_S
    seq = frames = 0
    mean = 0.0
    snapshot_done = not args.snapshot
    print(f"rig sim: renderer={args.renderer} {CAM_W}x{CAM_H} target_fps={args.fps} -> {args.sim}", flush=True)

    while app.is_running():
        now = time.monotonic()
        t = now - start
        spin(scene, t)
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)
        data = np.asarray(ann.get_data())
        if data.ndim == 3 and data.size:
            rgb = np.ascontiguousarray(data[..., :3], dtype=np.uint8)
            mean = float(rgb.mean())
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "JPEG", quality=args.quality)
            sender.send({"type": "frame", "w": rgb.shape[1], "h": rgb.shape[0], "seq": seq, "t": t}, buf.getvalue())
            seq += 1
            if not snapshot_done and mean > 3:
                Image.fromarray(rgb).save(args.snapshot)
                snapshot_done = True
                print(f"snapshot saved to {args.snapshot}", flush=True)
        frames += 1
        if now >= stats_at:
            span = now - start
            print(
                f"t={span:.0f}s fps={frames / span:.1f} sent={sender.sent} dropped={sender.dropped} "
                f"connected={sender.connected.is_set()} mean={mean:.1f}",
                flush=True,
            )
            stats_at = now + STATS_EVERY_S
        if args.duration and t >= args.duration:
            break
        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()

    sender.stop()
    os._exit(0)


if __name__ == "__main__":
    main()

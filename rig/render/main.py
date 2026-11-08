import io
import json
import os
import time
from pathlib import Path

from rig.render.cli import parse_args

CAMERA = "/RigRender/Camera"
WARMUP_STEPS = 8
STATS_EVERY_S = 5.0


class Post:
    def __init__(self, stage, scene: dict) -> None:
        import cinematic

        self.grade = self.atm = None
        self.stage, self.cinematic = stage, cinematic
        self.beams, self.weather, self.lens = [], None, bool(scene.get("lens"))
        tier = scene.get("look") or "none"
        if tier != "none":
            import carb
            import look
            import shafts

            st = carb.settings.get_settings()
            for key, value in look.rtx_settings(tier).items():
                st.set(key, value)
            self.grade = lambda rgb, t: look.grade(rgb, tier, t, cinematic.CURRENT["time"])
            if cinematic.PRESETS[scene["cinematic"]]["lights"] > 0:
                self.beams = shafts.specs(cinematic.AREA_LIGHTS, cinematic.SEA_FLOODS, cinematic.DECK_FLOODS, tier)
            self.shafts = shafts
        if scene.get("atmos"):
            from atmos_fx import Atmosphere

            self.atm = Atmosphere(stage, scene["cinematic"], scene["atmos"], None, cam_path=CAMERA)

    def tick(self, head: dict) -> None:
        tm = head.get("time")
        if tm in self.cinematic.PRESETS and tm != self.cinematic.CURRENT["time"]:
            self.cinematic.set_time(self.stage, tm, self.atm)
        if self.atm is None:
            return
        w = head.get("weather")
        if w and w != self.weather:
            self.weather = w
            self.atm.set_weather(w)
        self.atm.update(head.get("t", 0.0))

    def __call__(self, rgb, cam, t: float):
        if self.atm is not None and cam is not None:
            rgb = self.atm.post(rgb, cam, t, self.lens)
        if self.grade is not None:
            if self.beams and cam is not None:
                rgb = self.shafts.add(rgb, self.shafts.layer(cam, self.beams, t))
            rgb = self.grade(rgb, t)
        return rgb


def load_scene(app, ctx, scene: dict):
    stage, tick = _load_scene(app, ctx, scene)
    physics = [p for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"]
    for prim in physics:
        prim.SetActive(False)
    if physics:
        print(f"rig render: physics off, mirroring only ({len(physics)} PhysicsScene)", flush=True)
    return stage, tick


def _load_scene(app, ctx, scene: dict):
    import sys

    from rig.sim.main import load_environment, open_scene

    if scene.get("cinematic"):
        sys.path.insert(0, "/rig/sim")
        import cinematic
        import ocean

        path = cinematic.SCENE_PATH.with_name("rig_merged.usda") if scene.get("merged") else cinematic.SCENE_PATH
        open_scene(app, ctx, str(path))
        stage = ctx.get_stage()
        cinematic.apply_preset(stage, scene["cinematic"])
        rest, distance, _, _ = ocean.make_grid()
        height = cinematic.PRESETS[scene["cinematic"]]["wave_height"]
        print(f"rig render: cinematic {scene['cinematic']} scene {path.name}", flush=True)
        return stage, lambda t, ocean=None: cinematic.animate_ocean(stage, rest, distance, t, (ocean or {}).get("height", height))
    if scene.get("scene") == "test":
        from rig.sim.scene import build_scene

        build_scene(ctx.get_stage())
        return ctx.get_stage(), None
    open_scene(app, ctx, scene["scene"])
    stage = ctx.get_stage()
    env = load_environment(stage, scene.get("env", ""), scene.get("sea_level", 0.0))
    return stage, (lambda t, ocean=None: env.update(t, camera=CAMERA)) if env is not None else None


def apply_settings(values: dict) -> None:
    if not values:
        return
    import carb

    st = carb.settings.get_settings()
    for key, value in values.items():
        if isinstance(value, list):
            st.set_float_array(key, [float(v) for v in value])
        else:
            st.set(key, value)


def apply_state(applier, msg: dict) -> None:
    applier.apply(msg)
    apply_settings(msg.get("settings"))


def make_camera(stage):
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, CAMERA)
    op = cam.AddTransformOp()

    def set_view(v: dict) -> None:
        op.Set(Gf.Matrix4d(*v["m"]))
        cam.GetFocalLengthAttr().Set(v["focal"])
        cam.GetHorizontalApertureAttr().Set(v["hap"])
        cam.GetVerticalApertureAttr().Set(v["vap"])
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(*v["clip"]))

    return set_view


def camera_header(view: dict, w: int, h: int) -> dict:
    from rig.camera import from_usd

    m = view["m"]
    return from_usd([m[i * 4:i * 4 + 4] for i in range(4)], view["focal"], view["hap"], w, h)


def make_encoder(kind: str, quality: int):
    if kind == "cv2":
        import cv2

        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        return lambda rgb: cv2.imencode(".jpg", rgb[..., ::-1], params)[1].tobytes()
    from PIL import Image

    def encode(rgb) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    return encode


class Renderer:
    def __init__(self, args, stage) -> None:
        import omni.replicator.core as rep

        self.rep, self.args = rep, args
        self.rp = rep.create.render_product(CAMERA, (args.width, args.height))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach(self.rp)

    def frame(self, wait: bool):
        import numpy as np

        self.rep.orchestrator.step(rt_subframes=self.args.subframes, pause_timeline=False, wait_for_render=wait)
        data = np.asarray(self.ann.get_data())
        if data.ndim != 3 or not data.size:
            return None
        return np.ascontiguousarray(data[..., :3], dtype=np.uint8)


def boot(args):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})
    import carb
    import omni.timeline
    import omni.usd

    from rig.sim.main import disable_viewport, render_settings

    settings = carb.settings.get_settings()
    for key, value in render_settings(args).items():
        settings.set(key, value)
    disable_viewport()
    return app, omni.usd.get_context(), omni.timeline.get_timeline_interface()


def run_live(args) -> None:
    from rig.render.state import Inbox
    from rig.sim.sender import FrameSender

    app, ctx, timeline = boot(args)
    from rig.render.usdsync import StageApplier

    inbox = Inbox(record=args.record or None)
    host, port = args.relay.rsplit(":", 1)
    hello = {"type": "hello", "role": "render", "token": os.environ.get(args.token_env, "")}
    sender = FrameSender(host, int(port), hello=hello, on_message=inbox.put).start()
    print(f"rig render: waiting for scene state from {args.relay}", flush=True)
    first = None
    while first is None or not first.get("scene"):
        inbox.ready.wait(1.0)
        first = inbox.take() or first
        app.update()
    scene = first["scene"]
    stage, tick = load_scene(app, ctx, scene)
    set_view = make_camera(stage)
    post = Post(stage, scene) if scene.get("cinematic") else None
    applier = StageApplier(stage)
    renderer = Renderer(args, stage)
    timeline.play()
    encode = make_encoder(args.jpeg, args.quality)
    msg = first
    apply_state(applier, msg)
    for _ in range(WARMUP_STEPS):
        renderer.frame(False)
    period = 1.0 / args.fps if args.fps else 0.0
    start = next_tick = last_render = time.monotonic()
    stats_at = start + STATS_EVERY_S
    head = msg
    seq = frames = stale = 0
    acc = {"apply": 0.0, "render": 0.0, "encode": 0.0, "kb": 0.0, "lat": 0.0}
    snap_at = start if args.snapshot else None
    print(f"rig render: {args.renderer} {args.width}x{args.height} spp={args.spp} subframes={args.subframes} -> {args.relay}", flush=True)
    while app.is_running():
        now = time.monotonic()
        if msg is not None:
            if msg.get("scene") and msg["scene"] != scene:
                print(f"rig render: scene changed to {msg['scene']}, restarting", flush=True)
                os._exit(3)
            t0 = time.perf_counter()
            apply_state(applier, msg)
            head = msg
            acc["apply"] += time.perf_counter() - t0
        if head.get("view"):
            set_view(head["view"])
        if tick is not None:
            tick(head.get("t", 0.0), head.get("ocean"))
        if post is not None:
            post.tick(head)
        if msg is not None or now - last_render >= args.keepalive:
            t1 = time.perf_counter()
            rgb = renderer.frame(args.wait_render)
            if rgb is not None and post is not None:
                cam = camera_header(head["view"], rgb.shape[1], rgb.shape[0]) if head.get("view") else None
                rgb = post(rgb, cam, head.get("t", 0.0))
            t2 = time.perf_counter()
            if rgb is not None:
                mean = float(rgb[::8, ::8].mean())
                jpeg = encode(rgb)
                t3 = time.perf_counter()
                header = {"type": "frame", "w": rgb.shape[1], "h": rgb.shape[0], "seq": seq, "t": head.get("t", 0.0),
                          "mean": round(mean, 1), "state_seq": head.get("seq"), "state_ts": head.get("ts"),
                          "render_ms": round((t2 - t1) * 1000, 1), "encode_ms": round((t3 - t2) * 1000, 1),
                          "renderer": args.renderer}
                if head.get("view"):
                    header["camera"] = camera_header(head["view"], rgb.shape[1], rgb.shape[0])
                sender.send(header, jpeg)
                seq += 1
                frames += 1
                acc["render"] += t2 - t1
                acc["encode"] += t3 - t2
                acc["kb"] += len(jpeg) / 1024
                if head.get("ts"):
                    acc["lat"] += time.time() - head["ts"]
                if snap_at is not None and now >= snap_at and mean > 3:
                    from PIL import Image

                    path = args.snapshot if not args.snapshot_every else args.snapshot.replace(".png", f"_{int(now - start):04d}.png")
                    Image.fromarray(rgb).save(path)
                    print(f"snapshot saved to {path}", flush=True)
                    snap_at = now + args.snapshot_every if args.snapshot_every else None
            last_render = now
        else:
            stale += 1
        if now >= stats_at:
            span = now - stats_at + STATS_EVERY_S
            n = max(frames, 1)
            print(f"t={now - start:.0f}s fps={frames / span:.1f} render={acc['render'] / n * 1000:.1f}ms "
                  f"encode={acc['encode'] / n * 1000:.1f}ms apply={acc['apply'] / n * 1000:.1f}ms jpeg={acc['kb'] / n:.0f}KB "
                  f"state_to_send={acc['lat'] / n * 1000:.0f}ms states={inbox.received} applied={applier.applied} "
                  f"failed={applier.failed} sent={sender.sent} dropped={sender.dropped} connected={sender.connected.is_set()}",
                  flush=True)
            stats_at, frames = now + STATS_EVERY_S, 0
            acc = dict.fromkeys(acc, 0.0)
        if args.duration and now - start >= args.duration:
            break
        if period:
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                inbox.ready.wait(delay)
            else:
                next_tick = time.monotonic()
        msg = inbox.take()
        if msg is None and not period:
            inbox.ready.wait(0.05)
            msg = inbox.take()
    sender.stop()
    inbox.close()
    os._exit(0)


def run_offline(args) -> None:
    from rig.render.state import replay_plan

    log = [json.loads(line) for line in Path(args.offline).read_text().splitlines() if line.strip()]
    log = [m for m in log if m.get("type") == "state"]
    if not log:
        raise SystemExit(f"rig render: no state records in {args.offline}")
    app, ctx, timeline = boot(args)
    from PIL import Image

    from rig.render.usdsync import StageApplier

    stage, tick = load_scene(app, ctx, log[0]["scene"])
    set_view = make_camera(stage)
    post = Post(stage, log[0]["scene"]) if log[0]["scene"].get("cinematic") else None
    applier = StageApplier(stage)
    renderer = Renderer(args, stage)
    timeline.play()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = True
    n = 0
    t_start = time.monotonic()
    for deltas, state in replay_plan(log, args.fps):
        for d in deltas:
            apply_state(applier, d)
        t = state.get("t", 0.0)
        if t < args.start:
            continue
        if args.end and t > args.end:
            break
        if args.fps:
            applier.apply({"attrs": state["attrs"]})
        if state.get("view"):
            set_view(state["view"])
        if tick is not None:
            tick(t, state.get("ocean"))
        if post is not None:
            post.tick(state)
        if first:
            for _ in range(WARMUP_STEPS):
                renderer.frame(True)
            first = False
        t0 = time.perf_counter()
        rgb = renderer.frame(True)
        if rgb is None:
            continue
        if post is not None and state.get("view"):
            rgb = post(rgb, camera_header(state["view"], rgb.shape[1], rgb.shape[0]), t)
        Image.fromarray(rgb).save(out / f"frame_{n:05d}.png")
        print(f"offline frame {n} t={t:.2f}s {time.perf_counter() - t0:.2f}s/frame", flush=True)
        n += 1
    wall = time.monotonic() - t_start
    print(f"rig render: offline wrote {n} frames to {out} in {wall:.0f}s ({wall / max(n, 1):.2f}s/frame)", flush=True)
    write_mp4(out, args.fps or 30.0)
    os._exit(0)


def write_mp4(out: Path, fps: float) -> None:
    try:
        import cv2
    except ImportError:
        return
    frames = sorted(out.glob("frame_*.png"))
    if not frames:
        return
    h, w = cv2.imread(str(frames[0])).shape[:2]
    for name, codec in (("offline.mp4", "mp4v"), ("offline.avi", "MJPG")):
        writer = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(cv2.imread(str(f)))
            writer.release()
            print(f"rig render: wrote {out / name}", flush=True)
            return
    print(f"rig render: no video codec available, PNG sequence only in {out}", flush=True)


def main() -> None:
    args = parse_args()
    if args.offline:
        run_offline(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()

import inspect
import io
import os
import time

from rig.sim.cli import parse_args, parse_setting

STATS_EVERY_S = 5.0
WARMUP_UPDATES = 30
KEEPALIVE_S = 1.0
LOAD_TIMEOUT_S = 300.0
ORBIT_CAMERA = "/RigStream/OrbitCamera"


def render_settings(args) -> dict:
    pt = "/rtx/pathtracing" if args.renderer == "PathTracing" else "/rtx/rtpt"
    out = {
        "/rtx/post/motionblur/enabled": False,
        "/rtx/post/dof/enabled": False,
        "/rtx/post/lensFlares/enabled": False,
        f"{pt}/spp": args.spp,
    }
    if args.renderer == "PathTracing":
        out["/rtx/pathtracing/totalSpp"] = args.spp * args.subframes
    if args.bounces:
        out[f"{pt}/maxBounces"] = args.bounces
    out.update(parse_setting(item) for item in args.set)
    return out


def open_scene(app, ctx, path: str) -> None:
    if not ctx.open_stage(path):
        raise SystemExit(f"rig sim: could not open {path}")
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        app.update()
        _, loaded, total = ctx.get_stage_loading_status()
        if total == 0 or loaded >= total:
            return
    print(f"rig sim: stage still loading after {LOAD_TIMEOUT_S:.0f}s, continuing", flush=True)


def hide_builtin_environment(stage) -> list[str]:
    from rig.sim.envhide import is_builtin_environment

    hidden = [p for p in stage.Traverse() if is_builtin_environment(p.GetName(), p.GetTypeName())]
    for prim in hidden:
        prim.SetActive(False)
    return [str(p.GetPath()) for p in hidden]


def load_environment(stage, preset: str, sea_level: float):
    if not preset:
        return None
    try:
        from sim.environment import add_environment
    except ImportError as exc:
        print(f"rig sim: --env {preset} skipped, sim.environment unavailable ({exc})", flush=True)
        return None
    hidden = hide_builtin_environment(stage)
    env = add_environment(stage, preset, sea_level_z=sea_level)
    print(f"rig sim: environment preset {preset} added at sea level {sea_level}, hid {hidden}", flush=True)
    return env


def merged_camera(args, scene_path):
    import json
    import math

    from rig.sim.zoom import ZoomPath

    meta = json.loads(scene_path.with_name("rig_merged_meta.json").read_text())
    zoom = ZoomPath(wide_target=tuple(meta["zoom"]["wide_target"]), close_target=tuple(meta["zoom"]["close_target"]),
                    hold_s=args.merged_hold or math.inf)
    return lambda t: zoom.at(t + args.merged_start)


def load_cinematic(app, ctx, preset: str, args=None):
    import sys

    sys.path.insert(0, "/rig/sim")
    import cinematic
    import ocean
    from pxr import Gf, UsdGeom

    merged = bool(args is not None and args.merged)
    scene_path = cinematic.SCENE_PATH.with_name("rig_merged.usda") if merged else cinematic.SCENE_PATH
    open_scene(app, ctx, str(scene_path))
    stage = ctx.get_stage()
    path = "/World/Cinematic/Cameras/stream"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05 if merged else 0.5, 60000))
    cam.CreateFocalLengthAttr(28.0)
    op = UsdGeom.Xformable(cam).AddTransformOp()
    op.Set(Gf.Matrix4d(1))
    cinematic.apply_preset(stage, preset)
    rest, distance, _, _ = ocean.make_grid()
    wave = {"height": cinematic.PRESETS[preset]["wave_height"]}
    view = merged_camera(args, scene_path) if merged else None

    remote = args is not None and getattr(args, "render_mode", "local") == "remote"

    def update(t: float) -> None:
        if not remote:
            cinematic.animate_ocean(stage, rest, distance, t, wave["height"])
        cinematic.animate_effects(stage, t)
        if view is None:
            cinematic.set_camera(stage, path, t % cinematic.SEQUENCE_SECONDS)
            update.eye_target = cinematic.camera_at(t % cinematic.SEQUENCE_SECONDS)[:2]
            return
        eye, target, focal = view(t)
        update.eye_target = (eye, target)
        cam.GetFocalLengthAttr().Set(focal)
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())

    update.wave = wave
    update(0.0)
    print(f"rig sim: cinematic preset {preset} scene {scene_path.name}", flush=True)
    return path, update


def load_atmosphere(stage, cam_path: str, args):
    import sys

    sys.path.insert(0, "/rig/sim")
    import atmos
    from atmos_fx import Atmosphere

    import weather as weather_mod

    schedule = atmos.parse_schedule(weather_mod.atmos_kind(args.weather))
    first = atmos.schedule_at(schedule, 0.0)
    atm = Atmosphere(stage, args.cinematic, args.atmos, first, cam_path=cam_path)
    if first is not None:
        atm.set_weather(first, args.wind_dir, args.gust)
    state = {"kind": first}

    def tick(t: float) -> None:
        kind = atmos.schedule_at(schedule, t)
        if kind is not None and kind != state["kind"]:
            state["kind"] = kind
            atm.set_weather(kind, args.wind_dir, args.gust)
            print(f"rig sim: weather {kind}", flush=True)
        atm.update(t)

    print(f"rig sim: atmosphere tier={args.atmos} weather={args.weather or 'preset'}", flush=True)
    return atm, tick


def load_fleet_log(path: str):
    import json

    ticks = [json.loads(line) for line in open(path) if line.startswith('{"type": "tick"')]
    return [x["t"] for x in ticks], ticks


def make_obstacles(stage, cine_update, args):
    from rig.sim.obstacle_math import tick_index
    from rig.sim.obstacles import ObstacleScene

    scene = ObstacleScene(stage, wave=getattr(cine_update, "wave", None), radius=args.path_radius)
    replay = None
    if args.fleet_log:
        times, ticks = load_fleet_log(args.fleet_log)
        replay = {"times": times, "ticks": ticks, "i": -1}
        print(f"rig sim: replaying {len(ticks)} fleet ticks from {args.fleet_log} at x{args.fleet_speed}", flush=True)

    def tick(t: float) -> None:
        if replay:
            i = tick_index(replay["times"], (t * args.fleet_speed) % (replay["times"][-1] + 1.0))
            if i != replay["i"]:
                replay["i"] = i
                scene.sync(replay["ticks"][i])
        scene.tick(t)

    return scene, tick


def weather_kw(fn, args: dict) -> dict:
    ps = inspect.signature(fn).parameters
    anykw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values())
    return {k: args[k] for k in ("level", "wind_dir", "gust", "time", "rain", "fog") if k in args and (anykw or k in ps)}


def handle_event(scene, event: dict, atm=None, weather=None) -> None:
    name, args = event.get("name"), event.get("args") or {}
    if weather is not None and name == "set_weather":
        weather.set(**weather_kw(weather.set, args))
    if atm is not None and name == "set_weather" and "level" in args:
        atm.set_weather(**weather_kw(atm.set_weather, args))
    if scene is not None and name == "fleet":
        scene.sync(args)
        return
    if scene is not None and name == "set_weather" and "level" in args:
        scene.set_weather(**{k: args[k] for k in ("level", "wind_dir", "gust") if k in args})
    print(f"rig sim: event {name} {args}", flush=True)


def make_orbit_camera(stage, src_path: str, args):
    from pxr import Gf, Usd, UsdGeom

    from rig.sim.orbit import Orbit, look_at_rows, orbit_eye, view_target

    src = UsdGeom.Camera(stage.GetPrimAtPath(src_path))
    m = src.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rows = [tuple(m.GetRow(i)) for i in range(4)]
    target = args.orbit_target or view_target(rows, args.orbit_z)
    period = args.orbit
    orbit = Orbit.from_eye(rows[3][:3], target, period, radius=args.orbit_radius, z=args.orbit_eye_z)
    cam = UsdGeom.Camera.Define(stage, ORBIT_CAMERA)
    for name in ("focalLength", "horizontalAperture", "verticalAperture", "clippingRange"):
        value = src.GetPrim().GetAttribute(name).Get()
        if value is not None:
            cam.GetPrim().GetAttribute(name).Set(value)
    op = cam.AddTransformOp()

    def update(t: float) -> None:
        op.Set(Gf.Matrix4d(look_at_rows(orbit_eye(orbit, t), orbit.target)))

    update(0.0)
    print(f"rig sim: orbit r={orbit.radius:.1f} z={orbit.z:.1f} target={tuple(round(c, 2) for c in orbit.target)} period={period}s", flush=True)
    return ORBIT_CAMERA, update


def make_camera_header(stage, cam_path: str, w: int, h: int):
    from pxr import Usd, UsdGeom

    from rig.camera import from_usd

    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))

    def header() -> dict:
        m = cam.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return from_usd([list(m.GetRow(i)) for i in range(4)], cam.GetFocalLengthAttr().Get(),
                        cam.GetHorizontalApertureAttr().Get(), w, h)

    return header


def scene_info(args) -> dict:
    return {"cinematic": args.cinematic, "merged": bool(args.merged), "scene": args.scene, "env": args.env,
            "sea_level": args.sea_level, "look": args.look if args.cinematic else "none",
            "atmos": args.atmos if args.cinematic else "", "lens": bool(args.lens)}


def make_view(stage, cam_path: str):
    from pxr import Usd, UsdGeom

    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))

    def view() -> dict:
        m = cam.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return {"m": [round(float(m[i][j]), 6) for i in range(4) for j in range(4)],
                "focal": float(cam.GetFocalLengthAttr().Get()), "hap": float(cam.GetHorizontalApertureAttr().Get()),
                "vap": float(cam.GetVerticalApertureAttr().Get()), "clip": list(cam.GetClippingRangeAttr().Get())}

    return view


def disable_viewport() -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        vp = get_active_viewport()
        if vp is not None:
            vp.updates_enabled = False
    except Exception as exc:
        print(f"rig sim: viewport left on ({exc})", flush=True)


def wave_of(tickers: list):
    for tick in tickers:
        if isinstance(getattr(tick, "wave", None), dict):
            return tick.wave
    return None


def drain(sender, leak, obstacles, atm, sensors, weather=None) -> None:
    while not sender.inbox.empty():
        ev = sender.inbox.get_nowait()
        if sensors is not None and sensors.event(ev):
            continue
        if leak and ev["name"] == "gas_leak":
            print(f"rig sim: gas_leak {ev['args']} -> {leak.event(ev['args'])}", flush=True)
        else:
            handle_event(obstacles, ev, atm, weather)


def make_sensors(stage, obstacles, leak, args):
    if not args.views or obstacles is None:
        return None
    from rig.sim.sensors import SensorRig

    return SensorRig(stage, obstacles, leak, size=args.views_size, rate=args.views_rate, mode=args.views_mode)


def run_remote(app, args, stage, cam_path: str, tickers: list, obstacles=None, leak=None, wave=None, atm=None,
               sensors=None, weather=None) -> None:
    import json

    import carb

    from rig.render.state import SYNC_SETTINGS, SettingsWatch
    from rig.render.usdsync import StageCapture
    from rig.sim.sender import FrameSender

    disable_viewport()
    omni_timeline().play()
    for _ in range(WARMUP_UPDATES):
        app.update()
    capture = StageCapture(stage)
    if obstacles is not None:
        obstacles.republish()
    settings = SettingsWatch(carb.settings.get_settings().get, SYNC_SETTINGS)
    view = make_view(stage, cam_path)
    scene = scene_info(args)
    host, port = args.sim.rsplit(":", 1)
    sender = FrameSender(host, int(port)).start()
    period = 1.0 / args.fps
    start = next_tick = time.monotonic()
    stats_at = start + STATS_EVERY_S
    seq = steps = 0
    connects = -1
    nattrs = 0
    print(f"rig sim: remote render mode, publishing scene state at {args.fps:.0f} Hz -> {args.sim}", flush=True)
    while app.is_running():
        now = time.monotonic()
        t = now - start
        drain(sender, leak, obstacles, atm, sensors, weather)
        for tick in tickers:
            tick(t)
        app.update()
        delta = capture.collect()
        changed = settings.poll()
        if sender.connects != connects:
            connects = sender.connects
            delta, changed = capture.keyframe(), settings.all()
        msg = {"type": "state", "seq": seq, "t": t, "ts": time.time(), "scene": scene, "view": view(), **delta}
        if changed:
            msg["settings"] = changed
        if wave is not None:
            msg["ocean"] = {"height": round(float(wave["height"]), 4)}
        if atm is not None:
            msg["weather"] = atm.w
        plume = leak.header(t) if leak else None
        if plume:
            msg["plume"] = plume
        views = sensors.take() if sensors is not None else None
        if views:
            msg["views"] = views
        sender.send(msg, b"")
        seq += 1
        steps += 1
        nattrs += len(delta["attrs"])
        if now >= stats_at:
            nbytes = len(json.dumps(msg, separators=(",", ":")))
            span = now - stats_at + STATS_EVERY_S
            print(f"t={t:.0f}s state_hz={steps / span:.1f} attrs/msg={nattrs / max(steps, 1):.0f} msg={nbytes / 1024:.1f}KB "
                  f"sent={sender.sent} dropped={sender.dropped} connected={sender.connected.is_set()} skipped={capture.dropped}"
                  + (f" {sensors.summary()}" if sensors is not None else ""),
                  flush=True)
            stats_at, steps, nattrs = now + STATS_EVERY_S, 0, 0
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


def omni_timeline():
    import omni.timeline

    return omni.timeline.get_timeline_interface()


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})

    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image

    from rig.sim.sender import FrameSender

    settings = carb.settings.get_settings()
    for key, value in render_settings(args).items():
        settings.set(key, value)

    ctx = omni.usd.get_context()
    tickers = []
    obstacles = None
    weather = None
    if args.scene == "test":
        from rig.sim.scene import CAMERA_PATH, build_scene, spin

        scene = build_scene(ctx.get_stage())
        cam_path = CAMERA_PATH
        tickers.append(lambda t: spin(scene, t))
    elif args.cinematic:
        cam_path, cine_update = load_cinematic(app, ctx, args.cinematic, args)
        tickers.append(cine_update)
        obstacles, obstacle_tick = make_obstacles(ctx.get_stage(), cine_update, args)
        tickers.append(obstacle_tick)
        import weather as weather_mod

        weather = weather_mod.Controller(ctx.get_stage(), getattr(cine_update, "wave", None), args.cinematic)
        tickers.append(weather.tick)
        if weather_mod.parse_default(args.weather):
            weather.set(**weather_mod.parse_default(args.weather))
    else:
        open_scene(app, ctx, args.scene)
        cam_path = args.camera
    stage = ctx.get_stage()
    if not stage.GetPrimAtPath(cam_path).IsValid():
        raise SystemExit(f"rig sim: camera {cam_path} not in scene")
    if args.orbit:
        cam_path, orbit_update = make_orbit_camera(stage, cam_path, args)
        tickers.append(orbit_update)
    camera_header = make_camera_header(stage, cam_path, args.width, args.height)
    grade = None
    if args.cinematic and args.look != "none":
        import cinematic
        import look

        for key, value in look.rtx_settings(args.look).items():
            settings.set(key, value)

        def expose(t: float) -> None:
            eye, target = getattr(cine_update, "eye_target", ((0.0, 0.0, 0.0), (0.0, 0.0, 30.0)))
            ev = look.view_look(None, float(np.linalg.norm(np.subtract(target, eye))), bool(args.atmos), cinematic.CURRENT["time"])["ev"]
            for key, value in look.exposure(ev).items():
                settings.set(key, value)

        tickers.append(expose)
        grade = look.grade
        import shafts

        beams = (shafts.specs(cinematic.AREA_LIGHTS, cinematic.SEA_FLOODS, cinematic.DECK_FLOODS, args.look) if cinematic.PRESETS[args.cinematic]["lights"] > 0 else [])
    atm = None
    if args.atmos and args.cinematic:
        atm, atm_tick = load_atmosphere(stage, cam_path, args)
        tickers.append(atm_tick)
        if weather is not None:
            weather.atm = atm
    env = load_environment(stage, args.env, args.sea_level)
    if env is not None:
        tickers.append(lambda t: env.update(t, camera=cam_path))

    leak = None
    if args.leak:
        from sim.plume.isaac import LeakStream

        leak = LeakStream(stage, args)
        tickers.append(leak.tick)
    sensors = make_sensors(stage, obstacles, leak, args)
    if sensors is not None:
        tickers.append(sensors.tick)
    if args.render_mode == "remote":
        run_remote(app, args, stage, cam_path, tickers, obstacles, leak, wave_of(tickers), atm, sensors, weather)

    rp = rep.create.render_product(cam_path, (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    dist_ann = None
    if grade is not None:
        dist_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        dist_ann.attach(rp)
    omni.timeline.get_timeline_interface().play()
    for _ in range(WARMUP_UPDATES):
        app.update()

    host, port = args.sim.rsplit(":", 1)
    sender = FrameSender(host, int(port)).start()
    period = 1.0 / args.fps
    start = next_tick = time.monotonic()
    stats_at = start + STATS_EVERY_S
    snap_at = None if not args.snapshot else start
    seq = frames = 0
    stats_frames, stats_fresh, stats_start = 0, 0, start
    last_sample = b""
    last_sent = 0.0
    mean = 0.0
    jpeg_kb = 0.0
    print(
        f"rig sim: scene={args.scene} camera={cam_path} renderer={args.renderer} {args.width}x{args.height} "
        f"spp={args.spp} subframes={args.subframes} wait={args.wait_render} -> {args.sim}",
        flush=True,
    )

    while app.is_running():
        now = time.monotonic()
        t = now - start
        drain(sender, leak, obstacles, atm, sensors, weather)
        for tick in tickers:
            tick(t)
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False, wait_for_render=args.wait_render)
        data = np.asarray(ann.get_data())
        sample = data[::24, ::24, :3].tobytes() if data.ndim == 3 and data.size else b""
        fresh = bool(sample) and sample != last_sample
        last_sample = sample or last_sample
        stats_fresh += fresh
        if fresh or (sample and now - last_sent >= KEEPALIVE_S):
            last_sent = now
            rgb = np.ascontiguousarray(data[..., :3], dtype=np.uint8)
            if atm is not None:
                rgb = atm.post(rgb, camera_header(), t, args.lens)
            if grade is not None:
                now = cinematic.CURRENT["time"]
                rgb = shafts.add(rgb, shafts.layer(camera_header(), beams, t)) if beams else rgb
                mist = look.mist_level(settings.get("/rtx/fog/fogDistanceDensity"))
                if now in look.INSCATTER or mist > 0.0:
                    rgb = look.inscatter(rgb, lambda: np.asarray(dist_ann.get_data()), camera_header(), now,
                                         look.sun_direction(cinematic.PRESETS[now]["sun_rotation"]), tier=args.look, mist=mist)
                rgb = grade(rgb, args.look, t, now)
            mean = float(rgb.mean())
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "JPEG", quality=args.quality)
            jpeg_kb = buf.tell() / 1024
            head = {"type": "frame", "w": rgb.shape[1], "h": rgb.shape[0], "seq": seq, "t": t, "mean": round(mean, 1),
                    "camera": camera_header()}
            plume = leak.header(t) if leak else None
            if plume:
                head["plume"] = plume
            views = sensors.take() if sensors is not None else None
            if views:
                head["views"] = views
            sender.send(head, buf.getvalue())
            seq += 1
            if snap_at is not None and now >= snap_at and mean > 3:
                path = args.snapshot if not args.snapshot_every else args.snapshot.replace(".png", f"_{int(t):04d}.png")
                Image.fromarray(rgb).save(path)
                print(f"snapshot saved to {path}", flush=True)
                snap_at = now + args.snapshot_every if args.snapshot_every else None
        frames += 1
        stats_frames += 1
        if now >= stats_at:
            print(
                f"t={t:.0f}s fps={stats_fresh / (now - stats_start):.1f} steps={stats_frames / (now - stats_start):.1f} sent={sender.sent} dropped={sender.dropped} "
                f"connected={sender.connected.is_set()} mean={mean:.1f} jpeg={jpeg_kb:.0f}KB"
                + (f" plume_ms={leak.sim_ms:.1f} pts={leak.visual.count} vol10={leak.stats['vol_10lel_m3']:.2f}" if leak and leak.stats else "")
                + (f" {sensors.summary()}" if sensors is not None else ""),
                flush=True,
            )
            stats_at = now + STATS_EVERY_S
            stats_frames, stats_fresh, stats_start = 0, 0, now
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

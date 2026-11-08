from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DARK = (7, 9, 13)
EDGE = (190, 222, 255)
CAM_PATH = "/World/Cinematic/Cameras/stream"
WIDE = {"at": (16.6, -9.7, 42.9), "dist": 8.5, "elev": 50.0, "focal": 18.0, "az": -78.0}
CLOSE = {"at": (20.0, -9.4, 43.3), "dist": 4.2, "elev": 48.0, "focal": 20.0, "az": -68.0}


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def shot(s: dict):
    a, e = math.radians(s["az"]), math.radians(s["elev"])
    b = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    at = np.asarray(s["at"], float)
    return at + s["dist"] * b, at, s["focal"]


def camera_at(fi: int, F: int, push: tuple[int, int], drift: float = 4.0, wide=WIDE, close=CLOSE):
    u = smooth((fi - push[0]) / max(push[1] - push[0], 1))
    k = fi / max(F - 1, 1)
    s = {key: (1 - u) * np.asarray(wide[key], float) + u * np.asarray(close[key], float) for key in wide}
    s["az"] = float(s["az"]) - drift * (0.5 - k)
    s = {key: (tuple(v) if np.ndim(v) else float(v)) for key, v in s.items()}
    return shot(s)


def wire_layer(e, depth, valid):
    near = np.clip(1.6 - depth / 14.0, 0.35, 1.0) * valid
    return np.asarray(DARK, np.float32) + (np.asarray(EDGE, np.float32) - np.asarray(DARK, np.float32)) * (e * near)[..., None]


def geo_layer(nrm, depth, valid, light=(0.55, -0.65, 0.5)):
    l_ = np.asarray(light) / np.linalg.norm(light)
    lam = np.clip(nrm @ l_, 0, 1)
    floor = np.where(nrm[..., 2] > 0.95, 0.3, 1.0)
    shade = ((0.2 + 0.75 * lam) * floor)[..., None] * np.asarray([150, 160, 176], np.float32)
    geo = np.where(valid[..., None], shade * np.clip(1.25 - depth / 18.0, 0.3, 1.0)[..., None], np.asarray(DARK, np.float32))
    return np.clip(geo, 0, 255).astype(np.uint8)


def clear_path(st, path, stop, r=0.6, r_stop=1.1, south=-9.45, deck=42.4, top=43.6):
    from pxr import Usd, UsdGeom

    path = np.asarray(path, float)[::3, :2]
    n = 0
    for prim in st.Traverse():
        if prim.GetParent() and prim.GetParent().GetName() == "Clutter" and prim.GetName().split("_")[0] in ("tarp", "strap"):
            prim.SetActive(False)
            continue
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        pi = UsdGeom.PointInstancer(prim)
        P = np.asarray(pi.GetPositionsAttr().Get() or [], float)
        if not len(P):
            continue
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        Wp = P @ M[:3, :3] + M[3, :3]
        d = np.min(np.linalg.norm(Wp[:, None, :2] - path[None], axis=2), axis=1)
        ds = np.linalg.norm(Wp[:, :2] - np.asarray(stop, float)[:2], axis=1)
        hide = np.flatnonzero(((d < r) | (ds < r_stop)) & (Wp[:, 1] < south) & (Wp[:, 2] > deck) & (Wp[:, 2] < top))
        if len(hide):
            ids = list(pi.GetInvisibleIdsAttr().Get() or [])
            pi.GetInvisibleIdsAttr().Set(sorted(set(ids) | set(int(i) for i in hide)))
            n += len(hide)
    return n


def compose(layers, fi, cuts, W, H):
    from policy.pickcan.layers import wipe_mask

    if fi < cuts[0]:
        return layers[0]
    if fi < cuts[1]:
        m = wipe_mask(W, H, (fi - cuts[0]) / (cuts[1] - cuts[0]))
        return layers[1] * m + layers[0] * (1 - m)
    if fi < cuts[2]:
        return layers[1]
    if fi < cuts[3]:
        m = wipe_mask(W, H, (fi - cuts[2]) / (cuts[3] - cuts[2]))
        return layers[2] * m + layers[1] * (1 - m)
    return layers[2]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="runs/gas/plan.npz")
    p.add_argument("--out_dir", default="runs/gas/layers")
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_merged.usda"))
    p.add_argument("--name", default="gas_layers")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--cuts", default="")
    p.add_argument("--push", default="")
    p.add_argument("--renderer", default="RealTimePathTracing")
    p.add_argument("--subframes", type=int, default=6)
    p.add_argument("--rain", type=float, default=0.8)
    p.add_argument("--wet", type=float, default=0.35)
    p.add_argument("--fog", type=float, default=0.25)
    p.add_argument("--ev", type=float, default=0.0)
    p.add_argument("--probe", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wide", default="")
    p.add_argument("--close", default="")
    args = p.parse_args(argv)
    W, H = args.width, args.height
    wide = {**WIDE, **json.loads(args.wide or "{}")}
    close = {**CLOSE, **json.loads(args.close or "{}")}

    from policy.gasbot import clip, fx, sees

    plan = clip.load(args.plan)
    frames = plan["frames"]
    F = len(frames)
    cuts = [int(c) for c in (args.cuts or ",".join(str(c) for c in plan.get("cuts", (75, 90, 165, 180)))).split(",")]
    push = tuple(int(c) for c in args.push.split(",")) if args.push else (cuts[2] - 6, min(cuts[3] + 40, F - 1))
    os.makedirs(args.out_dir, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": W, "height": H})

    import carb
    import imageio
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw, ImageFilter
    from pxr import Gf, Sdf, UsdGeom

    from integration import repair_scene as rsc
    from policy.pickcan.layers import edges
    from policy.pickcan.video import cv_pose, intrinsics
    from rig.sim.gait import ARM, LEGS, PARK
    from rig.sim.joints import RobotJoints

    sys.path.insert(0, os.path.join(ROOT, "sim"))
    import cinematic
    import look
    import weather

    ctx = omni.usd.get_context()
    if not ctx.open_stage(args.scene):
        raise SystemExit(f"could not open {args.scene}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300:
        app.update()
        _, loaded, total = ctx.get_stage_loading_status()
        if total == 0 or loaded >= total:
            break
    st = ctx.get_stage()
    cinematic.apply_preset(st, "dusk")
    for path in ("/World/Robots", "/World/Obstacles"):
        pr = st.GetPrimAtPath(path)
        if pr.IsValid():
            pr.SetActive(False)

    print(f"[gas-layers] cleared {clear_path(st, [f['root'] for f in frames], frames[-1]['root'])} instances on the path", flush=True)
    root = "/World/GasBot"
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(st, root))
    top, rz = xf.AddTranslateOp(), xf.AddRotateZOp()
    body = UsdGeom.Xform.Define(st, f"{root}/body")
    body.GetPrim().GetReferences().AddReference(os.path.join(ROOT, "sim/output/robot/go2_ur3_lift/go2_ur3_lift.usda"))
    vs = body.GetPrim().GetVariantSets()
    if vs.HasVariantSet("Physics"):
        vs.GetVariantSet("Physics").SetVariantSelection("none")
    for _ in range(10):
        app.update()
    joints = RobotJoints(st, f"{root}/body")
    if hasattr(fx, "dress_tool"):
        fx.dress_tool(st, f"{root}/body")
    wheel = fx.Handwheel(st)
    vap = fx.Vapour(seed=args.seed, puff_rate=60.0, puff_alpha=0.75, jet_alpha=0.55)

    wc = weather.Controller(st, None, "dusk")
    wc._rain = lambda t, a: None
    wc.set(rain=args.wet, fog=args.fog, wind_dir=240.0)
    wc.tick(weather.RAMP_S + 1.0)
    cur = st.GetPrimAtPath(weather.RAIN_ROOT)
    if cur.IsValid():
        UsdGeom.Imageable(cur).MakeInvisible()

    cam = rsc.Camera(st, CAM_PATH, focal=WIDE["focal"], clip=(0.05, 20000.0))
    eye, at, focal = camera_at(0, F, push, wide=wide, close=close)
    cam.look(eye, at)
    s = carb.settings.get_settings()
    for k_, v_ in {**look.rtx_settings("offline"), "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False,
                   "/rtx/post/chromaticAberration/enabled": False}.items():
        s.set(k_, v_)
    rp = rep.create.render_product(cam.path, (W, H))
    ann = {}
    for kind in ("rgb", "distance_to_image_plane", "normals"):
        a_ = rep.AnnotatorRegistry.get_annotator(kind)
        a_.attach(rp)
        ann[kind] = a_
    sun_dir = look.sun_direction(cinematic.PRESETS["dusk"]["sun_rotation"])

    def place(fr, arm):
        x, y, z, yaw = fr["root"]
        with Sdf.ChangeBlock():
            top.Set(Gf.Vec3d(float(x), float(y), float(z)))
            rz.Set(math.degrees(float(yaw)))
        joints.set(LEGS, [float(v) for v in fr["legs"]])
        joints.set(ARM, [float(v) for v in arm])

    def grab():
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=False)
        rgb = np.asarray(ann["rgb"].get_data())[..., :3].copy()
        depth = np.asarray(ann["distance_to_image_plane"].get_data()).squeeze().astype(np.float32)
        nrm = np.asarray(ann["normals"].get_data()).reshape(H, W, -1)[..., :3].astype(np.float32)
        return rgb, depth, nrm

    place(frames[0], [PARK[j] for j in ARM])
    for _ in range(20):
        app.update()
    rng = np.random.default_rng(0)
    streaks = np.column_stack([rng.uniform(0, W, 520), rng.uniform(0, H, 520), rng.uniform(0.5, 1.0, 520)])
    sc = W / 1920.0
    order = [int(v) for v in args.probe.split(",")] if args.probe else list(range(F))
    writer = None if args.probe else imageio.get_writer(os.path.join(args.out_dir, f"{args.name}.mp4"), fps=args.fps,
                                                        quality=9, macro_block_size=8)
    arm = np.array([PARK[j] for j in ARM])
    stills, log = {}, []
    at_still = {"wire": cuts[0] // 2, "geo": (cuts[1] + cuts[2]) // 2, "full": F - 50}
    t_start = time.monotonic()
    last_v = None
    for fi in range(F):
        fr = frames[fi]
        if fr.get("valve_s") is not None:
            last_v = float(fr["valve_s"])
        arm = np.asarray(fx.valve_arm(last_v)) if last_v is not None else arm
        place(fr, arm)
        wheel.set(float(fx.wheel_angle(last_v)) if last_v is not None else 0.0)
        src = plan["src"]
        vap.step(float(fr["dt"]), src, fr["wind"], float(fr["wind_speed"]), bool(fr["leak_on"]))
        if fi not in order:
            continue
        eye, at, focal = camera_at(fi, F, push, wide=wide, close=close)
        cam.camera.GetFocalLengthAttr().Set(float(focal))
        cam.look(eye, at)
        ev = look.view_look(None, float(np.linalg.norm(np.subtract(at, eye))), False, "dusk")["ev"] + args.ev
        for k_, v_ in look.exposure(ev).items():
            s.set(k_, v_)
        wc.tick(weather.RAMP_S + 1.0 + fi / args.fps)
        K = intrinsics(float(focal), cam.aperture, W, H)
        T = cv_pose(cam.world_matrix())
        rgb, depth, nrm = grab()
        camd = {"K": K, "T": T}
        mist = look.mist_level(s.get("/rtx/fog/fogDistanceDensity"))
        full = look.inscatter(rgb, depth, camd, "dusk", sun_dir, 1.0, mist=mist)
        full = vap.draw(np.ascontiguousarray(full), K, T, depth)
        full = look.grade(full, "offline", fi / args.fps, "dusk")
        im = Image.fromarray(full).convert("RGBA")
        rl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(rl)
        for sx, sy, sv in streaks:
            y0 = (sy + fi * 55 * sv * sc) % H
            x0 = (sx - fi * 12 * sv * sc) % W
            dr.line([x0, y0, x0 - 12 * sv * sc, y0 + 55 * sv * sc], fill=(205, 212, 225, int(40 * args.rain * sv)),
                    width=max(1, int(2 * sc)))
        full = np.asarray(Image.alpha_composite(im, rl.filter(ImageFilter.GaussianBlur(0.8))).convert("RGB"))
        valid = np.isfinite(depth) & (depth > 0) & (depth < 60)
        e = edges(nrm, depth)
        wire = wire_layer(e, depth, valid)
        geo = sees.draw(geo_layer(nrm, depth, valid), K, T, depth, fr, plan, scale=sc)
        layers = [wire.astype(np.float32), np.asarray(geo, np.float32), np.asarray(full, np.float32)]
        if args.probe:
            for k_, L_ in zip(("wire", "geo", "full"), layers):
                Image.fromarray(np.clip(L_, 0, 255).astype(np.uint8)).save(
                    os.path.join(args.out_dir, f"probe_{fi:03d}_{k_}.jpg"), quality=92)
            print(f"[gas-layers] probe {fi}", flush=True)
            continue
        arr = np.clip(compose(layers, fi, cuts, W, H), 0, 255).astype(np.uint8)
        writer.append_data(arr)
        for key, at_ in at_still.items():
            if fi == at_:
                stills[key] = arr
        log.append({"frame": fi, "phase": fr["phase"], "t": round(float(fr["t"]), 3), "leak_on": bool(fr["leak_on"])})
        if fi % 10 == 0:
            el = time.monotonic() - t_start
            print(f"[gas-layers] frame {fi}/{F} {el:.0f}s", flush=True)
    if writer is not None:
        writer.close()
    if len(stills) == 3:
        strip = np.concatenate([np.asarray(Image.fromarray(stills[k_]).resize((W // 2, H // 2))) for k_ in ("wire", "geo", "full")], 1)
        Image.fromarray(strip).save(os.path.join(args.out_dir, f"{args.name}.png"))
    with open(os.path.join(args.out_dir, "layers.json"), "w") as f:
        json.dump({"metrics": plan.get("metrics"), "cuts": cuts, "push": push, "frames": log}, f)
    print("[gas-layers] RESULT " + json.dumps({"frames": F, "metrics": plan.get("metrics")}, default=str), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

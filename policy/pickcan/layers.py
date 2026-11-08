from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CORAL = (217, 119, 87)
DARK = (7, 9, 13)
EDGE = (190, 222, 255)
HDRI = os.path.join(ROOT, "sim/assets/third_party/hdris/belfast_sunset_puresky.hdr")


def edges(normals: np.ndarray, depth: np.ndarray) -> np.ndarray:
    n = normals[..., :3].astype(np.float32)
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 1e3).astype(np.float32)
    e = np.zeros(d.shape, np.float32)
    for ax in (0, 1):
        dn = np.abs(np.diff(n, axis=ax)).sum(-1)
        dd = np.abs(np.diff(d, axis=ax)) / np.minimum(d[:-1, :] if ax == 0 else d[:, :-1], 50.0)
        m = np.clip((dn - 0.25) * 2.5, 0, 1) + np.clip((dd - 0.03) * 20.0, 0, 1)
        if ax == 0:
            e[:-1, :] = np.maximum(e[:-1, :], m)
        else:
            e[:, :-1] = np.maximum(e[:, :-1], m)
    return np.clip(e, 0, 1)


def wipe_mask(w: int, h: int, s: float, feather: float = 0.08) -> np.ndarray:
    x = np.linspace(0, 1, w)[None, :] * 0.85 + np.linspace(0, 1, h)[:, None] * 0.15
    edge = s * (1 + feather) - feather
    return np.clip((edge - x) / feather + 1, 0, 1)[..., None]


def project(p, K, T):
    q = (np.asarray(p, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = q[..., 2]
    uv = np.stack([K[0, 0] * q[..., 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * q[..., 1] / np.maximum(z, 1e-6) + K[1, 2]], -1)
    return uv, z


def depth_colour(t: np.ndarray) -> np.ndarray:
    stops = np.array([[70, 140, 255], [80, 230, 210], [250, 230, 120], [255, 140, 90]], np.float32)
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="runs/pickup/layers")
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_deck.usda"))
    p.add_argument("--policy", default="mlp", choices=["mlp", "diffusion"])
    p.add_argument("--ckpt", default=os.path.join(ROOT, "policy/pickcan/weights/mlp.npz"))
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--max_steps", type=int, default=480)
    p.add_argument("--tail", type=int, default=24)
    p.add_argument("--cuts", default="36,44,82,90")
    p.add_argument("--dome", type=float, default=380.0)
    p.add_argument("--sun", type=float, default=900.0)
    p.add_argument("--key", type=float, default=3.0e4)
    p.add_argument("--rain", type=float, default=1.0)
    p.add_argument("--replan", type=int, default=4)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless, args.enable_cameras = True, True
    AppLauncher(args)

    import carb
    import gymnasium as gym
    import imageio
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image, ImageDraw, ImageFilter
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    from isaaclab.sim import RenderCfg

    from integration import repair_scene as rsc
    from policy import night_look
    from policy.envs import pickcan_env as PE
    from policy.envs.pickcan_env_cfg import CANS, PickCanEnvCfg
    from policy.envs.scrub_env import _add_backdrop
    from policy.pickcan import core
    from policy.pickcan.diffusion import DiffusionPolicy, MLPPolicy, load_weights
    from policy.pickcan.video import CAN_XY, ROBOT_XY, cv_pose, intrinsics
    from policy.storm import Lens

    sys.path.insert(0, os.path.join(ROOT, "sim"))
    import look

    W, H = args.width, args.height
    cfg = PickCanEnvCfg()
    cfg.scene.num_envs = 1
    cfg.init_joint_noise = 0.0
    cfg.episode_length_s = 1.0e5
    cfg.det_sigma_xy, cfg.det_sigma_z, cfg.det_jitter = 0.0, 0.0, 0.0
    cfg.robot_cfg.spawn.make_instanceable = False
    cfg.sim.render_interval = 10 ** 9
    cfg.sim.render = RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", carb_settings={
        "/rtx/rendermode": "RaytracedLighting", "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False})
    setup0 = PE.PickCanEnv._setup_scene

    def setup(self):
        setup0(self)
        st = omni.usd.get_context().get_stage()
        _add_backdrop(os.path.abspath(args.scene), (-ROBOT_XY[0], -ROBOT_XY[1], 0.0))
        UsdGeom.Imageable(st.GetPrimAtPath("/World/ground")).MakeInvisible()
        st.GetPrimAtPath("/World/Light").SetActive(False)
        for path in ("/World/Backdrop/Robots", "/World/Backdrop/Pipes/damage_patch"):
            prim = st.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetActive(False)
        night_look.wet_down(st, "/World/Backdrop")
        night_look.paint_robot(st, "/World/envs/env_0/Robot", "graphite")
        sky = UsdLux.DomeLight(st.GetPrimAtPath("/World/Backdrop/Environment/sky"))
        sky.GetTextureFileAttr().Set(Sdf.AssetPath(HDRI))
        sky.GetIntensityAttr().Set(args.dome)
        sun = UsdLux.DistantLight(st.GetPrimAtPath("/World/Backdrop/Environment/sun"))
        sun.GetIntensityAttr().Set(args.sun)
        sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.58, 0.3))
        xf = UsdGeom.Xformable(sun.GetPrim())
        ops = xf.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3f(84.0, 0.0, 215.0))
        night_look._lamp(st, "/World/PickKey", (0.9, 1.6, 1.9), (1.0, 0.86, 0.7), 0.12, args.key)
        night_look._lamp(st, "/World/PickRim", (-1.2, 0.8, 1.4), night_look.LED, 0.1, 0.35 * args.key)
        s = carb.settings.get_settings()
        s.set_bool("/rtx/fog/enabled", True)
        s.set_bool("/rtx/fog/fogZup/enabled", True)
        s.set_float_array("/rtx/fog/fogColor", [0.42, 0.27, 0.17])
        s.set_float("/rtx/fog/fogColorIntensity", 1.0)
        s.set_float("/rtx/fog/fogStartDist", 4.0)
        s.set_float("/rtx/fog/fogEndDist", 60.0)
        s.set_float("/rtx/fog/fogDistanceDensity", 0.55)
        s.set_float("/rtx/fog/fogHeightDensity", 0.0)

    PE.PickCanEnv._setup_scene = setup
    env = gym.make("Rig-PickCan-Go2UR3Lift-v0", cfg=cfg)
    u = env.unwrapped
    dev = u.device
    get_dones = u._get_dones

    def no_done():
        s_, t_ = get_dones()
        return torch.zeros_like(s_), torch.zeros_like(t_)

    u._get_dones = no_done
    settings = carb.settings.get_settings()
    for k_, v_ in {**cfg.sim.render.carb_settings, **look.TIERS["offline"]["rtx"],
                   "/rtx/post/motionblur/enabled": False}.items():
        settings.set(k_, v_)
    ids = torch.tensor([0], device=dev)
    name, can_r, can_h, _ = CANS[0]
    net = load_weights((DiffusionPolicy if args.policy == "diffusion" else MLPPolicy)(42), args.ckpt, dev)
    net.eval()
    gen = torch.Generator(device=dev).manual_seed(0)

    def start():
        with torch.inference_mode():
            env.reset(seed=3)
            u.place_can(ids, torch.tensor([CAN_XY], device=dev), torch.tensor([0.7], device=dev),
                        torch.tensor([True], device=dev))
            u.det_bias[:] = 0.0
            u.det_yaw_bias[:] = 0.0
            for _ in range(2):
                env.step(torch.tensor([[0.0, 0.0, 0.0, -1.0, -1.0]], device=dev))

    def policy_step(state):
        if state["plan"] is None or state["k"] >= args.replan:
            o = u.geo_obs()
            state["plan"] = net.sample(o, n_steps=8, generator=gen) if args.policy == "diffusion" else net.sample(o)
            state["k"] = 0
        a = state["plan"][:, state["k"]]
        state["k"] += 1
        return a

    def landed():
        cp, _, cv, _ = u._can_state()
        return bool(core.in_tote(cp)[0]) and not bool(u.sealed[0]) and float(cv[0].norm()) < 0.05

    start()
    state = {"plan": None, "k": 0}
    n_end, n_seal = None, None
    with torch.inference_mode():
        for step in range(args.max_steps):
            env.step(policy_step(state))
            if n_seal is None and bool(u.sealed[0]):
                n_seal = step
            if landed():
                n_end = step
                break
    print(f"[layers] pass1 sealed at step {n_seal}, landed at step {n_end}", flush=True)
    n_seal = n_seal or 130
    if n_end is None:
        n_end = args.max_steps - args.tail
    total = n_end + args.tail

    stage = omni.usd.get_context().get_stage()
    cam = rsc.Camera(stage, "/World/Cameras/layers", focal=22.0)
    eye = np.array([math.cos(1.05) * 1.95, 0.2 + math.sin(1.05) * 1.95, 1.42])
    look_at = np.array([-0.1, 0.12, 0.4])
    cam.look(eye, look_at)
    rp = rep.create.render_product(cam.path, (W, H))
    ann = {}
    for kind in ("rgb", "distance_to_image_plane", "normals"):
        a_ = rep.AnnotatorRegistry.get_annotator(kind)
        a_.attach(rp)
        ann[kind] = a_
    K = intrinsics(22.0, cam.aperture, W, H)
    T = cv_pose(cam.world_matrix())
    sun_dir = look.sun_direction((84.0, 0.0, 215.0))
    camd = {"K": K, "T": T}

    start()
    state = {"plan": None, "k": 0}
    for _ in range(6):
        u.sim.render()
    cuts = [int(c) for c in args.cuts.split(",")]
    F = args.frames
    knee_f, knee_s = cuts[2] - 4, min(n_seal + 12, total - 2)
    frame_step = [int(round(f * knee_s / knee_f)) if f <= knee_f else
                  int(round(knee_s + (f - knee_f) * (total - 1 - knee_s) / (F - 1 - knee_f))) for f in range(F)]
    os.makedirs(args.out_dir, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.out_dir, "pickcan_layers.mp4"), fps=args.fps, quality=9,
                                macro_block_size=8)
    lens = Lens(seed=4, rate=4.0)
    rng = np.random.default_rng(0)
    streaks = np.column_stack([rng.uniform(0, W, 520), rng.uniform(0, H, 520), rng.uniform(0.5, 1.0, 520)])
    gx = np.arange(-3.0, 3.01, 0.5)
    stills, log, step, fi = {}, [], 0, 0
    fit_shown = None
    while fi < F:
        with torch.inference_mode():
            while step < frame_step[fi]:
                env.step(policy_step(state))
                step += 1
            if not bool(u.ever_sealed[0]):
                fit_shown = {k_: v_[0].cpu().numpy() for k_, v_ in u.geom_fit.items()}
            pts = u.geom_pts[0][u.geom_hit[0]].cpu().numpy()
            deye = u.geom_eye[0].cpu().numpy()
            can_p = u._can_state()[0][0].cpu().numpy()
        reps = 0
        while fi < F and frame_step[fi] == step:
            if reps == 0:
                u.sim.render()
                u.sim.render()
                rgb = np.asarray(ann["rgb"].get_data())[..., :3].copy()
                depth = np.asarray(ann["distance_to_image_plane"].get_data()).squeeze().astype(np.float32)
                nrm = np.asarray(ann["normals"].get_data())
                nrm = nrm.reshape(H, W, -1)[..., :3].astype(np.float32)
                full = look.inscatter(rgb, depth, camd, "dusk", sun_dir, 1.0)
                full = look.grade(full, "offline", fi / args.fps, "dusk")
                im = Image.fromarray(full)
                rl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dr = ImageDraw.Draw(rl)
                for sx, sy, sv in streaks:
                    y0 = (sy + fi * 55 * sv) % H
                    x0 = (sx + fi * 9 * sv) % W
                    dr.line([x0, y0, x0 + 9 * sv, y0 + 55 * sv], fill=(205, 212, 225, int(46 * args.rain * sv)), width=2)
                im = Image.alpha_composite(im.convert("RGBA"), rl.filter(ImageFilter.GaussianBlur(0.8))).convert("RGB")
                full = lens.apply(np.asarray(im), 0.06)
                e = edges(nrm, depth)
                valid = np.isfinite(depth) & (depth > 0) & (depth < 60)
                near = np.clip(1.6 - depth / 6.0, 0.35, 1.0) * valid
                wire = np.asarray(DARK, np.float32) + (np.asarray(EDGE, np.float32) - np.asarray(DARK, np.float32)) * (e * near)[..., None]
                wim = Image.fromarray(np.clip(wire, 0, 255).astype(np.uint8))
                dw = ImageDraw.Draw(wim, "RGBA")
                for g in gx:
                    for a_, b_ in (((g, -3.0, 0.0), (g, 3.0, 0.0)), ((-3.0, g, 0.0), (3.0, g, 0.0))):
                        seg = np.linspace(a_, b_, 40)
                        uv, z = project(seg, K, T)
                        okz = z > 0.1
                        if okz.sum() > 1:
                            dw.line([tuple(q) for q in uv[okz]], fill=(70, 110, 150, 70), width=1)
                l_ = np.asarray([0.55, -0.65, 0.5]) / np.linalg.norm([0.55, -0.65, 0.5])
                lam = np.clip(nrm @ l_, 0, 1)
                floor = np.where(nrm[..., 2] > 0.95, 0.3, 1.0)
                shade = ((0.2 + 0.75 * lam) * floor)[..., None] * np.asarray([150, 160, 176], np.float32)
                geo = np.where(valid[..., None], shade * np.clip(1.25 - depth / 8.0, 0.3, 1.0)[..., None], np.asarray(DARK, np.float32))
                gim = Image.fromarray(np.clip(geo, 0, 255).astype(np.uint8))
                dg = ImageDraw.Draw(gim, "RGBA")
                for g in gx:
                    for a_, b_ in (((g, -3.0, 0.0), (g, 3.0, 0.0)), ((-3.0, g, 0.0), (3.0, g, 0.0))):
                        seg = np.linspace(a_, b_, 40)
                        uv, z = project(seg, K, T)
                        okz = z > 0.1
                        if okz.sum() > 1:
                            dg.line([tuple(q) for q in uv[okz]], fill=(90, 120, 150, 60), width=1)
                euv, ez = project(deye, K, T)
                if len(pts):
                    puv, pz = project(pts, K, T)
                    rng_ = np.linalg.norm(pts - deye, axis=-1)
                    col = depth_colour((rng_ - rng_.min()) / max(np.ptp(rng_), 1e-3))
                    if ez > 0.1:
                        for q in puv[::5]:
                            dg.line([tuple(euv), tuple(q)], fill=(120, 200, 255, 26), width=1)
                    for q, c_ in zip(puv, col):
                        dg.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=tuple(int(v) for v in c_) + (240,))
                if ez > 0.1:
                    dg.ellipse([euv[0] - 6, euv[1] - 6, euv[0] + 6, euv[1] + 6], outline=(120, 200, 255, 255), width=2)
                if fit_shown is not None and fit_shown["conf"] > 0.1 and not bool(u.ever_sealed[0]):
                    c0, ax, r0, L0 = fit_shown["centre"], fit_shown["axis"], float(fit_shown["radius"]), float(fit_shown["length"])
                    ref = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                    b1 = np.cross(ax, ref)
                    b1 /= np.linalg.norm(b1)
                    b2 = np.cross(ax, b1)
                    th = np.linspace(0, 2 * math.pi, 48)
                    for sgn in (-1, 1):
                        ring = c0 + sgn * ax * L0 / 2 + r0 * (np.cos(th)[:, None] * b1 + np.sin(th)[:, None] * b2)
                        uv, _ = project(ring, K, T)
                        dg.line([tuple(q) for q in uv], fill=CORAL + (255,), width=4)
                    for a0 in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
                        off = r0 * (math.cos(a0) * b1 + math.sin(a0) * b2)
                        uv, _ = project(np.stack([c0 - ax * L0 / 2 + off, c0 + ax * L0 / 2 + off]), K, T)
                        dg.line([tuple(q) for q in uv], fill=CORAL + (220,), width=3)
                    uv, _ = project(np.stack([c0 - ax * (L0 / 2 + 0.05), c0 + ax * (L0 / 2 + 0.05)]), K, T)
                    for t0 in np.linspace(0, 1, 12)[::2]:
                        q0 = uv[0] + (uv[1] - uv[0]) * t0
                        q1 = uv[0] + (uv[1] - uv[0]) * min(t0 + 1 / 12, 1)
                        dg.line([tuple(q0), tuple(q1)], fill=CORAL + (255,), width=2)
                    gp, _ = project(fit_shown["grasp"], K, T)
                    dg.ellipse([gp[0] - 12, gp[1] - 12, gp[0] + 12, gp[1] + 12], fill=CORAL + (255,), outline=(255, 235, 225, 255), width=3)
                layers = [np.asarray(wim, np.float32), np.asarray(gim, np.float32), np.asarray(full, np.float32)]
            if fi < cuts[0]:
                out = layers[0]
            elif fi < cuts[1]:
                m = wipe_mask(W, H, (fi - cuts[0]) / (cuts[1] - cuts[0]))
                out = layers[1] * m + layers[0] * (1 - m)
            elif fi < cuts[2]:
                out = layers[1]
            elif fi < cuts[3]:
                m = wipe_mask(W, H, (fi - cuts[2]) / (cuts[3] - cuts[2]))
                out = layers[2] * m + layers[1] * (1 - m)
            else:
                out = layers[2]
            frame = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
            arr = np.asarray(frame)
            writer.append_data(arr)
            for key, at in (("wire", cuts[0] // 2), ("geo", (cuts[1] + cuts[2]) // 2), ("full", F - 12)):
                if fi == at:
                    stills[key] = arr
            log.append({"frame": fi, "step": step, "sealed": bool(u.sealed[0]), "can": can_p.round(4).tolist()})
            fi += 1
            reps += 1
    writer.close()
    if len(stills) == 3:
        strip = np.concatenate([np.asarray(Image.fromarray(stills[k_]).resize((W // 2, H // 2))) for k_ in ("wire", "geo", "full")], 1)
        Image.fromarray(strip).save(os.path.join(args.out_dir, "pickcan_layers.png"))
    res = {"policy": args.policy, "landed_step_pass1": n_end, "sealed_step_pass1": n_seal, "in_tote_end": bool(core.in_tote(u._can_state()[0])[0]),
           "frames": F, "sim_steps": total, "speed": round(total / 30.0 / (F / args.fps), 2)}
    with open(os.path.join(args.out_dir, "layers.json"), "w") as f:
        json.dump({"result": res, "frames": log}, f)
    print("[layers] RESULT " + json.dumps(res), flush=True)
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

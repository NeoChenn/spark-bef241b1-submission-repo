from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ROBOT_XY = (1.5, -1.5)
CAN_XY = (0.12, -0.40)
TEAL, AMBER, RED, WHITE, GREY = (80, 220, 210), (255, 190, 60), (255, 80, 60), (232, 234, 236), (170, 176, 182)


def cv_pose(usd_world):
    import numpy as np

    T = np.asarray(usd_world, float).T.copy()
    T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return T


def intrinsics(focal, aperture, w, h):
    import numpy as np

    f = focal / aperture * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="runs/pickup/video")
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_deck.usda"))
    p.add_argument("--kind", type=int, default=0)
    p.add_argument("--yaw", type=float, default=0.7)
    p.add_argument("--upright", action="store_true")
    p.add_argument("--seconds", type=float, default=13.0)
    p.add_argument("--intro", type=float, default=1.6)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--look", default="night", choices=["night", "day"])
    p.add_argument("--dome", type=float, default=220.0)
    p.add_argument("--lamp_gain", type=float, default=1.0)
    p.add_argument("--renderer", default="RaytracedLighting")
    p.add_argument("--still", action="store_true")
    p.add_argument("--key", type=float, default=2.5e4)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless, args.enable_cameras = True, True
    AppLauncher(args)

    import carb
    import gymnasium as gym
    import imageio
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import UsdGeom

    from isaaclab.sim import RenderCfg

    from integration import repair_scene as rsc
    from policy.envs import pickcan_env as PE
    from policy.envs.pickcan_env_cfg import CANS, PickCanEnvCfg
    from policy.envs.scrub_env import _add_backdrop
    from policy.night_look import grade
    from policy.pickcan import core
    from policy.pickcan import detect as DT
    from policy.pickcan.detect import detect_cans
    DT.S_MIN = 0.8
    from policy.pickcan.script import NAMES, ScriptedPicker

    CANS_ROT = CANS[args.kind:] + CANS[:args.kind]
    import policy.envs.pickcan_env_cfg as PC
    PC.CANS = CANS_ROT
    PE.CANS = CANS_ROT
    name, can_r, can_h, _ = CANS_ROT[0]

    cfg = PickCanEnvCfg()
    cfg.scene.num_envs = 1
    cfg.init_joint_noise = 0.0
    cfg.episode_length_s = 1.0e5
    cfg.det_sigma_xy, cfg.det_sigma_z, cfg.det_jitter = 0.0, 0.0, 0.0
    cfg.robot_cfg.spawn.make_instanceable = False
    cfg.sim.render = RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", carb_settings={
        "/rtx/rendermode": args.renderer, "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False})
    work = (ROBOT_XY[0], ROBOT_XY[1], 0.3)
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
        if args.look == "night":
            from policy import night_look

            w = (0.0, 0.0, 0.3)
            print("[video] night look", night_look.apply(st, "/World/Backdrop", w, dome=args.dome,
                                                          lamp_gain=args.lamp_gain,
                                                          robot_root="/World/envs/env_0/Robot"), flush=True)
            night_look._lamp(st, "/World/PickKey", (0.9, -1.6, 1.9), (1.0, 0.86, 0.7), 0.12, args.key)
            night_look._lamp(st, "/World/PickRim", (-1.2, 0.8, 1.4), night_look.LED, 0.1, 0.4 * args.key)

    PE.PickCanEnv._setup_scene = setup
    env = gym.make("Rig-PickCan-Go2UR3Lift-v0", cfg=cfg)
    u = env.unwrapped
    dev = u.device
    get_dones = u._get_dones

    def no_done():
        s, t = get_dones()
        return torch.zeros_like(s), torch.zeros_like(t)

    u._get_dones = no_done
    settings = carb.settings.get_settings()
    for k, v in cfg.sim.render.carb_settings.items():
        settings.set(k, v)
    env.reset(seed=3)
    ids = torch.tensor([0], device=dev)
    lying = torch.tensor([not args.upright], device=dev)
    u.place_can(ids, torch.tensor([CAN_XY], device=dev), torch.tensor([args.yaw], device=dev), lying)
    u.det_bias[:] = 0.0
    u.det_yaw_bias[:] = 0.0

    stage = omni.usd.get_context().get_stage()
    origin = u.scene.env_origins[0].cpu().numpy()
    cams = {}
    W, H, WW, WH = 1280, 720, 640, 360
    wide = rsc.Camera(stage, "/World/Cameras/pick_wide", focal=26.0)
    wrist = rsc.Camera(stage, "/World/Cameras/pick_wrist", focal=14.0)
    for key, cam, size, kinds in (("wide", wide, (W, H), ("rgb",)), ("wrist", wrist, (WW, WH), ("rgb", "distance_to_image_plane"))):
        rp = rep.create.render_product(cam.path, size)
        for kd in kinds:
            ann = rep.AnnotatorRegistry.get_annotator(kd)
            ann.attach(rp)
            cams[(key, kd)] = ann
    K = intrinsics(14.0, wrist.aperture, WW, WH)

    def place_cams(t):
        a = -1.05 - 0.2 * math.sin(0.3 * t)
        eye = np.array([0.1 + math.cos(a) * 2.3, -0.2 + math.sin(a) * 2.3, 1.35 + 0.08 * math.sin(0.25 * t)])
        wide.look(eye + origin, np.array([0.0, -0.22, 0.22]) + origin)
        tip, R = (x[0].cpu().numpy() for x in u._tip()[:2])
        out = np.array([tip[0] + 0.01, tip[1], 0.0])
        out = out / max(np.linalg.norm(out), 1e-6)
        side = np.array([-out[1], out[0], 0.0])
        e = tip + np.array([0.0, 0.0, 0.2]) - 0.1 * out + 0.09 * side
        wrist.look(e + origin, tip + np.array([0.0, 0.0, -0.45]) + 0.22 * out + origin)

    for _ in range(3):
        place_cams(0.0)
        u.sim.render()

    fonts = {}
    for key, fn, size in (("big", "DejaVuSans-Bold.ttf", 24), ("mid", "DejaVuSans.ttf", 17), ("small", "DejaVuSans.ttf", 14)):
        try:
            fonts[key] = ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{fn}", size)
        except OSError:
            fonts[key] = ImageFont.load_default()

    picker = ScriptedPicker(1, dev, dt=u.ctrl_dt, scale=cfg.action_scale)
    os.makedirs(args.out_dir, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.out_dir, "pickcan.mp4"), fps=args.fps, quality=8, macro_block_size=8)
    n_steps = int(args.seconds * args.fps)
    det_best, grasp_det, log, keys = None, None, [], {}
    hold = torch.zeros(1, 5, device=dev)
    hold[:, 3] = -1.0
    hold[:, 4] = -1.0
    for step in range(n_steps):
        t = step / args.fps
        place_cams(t)
        with torch.inference_mode():
            tip = u._tip()[0]
            if t < args.intro:
                act = hold.clone()
            else:
                g = grasp_det if grasp_det is not None else u.grasp_estimate()
                act = picker.act(tip, u.p_tgt, g, u.sealed, u._robot.data.joint_pos[:, u.lift_id])
            env.step(act)
        u.sim.render()
        rgb = np.asarray(cams[("wide", "rgb")].get_data())
        wr = np.asarray(cams[("wrist", "rgb")].get_data())
        dp = np.asarray(cams[("wrist", "distance_to_image_plane")].get_data()).squeeze()
        if rgb.size == 0 or wr.size == 0:
            continue
        rgb, wr = rgb[..., :3].copy(), wr[..., :3].copy()
        dets = []
        phase = "DETECT" if t < args.intro else NAMES[int(picker.phase[0])]
        if phase in ("DETECT", "APPROACH", "DESCEND") and dp.size:
            T = cv_pose(wrist.world_matrix())
            T[:3, 3] -= origin
            if step == 3:
                np.savez_compressed(os.path.join(args.out_dir, "wrist_frame.npz"), rgb=wr, depth=dp, K=K, T=T,
                                    usd=np.asarray(wrist.world_matrix()), can=u._can_state()[0][0].cpu().numpy())
            dets = detect_cans(wr, np.nan_to_num(dp.astype(np.float64), posinf=50.0), K, T)
            dets = [q for q in dets if 0.28 < math.hypot(q["xyz"][0] + 0.01, q["xyz"][1]) < 0.55 and q["xyz"][2] < 0.25]
            if dets and t < args.intro:
                det_best = dets[0]
        can_p = u._can_state()[0][0].cpu().numpy()
        if det_best is not None and grasp_det is None and t >= args.intro - 1.0 / args.fps:
            xyz = np.asarray(det_best["xyz"], float)
            if np.linalg.norm(xyz[:2] - can_p[:2]) < 0.04:
                top = det_best.get("diameter", 2 * can_r) / 2 if det_best.get("lying", True) else det_best.get("length", can_h) / 2
                grasp_det = torch.tensor([[xyz[0], xyz[1], xyz[2] + top]], device=dev, dtype=torch.float32)
                print(f"[video] grasp from detector {grasp_det.tolist()} true can {can_p.round(3).tolist()}", flush=True)
            else:
                print(f"[video] detector off by {np.linalg.norm(xyz[:2] - can_p[:2]):.3f} m, using env estimate", flush=True)
                grasp_det = u.grasp_estimate()
        if args.look == "night":
            rgb, wr = grade(rgb), grade(wr)
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im, "RGBA")
        inset = Image.fromarray(wr).resize((480, 270))
        di = ImageDraw.Draw(inset, "RGBA")
        sx, sy = 480 / WW, 270 / WH
        for k, dt_ in enumerate(dets[:2]):
            x0, y0, x1, y1 = dt_["bbox"]
            col = TEAL if k == 0 else GREY
            di.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=col + (255,), width=2)
            lab = f"{dt_.get('colour', 'can')} can {dt_.get('score', 0):.2f}"
            di.rectangle([x0 * sx, max(0, y0 * sy - 18), x0 * sx + 8 * len(lab), y0 * sy], fill=(10, 12, 16, 190))
            di.text((x0 * sx + 3, max(0, y0 * sy - 17)), lab, fill=col, font=fonts["small"])
        X0, Y0 = W - 480 - 16, 16
        d.rectangle([X0 - 2, Y0 - 2, X0 + 481, Y0 + 271], outline=(255, 255, 255, 110))
        im.paste(inset, (X0, Y0))
        d.rectangle([X0, Y0 + 270 - 22, X0 + 250, Y0 + 270], fill=(10, 12, 16, 180))
        d.text((X0 + 8, Y0 + 270 - 19), "WRIST CAMERA  can detector", fill=WHITE, font=fonts["small"])
        sealed = bool(u.sealed[0])
        col = AMBER if phase in ("DETECT",) else (TEAL if phase != "DONE" else (120, 230, 120))
        d.rounded_rectangle([16, 16, 600, 58], radius=10, fill=(10, 12, 16, 170))
        d.ellipse([30, 30, 44, 44], fill=col)
        title = {"DETECT": "DEBRIS: CAN ON WALKWAY", "APPROACH": "PICK CAN: APPROACH", "DESCEND": "PICK CAN: DESCEND",
                 "LIFT": "PICK CAN: LIFT", "CARRY": "PICK CAN: STOW", "LOWER": "PICK CAN: STOW",
                 "RELEASE": "PICK CAN: RELEASE", "DONE": "WALKWAY CLEAR"}[phase]
        d.text((54, 24), title, fill=col, font=fonts["big"])
        d.text((480, 30), f"t {t:5.1f} s", fill=WHITE, font=fonts["mid"])
        lift_dz = float(can_p[2] - u.rest_z[0])
        lines = [f"{name.replace('_', ' ')} can  d {2 * can_r * 100:.1f} cm  L {can_h * 100:.0f} cm  "
                 f"{float(u.mass[0]):.2f} kg  {'lying' if not args.upright else 'upright'}",
                 f"suction {'SEALED' if sealed else ('on' if bool(u.suction[0]) else 'off')}   "
                 f"lift {max(lift_dz, 0) * 100:5.1f} cm   mast {float(u._robot.data.joint_pos[0, u.lift_id]) * 100:4.1f} cm",
                 f"tool contact {float(u.tool_force[0]):5.1f} N  (limit 20 N)   drops {int(u.drops[0])}"]
        if det_best is not None:
            e = np.linalg.norm(np.asarray(det_best["xyz"])[:2] - can_p[:2]) if phase == "DETECT" else None
            lines.insert(1, f"detector: {det_best.get('colour')} can at ({det_best['xyz'][0]:.2f}, {det_best['xyz'][1]:.2f}) m"
                            + (f"  err {e * 100:.1f} cm" if e is not None else ""))
        y = H - 24 - 22 * len(lines)
        d.rounded_rectangle([16, y - 12, 720, H - 16], radius=10, fill=(10, 12, 16, 165))
        for k, line in enumerate(lines):
            d.text((30, y + 22 * k), line, fill=WHITE, font=fonts["mid"])
        d.text((W - 430, H - 30), "Isaac Lab  Rig-PickCan-Go2UR3Lift-v0  scripted grasp", fill=GREY, font=fonts["small"])
        frame = np.asarray(im)
        writer.append_data(frame)
        if phase not in keys or (phase == "LIFT" and lift_dz > 0.1 and "LIFT_hi" not in keys):
            keys["LIFT_hi" if phase == "LIFT" and phase in keys else phase] = frame
            Image.fromarray(frame).save(os.path.join(args.out_dir, f"key_{len(keys):02d}_{phase.lower()}.png"))
        log.append({"t": t, "phase": phase, "sealed": sealed, "lift_dz": lift_dz, "force": float(u.tool_force[0]),
                    "can": can_p.round(4).tolist(), "dets": len(dets)})
        if args.still and step > 5:
            break
    writer.close()
    in_tote = bool(core.in_tote(u._can_state()[0])[0])
    res = {"in_tote": in_tote, "drops": int(u.drops[0]), "max_force": float(u.max_force[0]),
           "lift_max": float(u.lift_max[0]), "detector_used": grasp_det is not None, "frames": len(log)}
    print("[video] RESULT " + json.dumps(res), flush=True)
    with open(os.path.join(args.out_dir, "log.json"), "w") as f:
        json.dump({"result": res, "frames": log}, f)
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

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WIDE = ((2.9, 2.3, 2.0), (-0.6, -0.3, 0.75))
CLOSE = ((0.75, -0.02, 1.62), (0.0, -0.42, 1.18))


def worker(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--meta", default=os.path.join(ROOT, "sim/output/rig_deck_meta.json"))
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_deck.usda"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--abrasion", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--start_x", type=float, default=-2.6)
    p.add_argument("--bead_height", type=float, default=0.012)
    p.add_argument("--no_events", action="store_true")
    p.add_argument("--no-gemini", dest="no_gemini", action="store_true")
    p.add_argument("--still", action="store_true")
    p.add_argument("--spp", type=int, default=1)
    p.add_argument("--hue_lo", type=float, default=0.3)
    p.add_argument("--fps", type=int, default=15)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless, args.enable_cameras = True, True
    app = AppLauncher(args).app

    import carb
    import gymnasium as gym
    import imageio
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.sim import RenderCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from detection.localize import RustLocalizer, colour_classifier
    from integration.lab_bridge import ExternalScrub, LabRepairBridge, LabWorld, frame_telemetry, write_jsonl
    from integration.lab_video import FrameComposer
    from policy.envs import scrub_env_cfg as C
    from policy.envs.rust_state import random_patches
    from policy.envs.scrub_env import TASKS
    from policy.envs.scrub_env_cfg import ScrubPPORunnerCfg
    from policy.force_tools import force_stats, intervals
    from policy.orchestrator import Orchestrator, OrchestratorConfig, S
    from policy.play_scrub import apply_meta
    from policy.render_demo import RustPainter, build_overlay
    from policy.rust_overlay import grid_to_mask, repair_rgba, rig_translation, rust_rgba

    task, cfg_name = TASKS["lift_v2"]
    cfg = getattr(C, cfg_name)()
    cfg.scene.num_envs = 1
    meta = apply_meta(cfg, args.meta)
    cfg.brush_sensor = True
    cfg.sensor_in_loop = True
    cfg.spring_k_range = (cfg.contact_stiffness, cfg.contact_stiffness)
    cfg.abrasion_coef_range = (args.abrasion, args.abrasion)
    cfg.prob_new_patch = 0.0
    cfg.init_joint_noise = 0.0
    cfg.episode_length_s = 1.0e6
    cfg.viz = False
    cfg.spawn_pipe_visual = False
    cfg.weld_bead_radius = 0.014
    cfg.backdrop_usd = os.path.abspath(args.scene)
    cfg.backdrop_translation = tuple(rig_translation(meta))
    cfg.sim.render = RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", carb_settings={
        "/rtx/rendermode": "RaytracedLighting", "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False})
    agent_cfg = ScrubPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=base.device)
    get_dones = base._get_dones

    def no_terminate():
        s, t = get_dones()
        return torch.zeros_like(s), torch.zeros_like(t)

    base._get_dones = no_terminate
    base.cfg.sim.render_interval = 10 ** 9
    settings = carb.settings.get_settings()
    for key, val in cfg.sim.render.carb_settings.items():
        settings.set(key, val)

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        base.reset(seed=args.seed)
        rs_ = base.rust
        g = torch.Generator(device=base.device).manual_seed(args.seed)
        rust = random_patches(1, rs_.H, rs_.W, rs_.radius[:1], rs_.length, n_min=2, n_max=2, size_range=(0.05, 0.07),
                              device=base.device, generator=g, theta_window=rs_.theta_window) * rs_.reachable[:1].float()
        rs_.reset(torch.tensor([0], device=base.device), rust=rust, spray_all_rust=False, abrasion_coef=args.abrasion)
    base.rust.rust = base.rust.rust.clone()
    base.rust.sprayed = base.rust.sprayed.clone()
    base.rust.soak_time = base.rust.soak_time.clone()
    base.rust.initial_total = base.rust.initial_total.clone()

    stage = omni.usd.get_context().get_stage()
    for path in ("/World/Backdrop/Pipes/damage_patch", "/World/Backdrop/RepairZone/Pipes/damage_patch",
                 "/World/Backdrop/Robots/scrubber_dog", "/World/Backdrop/RepairZone/Robots"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
    tex_dir = os.path.join(ROOT, "sim/assets/third_party/textures/rust_coarse_01")
    painter = RustPainter(build_overlay(stage, base, tex_dir), tex_dir, flip_v=True)
    from integration import repair_scene as rsc

    origin = base.scene.env_origins[0].cpu().numpy()
    cams = {}
    for name, (eye, look), focal, size in (("wide", WIDE, 22.0, (1280, 720)), ("close", CLOSE, 40.0, (1280, 720))):
        cam = rsc.Camera(stage, f"/World/Cameras/video_{name}", focal=focal)
        cam.look(np.asarray(eye) + origin, np.asarray(look) + origin)
        rp = rep.create.render_product(cam.path, size)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        cams[name] = ann
    world = LabWorld(base.rust)

    os.makedirs(args.out_dir, exist_ok=True)
    composer = FrameComposer()
    writers = {"combined": imageio.get_writer(os.path.join(args.out_dir, "repair_live.mp4"), fps=args.fps, quality=8,
                                              macro_block_size=8),
               "close": imageio.get_writer(os.path.join(args.out_dir, "repair_live_close.mp4"), fps=args.fps, quality=8,
                                           macro_block_size=8)}
    tele, history, orch_ref, keyframes = [], [], {}, {}
    H, W = 512, 1024

    def paint_overlay(snap):
        rust_m = grid_to_mask(snap["rust"], H, W, flip_v=True)
        rgba = rust_rgba(rust_m, painter.tex, painter.noise)
        wet = grid_to_mask((snap["sprayed"] & ~snap["painted"]).astype(np.float32), H, W, flip_v=True)
        primer = grid_to_mask(snap["painted"].astype(np.float32), H, W, flip_v=True)
        rgba = rgba.astype(np.float32)
        rgba[..., :3] *= 1.0 - 0.4 * wet[..., None]
        rgba = repair_rgba(np.where((primer > 0.5)[..., None], 0, rgba).astype(np.uint8), np.zeros_like(wet), primer)
        painter.provider.set_data_array(rgba, [W, H])

    def render(snap):
        paint_overlay(snap)
        base.sim.render()
        imgs = {k: np.asarray(a.get_data()) for k, a in cams.items()}
        history.append((snap["t"], snap["force_n"]))
        orch = orch_ref.get("o")
        conf = None
        if orch is not None and orch.last is not None:
            cs = [q.mean_conf for q in orch.last.result.patches]
            conf = max(cs) if cs else orch.last.result.max_conf
        tele.append(frame_telemetry(snap, None if orch is None else orch.rust_frac, conf))
        if any(v.size == 0 for v in imgs.values()):
            return
        wide, close = (np.ascontiguousarray(imgs[k][..., :3]) for k in ("wide", "close"))
        if wide.mean() < 2:
            return
        lines = [l for l in snap["hud"][1:4]] + [f"true rust left {100 * snap['true_rust_fraction']:5.1f} %   "
                                                  f"model force {snap['force_model_n']:4.1f} N"]
        spark = [(r["t"], r["force_n"]) for r in bridge.force_log[-1500:]]
        frame = composer.compose(wide, close, snap["state"], lines, snap["force_peak_n"], spark, snap["t"],
                                 snap["speed"])
        writers["combined"].append_data(frame)
        writers["close"].append_data(close)
        st = snap["state"]
        if st not in keyframes or (st == "SCRUB" and snap["force_n"] > keyframes.get("_fmax", 0)):
            if st == "SCRUB":
                keyframes["_fmax"] = snap["force_n"]
            keyframes[st] = frame

    with torch.inference_mode():
        bridge = LabRepairBridge(env, base, policy, stage, world, render, None, start_x=args.start_x, fps=args.fps)
    insp_rp = rep.create.render_product(rsc.INSPECTION_CAMERA, (640, 360))
    insp_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    insp_ann.attach(insp_rp)
    bridge.insp_annot = insp_ann
    for _ in range(20):
        base.sim.render()

    localizer = RustLocalizer(colour_classifier(args.hue_lo), patch=32, stride=16, threshold=0.5, temperature=1.0)
    print(f"DETECTOR colour stand-in hue_lo={args.hue_lo} (no detector checkpoint on this box)", flush=True)
    advisor = None
    if not args.no_gemini:
        from policy.gemini_verifier import GeminiVerifier

        advisor = GeminiVerifier()
        advisor = advisor if advisor.available else None
    print(f"GEMINI {'on' if advisor else 'off (--no-gemini or no GEMINI_API_KEY): on-board detector only'}", flush=True)
    orch = Orchestrator(bridge, bridge, localizer, world, bridge, ExternalScrub(),
                        OrchestratorConfig(log_path=os.path.join(args.out_dir, "decisions.jsonl"), scrub_steps=200,
                                           scrub_redetect_every=40), advisor=advisor)
    orch_ref["o"] = orch
    bridge.state_fn = lambda: orch.state.value
    rec0 = orch._record

    def record(to, reason, evidence):
        rec0(to, reason, evidence)
        bridge.record_decision(orch.records[-1].to_json())
        print(f"t={orch.t:6.1f}s -> {to.value:<17} F={evidence.get('force')} | {reason}", flush=True)

    orch._record = record
    fired, started = set(), time.time()
    th0 = bridge.damage_theta
    for step in range(args.max_steps):
        if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN):
            break
        prev = orch.state
        with torch.inference_mode():
            orch.step()
        now = bridge.sim_time()
        if args.no_events:
            continue
        if prev == S.SPRAY and "patch" not in fired:
            fired.add("patch")
            th_new, z_new = th0 + 0.30, 0.13
            bridge.schedule(now + 6.0, f"NEW rust patch at theta={th_new:.2f}, z={z_new:+.2f}",
                            lambda: bridge.spawn_patch(th_new, z_new, size=0.035, level=0.85))
        if orch.state == S.SCRUB and prev != S.SCRUB and "bead" not in fired and "patch" in fired:
            fired.add("bead")
            bridge.schedule(now + 5.0, "weld bead under the brush (force spike)",
                            lambda: bridge.force_spike(args.bead_height))
        if prev == S.SCRUB and orch.state == S.VERIFY and "mist" not in fired and "bead" in fired:
            fired.add("mist")
            bridge.occlude_camera(True)
            bridge.schedule(now + 1.5, "lens mist cleared", lambda: bridge.occlude_camera(False))
            bridge.schedule(now + 0.5, "worker walks up to 2.4 m", lambda: bridge.person_at(2.4))
            bridge.schedule(now + 2.5, "worker leaves", lambda: bridge.person_at(None))
        if step % 5 == 0:
            print(f"step {step} t={now:.1f}s state={orch.state.value} frames={len(history)} "
                  f"wall={time.time() - started:.0f}s", flush=True)
    with torch.inference_mode():
        for _ in range(45):
            bridge._step()
    for w in writers.values():
        w.close()
    write_jsonl(os.path.join(args.out_dir, "telemetry.jsonl"), tele)
    write_jsonl(os.path.join(args.out_dir, "force_physics.jsonl"), bridge.force_log)
    for k, fr in keyframes.items():
        if not k.startswith("_"):
            Image.fromarray(fr).save(os.path.join(args.out_dir, f"key_{k}.png"))
    os.makedirs(os.path.join(args.out_dir, "inspection"), exist_ok=True)
    for t_i, img in bridge.inspections:
        Image.fromarray(img).save(os.path.join(args.out_dir, "inspection", f"t{t_i:07.2f}.jpg"), quality=88)
    if bridge.last_inspection is not None:
        Image.fromarray(bridge.last_inspection).save(os.path.join(args.out_dir, "inspection_last.png"))
    ft = np.array([r["t"] for r in bridge.force_log])
    ff = np.array([r["force_n"] for r in bridge.force_log])
    fs = np.array([r["state"] for r in bridge.force_log])
    scrub = fs == "SCRUB"
    summary = {
        "final_state": orch.state.value, "success": orch.state == S.DONE, "sim_time_s": round(bridge.t, 2),
        "decisions": len(orch.records), "back_off_count": sum(r.to == "BACK_OFF" for r in orch.records),
        "replans": sum("RE-PLAN" in r.reason for r in orch.records),
        "true_rust_left": float(world.fraction_remaining()[0]),
        "rust_removed_fraction": 1.0 - float(world.fraction_remaining()[0]),
        "painted_cells": int(world.painted.sum()),
        "force_all": force_stats(ff, bridge.phys_dt), "force_scrub": force_stats(ff[scrub], bridge.phys_dt),
        "back_off_windows": intervals(list(ft), list(fs == "BACK_OFF")),
        "bead": bridge.bead_cells, "gemini": bool(advisor), "detector": "colour stand-in",
        "setpoint_cap_n": 38.0, "force_limit_n": 40.0, "frames": len(history), "wall_s": round(time.time() - started, 1),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        worker(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

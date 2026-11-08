from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WIDE = ((2.9, 2.3, 2.0), (-0.9, -0.3, 0.75))
CLOSE = ((0.75, -0.02, 1.62), (0.0, -0.42, 1.18))
SOAK_NOTE = "soak shortened for the demo"


def worker(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--scrub_onnx", default="/ckpt/scrub/policy.onnx")
    p.add_argument("--walk_onnx", default="/ckpt/walk/policy.onnx")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--meta", default=os.path.join(ROOT, "sim/output/rig_deck_meta.json"))
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_deck.usda"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--abrasion", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--start_x", type=float, default=-2.6)
    p.add_argument("--soak_s", type=float, default=5.0)
    p.add_argument("--scrub_steps", type=int, default=200)
    p.add_argument("--paint_steps", type=int, default=3)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--size", default="960x540")
    p.add_argument("--hue_lo", type=float, default=0.3)
    p.add_argument("--no_video", action="store_true")
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
    from PIL import Image

    from isaaclab.sim import RenderCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from detection.localize import RustLocalizer, colour_classifier
    from integration import repair_scene as rsc
    from integration.lab_bridge import ExternalScrub, LabWorld, frame_telemetry, write_jsonl
    from integration.lab_video import FrameComposer
    from integration.loop_bridge import LoopBridge, summary_motions
    from integration.loop_timeline import replay_lines
    from integration.onnx_mlp import OnnxMLP
    from policy.envs import scrub_env_cfg as C
    from policy.envs.rust_state import random_patches
    from policy.envs.scrub_env import TASKS
    from policy.envs.scrub_env_cfg import ScrubPPORunnerCfg
    from policy.force_tools import force_stats
    from policy.orchestrator import Orchestrator, OrchestratorConfig, S
    from policy.play_scrub import apply_meta
    from policy.render_demo import RustPainter, build_overlay
    from policy.rust_overlay import grid_to_mask, repair_rgba, rust_rgba

    W_out, H_out = (int(v) for v in args.size.split("x"))
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
    cfg.theta_limits = (-0.3, 2.5)
    cfg.backdrop_usd = os.path.abspath(args.scene)
    from policy.rust_overlay import rig_translation

    cfg.backdrop_translation = tuple(rig_translation(meta))
    cfg.sim.render = RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", carb_settings={
        "/rtx/rendermode": "RaytracedLighting", "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False})
    agent_cfg = ScrubPPORunnerCfg()
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    scrub_net = OnnxMLP(args.scrub_onnx)
    walk_net = OnnxMLP(args.walk_onnx)
    print(f"ONNX scrub in={scrub_net.in_dim} walk in={walk_net.in_dim}", flush=True)

    def scrub_policy(obs):
        x = obs["policy"] if not torch.is_tensor(obs) else obs
        return torch.as_tensor(scrub_net(x.detach().cpu().numpy()), device=base.device)

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
    rust_cells0 = int((base.rust.rust[0] > 0.1).sum())

    stage = omni.usd.get_context().get_stage()
    for path in ("/World/Backdrop/Pipes/damage_patch", "/World/Backdrop/RepairZone/Pipes/damage_patch",
                 "/World/Backdrop/Robots/scrubber_dog", "/World/Backdrop/RepairZone/Robots"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
    tex_dir = os.path.join(ROOT, "sim/assets/third_party/textures/rust_coarse_01")
    painter = RustPainter(build_overlay(stage, base, tex_dir), tex_dir, flip_v=True)
    origin = base.scene.env_origins[0].cpu().numpy()
    cams = {}
    if not args.no_video:
        for name, (eye, look), focal in (("wide", WIDE, 20.0), ("close", CLOSE, 40.0)):
            cam = rsc.Camera(stage, f"/World/Cameras/loop_{name}", focal=focal)
            cam.look(np.asarray(eye) + origin, np.asarray(look) + origin)
            rp = rep.create.render_product(cam.path, (W_out, H_out) if name == "wide" else (W_out // 2, H_out // 2))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach(rp)
            cams[name] = ann
    world = LabWorld(base.rust)

    os.makedirs(args.out_dir, exist_ok=True)
    composer = FrameComposer(size=(W_out, H_out))
    writer = None if args.no_video else imageio.get_writer(os.path.join(args.out_dir, "mission_loop.mp4"), fps=args.fps,
                                                            quality=7, macro_block_size=8)
    tele, hud, history, orch_ref, keyframes = [], [], [], {}, {}
    H, W = 512, 1024

    def paint_overlay(snap):
        rust_m = grid_to_mask(snap["rust"], H, W, flip_v=True)
        wet = grid_to_mask((snap["sprayed"] & ~snap["painted"]).astype(np.float32), H, W, flip_v=True)
        primer = grid_to_mask(snap["painted"].astype(np.float32), H, W, flip_v=True)
        rgba = rust_rgba(rust_m, painter.tex, painter.noise).astype(np.float32)
        rgba[..., :3] *= 1.0 - 0.4 * wet[..., None]
        rgba = repair_rgba(np.where((primer > 0.5)[..., None], 0, rgba).astype(np.uint8), np.zeros_like(wet), primer,
                           primer_rgb=(38, 52, 40))
        painter.provider.set_data_array(rgba, [W, H])

    def caption(snap):
        ph, st = snap["phase"], snap["state"]
        label = ph if ph == st else f"{ph} · {st}"
        o = orch_ref.get("o")
        if st == "SOAK" and o is not None:
            label = f"SOAK {max(0.0, o.soak_until - o.t):.0f} s left"
        return label

    def render(snap):
        orch = orch_ref.get("o")
        conf = None
        if orch is not None and orch.last is not None:
            cs = [q.mean_conf for q in orch.last.result.patches]
            conf = max(cs) if cs else orch.last.result.max_conf
        tele.append(frame_telemetry(snap, None if orch is None else orch.rust_frac, conf) | {"phase": snap["phase"]})
        bx = snap["base"]
        hud.append(bridge.timeline.hud_record(snap["t"], snap["phase"], snap["force_n"], (bx[0], bx[1], bx[2]),
                                              {"rust": snap["true_rust_fraction"]}))
        history.append((snap["t"], snap["force_n"]))
        paint_overlay(snap)
        if writer is None:
            return
        base.sim.render()
        imgs = {k: np.asarray(a.get_data()) for k, a in cams.items()}
        if any(v.size == 0 for v in imgs.values()):
            return
        wide, close = (np.ascontiguousarray(imgs[k][..., :3]) for k in ("wide", "close"))
        if wide.mean() < 2:
            return
        note = f"soak {args.soak_s:.0f} s: {SOAK_NOTE} (real 30 s)" if snap["state"] == "SOAK" else \
            "walk + scrub policies: ONNX, force cap 38 N"
        lines = [l for l in snap["hud"][1:3]] + [f"true rust left {100 * snap['true_rust_fraction']:5.1f} %", note]
        spark = [(r["t"], r["force_n"]) for r in bridge.force_log[-1500:]]
        frame = composer.compose(wide, close, caption(snap), lines, snap["force_n"], spark, snap["t"], snap["speed"])
        writer.append_data(frame)
        ph = snap["phase"]
        if ph not in keyframes or (ph == "SCRUB" and snap["force_n"] > keyframes.get("_fmax", 0)):
            if ph == "SCRUB":
                keyframes["_fmax"] = snap["force_n"]
            keyframes[ph] = frame

    with torch.inference_mode():
        bridge = LoopBridge(env, base, scrub_policy, stage, world, render, None, start_x=args.start_x, fps=args.fps,
                            walk_net=walk_net, paint_steps=args.paint_steps)
    insp_rp = rep.create.render_product(rsc.INSPECTION_CAMERA, (640, 360))
    insp_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    insp_ann.attach(insp_rp)
    bridge.insp_annot = insp_ann
    for _ in range(20):
        base.sim.render()

    localizer = RustLocalizer(colour_classifier(args.hue_lo), patch=32, stride=16, threshold=0.5, temperature=1.0)
    print(f"DETECTOR colour stand-in hue_lo={args.hue_lo}", flush=True)
    orch = Orchestrator(bridge, bridge, localizer, world, bridge, ExternalScrub(),
                        OrchestratorConfig(log_path=os.path.join(args.out_dir, "decisions.jsonl"), soak_s=args.soak_s,
                                           scrub_steps=args.scrub_steps, scrub_redetect_every=40), advisor=None)
    orch_ref["o"] = orch
    bridge.state_fn = lambda: orch.state.value
    rec0 = orch._record

    def record(to, reason, evidence):
        rec0(to, reason, evidence)
        bridge.record_decision(orch.records[-1].to_json())
        print(f"t={orch.t:6.1f}s -> {to.value:<17} F={evidence.get('force')} | {reason}", flush=True)

    orch._record = record
    started = time.time()
    with torch.inference_mode():
        bridge._step()
    for step in range(args.max_steps):
        if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN):
            break
        with torch.inference_mode():
            orch.step()
        if step % 5 == 0:
            print(f"step {step} t={bridge.t:.1f}s state={orch.state.value} phase={bridge.phase()} "
                  f"frames={len(history)} wall={time.time() - started:.0f}s", flush=True)
    with torch.inference_mode():
        for _ in range(45):
            bridge._step()
    bridge.timeline.close(bridge.t)
    if writer is not None:
        writer.close()
    write_jsonl(os.path.join(args.out_dir, "telemetry.jsonl"), tele)
    write_jsonl(os.path.join(args.out_dir, "hud_timeline.jsonl"), hud)
    write_jsonl(os.path.join(args.out_dir, "force_physics.jsonl"), bridge.force_log)
    write_jsonl(os.path.join(args.out_dir, "phases.jsonl"), bridge.timeline.spans)
    decs = [r.to_json() for r in orch.records] + [{"t": r["t"], "from": "PAINT", "to": "REPOSITION",
                                                    "reason": f"base dx {r['dx']:+.2f} m, lift {r['lift']:.2f} m"}
                                                   for r in bridge.reposition_log]
    write_jsonl(os.path.join(args.out_dir, "replay.jsonl"),
                replay_lines(hud, decs, offset=tuple(meta.get("env_to_world", {}).get("translation", (0, 0, 0)))))
    for k, fr in keyframes.items():
        if not k.startswith("_"):
            Image.fromarray(fr).save(os.path.join(args.out_dir, f"key_{k}.png"))
    ff = np.array([r["force_n"] for r in bridge.force_log])
    mo = np.array([r.get("motion") or "" for r in bridge.force_log])
    ph = np.array([r.get("phase") or "" for r in bridge.force_log])
    paint = orch.paint_skill
    tgt = paint.target_for(orch.target) if orch.target is not None else None
    summary = {
        "final_state": orch.state.value, "success": orch.state == S.DONE, "sim_time_s": round(bridge.t, 2),
        "phase_order_ok": bridge.timeline.in_order(), "phase_durations_s": bridge.timeline.durations(),
        "decisions": len(orch.records), "back_off_count": sum(r.to == "BACK_OFF" for r in orch.records),
        "rust_cells_initial": rust_cells0, "true_rust_before": 1.0,
        "true_rust_after": float(world.fraction_remaining()[0]),
        "paint_coverage": None if tgt is None else paint.coverage(tgt), "paint_passes": paint.passes,
        "painted_cells": int(world.painted.sum()),
        "force_all": force_stats(ff, bridge.phys_dt), "force_scrub": force_stats(ff[ph == "SCRUB"], bridge.phys_dt),
        "force_park": force_stats(ff[mo == "PARK"], bridge.phys_dt) if (mo == "PARK").any() else None,
        "force_unpark": force_stats(ff[mo == "UNPARK"], bridge.phys_dt) if (mo == "UNPARK").any() else None,
        "motions": summary_motions(bridge.motions), "reposition": bridge.reposition_log,
        "paint_misses": bridge.paint_misses[:60], "fk_err_m": bridge.fk_err, "soak_s": args.soak_s,
        "soak_note": SOAK_NOTE, "walk_policy": args.walk_onnx, "scrub_policy": args.scrub_onnx,
        "setpoint_cap_n": 38.0, "force_limit_n": 40.0, "frames": len(history), "wall_s": round(time.time() - started, 1),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1, default=float)
    print("SUMMARY", json.dumps(summary, default=float)[:3000], flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        worker(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

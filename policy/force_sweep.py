from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def worker(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", default="sweep", choices=["sweep", "eval", "probe"])
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--setpoints", default="5,10,15,20,25,30,35,38,40,45")
    p.add_argument("--window_s", type=float, default=8.0)
    p.add_argument("--rate_s", type=float, default=3.0)
    p.add_argument("--meta", default=os.path.join(ROOT, "sim/output/rig_deck_meta.json"))
    p.add_argument("--analytic", action="store_true", help="policy sees the analytic spring force, not the sensor")
    p.add_argument("--seed", type=int, default=11)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from policy.envs import scrub_env_cfg as C
    from policy.envs.scrub_env import TASKS
    from policy.envs.scrub_env_cfg import ScrubPPORunnerCfg
    from policy.force_tools import clamp_force_action, force_stats, setpoint_action
    from policy.play_scrub import apply_meta

    task, cfg_name = TASKS["lift_v2"]
    cfg = getattr(C, cfg_name)()
    cfg.scene.num_envs = args.num_envs
    apply_meta(cfg, args.meta)
    cfg.brush_sensor = True
    cfg.sensor_in_loop = not args.analytic
    cfg.spring_k_range = (cfg.contact_stiffness, cfg.contact_stiffness)
    cfg.viz = False
    cfg.spawn_pipe_visual = False
    if args.mode == "sweep":
        cfg.force_set_max = 50.0
        cfg.prob_new_patch = 0.0
        cfg.prob_not_soaked = 0.0
        cfg.abrasion_coef_range = (1.0, 1.0)
        cfg.init_joint_noise = 0.0
        cfg.episode_length_s = args.window_s + 5.0
    else:
        cfg.prob_new_patch = 1.0
    agent_cfg = ScrubPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=base.device)
    dt = base.step_dt
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "analytic" if args.analytic else "sensor"
    t0 = time.time()

    if args.mode == "probe":
        torch.manual_seed(args.seed)
        base.reset(seed=args.seed)
        obs = env.get_observations()
        log = []
        for k in range(int(args.window_s / dt)):
            with torch.inference_mode():
                act = policy(obs)
                act[:, 3] = torch.clamp(act[:, 3], max=clamp_force_action(1.0, 38.0))
                if k == int(3.0 / dt):
                    base.place_bead(torch.arange(base.num_envs, device=base.device), base.theta, base.axial, 0.012)
                obs, _, _, _ = env.step(act)
            log.append({"t": round(k * dt, 3), "sensor": base.force_sensor[:4].tolist(),
                        "peak": base.force_peak[:4].tolist(), "model": base.force_model[:4].tolist(),
                        "fset": base.f_set[:4].tolist(), "gap": base.gap[:4].tolist(),
                        "frac": base.rust.fraction_remaining()[:4].tolist()})
            if k % 15 == 0:
                print(f"[probe] t={k * dt:5.2f} sensor={base.force_sensor[0]:6.2f} peak={base.force_peak[0]:6.2f} "
                      f"model={base.force_model[0]:6.2f} fset={base.f_set[0]:5.1f} gap={base.gap[0]:+.4f} "
                      f"frac={base.rust.fraction_remaining()[0]:.3f}", flush=True)
        json.dump(log, open(os.path.join(args.out_dir, "probe.json"), "w"))
        os._exit(0)

    if args.mode == "eval":
        cap = clamp_force_action(1.0, 38.0)
        recs = {k: [] for k in ("max_force", "max_force_model", "success", "fraction_removed", "time_to_clean_s")}
        pooled = []
        run_model = torch.zeros(base.num_envs, device=base.device)
        base.reset(seed=args.seed)
        obs = env.get_observations()
        while len(recs["success"]) < args.episodes:
            with torch.inference_mode():
                act = policy(obs)
                act[:, 3] = torch.clamp(act[:, 3], max=cap)
                obs, _, dones, _ = env.step(act)
            run_model = torch.maximum(run_model, base.force_model)
            pooled.append(base.force_peak.cpu().numpy().copy())
            st = getattr(base, "last_episode_stats", None)
            if dones.any() and st is not None:
                ids = st["env_ids"]
                recs["max_force"] += st["max_force"].float().cpu().tolist()
                recs["max_force_model"] += run_model[ids].cpu().tolist()
                recs["success"] += st["success"].float().cpu().tolist()
                recs["fraction_removed"] += st["fraction_removed"].float().cpu().tolist()
                recs["time_to_clean_s"] += st["time_to_clean_s"].float().cpu().tolist()
                run_model[ids] = 0.0
                base.last_episode_stats = None
        mf = np.array(recs["max_force"])
        pf = np.concatenate(pooled)
        res = {"ckpt": args.ckpt, "force": tag, "episodes": len(mf), "setpoint_cap_n": 38.0,
               "success_rate": float(np.mean(recs["success"])),
               "mean_episode_peak_n": float(mf.mean()), "p95_episode_peak_n": float(np.percentile(mf, 95)),
               "max_peak_n": float(mf.max()), "frac_episodes_over_40n": float((mf > 40).mean()),
               "mean_episode_peak_model_n": float(np.mean(recs["max_force_model"])),
               "step_stats": force_stats(pf, dt), "wall_s": round(time.time() - t0, 1)}
        print("[sweep] EVAL", json.dumps(res), flush=True)
        json.dump(res, open(os.path.join(args.out_dir, f"eval_v2_{tag}.json"), "w"), indent=1)
        json.dump(recs, open(os.path.join(args.out_dir, f"episodes_v2_{tag}.json"), "w"))
        os._exit(0)

    get_dones = base._get_dones

    def no_terminate():
        s, t = get_dones()
        return torch.zeros_like(s), torch.zeros_like(t)

    base._get_dones = no_terminate
    rows = []
    n_win, n_rate = int(args.window_s / dt), int(args.rate_s / dt)

    @torch.inference_mode()
    def run_setpoint(F):
        torch.manual_seed(args.seed)
        base.reset(seed=args.seed)
        obs = env.get_observations()
        a3 = setpoint_action(F)
        forces, models, frac = [], [], []
        f0 = base.rust.fraction_remaining().clone()
        t_clean = torch.full((base.num_envs,), float("nan"), device=base.device)
        first_contact = torch.full((base.num_envs,), float("nan"), device=base.device)
        for k in range(n_win):
            act = policy(obs)
            act[:, 3] = a3
            obs, _, _, _ = env.step(act)
            fr = base.rust.fraction_remaining()
            newly = torch.isnan(t_clean) & (fr < cfg.success_fraction)
            t_clean[newly] = (k + 1) * dt
            touch = torch.isnan(first_contact) & (base.force_peak > 2.0)
            first_contact[touch] = k * dt
            forces.append(base.force_peak.cpu().numpy().copy())
            models.append(base.force_model.cpu().numpy().copy())
            frac.append(fr.cpu().numpy().copy())
        Fp = np.stack(forces)
        frac = np.stack(frac)
        removed_rate = (f0.cpu().numpy() - frac[n_rate - 1]) / args.rate_s
        tc = t_clean.cpu().numpy()
        stats = force_stats(Fp.reshape(-1), dt)
        on = Fp > 2.0
        row = {"setpoint_n": F, "removal_rate_per_s": float(removed_rate.mean()),
               "removal_rate_std": float(removed_rate.std()),
               "fraction_removed_window": float((f0.cpu().numpy() - frac[-1]).mean()),
               "success_rate": float(np.isfinite(tc).mean()),
               "time_to_clean_s": float(np.nanmean(tc)) if np.isfinite(tc).any() else None,
               "clean_rate_per_s": float(np.nanmean(1.0 / tc)) if np.isfinite(tc).any() else 0.0,
               "mean_contact_n": stats["mean_contact_n"], "p95_contact_n": stats["p95_contact_n"],
               "peak_n": stats["peak_n"], "frac_contact_over_40n": stats["frac_contact_over_limit"],
               "env_peak_p95_n": float(np.percentile(Fp.max(0), 95)),
               "mean_contact_model_n": float(np.stack(models)[on].mean()) if on.any() else 0.0}
        print("[sweep] ROW", json.dumps(row), flush=True)
        return row

    for F in [float(v) for v in args.setpoints.split(",")]:
        rows.append(run_setpoint(F))
    out = {"ckpt": args.ckpt, "force": tag, "num_envs": base.num_envs, "window_s": args.window_s,
           "rate_s": args.rate_s, "contact_stiffness": cfg.contact_stiffness, "rows": rows,
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(args.out_dir, f"sweep_{tag}.json"), "w"), indent=1)
    print("[sweep] DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        worker(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

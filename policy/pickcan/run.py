from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def summarize(eps: list[dict], dt: float) -> dict:
    import numpy as np

    s = np.array([e["success"] for e in eps], bool)
    t = np.array([e["time_s"] for e in eps])
    f = np.array([e["max_force"] for e in eps])
    dk = np.array([e["max_deck_force"] for e in eps])
    dr = np.array([e["drops"] for e in eps])
    ly = np.array([e["lying"] for e in eps], bool)
    out = {"episodes": int(len(eps)), "success": float(s.mean()),
           "success_lying": float(s[ly].mean()) if ly.any() else None,
           "success_upright": float(s[~ly].mean()) if (~ly).any() else None,
           "time_s_mean": float(t[s].mean()) if s.any() else None,
           "time_s_p95": float(np.percentile(t[s], 95)) if s.any() else None,
           "drops_per_episode": float(dr.mean()), "episodes_with_drop": float((dr > 0).mean()),
           "max_force_mean": float(f.mean()), "max_force_p95": float(np.percentile(f, 95)),
           "max_force_max": float(f.max()), "deck_force_p95": float(np.percentile(dk, 95)),
           "over_20N": float((f > 20).mean()), "sealed": float(np.mean([e["sealed"] for e in eps])),
           "lifted": float(np.mean([e["lift_max"] >= 0.15 for e in eps]))}
    kinds = {}
    for e in eps:
        kinds.setdefault(e["kind"], []).append(e["success"])
    out["success_by_can"] = {str(k): round(float(np.mean(v)), 3) for k, v in sorted(kinds.items())}
    return out


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="script", choices=["script", "rl", "smoke"])
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--ckpt", default="")
    p.add_argument("--out", default="runs/pickup/eval_script.json")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_steps", type=int, default=2000)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)

    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    importlib.import_module("policy.envs.pickcan_env")
    from policy.envs.pickcan_env_cfg import PickCanEnvCfg, PickCanPPORunnerCfg
    from policy.pickcan.script import NAMES, ScriptedPicker

    cfg = PickCanEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = RslRlVecEnvWrapper(gym.make("Rig-PickCan-Go2UR3Lift-v0", cfg=cfg), clip_actions=1.0)
    u = env.unwrapped
    N, dev = u.num_envs, u.device
    obs = env.get_observations()
    policy = None
    if args.mode == "rl":
        from rsl_rl.runners import OnPolicyRunner

        agent = PickCanPPORunnerCfg()
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=dev)
        runner.load(args.ckpt)
        policy = runner.get_inference_policy(device=dev)
    picker = ScriptedPicker(N, dev, dt=u.ctrl_dt, scale=cfg.action_scale)
    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    eps: list[dict] = []
    last = None
    t0 = time.time()
    for step in range(args.max_steps):
        with torch.inference_mode():
            if policy is not None:
                act = policy(obs)
            else:
                act = picker.act(u._tip()[0], u.p_tgt, u.grasp_estimate(), u.sealed, u._robot.data.joint_pos[:, u.lift_id])
            obs, _, dones, _ = env.step(act)
        if args.mode == "smoke" and step % 15 == 0:
            tip = u._tip()[0][0].tolist()
            can = u._can_state()[0][0].tolist()
            print(f"[smoke] step {step} phase {NAMES[int(picker.phase[0])]} tip {[round(v, 3) for v in tip]} "
                  f"tgt {[round(v, 3) for v in u.p_tgt[0].tolist()]} can {[round(v, 3) for v in can]} "
                  f"sealed {bool(u.sealed[0])} force {float(u.tool_force[0]):.1f} lift_max {float(u.lift_max[0]):.3f}"
                  f" ph_hist {torch.bincount(picker.phase, minlength=7).tolist()}", flush=True)
        st = getattr(u, "last_episode_stats", None)
        if st is not None and st is not last:
            last = st
            ids = st["env_ids"]
            fresh = ~done_once[ids]
            for j in fresh.nonzero(as_tuple=False).squeeze(-1).tolist():
                eps.append({k: (v[j].item() if hasattr(v, "dim") else v) for k, v in st.items() if k != "env_ids"})
            done_once[ids] = True
        d = dones.bool()
        if d.any():
            picker.reset(d.nonzero(as_tuple=False).squeeze(-1))
        if bool(done_once.all()):
            break
    res = summarize(eps, u.ctrl_dt)
    res.update({"mode": args.mode, "ckpt": args.ckpt, "num_envs": N, "steps": step + 1,
                "wall_s": round(time.time() - t0, 1)})
    print("[pickcan] RESULT " + json.dumps(res), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": res, "episodes": eps}, f, indent=1)
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

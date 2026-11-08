from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--warm_s", type=float, default=6.0)
    p.add_argument("--rec_s", type=float, default=8.0)
    p.add_argument("--cycles", type=int, default=2)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)
    import numpy as np
    import torch

    from policy.walk.gait import best_loop, gait_period
    from policy.walk.walk_env_cfg import LEGS, WalkEnvCfg
    from policy.walk.walk_sim import deck_cfg, load_policy, make_env, place, set_cmd

    cfg = deck_cfg(WalkEnvCfg(), 1)
    env, policy = load_policy(make_env(cfg), args.ckpt)
    base = env.unwrapped
    r = base.scene["robot"]
    dt = base.step_dt
    q, pos, quat = [], [], []
    with torch.inference_mode():
        env.reset()
        place(base, (0.0, 0.0), 0.0)
        obs = env.get_observations()
        for k in range(int((args.warm_s + args.rec_s) / dt)):
            set_cmd(base, [[args.vx, 0.0, 0.0]])
            obs, _, _, _ = env.step(policy(obs))
            if k * dt >= args.warm_s:
                q.append(r.data.joint_pos[0].cpu().numpy())
                pos.append(r.data.root_pos_w[0].cpu().numpy())
                quat.append(r.data.root_quat_w[0].cpu().numpy())
    q, pos, quat = np.array(q), np.array(pos), np.array(quat)
    names = list(r.joint_names)
    legs = [names.index(n) for n in LEGS]
    period = gait_period(q[:, legs], dt)
    s0, n = best_loop(q[:, legs], period, args.cycles)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, joint_names=np.array(names), dt=dt, period_steps=period, cmd_vx=args.vx,
                        loop_start=s0, q_loop=q[s0:s0 + n], root_pos_loop=pos[s0:s0 + n],
                        root_quat_loop=quat[s0:s0 + n],
                        q_full=q, root_pos_full=pos, root_quat_full=quat)
    v = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]) / (len(pos) * dt))
    print("[gait] " + json.dumps({"period_s": period * dt, "loop_steps": n, "mean_speed_mps": v,
                                  "base_z": float(pos[:, 2].mean()), "joints": len(names)}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

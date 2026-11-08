from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=42)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx

    from policy.walk.walk_env_cfg import TASK, WalkEnvCfg, WalkPPORunnerCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    env_cfg = WalkEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent = WalkPPORunnerCfg()
    agent.seed, agent.max_iterations, agent.device = args.seed, args.iters, "cuda:0"
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent.clip_actions)
    os.makedirs(args.run_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=args.run_dir, device=agent.device)
    if args.resume:
        runner.load(args.resume)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"env": env_cfg.to_dict(), "agent": agent.to_dict()}, f, default=str, indent=1)
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    wall = time.time() - t0
    try:
        pol = runner.alg.policy
        export_policy_as_onnx(pol, normalizer=getattr(pol, "actor_obs_normalizer", None), path=args.run_dir,
                              filename="policy.onnx")
    except Exception as e:
        print(f"[walk] onnx export failed: {e}", flush=True)
    with open(os.path.join(args.run_dir, "train_time.json"), "w") as f:
        json.dump({"wall_s": round(wall), "iters": args.iters, "num_envs": args.num_envs}, f)
    print(f"[walk] TRAINING_DONE {wall:.0f}s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

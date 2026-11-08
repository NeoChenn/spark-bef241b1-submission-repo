from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--run_dir", default="runs/pickup/rl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minutes", type=float, default=60.0)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    importlib.import_module("policy.envs.pickcan_env")
    from policy.envs.pickcan_env_cfg import PickCanEnvCfg, PickCanPPORunnerCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = PickCanEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    agent = PickCanPPORunnerCfg()
    agent.seed = args.seed
    agent.max_iterations = args.iters
    agent.device = "cuda:0"
    env = RslRlVecEnvWrapper(gym.make("Rig-PickCan-Go2UR3Lift-v0", cfg=cfg), clip_actions=agent.clip_actions)
    os.makedirs(args.run_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=args.run_dir, device=agent.device)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"env": cfg.to_dict(), "agent": agent.to_dict()}, f, default=str, indent=1)
    t0 = time.time()
    chunk = 50
    done = 0
    while done < args.iters and time.time() - t0 < args.minutes * 60:
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=done == 0)
        done += chunk
        runner.save(os.path.join(args.run_dir, f"model_{done}.pt"))
        print(f"[pickcan] chunk {done} iters {time.time() - t0:.0f}s", flush=True)
    print(f"[pickcan] TRAINING_DONE {done} iters {time.time() - t0:.0f}s", flush=True)
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

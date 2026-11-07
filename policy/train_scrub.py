"""Train the SCRUB skill (PPO, rsl_rl) on Modal L40S with Isaac Lab.

    .venv/bin/modal run policy/train_scrub.py --smoke                        # 64 envs, 20 iters
    .venv/bin/modal run --detach policy/train_scrub.py --run scrub_v1        # full run (2048 envs)
    .venv/bin/modal run --detach policy/train_scrub.py --run scrub_v1 --resume --iters 2000

Checkpoints + tensorboard: rust-ckpt volume, /ckpt/scrub/<run>/model_<it>.pt (committed every ~2 min).
Monitor:  .venv/bin/modal app logs <app-id>    |   .venv/bin/modal volume ls rust-ckpt scrub/<run>
          .venv/bin/modal volume get rust-ckpt scrub/<run>/progress.jsonl .

The same file is the in-container worker:  python -m policy.train_scrub --worker ...  (run under ISAACLAB_PY).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TASK = "Rig-Scrub-Franka-v0"


def latest_checkpoint(run_dir: str) -> str | None:
    cks = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not cks:
        return None
    return max(cks, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


# =============================================================================== worker (inside Isaac python)
def worker(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--urdf", type=str, default="")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import policy.envs.scrub_env  # noqa: F401  (registers the task)
    from policy.envs.scrub_env_cfg import ScrubEnvCfg, ScrubPPORunnerCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    env_cfg = ScrubEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.spawn_pipe_visual = False
    env_cfg.spawn_dog_visual = False
    if args.urdf:
        from policy.envs.scrub_env_cfg import urdf_robot_cfg

        env_cfg.robot_cfg = urdf_robot_cfg(args.urdf)
    agent_cfg = ScrubPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iters
    agent_cfg.device = "cuda:0"

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    os.makedirs(args.run_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.run_dir, device=agent_cfg.device)
    if args.resume:
        ck = latest_checkpoint(args.run_dir)
        if ck:
            print(f"[scrub] resuming from {ck}", flush=True)
            runner.load(ck)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}, f, default=str, indent=1)
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    print(f"[scrub] TRAINING_DONE {time.time() - t0:.0f}s", flush=True)
    env.close()
    app.close()


# =============================================================================== Modal app
try:
    if "--worker" in sys.argv:
        raise ImportError  # the worker never needs modal
    import modal

    from infra.common import CKPT_DIR, CKPT_VOL, ASSET_DIR, ASSET_VOL
    from infra.isaac_image import ISAAC_GPU, ISAACLAB_PY, isaac_env, isaac_image

    app = modal.App("rig-scrub-train")

    _ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
    _KEYS = {
        "reward": re.compile(r"Mean reward:\s*([-\d.eE+]+)"),
        "ep_len": re.compile(r"Mean episode length:\s*([-\d.eE+]+)"),
        "frac_removed": re.compile(r"Episode/fraction_removed:\s*([-\d.eE+]+)"),
        "success": re.compile(r"Episode/success:\s*([-\d.eE+]+)"),
        "max_force": re.compile(r"Episode/max_force:\s*([-\d.eE+]+)"),
        "sps": re.compile(r"Computation:\s*([\d.]+)\s*steps/s"),
    }

    @app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=6 * 3600,
                  volumes={CKPT_DIR: CKPT_VOL, ASSET_DIR: ASSET_VOL})
    def train(run: str, num_envs: int, iters: int, resume: bool = False, urdf: str = "") -> dict:
        run_dir = f"{CKPT_DIR}/scrub/{run}"
        os.makedirs(run_dir, exist_ok=True)
        cmd = [ISAACLAB_PY, "-m", "policy.train_scrub", "--worker", "--num_envs", str(num_envs),
               "--iters", str(iters), "--run_dir", run_dir, "--headless"]
        if resume:
            cmd.append("--resume")
        if urdf:
            cmd += ["--urdf", urdf]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd="/root", env=isaac_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_commit, cur, history = time.time(), {}, []
        prog = open(os.path.join(run_dir, "progress.jsonl"), "a")
        for line in proc.stdout:
            sys.stdout.write(line)
            m = _ITER_RE.search(line)
            if m:
                if cur.get("iter") is not None:
                    history.append(cur)
                    prog.write(json.dumps(cur) + "\n")
                    prog.flush()
                    print(f"[scrub] it {cur['iter']} reward {cur.get('reward')} frac_removed "
                          f"{cur.get('frac_removed')} success {cur.get('success')} max_force {cur.get('max_force')} "
                          f"sps {cur.get('sps')}", flush=True)
                cur = {"iter": int(m.group(1))}
            for k, rx in _KEYS.items():
                mm = rx.search(line)
                if mm:
                    cur[k] = float(mm.group(1))
            if time.time() - last_commit > 120:
                prog.flush()
                CKPT_VOL.commit()
                last_commit = time.time()
        proc.wait()
        if cur.get("iter") is not None:
            history.append(cur)
            prog.write(json.dumps(cur) + "\n")
        prog.close()
        CKPT_VOL.commit()
        ck = latest_checkpoint(run_dir)
        res = {"returncode": proc.returncode, "run_dir": run_dir, "latest_ckpt": ck, "iters_logged": len(history),
               "first": history[0] if history else None, "last": history[-1] if history else None}
        print("[scrub] RESULT", json.dumps(res), flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"worker failed rc={proc.returncode}")
        return res

    @app.local_entrypoint()
    def main(run: str = "", smoke: bool = False, num_envs: int = 2048, iters: int = 3000, resume: bool = False,
             urdf: str = ""):
        if smoke:
            num_envs, iters = 64, 20
            run = run or "smoke"
        run = run or time.strftime("scrub_%m%d_%H%M")
        print(f"[scrub] run={run} num_envs={num_envs} iters={iters} resume={resume}")
        print(train.remote(run, num_envs, iters, resume, urdf))

except ImportError:  # inside the Isaac worker without modal/infra available
    app = None


if __name__ == "__main__" and "--worker" in sys.argv:
    worker(sys.argv[1:])

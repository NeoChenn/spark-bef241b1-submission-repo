"""Train the PICK skill (litter pickup, PPO, rsl_rl) in Isaac Lab. A copy of policy/train_scrub.py for the pick env.

    MODAL_PROFILE=jacobyuan99 .venv/bin/modal run policy/train_pick.py --smoke                           # 64 envs, 20 iters
    MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach policy/train_pick.py --run pick_v1 --init_from <cpu ckpt>
GB10 (watson):
    infra/watson/run.sh policy.train_pick --worker --num_envs 2048 --iters 1500 --run_dir /repo/runs/pick_isaac \
        --init_from /repo/runs/pick_v1/model_1499.pt

--init_from takes a checkpoint from the CPU trainer (policy/train_pick_cpu.py): same observations, actions, rewards
and network, so the Isaac run starts from a policy that already picks and adapts it to the real arm.
Checkpoints: /ckpt/pick/<run>/model_<it>.pt on the rust-ckpt volume (Modal) or --run_dir (GB10).
NOT YET RUN: the pick env (policy/envs/pick_env.py) has not been run in Isaac; see its docstring.
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
    p.add_argument("--init_from", type=str, default="", help="checkpoint to initialise weights from (fine-tune)")
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

    from policy.envs.pick_env import TASK  # registers the task
    from policy.envs.pick_env_cfg import PickEnvCfg, PickPPORunnerCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    task = TASK
    env_cfg = PickEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.urdf:
        from policy.envs.pick_env_cfg import PICK_INIT
        from policy.envs.scrub_env_cfg import urdf_robot_cfg
        env_cfg.robot_cfg = urdf_robot_cfg(args.urdf, init_joint_pos=PICK_INIT)
    agent_cfg = PickPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iters
    agent_cfg.device = "cuda:0"

    print(f"[pick] task={task} obs={env_cfg.observation_space}", flush=True)
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    os.makedirs(args.run_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.run_dir, device=agent_cfg.device)
    if args.resume:
        ck = latest_checkpoint(args.run_dir)
        if ck:
            print(f"[pick] resuming from {ck}", flush=True)
            runner.load(ck)
    elif args.init_from:
        print(f"[pick] initialising weights from {args.init_from}", flush=True)
        runner.load(args.init_from, load_optimizer=False)
        runner.current_learning_iteration = 0
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}, f, default=str, indent=1)
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    print(f"[pick] TRAINING_DONE {time.time() - t0:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)  # simulation_app.close() can hang on Modal


# =============================================================================== Modal app
try:
    if "--worker" in sys.argv:
        raise ImportError  # the worker never needs modal
    import modal

    from infra.common import CKPT_DIR, CKPT_VOL, ASSET_DIR, ASSET_VOL
    from infra.isaac_image import ISAAC_GPU, ISAACLAB_PY, isaac_env, isaac_image

    app = modal.App("rig-pick-train")

    _ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
    _KEYS = {
        "reward": re.compile(r"Mean reward:\s*([-\d.eE+]+)"),
        "ep_len": re.compile(r"Mean episode length:\s*([-\d.eE+]+)"),
        "grasped": re.compile(r"Episode/grasped:\s*([-\d.eE+]+)"),
        "success": re.compile(r"Episode/success:\s*([-\d.eE+]+)"),
        "knocks": re.compile(r"Episode/knocks:\s*([-\d.eE+]+)"),
        "sps": re.compile(r"Computation:\s*([\d.]+)\s*steps/s"),
    }

    @app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=6 * 3600,
                  volumes={CKPT_DIR: CKPT_VOL, ASSET_DIR: ASSET_VOL})
    def train(run: str, num_envs: int, iters: int, resume: bool = False, urdf: str = "", init_from: str = "") -> dict:
        run_dir = f"{CKPT_DIR}/pick/{run}"
        os.makedirs(run_dir, exist_ok=True)
        cmd = [ISAACLAB_PY, "-m", "policy.train_pick", "--worker", "--num_envs", str(num_envs),
               "--iters", str(iters), "--run_dir", run_dir, "--headless"]
        if resume:
            cmd.append("--resume")
        if urdf:
            cmd += ["--urdf", urdf]
        if init_from:
            cmd += ["--init_from", init_from if init_from.startswith("/") else f"{CKPT_DIR}/pick/{init_from}"]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd="/root", env=isaac_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_commit, cur, history, done = time.time(), {}, [], False
        prog = open(os.path.join(run_dir, "progress.jsonl"), "a")
        for line in proc.stdout:
            sys.stdout.write(line)
            done = done or "[pick] TRAINING_DONE" in line
            m = _ITER_RE.search(line)
            if m:
                if cur.get("iter") is not None:
                    history.append(cur)
                    prog.write(json.dumps(cur) + "\n")
                    prog.flush()
                    print(f"[pick] it {cur['iter']} reward {cur.get('reward')} grasped {cur.get('grasped')} "
                          f"success {cur.get('success')} knocks {cur.get('knocks')} sps {cur.get('sps')}", flush=True)
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
        res = {"returncode": proc.returncode, "done": done, "run_dir": run_dir, "latest_ckpt": ck, "iters_logged": len(history),
               "first": history[0] if history else None, "last": history[-1] if history else None}
        print("[pick] RESULT", json.dumps(res), flush=True)
        if proc.returncode != 0 or not done:  # Kit can exit 0 after a Python exception
            raise RuntimeError(f"worker failed rc={proc.returncode} done={done}")
        return res

    @app.local_entrypoint()
    def main(run: str = "", smoke: bool = False, num_envs: int = 2048, iters: int = 1500, resume: bool = False,
             urdf: str = "", init_from: str = ""):
        if smoke:
            num_envs, iters = 64, 20
            run = run or "smoke"
        run = run or time.strftime("pick_%m%d_%H%M")
        print(f"[pick] run={run} num_envs={num_envs} iters={iters} resume={resume} init_from={init_from}")
        print(train.remote(run, num_envs, iters, resume, urdf, init_from))

except ImportError:  # inside the Isaac worker without modal/infra available
    app = None


if __name__ == "__main__" and "--worker" in sys.argv:
    try:
        worker(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

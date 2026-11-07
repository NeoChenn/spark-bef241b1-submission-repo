"""Evaluate a trained SCRUB policy (and export it) on Modal.

    .venv/bin/modal run policy/play_scrub.py --run scrub_v1                         # 256 episodes, stats
    .venv/bin/modal run policy/play_scrub.py --run scrub_v1 --video --meta          # rig-scene demo video
    .venv/bin/modal run policy/play_scrub.py --run scrub_v1 --force-patch           # every episode gets a mid-run patch

Reports success rate, mean time to clean, max contact force, % of episodes with a mid-run patch that were still
completed. Writes /ckpt/scrub/<run>/eval.json, exported/policy.pt (TorchScript) + policy.onnx (obs normaliser
baked in; obs/action layout = ScrubEnv._get_observations / ScrubEnvCfg docs) and optionally videos/*.mp4.
--meta sets the geometry exactly from sim/output/rig_deck_meta.json (pipe r=0.15 at z=1.1, damage side).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

META_PATH_IN_IMAGE = "/root/rig_deck_meta.json"


def apply_meta(env_cfg, meta_path: str):
    """Collapse the domain randomisation onto the sim team's rig scene."""
    from policy.envs.rust_state import rig_meta_scene

    m = rig_meta_scene(meta_path, standoff=0.30, base_height=env_cfg.base_height)
    r = m["pipe_radius"]
    env_cfg.radius_range = (r, r)
    env_cfg.height_range = (m["pipe_origin"][2],) * 2
    env_cfg.standoff_range = (m["standoff"],) * 2
    env_cfg.axial_range = (0.0, 0.0)
    env_cfg.yaw_range = (0.0, 0.0)
    env_cfg.rust_theta_window = (m["damage_theta"], min(0.6, m["damage_radius"] / r * 0.5))
    return m


def worker(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--video", action="store_true")
    p.add_argument("--video_steps", type=int, default=600)
    p.add_argument("--meta", type=str, default="")
    p.add_argument("--force_patch", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

    import policy.envs.scrub_env  # noqa: F401
    from policy.envs.scrub_env_cfg import ScrubEnvCfg, ScrubPPORunnerCfg

    env_cfg = ScrubEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.meta:
        print("[play] rig meta:", apply_meta(env_cfg, args.meta), flush=True)
    else:
        # visuals sit at the nominal pose; pin the pipe there when filming so the picture matches the physics
        if args.video:
            r = 0.5 * sum(env_cfg.radius_range)
            env_cfg.radius_range = (r, r)
            env_cfg.height_range = (0.5 * sum(env_cfg.height_range),) * 2
            env_cfg.standoff_range = (0.5 * sum(env_cfg.standoff_range),) * 2
            env_cfg.axial_range = (0.0, 0.0)
            env_cfg.yaw_range = (0.0, 0.0)
    if args.force_patch:
        env_cfg.prob_new_patch = 1.0
    env_cfg.viz = args.video
    env_cfg.spawn_pipe_visual = args.video
    env_cfg.spawn_dog_visual = args.video
    agent_cfg = ScrubPPORunnerCfg()
    agent_cfg.device = "cuda:0"

    env = gym.make("Rig-Scrub-Franka-v0", cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=base.device)
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    exp_dir = os.path.join(args.out_dir, "exported")
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=exp_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=exp_dir, filename="policy.onnx")
        print(f"[play] exported policy to {exp_dir}", flush=True)
    except Exception as e:  # export is a nice-to-have
        print(f"[play] export failed: {e}", flush=True)

    recs = {k: [] for k in ("fraction_removed", "success", "max_force", "time_to_clean_s", "new_patch")}
    frames = []
    obs = env.get_observations()
    max_steps = args.video_steps if args.video else 10 ** 9
    step = 0
    t0 = time.time()
    while len(recs["success"]) < args.episodes and step < max_steps:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        step += 1
        if dones.any() and getattr(base, "last_episode_stats", None) is not None:
            st = base.last_episode_stats
            for k in recs:
                recs[k] += st[k].float().cpu().tolist()
            base.last_episode_stats = None
        if args.video:
            frames.append(base.render())
    n = len(recs["success"])
    succ = recs["success"]
    np_ = [s for s, e in zip(succ, recs["new_patch"]) if e]
    res = {
        "ckpt": args.ckpt, "episodes": n, "meta": bool(args.meta),
        "success_rate": sum(succ) / max(n, 1),
        "mean_fraction_removed": sum(recs["fraction_removed"]) / max(n, 1),
        "mean_time_to_clean_s": (sum(t for t, s in zip(recs["time_to_clean_s"], succ) if s) / max(sum(succ), 1)),
        "max_force_N": max(recs["max_force"]) if n else None,
        "mean_episode_max_force_N": sum(recs["max_force"]) / max(n, 1),
        "episodes_with_new_patch": len(np_),
        "new_patch_handled_rate": (sum(np_) / len(np_)) if np_ else None,
        "wall_s": round(time.time() - t0, 1),
    }
    print("[play] EVAL", json.dumps(res), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_meta.json" if args.meta else "eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    if frames:
        import imageio

        vdir = os.path.join(args.out_dir, "videos")
        os.makedirs(vdir, exist_ok=True)
        path = os.path.join(vdir, time.strftime("scrub_%m%d_%H%M%S.mp4"))
        imageio.mimsave(path, [f for f in frames if f is not None], fps=int(round(1 / base.step_dt)))
        print(f"[play] video -> {path}", flush=True)
    env.close()
    app.close()


try:
    if "--worker" in sys.argv:
        raise ImportError
    import modal

    from infra.common import CKPT_DIR, CKPT_VOL
    from infra.isaac_image import ISAAC_GPU, ISAACLAB_PY, isaac_env, isaac_image

    _meta_local = os.path.join(ROOT, "sim", "output", "rig_deck_meta.json")
    play_image = isaac_image.add_local_file(_meta_local, META_PATH_IN_IMAGE) if os.path.exists(_meta_local) else isaac_image
    app = modal.App("rig-scrub-play")

    @app.function(image=play_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=3600, volumes={CKPT_DIR: CKPT_VOL})
    def play(run: str, ckpt: str = "", episodes: int = 256, num_envs: int = 64, video: bool = False,
             meta: bool = False, force_patch: bool = False) -> dict:
        from policy.train_scrub import latest_checkpoint

        CKPT_VOL.reload()
        run_dir = f"{CKPT_DIR}/scrub/{run}"
        ck = os.path.join(run_dir, ckpt) if ckpt else latest_checkpoint(run_dir)
        if not ck:
            raise FileNotFoundError(f"no checkpoint in {run_dir}")
        cmd = [ISAACLAB_PY, "-m", "policy.play_scrub", "--worker", "--ckpt", ck, "--out_dir", run_dir,
               "--episodes", str(episodes), "--num_envs", str(4 if video else num_envs), "--headless"]
        if video:
            cmd.append("--video")
        if meta:
            cmd += ["--meta", META_PATH_IN_IMAGE]
        if force_patch:
            cmd.append("--force_patch")
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd="/root", env=isaac_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        res = None
        for line in proc.stdout:
            sys.stdout.write(line)
            m = re.search(r"\[play\] EVAL (\{.*\})", line)
            if m:
                res = json.loads(m.group(1))
        proc.wait()
        CKPT_VOL.commit()
        if proc.returncode != 0 and res is None:
            raise RuntimeError(f"play worker failed rc={proc.returncode}")
        return res

    @app.local_entrypoint()
    def main(run: str, ckpt: str = "", episodes: int = 256, num_envs: int = 64, video: bool = False,
             meta: bool = False, force_patch: bool = False):
        print(play.remote(run, ckpt, episodes, num_envs, video, meta, force_patch))

except ImportError:
    app = None


if __name__ == "__main__" and "--worker" in sys.argv:
    worker(sys.argv[1:])

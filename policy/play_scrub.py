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
CAMS = {"close": ((0.55, 0.05, 1.55), (0.0, -0.42, 1.18)),
        "wide": ((1.9, 1.5, 1.6), (0.0, -0.25, 0.75))}


def apply_meta(env_cfg, meta_path: str):
    from policy.envs.rust_state import rig_meta_scene

    standoff = 0.5 * sum(env_cfg.standoff_range)
    m = rig_meta_scene(meta_path, standoff=standoff, base_height=env_cfg.base_height)
    r = m["pipe_radius"]
    env_cfg.radius_range = (r, r)
    z = m["pipe_origin"][2]
    if z > env_cfg.max_reach_z:
        z_new = 0.5 * sum(env_cfg.height_range)
        print(f"[play] WARNING: rig pipe axis z={z:.2f} m is out of the arm's reach (> {env_cfg.max_reach_z} m); "
              f"using z={z_new:.2f} m instead", flush=True)
        m["pipe_origin"] = (m["pipe_origin"][0], m["pipe_origin"][1], z_new)
        m["robot_pos_world"] = (m["robot_pos_world"][0], m["robot_pos_world"][1], env_cfg.base_height)
        z = z_new
    env_cfg.height_range = (z, z)
    env_cfg.standoff_range = (m["standoff"],) * 2
    d_axis = (0.5 * sum(env_cfg.axis_dist_range)) if env_cfg.axis_dist_range is not None else m["standoff"] + r
    if env_cfg.axis_dist_range is not None:
        env_cfg.axis_dist_range = (d_axis, d_axis)
    po = m["pipe_origin"]
    m["robot_pos_world"] = (po[0], po[1] + d_axis, env_cfg.base_height)
    m["robot_yaw_world"] = env_cfg.robot_yaw
    m["env_to_world"] = {"translation": (po[0], po[1] + d_axis, 0.0), "yaw": 0.0}
    env_cfg.axial_range = (0.0, 0.0)
    env_cfg.yaw_range = (0.0, 0.0)
    env_cfg.rust_theta_window = (m["damage_theta"], min(env_cfg.rust_theta_window[1], m["damage_radius"] / r * 0.5))
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
    p.add_argument("--robot", type=str, default="lift", choices=["lift", "lift_v2", "lift_v3", "go2ur3", "franka"])
    p.add_argument("--storm", type=float, default=-1.0)
    p.add_argument("--gust_gate", type=float, nargs=2, default=None, metavar=("LIMIT", "RESUME"))
    p.add_argument("--fmax", type=float, default=0.0)
    p.add_argument("--storm_parts", default="gust,deck")
    p.add_argument("--cam", type=str, default="", choices=["", "close", "wide"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cbf", action="store_true")
    p.add_argument("--cbf_fmax", type=float, default=40.0)
    p.add_argument("--cbf_gamma", type=float, default=0.5)
    p.add_argument("--cbf_margin", type=float, default=2.0)
    p.add_argument("--cbf_alpha", type=float, default=1.0)
    p.add_argument("--tag", type=str, default="")
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

    from policy.envs import scrub_env_cfg as C
    from policy.envs.scrub_env import TASKS
    from policy.envs.scrub_env_cfg import ScrubPPORunnerCfg

    task, cfg_name = TASKS[args.robot]
    env_cfg = getattr(C, cfg_name)()
    env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.storm >= 0:
        env_cfg.storm_level_range = (args.storm, args.storm)
        if "deck" not in args.storm_parts:
            env_cfg.storm_deck_amp, env_cfg.storm_trunk_sway = (0.0, 0.0), 0.0
        if "gust" not in args.storm_parts:
            env_cfg.storm_gust_peak, env_cfg.storm_gust_mean = (0.0, 0.0), (0.0, 0.0)
    if args.meta:
        print("[play] rig meta:", apply_meta(env_cfg, args.meta), flush=True)
    else:
        if args.video:
            r = 0.5 * sum(env_cfg.radius_range)
            env_cfg.radius_range = (r, r)
            env_cfg.height_range = (0.5 * sum(env_cfg.height_range),) * 2
            env_cfg.standoff_range = (0.5 * sum(env_cfg.standoff_range),) * 2
            if env_cfg.axis_dist_range is not None:
                env_cfg.axis_dist_range = (0.5 * sum(env_cfg.axis_dist_range),) * 2
            env_cfg.axial_range = (0.0, 0.0)
            env_cfg.yaw_range = (0.0, 0.0)
    if args.force_patch:
        env_cfg.prob_new_patch = 1.0
    if args.cam:
        from isaaclab.envs import ViewerCfg

        eye, look = CAMS[args.cam]
        env_cfg.viewer = ViewerCfg(eye=eye, lookat=look, resolution=(1280, 720), origin_type="env", env_index=0)
    env_cfg.viz = args.video
    env_cfg.spawn_pipe_visual = args.video
    env_cfg.spawn_dog_visual = args.video and args.robot == "franka"
    agent_cfg = ScrubPPORunnerCfg()
    agent_cfg.device = "cuda:0"

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=base.device)
    shield = None
    if args.cbf:
        from policy.cbf_shield import CBFShield, attach

        shield = attach(base, CBFShield(base.num_envs, k0=env_cfg.spring_k_nominal, f_max=args.cbf_fmax,
                                        gamma=args.cbf_gamma, margin=args.cbf_margin, alpha=args.cbf_alpha,
                                        device=base.device))
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    exp_dir = os.path.join(args.out_dir, "exported")
    for fn, name in ((export_policy_as_onnx, "policy.onnx"), (export_policy_as_jit, "policy.pt")):
        try:
            fn(policy_nn, normalizer=normalizer, path=exp_dir, filename=name)
            print(f"[play] exported {exp_dir}/{name}", flush=True)
        except Exception as e:
            print(f"[play] export {name} failed: {e}", flush=True)

    recs = {k: [] for k in ("fraction_removed", "success", "max_force", "time_to_clean_s", "new_patch")}
    frames = []
    obs = env.get_observations()
    max_steps = args.video_steps if args.video else 10 ** 9
    step = 0
    t0 = time.time()
    paused = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    while len(recs["success"]) < args.episodes and step < max_steps:
        with torch.inference_mode():
            act = policy(obs)
            if args.fmax > 0:
                act[:, 3] = act[:, 3].clamp(max=(args.fmax - env_cfg.force_set_mid) / env_cfg.force_set_half)
            if args.gust_gate:
                g = base.gust_n
                paused = torch.where(paused, g >= args.gust_gate[1], g > args.gust_gate[0])
                th = base.theta
                retract = torch.stack([torch.zeros_like(th), torch.cos(th), torch.sin(th), -torch.ones_like(th)], -1)
                act = torch.where(paused[:, None], retract, act)
            obs, _, dones, _ = env.step(act)
            paused &= ~dones.bool()
            if shield is not None:
                shield.reset(dones)
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
    mf = sorted(recs["max_force"])
    np_ = [s for s, e in zip(succ, recs["new_patch"]) if e]
    res = {
        "ckpt": args.ckpt, "episodes": n, "meta": bool(args.meta), "storm": max(args.storm, 0.0), "gust_gate": args.gust_gate, "fmax": args.fmax, "storm_parts": args.storm_parts,
        "success_rate": sum(succ) / max(n, 1),
        "mean_fraction_removed": sum(recs["fraction_removed"]) / max(n, 1),
        "mean_time_to_clean_s": (sum(t for t, s in zip(recs["time_to_clean_s"], succ) if s) / max(sum(succ), 1)),
        "max_force_N": max(recs["max_force"]) if n else None,
        "mean_episode_max_force_N": sum(recs["max_force"]) / max(n, 1),
        "p95_episode_max_force_N": mf[min(int(0.95 * n), n - 1)] if n else None,
        "frac_episodes_over_40N": sum(f > 40.0 for f in mf) / max(n, 1),
        "episodes_with_new_patch": len(np_),
        "new_patch_handled_rate": (sum(np_) / len(np_)) if np_ else None,
        "wall_s": round(time.time() - t0, 1),
        "seed": args.seed, "cbf": bool(args.cbf),
    }
    if shield is not None:
        res.update({"cbf_fmax": args.cbf_fmax, "cbf_gamma": args.cbf_gamma, "cbf_margin": args.cbf_margin,
                    "cbf_alpha": args.cbf_alpha, "cbf_duty": shield.duty()})
    print("[play] EVAL", json.dumps(res), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"eval{'_meta' if args.meta else ''}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out_dir, f"episodes{'_meta' if args.meta else ''}{args.tag}.json"), "w") as f:
        json.dump(recs, f)
    if frames:
        import imageio

        vdir = os.path.join(args.out_dir, "videos")
        os.makedirs(vdir, exist_ok=True)
        path = os.path.join(vdir, time.strftime(f"scrub_{args.cam or 'view'}_%m%d_%H%M%S.mp4"))
        imageio.mimsave(path, [f for f in frames if f is not None], fps=int(round(1 / base.step_dt)))
        print(f"[play] video -> {path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


try:
    if "--worker" in sys.argv:
        raise ImportError
    import modal

    from infra.common import ASSET_DIR, ASSET_VOL, CKPT_DIR, CKPT_VOL
    from infra.isaac_image import ISAAC_GPU, ISAACLAB_PY, isaac_env, isaac_image

    _meta_local = os.path.join(ROOT, "sim", "output", "rig_deck_meta.json")
    play_image = isaac_image.add_local_file(_meta_local, META_PATH_IN_IMAGE) if os.path.exists(_meta_local) else isaac_image
    app = modal.App("rig-scrub-play")

    @app.function(image=play_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=3600,
                  volumes={CKPT_DIR: CKPT_VOL, ASSET_DIR: ASSET_VOL})
    def play(run: str, ckpt: str = "", episodes: int = 256, num_envs: int = 64, video: bool = False,
             meta: bool = False, force_patch: bool = False, robot: str = "lift") -> dict:
        from policy.train_scrub import latest_checkpoint

        CKPT_VOL.reload()
        run_dir = f"{CKPT_DIR}/scrub/{run}"
        ck = os.path.join(run_dir, ckpt) if ckpt else latest_checkpoint(run_dir)
        if not ck:
            raise FileNotFoundError(f"no checkpoint in {run_dir}")
        cmd = [ISAACLAB_PY, "-m", "policy.play_scrub", "--worker", "--ckpt", ck, "--out_dir", run_dir,
               "--episodes", str(episodes), "--num_envs", str(4 if video else num_envs), "--robot", robot,
               "--headless"]
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
             meta: bool = False, force_patch: bool = False, robot: str = "lift"):
        print(play.remote(run, ckpt, episodes, num_envs, video, meta, force_patch, robot))

except ImportError:
    app = None


if __name__ == "__main__" and "--worker" in sys.argv:
    try:
        worker(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class IsaacView:

    def __init__(self, base, first_done):
        self.b, self.first_done = base, first_done

    def __getattr__(self, k):
        return getattr(self.b, k)

    @property
    def N(self):
        return self.b.num_envs

    @property
    def done(self):
        return self.first_done

    def native_from_target(self, th, z, press):
        from bench.scrub_sim import native_from_target

        return native_from_target(self.b, th, z, press)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["ppo", "greedy", "random"])
    p.add_argument("--ckpt", default=".work/scrub_v2_best/model.pt")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--meta", default="")
    p.add_argument("--out", required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from bench.scrub_arms import GreedyArm, RandomArm
    from policy.envs import scrub_env_cfg as C
    from policy.envs.scrub_env import TASKS
    from policy.play_scrub import apply_meta

    task, cfg_name = TASKS["lift_v2"]
    cfg = getattr(C, cfg_name)()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.prob_new_patch = 1.0
    cfg.viz = False
    cfg.spawn_pipe_visual = False
    cfg.spawn_dog_visual = False
    if args.meta:
        apply_meta(cfg, args.meta)
    agent = C.ScrubPPORunnerCfg()
    agent.device = "cuda:0"
    agent.seed = args.seed
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    rust0 = base.rust.rust.sum(dim=(1, 2)).cpu().numpy().round(4).tobytes()
    first = torch.zeros(base.num_envs, dtype=torch.bool)
    view = IsaacView(base, first)
    if args.arm == "ppo":
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(args.ckpt)
        pol = runner.get_inference_policy(device=base.device)
        act = lambda o: pol(o)
    else:
        arm = GreedyArm() if args.arm == "greedy" else RandomArm(seed=args.seed)
        arm.reset(view)
        act = lambda o: arm.act(view)
    recs = {}
    obs = env.get_observations()
    ms = []
    t0 = time.time()
    step = 0
    while len(recs) < base.num_envs and step < 2000:
        t = time.perf_counter()
        with torch.inference_mode():
            a = act(obs)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000.0)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(a)
        step += 1
        st = getattr(base, "last_episode_stats", None)
        if dones.any() and st is not None:
            for j, eid in enumerate(st["env_ids"].tolist()):
                if eid in recs:
                    continue
                succ = bool(st["success"][j])
                t_clean = float(st["time_to_clean_s"][j])
                ep_s = t_clean if succ else cfg.episode_length_s
                fr = float(st["fraction_removed"][j])
                recs[eid] = {"env": eid, "success": succ, "time_to_clean_s": t_clean if succ else None,
                             "peak_force_N": float(st["max_force"][j]), "over_40N": float(st["max_force"][j]) > 40.0,
                             "fraction_removed": fr, "episode_s": ep_s, "removed_pct_per_s": 100.0 * fr / max(ep_s, 1e-6),
                             "new_patch": bool(st["new_patch"][j])}
                first[eid] = True
            base.last_episode_stats = None
    res = {"arm": args.arm, "task": task, "seed": args.seed, "num_envs": base.num_envs, "meta": bool(args.meta),
           "ckpt": args.ckpt if args.arm == "ppo" else None, "scenario_hash": hashlib.sha1(rust0).hexdigest()[:12],
           "batched_act_ms_median": sorted(ms)[len(ms) // 2], "wall_s": round(time.time() - t0, 1),
           "episodes": [recs[k] for k in sorted(recs)]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("[bench] done", args.arm, len(recs), res["scenario_hash"], flush=True)
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

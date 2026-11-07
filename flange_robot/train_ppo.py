"""Train the repair decision policy with MaskablePPO.

    python -m flange_robot.train_ppo --steps 200000 --out checkpoints/ppo_repair

Evaluates on held-out scenarios every `--eval-every` steps and keeps the best checkpoint as
`<out>/best.zip`. The final model is saved as `<out>/last.zip`. Checkpoints are gitignored; commit
a small final model deliberately if the submission needs it.

Needs: torch, gymnasium, stable-baselines3, sb3-contrib (CPU is fine).
"""

import argparse
import os
import time

import numpy as np

from .evaluate import heldout_specs, run_ppo, summarise


def make_env(seed: int):
    def _init():
        from .rl_env import RepairSkillEnv

        return RepairSkillEnv(seed=seed)

    return _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-n", type=int, default=120)
    ap.add_argument("--out", default="checkpoints/ppo_repair")
    a = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    os.makedirs(a.out, exist_ok=True)
    maker = SubprocVecEnv if a.n_envs > 1 else DummyVecEnv
    vec = maker([make_env(a.seed * 1000 + i) for i in range(a.n_envs)])
    model = MaskablePPO(
        "MlpPolicy", vec, n_steps=256, batch_size=256, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
        ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), seed=a.seed, device="cpu", verbose=0,
    )
    val = heldout_specs(a.eval_n, seed=424242)  # validation set, different from the final test set
    best, done, t0 = -1e9, 0, time.time()
    print(f"training {a.steps} steps on {a.n_envs} envs", flush=True)
    while done < a.steps:
        model.learn(a.eval_every, reset_num_timesteps=False)
        done = model.num_timesteps
        s = summarise([run_ppo(model, sp) for sp in val])
        flag = ""
        if s["mean_return"] > best:
            best = s["mean_return"]
            model.save(os.path.join(a.out, "best"))
            flag = "  <- best"
        print(f"[{done:>8d} steps, {time.time() - t0:5.0f}s] val return {s['mean_return']:+.2f} success "
              f"{s['success_rate']:.1%} steps {s['mean_steps']:.1f} false-clears {s['false_all_clears']} "
              f"unsafe {s['unsafe_tightenings']}{flag}", flush=True)
    model.save(os.path.join(a.out, "last"))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()

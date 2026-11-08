"""Train the PICK skill (litter pickup) with rsl_rl PPO on CPU, using the team's scrub PPO settings.

PPO from scratch never finds a grasp (the jaws must close for several steps in a row while centred within ~1 cm;
random actions do that about once in 10,000 steps), so the actor is first warm-started from the scripted pick by
behaviour cloning with DAgger (the student drives, the script labels), then PPO fine-tunes it on the reward, which
is where it can beat the script (camera error, knocks).

    python -m policy.train_pick_cpu --run_dir runs/pick_cpu --num_envs 1024 --iters 400
    python -m policy.train_pick_cpu --run_dir runs/pick_cpu --eval_only            # score the latest checkpoint

Writes model_<it>.pt checkpoints (rsl_rl format, loadable by the Isaac trainer's --init_from), progress.jsonl,
exported/policy.onnx (actor with its observation normaliser, like play_scrub.py) and eval.json, which compares the
policy with the scripted pick (policy/envs/pick_scripted.py) under the same camera errors.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import time

import torch

from policy.envs.pick_cpu_env import EPISODE_STEPS, PPO_CFG, PickCpuEnv
from policy.envs.pick_scripted import rollout, scripted_action
from policy.envs.pick_state import ACT_DIM, OBS_DIM, PickParams, PickState


def latest_checkpoint(run_dir: str) -> str | None:
    cks = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return max(cks, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1))) if cks else None


class ExportedActor(torch.nn.Module):
    """Observation normaliser + actor MLP -> mean action in [-1, 1] (what play_scrub exports for scrub)."""

    def __init__(self, policy):
        super().__init__()
        self.norm = copy.deepcopy(policy.actor_obs_normalizer).cpu()
        self.actor = copy.deepcopy(policy.actor).cpu()

    def forward(self, obs):
        return self.actor(self.norm(obs)).clamp(-1, 1)


def bc_warm_start(policy, rounds: int = 4, n: int = 1024, epochs: int = 3, lr: float = 1e-3, seed: int = 7,
                  params: PickParams | None = None) -> list[float]:
    """Behaviour cloning + DAgger from policy.envs.pick_scripted into the rsl_rl actor (and its obs normaliser)."""
    params = params or PickParams(hover_start=0.3)
    opt = torch.optim.Adam(policy.actor.parameters(), lr=lr)
    obs_all, act_all, success = [], [], []
    for k in range(rounds):
        st = PickState(n, params, seed=seed + k)
        last = torch.zeros(n, ACT_DIM)
        beta = 1.0 if k == 0 else 0.5 ** k          # share of steps where the script, not the student, drives
        for t in range(EPISODE_STEPS):
            o = st.observe(last, torch.full((n,), t / EPISODE_STEPS))
            label = scripted_action(st)
            with torch.no_grad():
                student = policy.actor(policy.actor_obs_normalizer(o)).clamp(-1, 1)
            drive = torch.where((torch.rand(n, 1) < beta), label, student)
            obs_all.append(o)
            act_all.append(label)
            st.step(drive)
            last = drive
            if bool(st.success.all()):
                break
        success.append(round(st.ever_grasped.float().mean().item(), 3))
        O, A = torch.cat(obs_all), torch.cat(act_all)
        # fit the observation normalisers NOW (rsl_rl updates them with .update(), not by calling them); if they
        # stay at identity the cloned actor learns raw inputs and breaks as soon as PPO starts normalising
        new = torch.cat(obs_all[-(len(obs_all) // (k + 1)):])
        for norm in (policy.actor_obs_normalizer, getattr(policy, "critic_obs_normalizer", None)):
            if hasattr(norm, "update"):
                norm.train()
                norm.update(new)
                norm.eval()
        for _ in range(epochs):
            perm = torch.randperm(len(O))
            for i in range(0, len(O), 4096):
                b = perm[i:i + 4096]
                loss = ((policy.actor(policy.actor_obs_normalizer(O[b])) - A[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        print(f"[pick] BC round {k}: {len(O)} samples, loss {loss.item():.4f}, grasp rate while collecting {success[-1]}",
              flush=True)
    return success


def evaluate(act_fn, n=1024, seed=123, params: PickParams | None = None) -> dict:
    s = PickState(n, params or PickParams(), seed=seed)
    r = rollout(s, act_fn, steps=EPISODE_STEPS)
    ok = r["success"]
    return {"success": round(ok.float().mean().item(), 4), "knocks_per_pick": round(r["knocks"].mean().item(), 3),
            "drops_per_pick": round(r["drops"].mean().item(), 3),
            "median_s": round(r["t_success"][ok].float().median().item() / 30, 2) if ok.any() else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/pick_cpu")
    ap.add_argument("--num_envs", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--hover_start", type=float, default=0.3, help="training curriculum (evaluation always uses 0)")
    ap.add_argument("--bc_rounds", type=int, default=4, help="behaviour-cloning warm start from the script (0 = off)")
    ap.add_argument("--ppo_std", type=float, default=0.3, help="action noise after the warm start")
    ap.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate when fine-tuning a warm start")
    ap.add_argument("--critic_warmup", type=int, default=20, help="iterations with the actor frozen after BC")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.run_dir, exist_ok=True)
    from rsl_rl.runners import OnPolicyRunner

    env = PickCpuEnv(a.num_envs, PickParams(hover_start=a.hover_start), seed=a.seed)
    cfg = copy.deepcopy(PPO_CFG)
    cfg["seed"] = a.seed
    if a.bc_rounds > 0:
        # fine-tuning a warm start: fixed, lower learning rate (the team's adaptive schedule keeps raising it
        # while KL is small, and the cloned policy then degrades), and checkpoints often enough to pick the best
        cfg["algorithm"]["learning_rate"] = a.lr
        cfg["algorithm"]["schedule"] = "fixed"
        cfg["save_interval"] = 10
    cfg_clean = copy.deepcopy(cfg)
    runner = OnPolicyRunner(env, cfg, log_dir=a.run_dir, device="cpu")
    ck = latest_checkpoint(a.run_dir)
    if (a.resume or a.eval_only) and ck:
        print(f"[pick] loading {ck}", flush=True)
        runner.load(ck)
    if not a.eval_only and not ck and a.bc_rounds > 0:
        bc_warm_start(runner.alg.policy, rounds=a.bc_rounds, params=PickParams(hover_start=a.hover_start))
        # freeze the normalisers at the statistics the actor was cloned with: rsl_rl keeps updating them on every
        # rollout otherwise, which shifts the actor's inputs and drops success even while the actor is frozen
        for norm in (runner.alg.policy.actor_obs_normalizer, getattr(runner.alg.policy, "critic_obs_normalizer", None)):
            if hasattr(norm, "until"):
                norm.until = int(norm.count)
        with torch.no_grad():
            if hasattr(runner.alg.policy, "std"):
                runner.alg.policy.std.fill_(a.ppo_std)
            elif hasattr(runner.alg.policy, "log_std"):
                runner.alg.policy.log_std.fill_(torch.log(torch.tensor(a.ppo_std)))
        actor = ExportedActor(runner.alg.policy).eval()
        with torch.no_grad():
            print("[pick] after BC:", json.dumps(evaluate(lambda st, o: actor(o), n=512)), flush=True)
        runner.save(os.path.join(a.run_dir, "warmstart_bc.pt"))
    if not a.eval_only:
        t0 = time.time()
        pol = runner.alg.policy
        if a.bc_rounds > 0 and not ck and a.critic_warmup > 0:
            # critic warm-up: a random critic would push the cloned actor anywhere in the first updates
            for prm in list(pol.actor.parameters()) + [pol.std]:
                prm.requires_grad_(False)
            runner.learn(num_learning_iterations=a.critic_warmup, init_at_random_ep_len=True)
            for prm in list(pol.actor.parameters()) + [pol.std]:
                prm.requires_grad_(True)
            # the adaptive schedule raised the learning rate while the frozen actor's KL was ~0: put it back
            runner.alg.learning_rate = a.lr
            for g in runner.alg.optimizer.param_groups:
                g["lr"] = a.lr
        runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
        print(f"[pick] TRAINING_DONE {time.time() - t0:.0f}s", flush=True)
        # keep the best of the warm start and the saved checkpoints on the fair test (normal starts)
        best, best_s = None, -1.0
        for path in sorted(glob.glob(os.path.join(a.run_dir, "*.pt"))):
            probe = OnPolicyRunner(PickCpuEnv(8, seed=a.seed), copy.deepcopy(cfg_clean), log_dir=None, device="cpu")
            probe.load(path)
            act = ExportedActor(probe.alg.policy).eval()
            with torch.no_grad():
                sc = evaluate(lambda st, o: act(o), n=512, seed=99)["success"]
            print(f"[pick] checkpoint {os.path.basename(path)}: success {sc}", flush=True)
            if sc > best_s:
                best, best_s = path, sc
        print(f"[pick] BEST {best} ({best_s})", flush=True)
        runner = OnPolicyRunner(PickCpuEnv(8, seed=a.seed), copy.deepcopy(cfg_clean), log_dir=None, device="cpu")
        runner.load(best)

    policy = runner.alg.policy
    policy.eval()
    actor = ExportedActor(policy).eval()
    with torch.no_grad():
        res = {"policy": evaluate(lambda st, o: actor(o)),
               "scripted": evaluate(lambda st, o: scripted_action(st)),
               "policy_perfect_camera": evaluate(lambda st, o: actor(o), params=PickParams(est_bias=(0, 0), est_yaw_bias=(0, 0))),
               "scripted_perfect_camera": evaluate(lambda st, o: scripted_action(st), params=PickParams(est_bias=(0, 0), est_yaw_bias=(0, 0)))}
    print("[pick] EVAL", json.dumps(res), flush=True)
    out = os.path.join(a.run_dir, "exported")
    os.makedirs(out, exist_ok=True)
    torch.onnx.export(actor, torch.zeros(1, OBS_DIM), os.path.join(out, "policy.onnx"), input_names=["obs"],
                      output_names=["actions"], dynamic_axes={"obs": {0: "n"}, "actions": {0: "n"}}, dynamo=False)
    meta = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "control_hz": 30, "episode_steps": EPISODE_STEPS,
            "obs": "policy.envs.pick_state.PickState.observe", "actions": "policy.envs.pick_cpu_env (docstring)",
            "checkpoint": os.path.basename(best) if not a.eval_only else latest_checkpoint(a.run_dir), "eval": res}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(a.run_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()

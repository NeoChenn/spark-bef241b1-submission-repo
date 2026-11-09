"""PPO for the path-following policy, plus a pure-pursuit baseline and the held-out evaluation.

  python -m patrol.ppo --iters 300 --envs 1024 --out runs/patrol/nav_v1      # local
  modal run patrol/modal_app.py                                             # Modal CPU container
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from patrol import features
from patrol.sim2d import PatrolSim, SimCfg


class RunningNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d**2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-5, 5)


def mlp(i, o, hidden=(256, 256)):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=features.OBS_DIM, act_dim=2):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x), self.log_std.exp())

    def act_deterministic(self, obs):
        return self.actor(self.norm(obs)).clamp(-1, 1)


class Deploy(nn.Module):
    """obs (raw, unnormalised) -> action in [-1, 1]. The normaliser is baked in."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.norm, self.actor = ac.norm, ac.actor

    def forward(self, obs):
        return self.actor(self.norm(obs)).clamp(-1, 1)


def pure_pursuit(sim: PatrolSim) -> torch.Tensor:
    """Baseline: steer at the 1.0 m lookahead point, push away from close lidar returns, slow near things."""
    look = sim.lookahead_robot()[:, 1]
    curv = 2 * look[:, 1] / (look ** 2).sum(-1).clamp(min=1e-3)
    r = sim.lidar(noise=True)
    ang = sim.beam_angles
    close = (0.7 - r).clamp(min=0)
    front = (ang.abs() < math.pi / 2).float()
    repel = -(close * ang.sin() * front).sum(-1) * 1.5
    ahead = torch.where(ang.abs() < 0.5, r, torch.full_like(r, features.LIDAR_RANGE)).amin(-1)
    v = (0.8 * ((ahead - 0.35) / 1.0).clamp(0, 1)).clamp(min=0.1)
    w = (curv * v + repel).clamp(-features.W_MAX, features.W_MAX)
    return torch.stack([v / features.V_MAX, w / features.W_MAX], -1)


@torch.no_grad()
def evaluate(policy, episodes=1000, seed=1234, cfg_overrides=None) -> dict:
    """Held-out worlds (fixed seed, different from training). policy: callable(sim, obs) -> action."""
    cfg = SimCfg(num_envs=min(episodes, 1000), seed=seed, start_frac=0.0, **(cfg_overrides or {}))
    sim = PatrolSim(cfg)
    obs = sim.observe()
    done_n, goals, cols, touts, lat_sum, spd_sum, n_steps = 0, 0, 0, 0, 0.0, 0.0, 0
    counted = torch.zeros(sim.N, dtype=torch.bool)
    ep_lat = torch.zeros(sim.N)
    ep_len = torch.zeros(sim.N)
    lat_eps = []
    while done_n < episodes:
        a = policy(sim, obs)
        obs, _, done, info = sim.step(a)
        live = ~counted
        ep_lat += info["lat"] * live
        ep_len += live.float()
        spd_sum += float((info["speed"] * live).sum())
        n_steps += int(live.sum())
        for k in done.nonzero()[:, 0].tolist():
            if counted[k]:
                continue
            goals += int(info["goal"][k])
            cols += int(info["collision"][k])
            touts += int(info["timeout"][k])
            lat_eps.append(float(ep_lat[k] / ep_len[k].clamp(min=1)))
            done_n += 1
            counted[k] = True
            if done_n >= episodes:
                break
        if counted.all():
            counted[:] = False
            ep_lat.zero_(), ep_len.zero_()
    return {"episodes": done_n, "goal_rate": goals / done_n, "collision_rate": cols / done_n,
            "timeout_rate": touts / done_n, "mean_lateral_err_m": sum(lat_eps) / len(lat_eps),
            "mean_speed_mps": spd_sum / max(n_steps, 1)}


def train(iters=300, envs=1024, horizon=64, out="runs/patrol/nav_v1", lr=3e-4, epochs=4, minibatches=8,
          gamma=0.99, lam=0.95, clip=0.2, ent=0.003, seed=0, threads=None, log=print,
          sim_overrides: dict | None = None, init: str | None = None, device: str = "cpu") -> dict:
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sim = PatrolSim(SimCfg(num_envs=envs, seed=seed, device=device, **(sim_overrides or {})))
    ac = ActorCritic()
    if init:
        ac.load_state_dict(torch.load(init, map_location="cpu")["model"])
    ac.to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    obs = sim.observe()
    hist = []
    t0 = time.time()
    for it in range(iters):
        O, A, LP, V, R, D = [], [], [], [], [], []
        stats = {"goal": 0, "collision": 0, "timeout": 0}
        with torch.no_grad():
            for _ in range(horizon):
                ac.norm.update(obs)
                x = ac.norm(obs)
                dist = ac.dist(x)
                a = dist.sample()
                O.append(x), A.append(a), LP.append(dist.log_prob(a).sum(-1)), V.append(ac.critic(x)[:, 0])
                obs, r, d, info = sim.step(a)
                R.append(r), D.append(d.float())
                for k in stats:
                    stats[k] += int(info[k].sum())
            last_v = ac.critic(ac.norm(obs))[:, 0]
        O, A, LP, V, R, D = map(torch.stack, (O, A, LP, V, R, D))
        adv = torch.zeros_like(R)
        g = torch.zeros(envs, device=device)
        for t in reversed(range(horizon)):
            nv = last_v if t == horizon - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            g = delta + gamma * lam * (1 - D[t]) * g
            adv[t] = g
        ret = adv + V
        O, A, LP, adv, ret = (z.reshape(-1, *z.shape[2:]) for z in (O, A, LP, adv, ret))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = O.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for mb in perm.chunk(minibatches):
                dist = ac.dist(O[mb])
                lp = dist.log_prob(A[mb]).sum(-1)
                ratio = (lp - LP[mb]).exp()
                pg = -torch.min(ratio * adv[mb], ratio.clamp(1 - clip, 1 + clip) * adv[mb]).mean()
                vl = 0.5 * ((ac.critic(O[mb])[:, 0] - ret[mb]) ** 2).mean()
                loss = pg + vl - ent * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        ended = max(sum(stats.values()), 1)
        rec = {"iter": it, "reward": float(R.mean()), "episodes": ended,
               "goal_rate": stats["goal"] / ended, "collision_rate": stats["collision"] / ended,
               "std": ac.log_std.exp().tolist(), "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        if it % 10 == 0 or it == iters - 1:
            log(json.dumps(rec))
            with open(out / "progress.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        if (it + 1) % 50 == 0 or it == iters - 1:
            torch.save({"model": ac.state_dict(), "iter": it}, out / "model.pt")
    ac.cpu()
    torch.save({"model": ac.state_dict(), "iter": iters - 1}, out / "model.pt")
    export(ac, out)
    sensor = {"lidar_beams": (sim_overrides or {}).get("lidar_beams", SimCfg.lidar_beams)}
    ppo_eval = evaluate(lambda s, o: ac.act_deterministic(o), cfg_overrides=sensor)
    base_eval = evaluate(lambda s, o: pure_pursuit(s), cfg_overrides=sensor)
    res = {"ppo": ppo_eval, "pure_pursuit_baseline": base_eval, "iters": iters, "envs": envs,
           "env_steps": iters * horizon * envs, "train_sec": round(time.time() - t0, 1)}
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    log(json.dumps(res, indent=2))
    return res


def export(ac: ActorCritic, out: Path):
    dep = Deploy(ac).eval()
    torch.save(dep.state_dict(), out / "deploy.pt")
    dummy = torch.zeros(1, features.OBS_DIM)
    torch.onnx.export(dep, dummy, out / "policy.onnx", input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    (out / "policy_meta.json").write_text(json.dumps({
        "obs_dim": features.OBS_DIM, "sectors": features.SECTORS, "lidar_range": features.LIDAR_RANGE,
        "lookahead_m": features.LOOKAHEAD, "v_max": features.V_MAX, "v_min": features.V_MIN,
        "w_max": features.W_MAX, "control_hz": 10, "spec": "patrol/features.py docstring"}, indent=2))


def load_actor(ckpt: str | Path) -> ActorCritic:
    ac = ActorCritic()
    ac.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return ac.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--out", default="runs/patrol/nav_v1")
    a = ap.parse_args()
    train(a.iters, a.envs, out=a.out)

from __future__ import annotations

import math

import torch
from torch import nn

CHUNK = 8
ACT = 5


def chunk_targets(actions: torch.Tensor, valid: torch.Tensor, k: int = CHUNK) -> torch.Tensor:
    T = actions.shape[0]
    idx = torch.arange(T, device=actions.device)[:, None] + torch.arange(k, device=actions.device)[None, :]
    last = valid.long().sum(0) - 1
    idx = torch.minimum(idx[:, :, None], last[None, None, :].clamp_min(0))
    n = actions.shape[1]
    return actions[idx, torch.arange(n, device=actions.device)[None, None, :]].permute(0, 2, 1, 3)


def cosine_alphas(steps: int) -> torch.Tensor:
    t = torch.linspace(0, 1, steps + 1)
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].clamp(1e-5, 1.0)


class Normalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))

    def fit(self, x: torch.Tensor):
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp_min(1e-3))
        return self

    def forward(self, x):
        return (x - self.mu) / self.sd


class Block(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.Mish(), nn.Linear(w, w))

    def forward(self, x):
        return x + self.net(x)


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim: int, width: int = 512, depth: int = 4, steps: int = 50, chunk: int = CHUNK,
                 act: int = ACT):
        super().__init__()
        self.chunk, self.act, self.steps = chunk, act, steps
        self.norm = Normalizer(obs_dim)
        self.t_emb = nn.Sequential(nn.Linear(32, width), nn.Mish())
        self.inp = nn.Linear(chunk * act + obs_dim, width)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk * act))
        self.register_buffer("ab", cosine_alphas(steps))

    def _temb(self, t):
        f = torch.exp(torch.arange(16, device=t.device) * (-math.log(1000.0) / 15))
        a = t.float()[:, None] * f[None, :]
        return self.t_emb(torch.cat([a.sin(), a.cos()], -1))

    def eps(self, x, obs, t):
        h = self.inp(torch.cat([x.flatten(1), self.norm(obs)], -1)) + self._temb(t)
        return self.out(self.blocks(h)).view(-1, self.chunk, self.act)

    def loss(self, obs, a0):
        n = obs.shape[0]
        t = torch.randint(0, self.steps, (n,), device=obs.device)
        ab = self.ab[t][:, None, None]
        noise = torch.randn_like(a0)
        x = ab.sqrt() * a0 + (1 - ab).sqrt() * noise
        return ((self.eps(x, obs, t) - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, n_steps: int = 8, generator=None):
        n = obs.shape[0]
        x = torch.randn(n, self.chunk, self.act, device=obs.device, generator=generator)
        ts = torch.linspace(self.steps - 1, 0, n_steps, device=obs.device).round().long()
        for i, t in enumerate(ts):
            ab = self.ab[t]
            e = self.eps(x, obs, t.expand(n))
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1.0, 1.0)
            if i + 1 < len(ts):
                ab_n = self.ab[ts[i + 1]]
                x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * e
            else:
                x = x0
        return x


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, width: int = 512, depth: int = 3, chunk: int = CHUNK, act: int = ACT):
        super().__init__()
        self.chunk, self.act = chunk, act
        self.norm = Normalizer(obs_dim)
        self.inp = nn.Linear(obs_dim, width)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk * act))

    def forward(self, obs):
        return torch.tanh(self.out(self.blocks(self.inp(self.norm(obs))))).view(-1, self.chunk, self.act)

    def loss(self, obs, a0):
        return ((self(obs) - a0) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, n_steps: int = 0, generator=None):
        return self(obs)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

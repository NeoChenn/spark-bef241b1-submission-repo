from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn

G = 9.81
HIST = 50
BASE_OBS = 236
PROPRIO = list(range(48)) + [235]
EXT_NAMES = ["payload_mass_kg", "com_x_m", "com_y_m", "com_z_m", "deck_ax", "deck_ay", "deck_az"]
EXT_DIM = len(EXT_NAMES)
EXT_SCALE = torch.tensor([3.0, 0.05, 0.05, 0.02, 1.0, 1.0, 1.0])


class History:
    def __init__(self, n: int, length: int, dim: int, device="cpu"):
        self.data = torch.zeros(n, length, dim, device=device)
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)

    def push(self, x: torch.Tensor):
        self.data = torch.cat([self.data[:, 1:], x[:, None]], 1)
        f = self.fresh
        if f.any():
            self.data[f] = x[f][:, None].expand(-1, self.data.shape[1], -1)
            self.fresh[:] = False

    def reset(self, ids: torch.Tensor):
        self.data[ids] = 0.0
        self.fresh[ids] = True


class Adapter(nn.Module):
    def __init__(self, in_dim: int, length: int, out_dim: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, 64), nn.ELU())
        k1, s1 = (8, 4) if length >= 32 else (3, 1)
        self.conv = nn.Sequential(nn.Conv1d(64, 32, k1, s1), nn.ELU(), nn.Conv1d(32, 32, 5, 1), nn.ELU(),
                                  nn.Conv1d(32, 32, 5, 1), nn.ELU(), nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 64, length)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, 64), nn.ELU(), nn.Linear(64, out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(self.embed(h).transpose(1, 2)))


@dataclass
class DeckMotion:
    tilt0: torch.Tensor
    amp: torch.Tensor
    w: torch.Tensor
    phase: torch.Tensor
    max_tilt: float = 0.1
    max_heave: float = 1.5

    @classmethod
    def sample(cls, n: int, gen: torch.Generator, max_tilt: float = 0.1, max_heave: float = 1.5, device="cpu"):
        u = lambda *s: torch.rand(*s, generator=gen)
        tilt0 = (u(n, 2) * 2 - 1) * 0.3 * max_tilt
        amp = torch.cat([u(n, 2) * 0.7 * max_tilt, u(n, 1) * max_heave], 1)
        w = 2 * torch.pi / (6.0 + 6.0 * u(n, 3))
        phase = u(n, 3) * 2 * torch.pi
        return cls(*(x.to(device) for x in (tilt0, amp, w, phase)), max_tilt, max_heave)

    def accel(self, t: torch.Tensor) -> torch.Tensor:
        s = torch.sin(self.w * t[:, None] + self.phase)
        roll = self.tilt0[:, 0] + self.amp[:, 0] * s[:, 0]
        pitch = self.tilt0[:, 1] + self.amp[:, 1] * s[:, 1]
        return torch.stack([-G * torch.sin(pitch), G * torch.sin(roll) * torch.cos(pitch), -self.amp[:, 2] * s[:, 2]], 1)

    def assign(self, ids: torch.Tensor, other: "DeckMotion"):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v[ids] = getattr(other, f.name).to(v.device)


def sample_payload(n: int, gen: torch.Generator, mass=(-1.0, 5.0), com=(0.05, 0.05, 0.02)):
    m = mass[0] + (mass[1] - mass[0]) * torch.rand(n, generator=gen)
    c = (torch.rand(n, 3, generator=gen) * 2 - 1) * torch.tensor(com)
    return m, c


def expand_first_layer(sd: dict, extra: int, prefixes=("actor.", "critic.")) -> dict:
    out = dict(sd)
    for p in prefixes:
        k = p + "0.weight"
        if k in sd:
            w = sd[k]
            out[k] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype, device=w.device)], 1)
    return out


def mlp_from_state(sd: dict, prefix: str = "actor.") -> nn.Sequential:
    idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)})
    layers = []
    for j, i in enumerate(idx):
        w = sd[f"{prefix}{i}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(sd[f"{prefix}{i}.bias"])
        layers.append(lin)
        if j < len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return torch.where(ss_tot > 1e-12, 1 - ss_res / ss_tot.clamp_min(1e-12), torch.ones_like(ss_tot))


class RMARuntime:
    def __init__(self, path: str, n: int = 1):
        import numpy as np
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.history = np.zeros((n, HIST, len(PROPRIO)), dtype=np.float32)
        self.fresh = np.ones(n, dtype=bool)

    def reset(self, ids):
        self.history[ids] = 0.0
        self.fresh[ids] = True

    def act(self, obs):
        import numpy as np

        obs = np.asarray(obs, dtype=np.float32)
        p = obs[:, PROPRIO]
        self.history = np.concatenate([self.history[:, 1:], p[:, None]], 1)
        self.history[self.fresh] = p[self.fresh][:, None]
        self.fresh[:] = False
        return self.sess.run(None, {"obs": obs[:, :BASE_OBS], "history": self.history})[0]

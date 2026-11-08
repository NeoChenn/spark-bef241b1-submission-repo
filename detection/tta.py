from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn

NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def norm_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, NORMS) and m.running_mean is not None]


def binary_entropy(logit: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    return -(p * torch.log(p.clamp_min(1e-6)) + (1 - p) * torch.log((1 - p).clamp_min(1e-6)))


class TTA(nn.Module):
    def __init__(self, model: nn.Module, mode: str = "bn", prior: float = 0.0, lr: float = 1e-3, steps: int = 1,
                 temperature: float = 1.0, gate=None):
        super().__init__()
        self.model, self.mode, self.prior, self.steps, self.temperature = model.eval(), mode, prior, steps, temperature
        self.norms = norm_layers(model)
        self.gate, self.last_shift = gate, None
        self.state0 = copy.deepcopy(model.state_dict())
        self.params = [p for m in self.norms for p in (m.weight, m.bias) if p is not None]
        self.opt = torch.optim.Adam(self.params, lr=lr) if mode == "tent" and self.params else None
        self.opt0 = copy.deepcopy(self.opt.state_dict()) if self.opt else None

    def reset(self) -> None:
        self.model.load_state_dict(self.state0)
        if self.opt:
            self.opt.load_state_dict(self.opt0)

    @contextmanager
    def _batch_stats(self):
        src = [(m.running_mean, m.running_var) for m in self.norms]
        hooks = []
        for m, (mu0, var0) in zip(self.norms, src):
            def pre(mod, inp, mu0=mu0, var0=var0):
                z = inp[0].detach()
                dims = [0] + list(range(2, z.dim()))
                mod.running_mean = self.prior * mu0 + (1 - self.prior) * z.mean(dims)
                mod.running_var = self.prior * var0 + (1 - self.prior) * z.var(dims, unbiased=False)
            hooks.append(m.register_forward_pre_hook(pre))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()
            for m, (mu0, var0) in zip(self.norms, src):
                m.running_mean, m.running_var = mu0, var0

    def shift(self, x: torch.Tensor) -> float:
        m = self.norms[0]
        got = {}

        def pre(mod, inp):
            z = inp[0].detach()
            dims = [0] + list(range(2, z.dim()))
            got["mu"], got["var"] = z.mean(dims), z.var(dims, unbiased=False)
            raise StopIteration
        h = m.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                self.model(x)
        except StopIteration:
            pass
        finally:
            h.remove()
        mu0, v0 = m.running_mean, m.running_var + m.eps
        mu, v = got["mu"], got["var"] + m.eps
        d2 = (mu - mu0) ** 2
        return float((0.5 * ((v + d2) / v0 + (v0 + d2) / v) - 1).mean())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.last_shift = self.shift(x) if self.gate is not None and self.norms else None
        if self.last_shift is not None and self.last_shift < self.gate:
            with torch.no_grad():
                return self.model(x)
        if self.mode == "off" or not self.norms:
            with torch.no_grad():
                return self.model(x)
        with self._batch_stats():
            for _ in range(self.steps if self.opt else 0):
                loss = binary_entropy(self.model(x) / self.temperature).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            with torch.no_grad():
                return self.model(x)

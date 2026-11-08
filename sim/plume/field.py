from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PPM = 1e6


@dataclass
class Field:
    fraction: np.ndarray
    origin: np.ndarray
    dx: float
    lel: float = 0.05
    t: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.fraction = np.asarray(self.fraction, np.float32)
        self.origin = np.asarray(self.origin, np.float64)

    @property
    def shape(self) -> tuple:
        return self.fraction.shape

    @property
    def ppm(self) -> np.ndarray:
        return self.fraction * PPM

    def centers(self, axis: int) -> np.ndarray:
        return self.origin[axis] + (np.arange(self.shape[axis]) + 0.5) * self.dx

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.origin.copy(), self.origin + np.asarray(self.shape) * self.dx

    def sample(self, points) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, np.float64))
        g = (p - self.origin) / self.dx - 0.5
        n = np.asarray(self.shape)
        inside = np.all((g >= -0.5) & (g <= n - 0.5), axis=1)
        g = np.clip(g, 0.0, n - 1.0)
        i0 = np.minimum(np.floor(g).astype(int), n - 2).clip(0)
        f = g - i0
        out = np.zeros(len(p))
        c = self.fraction
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    ii = np.minimum(i0 + (dx, dy, dz), n - 1)
                    out += w * c[ii[:, 0], ii[:, 1], ii[:, 2]]
        return np.where(inside, out, 0.0) * PPM

    def sample_ppm(self, x: float, y: float, z: float) -> float:
        return float(self.sample([(x, y, z)])[0])

    def lel_percent(self, points) -> np.ndarray:
        return self.sample(points) / PPM / self.lel * 100.0

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ppm=self.ppm.astype(np.float32), origin=self.origin, dx=self.dx, lel=self.lel,
                            t=self.t, meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path) -> "Field":
        d = np.load(path)
        return cls(d["ppm"] / PPM, d["origin"], float(d["dx"]), float(d["lel"]), float(d["t"]), json.loads(str(d["meta"])))

    def to_json(self, stride: int = 1) -> dict:
        s = slice(None, None, stride)
        return {"origin": self.origin.tolist(), "dx": self.dx * stride, "shape": list(self.fraction[s, s, s].shape),
                "lel": self.lel, "t": self.t, "ppm": np.round(self.ppm[s, s, s], 1).tolist(), "meta": self.meta}


def sample_ppm(snapshot: Field, x: float, y: float, z: float) -> float:
    return snapshot.sample_ppm(x, y, z)

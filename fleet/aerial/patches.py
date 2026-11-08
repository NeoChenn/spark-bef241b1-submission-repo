from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[2]
TEXTURES = REPO / "sim/assets/third_party/textures"
TESTDATA = REPO / "fleet/testdata/aerial"

RI_PCT = {0: 0.0, 1: 0.05, 2: 0.5, 3: 1.0, 4: 8.0, 5: 40.0}
RI_EDGES = [0.02, math.sqrt(0.05 * 0.5), math.sqrt(0.5 * 1.0), math.sqrt(1.0 * 8.0), math.sqrt(8.0 * 40.0)]
PAINTS = {
    "grey": (203, 208, 204),
    "white": (236, 233, 222),
    "green": (60, 110, 75),
    "yellow": (240, 170, 20),
}
BASES = ("blue_metal_plate", "grey", "white", "green", "yellow")
FILLS = ("rust_coarse_01", "Rust004")


def ri_from_pct(pct: float) -> int:
    return int(sum(pct >= e for e in RI_EDGES))


def _texture(name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    im = Image.open(TEXTURES / name / "diff.jpg").convert("RGB")
    side = int(min(im.size) * rng.uniform(0.45, 0.8))
    x0 = int(rng.integers(0, im.size[0] - side + 1))
    y0 = int(rng.integers(0, im.size[1] - side + 1))
    crop = im.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.BILINEAR)
    return np.asarray(crop.rotate(90 * int(rng.integers(0, 4))), np.float32)


def _base(base: str, size: int, rng: np.random.Generator) -> np.ndarray:
    plate = _texture("blue_metal_plate", size, rng)
    if base == "blue_metal_plate":
        return plate
    lum = plate.mean(axis=2, keepdims=True)
    shade = 1.0 + 0.35 * (lum / lum.mean() - 1.0)
    return np.clip(np.array(PAINTS[base], np.float32) * shade, 0, 255)


def rust_mask(pct: float, size: int, rng: np.random.Generator) -> np.ndarray:
    k = int(round(pct / 100.0 * size * size))
    mask = np.zeros((size, size), bool)
    if k <= 0:
        return mask
    small = max(size // 4, 8)
    src = _texture("rusty_metal_02", small, rng) / 255.0
    r, g, b = src[..., 0], src[..., 1], src[..., 2]
    score = (r - b) + 0.3 * (r - g) - 0.15 * (r + g + b) / 3.0
    score = np.asarray(Image.fromarray(score.astype(np.float32), "F").resize((size, size), Image.BILINEAR))
    score = score + rng.uniform(0, 1e-6, score.shape)
    idx = np.argpartition(score.ravel(), -k)[-k:]
    mask.ravel()[idx] = True
    return mask


def camera(img: np.ndarray, rng: np.random.Generator, quality: int = 88) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    a = rng.uniform(0, 2 * math.pi)
    light = 0.92 + 0.14 * (np.cos(a) * xx + np.sin(a) * yy)
    out = np.clip(img * light[..., None] + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
    im = Image.fromarray(out).filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, subsampling=0)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


@dataclass
class PatchSpec:
    id: str
    rust_pct: float
    base: str = "grey"
    fill: str = "rust_coarse_01"
    seed: int = 0
    size: int = 384
    repaired: bool = False

    @property
    def ri(self) -> int:
        return 0 if self.repaired else ri_from_pct(self.rust_pct)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        base = _base(self.base, self.size, rng)
        mask = rust_mask(self.rust_pct, self.size, rng)
        fill = _texture(self.fill, self.size, rng)
        if self.repaired:
            img = base.copy()
            if mask.any():
                ys, xs = np.nonzero(mask)
                y0, y1 = max(ys.min() - 15, 0), min(ys.max() + 16, self.size)
                x0, x1 = max(xs.min() - 15, 0), min(xs.max() + 16, self.size)
                colour = np.array(PAINTS.get(self.base, (60, 80, 120)), np.float32) * 1.04
                img[y0:y1, x0:x1] = np.clip(colour + rng.normal(0, 3, (y1 - y0, x1 - x0, 3)), 0, 255)
            mask = np.zeros_like(mask)
        else:
            img = np.where(mask[..., None], fill, base)
        return camera(img, rng), mask


def rust_pixels(img: np.ndarray) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(np.asarray(img, np.uint8)).convert("HSV"), np.float32)
    h, s, v = hsv[..., 0] * (360.0 / 255.0), hsv[..., 1], hsv[..., 2]
    return (h >= 4.0) & (h <= 36.0) & (s >= 95.0) & (v >= 25.0) & (v <= 190.0)


@dataclass
class ClassicalGrade:
    rust_pct: float
    ri: int

    def summary(self) -> dict:
        return {"rust_pct": round(self.rust_pct, 3), "ri": self.ri}


def grade_classical(img: np.ndarray) -> tuple[ClassicalGrade, np.ndarray]:
    mask = rust_pixels(img)
    m = mask.astype(np.int8)
    nb = np.zeros_like(m)
    nb[1:] += m[:-1]
    nb[:-1] += m[1:]
    nb[:, 1:] += m[:, :-1]
    nb[:, :-1] += m[:, 1:]
    mask = mask & (nb >= 2)
    pct = 100.0 * float(mask.mean())
    return ClassicalGrade(pct, ri_from_pct(pct)), mask


def test_set() -> list[PatchSpec]:
    specs = []
    n = 0
    for ri, pct in RI_PCT.items():
        for j in range(3):
            base = BASES[(ri + 2 * j) % len(BASES)]
            fill = FILLS[(ri + j) % len(FILLS)]
            specs.append(PatchSpec(f"ri{ri}_{j}", pct, base, fill, seed=1000 + n))
            n += 1
    return specs


def write_test_set(out: Path = TESTDATA) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in test_set():
        img, mask = spec.render()
        Image.fromarray(img).save(out / f"{spec.id}.jpg", quality=95, subsampling=0)
        rows.append({"id": spec.id, "file": f"{spec.id}.jpg", "true_rust_pct": round(100.0 * float(mask.mean()), 4),
                     "true_ri": spec.ri, "base": spec.base, "fill": spec.fill, "seed": spec.seed})
    return rows


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def accuracy(pairs: list[tuple[int, Optional[int]]]) -> dict:
    graded = [(t, p) for t, p in pairs if p is not None]
    if not graded:
        return {"n": 0, "exact": None, "within_1": None}
    exact = sum(t == p for t, p in graded) / len(graded)
    within = sum(abs(t - p) <= 1 for t, p in graded) / len(graded)
    return {"n": len(graded), "exact": round(exact, 3), "within_1": round(within, 3)}

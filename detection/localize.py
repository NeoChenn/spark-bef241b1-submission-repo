from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PatchClassifier = Callable[[np.ndarray], np.ndarray]

UNSURE_LO, UNSURE_HI = 0.35, 0.65


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


@dataclass
class RustPatch:
    id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    mean_conf: float
    cells: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class LocalizeResult:
    heatmap: np.ndarray
    prob_map: np.ndarray
    mask: np.ndarray
    patches: list[RustPatch]
    unsure: bool
    unsure_cells: list[tuple[int, int]]
    max_conf: float
    threshold: float

    @property
    def rust_pixel_fraction(self) -> float:
        return float(self.mask.mean())

    def summary(self) -> dict:
        return {"n_patches": len(self.patches), "max_conf": round(self.max_conf, 3),
                "unsure": self.unsure, "n_unsure_cells": len(self.unsure_cells),
                "rust_px_frac": round(self.rust_pixel_fraction, 4)}


def colour_classifier(hue_lo: float = 0.07):
    def classify(patches: np.ndarray) -> np.ndarray:
        p = patches.astype(np.float32) / 255.0
        r, g, b = p[..., 0], p[..., 1], p[..., 2]
        hue = (r - g) / (r + 1e-3)
        warm = (r > b) & (r > 0.12)
        score = np.clip((hue - hue_lo) / 0.2, 0.0, 1.0) * warm
        frac = score.reshape(len(p), -1).mean(axis=1)
        return (frac - 0.15) * 30.0
    return classify


fake_colour_classifier = colour_classifier(0.07)


class OnnxPatchClassifier:

    def __init__(self, model_path: str, meta: dict, batch_size: int = 64):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        size = meta.get("input_size", meta.get("img_size", meta.get("image_size", 224)))
        self.size = int(size[-1] if isinstance(size, (list, tuple)) else size)
        self.mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32).reshape(1, 3, 1, 1)
        self.batch_size = batch_size

    def _prep(self, patches: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(patches.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return ((x.numpy() - self.mean) / self.std).astype(np.float32)

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(patches), self.batch_size):
            x = self._prep(patches[s:s + self.batch_size])
            y = self.sess.run(None, {self.input_name: x})[0]
            out.append(np.asarray(y, np.float32).reshape(len(x), -1)[:, -1])
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class RustLocalizer:
    def __init__(self, classifier: PatchClassifier, patch: int = 32, stride: int = 16,
                 threshold: float = 0.5, temperature: float = 1.0, batch_size: int = 256,
                 min_cells: int = 1, unsure_band: tuple[float, float] = (UNSURE_LO, UNSURE_HI),
                 unsure_ignore_edges: bool = True, edge_conf: float = 0.9):
        self.classifier = classifier
        self.patch, self.stride = int(patch), int(stride)
        self.threshold, self.temperature = float(threshold), float(temperature)
        self.batch_size = batch_size
        self.min_cells = min_cells
        self.unsure_band = unsure_band
        self.unsure_ignore_edges = unsure_ignore_edges
        self.edge_conf = edge_conf

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str = "/ckpt/detector/best", **kw) -> "RustLocalizer":
        d = Path(ckpt_dir)
        meta = json.loads((d / "meta.json").read_text())
        clf = OnnxPatchClassifier(d / "model.onnx", meta)
        kw.setdefault("threshold", float(meta.get("threshold", 0.5)))
        kw.setdefault("temperature", float(meta.get("temperature", 1.0)))
        if "unsure_band" in meta:
            kw.setdefault("unsure_band", tuple(meta["unsure_band"]))
        return cls(clf, **kw)

    def _grid(self, H, W):
        p, s = self.patch, self.stride
        ys = list(range(0, max(H - p, 0) + 1, s))
        xs = list(range(0, max(W - p, 0) + 1, s))
        if ys[-1] + p < H:
            ys.append(H - p)
        if xs[-1] + p < W:
            xs.append(W - p)
        return ys, xs

    def localize(self, image: np.ndarray) -> LocalizeResult:
        img = np.ascontiguousarray(image[..., :3])
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        H, W = img.shape[:2]
        p = self.patch
        ys, xs = self._grid(H, W)
        win = np.lib.stride_tricks.sliding_window_view(img, (p, p, 3))[:, :, 0]
        patches = win[np.ix_(ys, xs)].reshape(-1, p, p, 3)
        logits = np.concatenate([np.asarray(self.classifier(patches[s:s + self.batch_size]), np.float32).reshape(-1)
                                 for s in range(0, len(patches), self.batch_size)])
        heat = _sigmoid(logits / self.temperature).reshape(len(ys), len(xs))

        acc = np.zeros((H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        for gi, y in enumerate(ys):
            for gj, x in enumerate(xs):
                acc[y:y + p, x:x + p] += heat[gi, gj]
                cnt[y:y + p, x:x + p] += 1
        prob_map = acc / np.maximum(cnt, 1)
        mask = prob_map >= self.threshold

        hot = heat >= self.threshold
        lo, hi = self.unsure_band
        mid = (heat >= lo) & (heat <= hi)
        if self.unsure_ignore_edges:
            conf = heat > self.edge_conf
            near = np.zeros_like(conf)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    near |= np.roll(np.roll(conf, di, 0), dj, 1)
            mid &= ~near
        unsure_cells = [tuple(map(int, c)) for c in np.argwhere(mid)]

        patches_out = []
        for comp in _components(hot):
            if len(comp) < self.min_cells:
                continue
            cm = np.zeros((H, W), bool)
            for (gi, gj) in comp:
                cm[ys[gi]:ys[gi] + p, xs[gj]:xs[gj] + p] = True
            cm &= mask
            if not cm.any():
                continue
            yy, xx = np.nonzero(cm)
            w = prob_map[yy, xx]
            patches_out.append(RustPatch(
                id=len(patches_out),
                bbox=(int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1),
                centroid=(float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())),
                area=int(cm.sum()),
                mean_conf=float(np.mean([heat[c] for c in comp])),
                cells=comp))
        return LocalizeResult(heatmap=heat, prob_map=prob_map, mask=mask, patches=patches_out,
                              unsure=bool(unsure_cells), unsure_cells=unsure_cells,
                              max_conf=float(heat.max()) if heat.size else 0.0,
                              threshold=self.threshold)


def _components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(binary, bool)
    comps = []
    H, W = binary.shape
    for i, j in np.argwhere(binary):
        if seen[i, j]:
            continue
        q, comp = deque([(i, j)]), []
        seen[i, j] = True
        while q:
            a, b = q.popleft()
            comp.append((int(a), int(b)))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                na, nb = a + da, b + db
                if 0 <= na < H and 0 <= nb < W and binary[na, nb] and not seen[na, nb]:
                    seen[na, nb] = True
                    q.append((na, nb))
        comps.append(comp)
    return comps


def pixel_rays(camera, stride: int = 1):
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    u = np.arange(0, camera.width, stride) + 0.5
    v = np.arange(0, camera.height, stride) + 0.5
    uu, vv = np.meshgrid(u, v)
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    d = d_cam @ T[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return T[:3, 3], d


def ray_cylinder(origin: np.ndarray, dirs: np.ndarray, pipe):
    a = pipe.axis
    w = origin - pipe.origin
    w_perp = w - w.dot(a) * a
    d_par = dirs @ a
    d_perp = dirs - d_par[..., None] * a
    A = (d_perp ** 2).sum(-1)
    B = 2 * (d_perp @ w_perp)
    C = w_perp.dot(w_perp) - pipe.radius ** 2
    disc = B ** 2 - 4 * A * C
    ok = (disc >= 0) & (A > 1e-12)
    t = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(A, 1e-12)), np.inf)
    ok &= t > 0
    pts = origin + np.where(ok, t, 0)[..., None] * dirs
    rel = pts - pipe.origin
    z = rel @ a
    radial = rel - z[..., None] * a
    b = np.cross(a, pipe.ref)
    theta = np.mod(np.arctan2(radial @ b, radial @ pipe.ref), 2 * np.pi)
    ok &= np.abs(z) <= pipe.length / 2
    return ok, theta, z, pts


def project_mask_to_grid(mask: np.ndarray, camera, pipe, prob_map: Optional[np.ndarray] = None,
                         min_frac: float = 0.5):
    o, d = pixel_rays(camera)
    hit, theta, z, _ = ray_cylinder(o, d, pipe)
    nT, nZ = pipe.n_theta, pipe.n_z
    j = (np.floor(theta / (2 * np.pi) * nT).astype(int)) % nT
    i = np.clip(np.floor((z + pipe.length / 2) / pipe.length * nZ).astype(int), 0, nZ - 1)
    flat = (i * nT + j)[hit]
    total = np.bincount(flat, minlength=nT * nZ)
    rust = np.bincount(flat, weights=mask[hit].astype(np.float64), minlength=nT * nZ)
    pm = prob_map if prob_map is not None else mask.astype(np.float64)
    psum = np.bincount(flat, weights=pm[hit].astype(np.float64), minlength=nT * nZ)
    visible = (total > 0).reshape(nZ, nT)
    frac = (rust / np.maximum(total, 1)).reshape(nZ, nT)
    conf = (psum / np.maximum(total, 1)).reshape(nZ, nT)
    return (frac >= min_frac) & visible, visible, conf


def pixel_to_theta_z(px: tuple[float, float], camera, pipe):
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    d = np.array([(px[0] - K[0, 2]) / K[0, 0], (px[1] - K[1, 2]) / K[1, 1], 1.0]) @ T[:3, :3].T
    d /= np.linalg.norm(d)
    hit, th, z, _ = ray_cylinder(T[:3, 3], d[None], pipe)
    return (float(th[0]), float(z[0])) if hit[0] else None

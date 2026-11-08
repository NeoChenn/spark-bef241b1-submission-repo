from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from detection.localize import UNSURE_HI, UNSURE_LO, OnnxPatchClassifier, PatchClassifier, _sigmoid, pixel_rays


@dataclass
class BoltReading:
    bolt: int
    angle: Optional[float]
    conf: float
    center_px: tuple[float, float]
    red_px: int


@dataclass
class LeakReading:
    bolts: list[BoltReading]
    crack_prob: float
    crack_unsure: bool
    crack_windows: int
    n_windows: int
    threshold: float = 0.5
    heat: list = field(default_factory=list)
    view_quality: float = 1.0

    def summary(self) -> dict:
        seen = [b for b in self.bolts if b.angle is not None]
        return {"bolts_read": len(seen), "max_abs_angle": round(max((abs(b.angle) for b in seen), default=0.0), 3),
                "crack_prob": round(self.crack_prob, 3), "crack_unsure": self.crack_unsure,
                "crack_windows": self.crack_windows, "view_quality": round(self.view_quality, 2)}


def _project(camera, pts):
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    pc = (np.asarray(pts, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = np.where(np.abs(pc[..., 2]) < 1e-9, 1e-9, pc[..., 2])
    return np.stack([K[0, 0] * pc[..., 0] / z + K[0, 2], K[1, 1] * pc[..., 1] / z + K[1, 2]], -1), pc[..., 2]


def _wrap_half_pi(a: float) -> float:
    return (a + math.pi / 2) % math.pi - math.pi / 2


def fake_crack_classifier(patches: np.ndarray) -> np.ndarray:
    g = patches.astype(np.float32).mean(axis=-1)
    med = np.median(g.reshape(len(g), -1), axis=1).reshape(-1, 1, 1)
    dark = g < med - 45.0
    frac = dark.reshape(len(g), -1).mean(axis=1)
    return (frac - 0.03) * 45.0


class LeakLocalizer:
    def __init__(self, crack_classifier: Optional[PatchClassifier] = None, patch: int = 32, stride: int = 16,
                 threshold: float = 0.5, temperature: float = 1.0,
                 unsure_band: tuple[float, float] = (UNSURE_LO, UNSURE_HI), min_coverage: float = 0.9,
                 line_contrast: float = 30.0, max_width_mm: float = 6.0, crack_len_mm: float = 12.0,
                 crack_len_scale_mm: float = 3.0, clear_view: float = 0.85):
        self.classifier = crack_classifier
        self.patch, self.stride = int(patch), int(stride)
        self.threshold, self.temperature = float(threshold), float(temperature)
        self.unsure_band = unsure_band
        self.min_coverage = min_coverage
        self.line_contrast = line_contrast
        self.max_width_mm = max_width_mm
        self.crack_len_mm = crack_len_mm
        self.crack_len_scale_mm = crack_len_scale_mm
        self.clear_view = clear_view

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str = "/ckpt/leak_detector/best", **kw) -> "LeakLocalizer":
        d = Path(ckpt_dir)
        meta = json.loads((d / "meta.json").read_text())
        kw.setdefault("threshold", float(meta.get("threshold", 0.5)))
        kw.setdefault("temperature", float(meta.get("temperature", 1.0)))
        if "unsure_band" in meta:
            kw.setdefault("unsure_band", tuple(meta["unsure_band"]))
        return cls(OnnxPatchClassifier(d / "model.onnx", meta), **kw)

    def read(self, image: np.ndarray, camera, flange) -> LeakReading:
        img = np.ascontiguousarray(np.asarray(image)[..., :3]).astype(np.uint8)
        bolts = [self._bolt(img, camera, flange, k) for k in range(flange.n_bolts)]
        crack = self._crack if self.classifier is not None else self._crack_lines
        prob, unsure, hot, n, heat = crack(img, camera, flange)
        seen = [b.conf for b in bolts if b.angle is not None]
        quality = float(np.mean(seen)) if seen else 0.0
        if prob < self.threshold and quality < self.clear_view:
            unsure = True
        return LeakReading(bolts, prob, unsure, hot, n, self.threshold, heat, quality)

    def _bolt(self, img, camera, flange, k) -> BoltReading:
        H, W = img.shape[:2]
        c = flange.bolt_center(k)
        (p0, p1), z = _project(camera, np.stack([c, c + flange.head_radius * flange.radial(k)]))
        if (z <= 0.05).any():
            return BoltReading(k, None, 0.0, (float("nan"), float("nan")), 0)
        d_exp = p1 - p0
        r_px = float(np.linalg.norm(d_exp))
        cx, cy = float(p0[0]), float(p0[1])
        if r_px < 2.0 or not (r_px <= cx < W - r_px and r_px <= cy < H - r_px):
            return BoltReading(k, None, 0.0, (cx, cy), 0)
        x0, x1 = int(math.floor(cx - r_px)), int(math.ceil(cx + r_px)) + 1
        y0, y1 = int(math.floor(cy - r_px)), int(math.ceil(cy + r_px)) + 1
        crop = img[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
        inside = dx * dx + dy * dy <= (0.95 * r_px) ** 2
        R, G, B = crop[..., 0], crop[..., 1], crop[..., 2]
        red = inside & (R > 110) & (R > 1.5 * G) & (R > 1.5 * B)
        n_red = int(red.sum())
        if n_red < 6:
            return BoltReading(k, None, 0.0, (cx, cy), n_red)
        pts = np.stack([dx[red], dy[red]], 1)
        pts = pts - pts.mean(0)
        evals, evecs = np.linalg.eigh(np.cov(pts.T) + 1e-9 * np.eye(2))
        e1 = evecs[:, 1]
        angle = _wrap_half_pi(math.atan2(e1[1], e1[0]) - math.atan2(d_exp[1], d_exp[0]))
        elong = math.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9))
        count_score = min(1.0, n_red / (0.6 * 0.43 * r_px * r_px))
        shape_score = min(1.0, max(0.0, (elong - 2.5) / 3.5))
        size_score = min(1.0, r_px / 8.0)
        sat = float((R[red] - 0.5 * (G[red] + B[red])).mean())
        sat_score = min(1.0, max(0.0, (sat - 40.0) / 60.0))
        conf = count_score * shape_score * size_score * sat_score
        return BoltReading(k, angle, float(conf), (cx, cy), n_red)

    def face_mask(self, camera, flange, shape, bolt_clear: float = 2.3) -> np.ndarray:
        H, W = shape
        o, d = pixel_rays(camera)
        n = np.asarray(flange.normal, np.float64)
        c = np.asarray(flange.center, np.float64)
        denom = d @ n
        t = ((c - o) @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        P = o + t[..., None] * d
        rel = P - c
        r = np.linalg.norm(rel, axis=-1)
        m = (t > 0) & (r >= 1.2 * flange.pipe_radius) & (r <= 0.95 * flange.outer_radius)
        on_face = flange.center + 0.0 * n
        for k in range(flange.n_bolts):
            b = on_face + flange.bolt_circle * flange.radial(k)
            m &= np.linalg.norm(P - b, axis=-1) >= bolt_clear * flange.head_radius
        return m[:H, :W]

    def _crack(self, img, camera, flange):
        H, W = img.shape[:2]
        m = self.face_mask(camera, flange, (H, W))
        p, s = self.patch, self.stride
        ys = list(range(0, max(H - p, 0) + 1, s))
        xs = list(range(0, max(W - p, 0) + 1, s))
        keep = [(y, x) for y in ys for x in xs if m[y:y + p, x:x + p].mean() >= self.min_coverage]
        if not keep:
            return 0.0, False, 0, 0, []
        patches = np.stack([img[y:y + p, x:x + p] for y, x in keep])
        logits = np.asarray(self.classifier(patches), np.float32).reshape(-1)
        heat = _sigmoid(logits / self.temperature)
        lo, hi = self.unsure_band
        hot = int((heat >= self.threshold).sum())
        unsure = hot == 0 and bool(((heat >= lo) & (heat <= hi)).any())
        cells = [(x + p / 2, y + p / 2, float(h)) for (y, x), h in zip(keep, heat)]
        return float(heat.max()), unsure, hot, len(keep), cells

    def _crack_lines(self, img, camera, flange):
        from scipy import ndimage

        H, W = img.shape[:2]
        face = self.face_mask(camera, flange, (H, W), bolt_clear=1.4)
        R, G, B = (img[..., i].astype(np.float32) for i in range(3))
        red = (R > 110) & (R > 1.5 * G) & (R > 1.5 * B)
        face &= ~ndimage.binary_dilation(red, iterations=3)
        face = ndimage.binary_erosion(face, iterations=2)
        if not face.any():
            return 0.0, False, 0, 0, []
        gray = img.astype(np.float32).mean(axis=-1)
        dark = ((ndimage.median_filter(gray, size=11) - gray) > self.line_contrast) & face
        lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
        (c0, c1), _ = _project(camera, np.stack([flange.center, flange.center + 0.01 * flange.ref]))
        px_per_mm = float(np.linalg.norm(c1 - c0)) / 10.0
        best, hits, cells = 0.0, 0, []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            ys, xs = np.nonzero(lab[sl] == i)
            if len(ys) < 6:
                continue
            pts = np.stack([xs, ys], 1).astype(np.float64)
            evals = np.linalg.eigvalsh(np.cov(pts.T) + 1e-9 * np.eye(2))
            length_px = math.sqrt(12.0 * max(evals[1], 0.0))
            width_px = len(ys) / max(length_px, 1.0)
            if width_px / px_per_mm > self.max_width_mm:
                continue
            length_mm = length_px / px_per_mm
            p = float(_sigmoid((length_mm - self.crack_len_mm) / self.crack_len_scale_mm))
            cells.append((float(sl[1].start + xs.mean()), float(sl[0].start + ys.mean()), p))
            hits += int(p >= self.threshold)
            best = max(best, p)
        lo, hi = self.unsure_band
        unsure = hits == 0 and lo <= best <= hi
        return best, unsure, hits, len(cells), cells

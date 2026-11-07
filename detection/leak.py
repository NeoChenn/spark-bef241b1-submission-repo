"""Leak perception from the wrist camera: torque-stripe reader + crack patch classifier.

Same process as rust (detection/localize.py):
  * CRACK: a sliding window over the flange face, every patch scored by a pluggable binary classifier
    (an ONNX model from /ckpt/leak_detector/best in production, a hand-written fake in tests and the mock),
    then temperature-scaled, thresholded, with an "unsure" probability band that triggers another view.
  * BOLTS: each bolt head is located by projecting the known flange geometry into the image (like
    project_mask_to_grid uses the known pipe). The red torque stripe on the head is segmented and its
    direction compared with where it should point. The angle between them is how far the bolt has turned.

Geometry objects are duck-typed (center, normal, ref, binormal, bolt_circle, head_radius, pipe_radius,
outer_radius, n_bolts, radial(k, extra), bolt_center(k)), see policy/leak/geometry.py, so this module has no
dependency on the policy package beyond that shape.

HONEST STATUS: the stripe reader is classical image processing, not a learned model. The fake crack
classifier only works on the mock renderer's images. A trained crack model is the planned replacement.
"""
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
    angle: Optional[float]          # signed, image frame, radians; None if not readable
    conf: float                     # 0..1
    center_px: tuple[float, float]
    red_px: int


@dataclass
class LeakReading:
    bolts: list[BoltReading]
    crack_prob: float               # max calibrated crack probability over face windows
    crack_unsure: bool              # some window in the unsure band and none above threshold
    crack_windows: int              # windows at or above threshold
    n_windows: int
    threshold: float = 0.5
    heat: list = field(default_factory=list)   # (x, y, prob) per window, for the HUD/debug

    def summary(self) -> dict:
        seen = [b for b in self.bolts if b.angle is not None]
        return {"bolts_read": len(seen), "max_abs_angle": round(max((abs(b.angle) for b in seen), default=0.0), 3),
                "crack_prob": round(self.crack_prob, 3), "crack_unsure": self.crack_unsure,
                "crack_windows": self.crack_windows}


def _project(camera, pts):
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    pc = (np.asarray(pts, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = np.where(np.abs(pc[..., 2]) < 1e-9, 1e-9, pc[..., 2])
    return np.stack([K[0, 0] * pc[..., 0] / z + K[0, 2], K[1, 1] * pc[..., 1] / z + K[1, 2]], -1), pc[..., 2]


def _wrap_half_pi(a: float) -> float:
    """Axis difference: a line has no direction, so wrap to (-pi/2, pi/2]."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


def fake_crack_classifier(patches: np.ndarray) -> np.ndarray:
    """Thin dark-line pixel fraction -> logit. Only meant for the mock renderer's images."""
    g = patches.astype(np.float32).mean(axis=-1)
    med = np.median(g.reshape(len(g), -1), axis=1).reshape(-1, 1, 1)
    dark = g < med - 45.0
    frac = dark.reshape(len(g), -1).mean(axis=1)
    return (frac - 0.03) * 45.0


class LeakLocalizer:
    def __init__(self, crack_classifier: PatchClassifier = fake_crack_classifier, patch: int = 32, stride: int = 16,
                 threshold: float = 0.5, temperature: float = 1.0,
                 unsure_band: tuple[float, float] = (UNSURE_LO, UNSURE_HI), min_coverage: float = 0.9):
        self.classifier = crack_classifier
        self.patch, self.stride = int(patch), int(stride)
        self.threshold, self.temperature = float(threshold), float(temperature)
        self.unsure_band = unsure_band
        self.min_coverage = min_coverage

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str = "/ckpt/leak_detector/best", **kw) -> "LeakLocalizer":
        """Same checkpoint layout as the rust detector: model.onnx + meta.json."""
        d = Path(ckpt_dir)
        meta = json.loads((d / "meta.json").read_text())
        kw.setdefault("threshold", float(meta.get("threshold", 0.5)))
        kw.setdefault("temperature", float(meta.get("temperature", 1.0)))
        if "unsure_band" in meta:
            kw.setdefault("unsure_band", tuple(meta["unsure_band"]))
        return cls(OnnxPatchClassifier(d / "model.onnx", meta), **kw)

    # -- public ----------------------------------------------------------------
    def read(self, image: np.ndarray, camera, flange) -> LeakReading:
        img = np.ascontiguousarray(np.asarray(image)[..., :3]).astype(np.uint8)
        bolts = [self._bolt(img, camera, flange, k) for k in range(flange.n_bolts)]
        prob, unsure, hot, n, heat = self._crack(img, camera, flange)
        return LeakReading(bolts, prob, unsure, hot, n, self.threshold, heat)

    # -- bolts -------------------------------------------------------------------
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
        # A dirty or hazy lens washes the red out: confidence follows how saturated the stripe still looks.
        sat = float((R[red] - 0.5 * (G[red] + B[red])).mean())
        sat_score = min(1.0, max(0.0, (sat - 50.0) / 90.0))
        conf = count_score * shape_score * size_score * sat_score
        return BoltReading(k, angle, float(conf), (cx, cy), n_red)

    # -- crack -------------------------------------------------------------------
    def face_mask(self, camera, flange, shape) -> np.ndarray:
        """Pixels showing the flange face itself (not the bore, not bolts or their stripes, not off the edge)."""
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
            m &= np.linalg.norm(P - b, axis=-1) >= 2.3 * flange.head_radius
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

"""Camera detectors for loose debris on the deck and for people, trained like the rust and crack models.

Same flow as detection/localize.py (rust) and detection/leak.py (crack): windows of the frame -> trained classifier
(MobileNetV4 + LoRA, ONNX) -> calibrated probability -> threshold -> "unsure" band -> another view.

  DebrisDetector   windows over the deck part of the wrist / body camera frame. A hit gives the pixel centre the arm
                   picks at.
  PersonDetector   windows at three sizes over the whole frame. A hit also gives a rough distance from the window
                   height (a person is about 1.7 m tall), so the robot's 3 m pause rule can come from its own camera.

Checkpoints ship in detection/checkpoints/{debris_v2,person_v1} (model.onnx + meta.json, the rust layout).
Results and limits: detection/RIG_VISION_RESULTS.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from detection.localize import UNSURE_HI, UNSURE_LO, OnnxPatchClassifier, _sigmoid

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
PERSON_HEIGHT_M = 1.7


@dataclass
class Hit:
    box: tuple[int, int, int, int]          # x0, y0, x1, y1 in pixels
    p: float
    point: Optional[tuple[float, float]] = None   # refined target (e.g. middle of the changed pixels)

    @property
    def centre(self) -> tuple[float, float]:
        if self.point is not None:
            return self.point
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2


@dataclass
class Detection:
    task: str
    prob: float                              # highest calibrated window probability
    found: bool                              # some window at or above threshold
    unsure: bool                             # nothing found, but some window in the unsure band
    hits: list[Hit] = field(default_factory=list)
    n_windows: int = 0
    threshold: float = 0.5
    distance_m: Optional[float] = None       # PersonDetector only: rough distance to the nearest hit

    def summary(self) -> dict:
        d = {"task": self.task, "prob": round(self.prob, 3), "found": self.found, "unsure": self.unsure,
             "hits": len(self.hits), "windows": self.n_windows}
        if self.hits:
            d["best_box"] = list(max(self.hits, key=lambda h: h.p).box)
        if self.distance_m is not None:
            d["distance_m"] = round(self.distance_m, 2)
        return d


def _nms(hits: list[Hit], iou_max: float = 0.3) -> list[Hit]:
    keep: list[Hit] = []
    for h in sorted(hits, key=lambda h: -h.p):
        x0, y0, x1, y1 = h.box
        ok = True
        for k in keep:
            a0, b0, a1, b1 = k.box
            iw, ih = max(0, min(x1, a1) - max(x0, a0)), max(0, min(y1, b1) - max(y0, b0))
            inter = iw * ih
            union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            if union > 0 and inter / union > iou_max:
                ok = False
                break
        if ok:
            keep.append(h)
    return keep


class WindowDetector:
    """Slide square windows of several sizes over a frame, classify each, keep calibrated hits."""

    task = "object"

    def __init__(self, classifier, sizes=(96, 160), stride_frac: float = 0.5, threshold: float = 0.5,
                 temperature: float = 1.0, unsure_band=(UNSURE_LO, UNSURE_HI), region=(0.0, 0.0, 1.0, 1.0)):
        self.classifier = classifier
        self.sizes = tuple(int(s) for s in sizes)
        self.stride_frac = float(stride_frac)
        self.threshold, self.temperature = float(threshold), float(temperature)
        self.unsure_band = tuple(unsure_band)
        self.region = tuple(region)          # fractions (x0, y0, x1, y1) of the frame to search

    @classmethod
    def from_checkpoint(cls, ckpt_dir, **kw):
        d = Path(ckpt_dir)
        meta = json.loads((d / "meta.json").read_text())
        kw.setdefault("threshold", float(meta.get("threshold", 0.5)))
        kw.setdefault("temperature", float(meta.get("temperature", 1.0)))
        if "unsure_band" in meta:
            kw.setdefault("unsure_band", tuple(meta["unsure_band"]))
        if "window_sizes" in meta:
            kw.setdefault("sizes", tuple(meta["window_sizes"]))
        return cls(OnnxPatchClassifier(d / "model.onnx", meta), **kw)

    def windows(self, H: int, W: int):
        rx0, ry0, rx1, ry1 = (int(round(f * n)) for f, n in zip(self.region, (W, H, W, H)))
        for s in self.sizes:
            s = min(s, rx1 - rx0, ry1 - ry0)
            st = max(8, int(s * self.stride_frac))
            ys = list(range(ry0, ry1 - s + 1, st)) or [ry0]
            xs = list(range(rx0, rx1 - s + 1, st)) or [rx0]
            if ys[-1] + s < ry1:
                ys.append(ry1 - s)
            if xs[-1] + s < rx1:
                xs.append(rx1 - s)
            for y in ys:
                for x in xs:
                    yield x, y, s

    def detect(self, image: np.ndarray) -> Detection:
        img = np.ascontiguousarray(np.asarray(image)[..., :3])
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        H, W = img.shape[:2]
        wins = list(self.windows(H, W))
        if not wins:
            return Detection(self.task, 0.0, False, False, threshold=self.threshold)
        import torch
        import torch.nn.functional as F
        crops = []
        for x, y, s in wins:                  # resize every window to one size so they batch together
            t = torch.from_numpy(img[y:y + s, x:x + s]).permute(2, 0, 1)[None].float()
            crops.append(F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)[0].permute(1, 2, 0))
        patches = torch.stack(crops).clamp(0, 255).to(torch.uint8).numpy()
        logits = np.asarray(self.classifier(patches), np.float32).reshape(-1)
        p = _sigmoid(logits / self.temperature)
        hits = _nms([Hit((x, y, x + s, y + s), float(q)) for (x, y, s), q in zip(wins, p) if q >= self.threshold])
        lo, hi = self.unsure_band
        unsure = not hits and bool(((p >= lo) & (p <= hi)).any())
        return Detection(self.task, float(p.max()), bool(hits), unsure, hits, len(wins), self.threshold)


def change_mask(image: np.ndarray, reference: np.ndarray, sharp: float = 0.25, colour: float = 0.06) -> np.ndarray:
    """Pixels where the deck really changed against the clean reference of this patrol stop.

    Shadows and lighting change brightness smoothly and keep colour, so they are ignored: the log brightness ratio
    is compared with its own blurred version (only sharp, local changes count) and colour is compared as
    chromaticity (brightness removed). An object changes both. On the held-out deck, shadow / exposure changes alone
    mark at most 0.07% of the frame; a dropped can marks 12% or more of the pixels around it."""
    from scipy import ndimage
    a = np.asarray(image, np.float32)[..., :3] + 4.0
    b = np.asarray(reference, np.float32)[..., :3] + 4.0
    ratio = np.log(a.mean(-1) / b.mean(-1))
    local = np.abs(ratio - ndimage.gaussian_filter(ratio, 12))
    chroma = np.abs(a / a.sum(-1, keepdims=True) - b / b.sum(-1, keepdims=True)).max(-1)
    return ndimage.gaussian_filter(((local > sharp) | (chroma > colour)).astype(np.float32), 1.5) > 0.5


class DebrisDetector(WindowDetector):
    """Loose objects on the deck. Searches the lower part of the frame, where the deck is.

    With a ``reference`` (a clean picture of the same patrol stop, as airport FOD systems keep), only windows where
    the deck has changed are classified, so fixed equipment (fire extinguishers, gas bottles, pipe supports) is never
    a candidate, and the model's job is to tell litter from harmless change (shadows, light, wet patches). Without a
    reference every window is classified, which on an unseen deck also fires on equipment; see RIG_VISION_RESULTS.md.
    """

    task = "debris"

    def __init__(self, classifier, sizes=(96, 160), region=(0.0, 0.25, 1.0, 1.0), min_change: float = 0.02, **kw):
        super().__init__(classifier, sizes=sizes, region=region, **kw)
        self.min_change = float(min_change)
        self._changed = None

    def windows(self, H: int, W: int):
        for x, y, s in super().windows(H, W):
            if self._changed is None or self._changed[y:y + s, x:x + s].mean() >= self.min_change:
                yield x, y, s

    def detect(self, image: np.ndarray, reference: Optional[np.ndarray] = None) -> Detection:
        self._changed = None if reference is None else change_mask(image, reference)
        try:
            det = super().detect(image)
            if self._changed is not None and det.hits:
                det.hits = self._refine(det.hits, self._changed)
            return det
        finally:
            self._changed = None

    @staticmethod
    def _refine(hits: list[Hit], changed: np.ndarray, merge_px: float = 30.0) -> list[Hit]:
        """Aim each hit at the middle of the largest patch of changed pixels inside it (where the object is, not the
        window centre), then merge hits that aim at the same object."""
        from scipy import ndimage
        out: list[Hit] = []
        for h in sorted(hits, key=lambda h: -h.p):
            x0, y0, x1, y1 = h.box
            lab, n = ndimage.label(changed[y0:y1, x0:x1])
            if n:
                k = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
                cy, cx = ndimage.center_of_mass(lab == k)
                h = Hit(h.box, h.p, (x0 + float(cx), y0 + float(cy)))
            if all(np.hypot(h.centre[0] - o.centre[0], h.centre[1] - o.centre[1]) > merge_px for o in out):
                out.append(h)
        return out

    @classmethod
    def default(cls, **kw) -> Optional["DebrisDetector"]:
        try:
            return cls.from_checkpoint(CKPT_DIR / "debris_v2", **kw)
        except (FileNotFoundError, ImportError):
            return None


class PersonDetector(WindowDetector):
    """People anywhere in the frame. Distance from window height: d = f_y * 1.7 m / height_px (rough)."""

    task = "person"

    def __init__(self, classifier, sizes=(150, 240, 360), **kw):
        super().__init__(classifier, sizes=sizes, **kw)

    @classmethod
    def default(cls, **kw) -> Optional["PersonDetector"]:
        try:
            return cls.from_checkpoint(CKPT_DIR / "person_v1", **kw)
        except (FileNotFoundError, ImportError):
            return None

    def detect(self, image: np.ndarray, camera=None, range_m: Optional[np.ndarray] = None) -> Detection:
        """``range_m``: optional (H, W) ground distance in metres per pixel from the depth camera. With it, the
        distance comes from depth (the model says *who*, depth says *how far*); without it, from the box alone."""
        det = super().detect(image)
        if det.hits and camera is not None:
            if range_m is not None:
                ds = [depth_distance(h.box, camera, range_m) for h in det.hits]
                ds = [d for d in ds if d is not None]
                det.distance_m = min(ds) if ds else min(person_distance(h.box, camera) for h in det.hits)
            else:
                det.distance_m = min(person_distance(h.box, camera) for h in det.hits)
        return det


def floor_range(camera, H: int, W: int) -> np.ndarray:
    """Ground distance at which each pixel's ray meets the deck (inf above the horizon). World z up."""
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    v, u = np.mgrid[:H, :W].astype(np.float64)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T @ T[:3, :3].T
    eye = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(rays[..., 2] < -1e-9, -eye[2] / rays[..., 2], np.inf)
    return np.where(np.isfinite(t), t * np.hypot(rays[..., 0], rays[..., 1]), np.inf)


def depth_distance(box, camera, range_m: np.ndarray, below_floor: float = 0.85) -> Optional[float]:
    """Distance to whatever stands up off the deck inside ``box``: pixels clearly nearer than the floor would be
    there (a person, not the deck behind or below them). 10th percentile, so a few noisy pixels do not decide."""
    H, W = range_m.shape
    x0, y0, x1, y1 = (int(max(0, min(v, lim))) for v, lim in zip(box, (W, H, W, H)))
    r = range_m[y0:y1, x0:x1]
    fl = floor_range(camera, H, W)[y0:y1, x0:x1]
    up = np.isfinite(r) & (r < below_floor * fl) & (r < 50.0)
    if up.sum() < 0.02 * r.size:
        return None
    return float(np.percentile(r[up], 10))


def person_distance(box, camera) -> float:
    """Rough distance to a person in ``box``: the nearer of two estimates, so errors lean towards pausing.

    * height: a 1.7 m person filling the window height. A window can be at most the frame height, so on its own
      this can never say "closer than f * 1.7 / H" (about 5 m for the 640x360 camera).
    * ground: the ray through the bottom-centre of the box hits the deck (z = 0) at the person's feet. Needs the
      camera pose (``camera.T_world_cam``, world z up, camera above the deck). A box that reaches the bottom of the
      frame means the feet are at or below it, so the person is at most that far away.
    """
    K = np.asarray(camera.K, np.float64)
    x0, y0, x1, y1 = box
    est = K[1, 1] * PERSON_HEIGHT_M / max(1.0, y1 - y0)
    T = np.asarray(camera.T_world_cam, np.float64)
    eye = T[:3, 3]
    if eye[2] > 0.05:                               # camera above the deck: intersect the foot ray with z = 0
        ray = T[:3, :3] @ np.linalg.solve(K, np.array([(x0 + x1) / 2, y1, 1.0]))
        if ray[2] < -1e-6:
            foot = eye + ray * (-eye[2] / ray[2])
            est = min(est, float(np.linalg.norm(foot[:2] - eye[:2])))
    return est

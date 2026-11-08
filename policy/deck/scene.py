"""Mock camera views of the rig deck for the debris and person detectors (CPU, no Isaac).

Background: the team's held-out deck-level render ``sim/output/preview_deck_end.png`` (never used for training).
It already has deck equipment on it (fire extinguishers, gas bottles, pipe supports): things that belong there and
must not be picked up. Onto it we drop:

  * debris: drink cans drawn by detection.prepare_rig_data.draw_can, or real litter cut-outs from the held-out test
    split (assets/debris_*.png), sized by distance on the deck;
  * a person: a real held-out test-split photo crop (assets/person_*.jpg), standing on the deck at a given distance,
    sized from a 1.7 m height and the camera focal length.

Camera: 640x360, 35 mm lens on a 20.955 mm sensor (the team's wrist camera), horizon at row HORIZON, camera 0.6 m
above the deck (Go2 body height).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from policy.interfaces import CameraModel

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
BACKGROUNDS = ["preview_deck_end.png"]
W, H = 640, 360
F = 35.0 / 20.955 * W
HORIZON = 128            # image row of the horizon in the background render at 640x360
CAM_H = 0.6              # camera height above the deck, m


def camera_model() -> CameraModel:
    """Camera 0.6 m above the deck at the origin, looking along world +x, tilted down so the horizon is at row
    HORIZON (world z up, OpenCV optical frame, as everywhere else in the repo)."""
    from policy.leak.geometry import look_at
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    eye = np.array([0.0, 0.0, CAM_H])
    down = (H / 2 - HORIZON) / F                    # tan of the tilt below the horizon
    return CameraModel(K, look_at(eye, eye + np.array([1.0, 0.0, -down])), W, H)


@lru_cache(maxsize=4)
def background(name: str = BACKGROUNDS[0]) -> np.ndarray:
    return np.asarray(Image.open(REPO / "sim" / "output" / name).convert("RGB").resize((W, H), Image.BILINEAR))


@lru_cache(maxsize=16)
def sprite(name: str) -> Image.Image:
    return Image.open(ASSETS / name).convert("RGBA")


def sprites(prefix: str) -> list[str]:
    return sorted(p.name for p in ASSETS.glob(f"{prefix}_*"))


@dataclass
class Debris:
    x: float                 # metres, left (-) / right (+) of the camera axis
    d: float                 # metres ahead on the deck
    kind: str = "can"        # "can" or a sprite file name from assets/
    length_m: float = 0.15
    seed: int = 0

    def pixel(self) -> tuple[float, float]:
        """Image position of the object's contact point on the deck."""
        return W / 2 + F * self.x / self.d, HORIZON + F * CAM_H / self.d


def _paste(base: Image.Image, obj: Image.Image, cx: float, bottom: float, shadow=True) -> None:
    """Paste an RGBA object centred at ``cx`` with its bottom at row ``bottom`` (may run off the frame)."""
    x, y = int(cx - obj.size[0] / 2), int(bottom - obj.size[1])
    if shadow:
        a = obj.split()[-1].filter(ImageFilter.GaussianBlur(3)).point(lambda v: v * 0.5)
        base.paste(Image.new("RGBA", obj.size, (0, 0, 0, 255)), (x + obj.size[0] // 12, y + obj.size[1] // 10), a)
    base.paste(obj, (x, y), obj)


def deck_view(items: list[Debris] = (), bg: str = BACKGROUNDS[0], dust: float = 0.0, light: float = 1.0,
              shadow_seed: int | None = None) -> np.ndarray:
    """What the body camera sees when it looks down the deck at a patrol stop.

    ``light`` scales exposure and ``shadow_seed`` lays a soft shadow (crane boom, a passing cloud edge, a person
    just out of view) across the deck: harmless changes against the clean reference ``deck_view([])``."""
    from detection.prepare_rig_data import draw_can
    base = Image.fromarray(background(bg)).convert("RGBA")
    for it in sorted(items, key=lambda i: -i.d):              # far first
        u, v = it.pixel()
        px = max(10, int(F * it.length_m / it.d))
        if it.kind == "can":
            obj = draw_can(np.random.default_rng(it.seed), 128)
            obj = obj.crop(obj.getbbox())
        else:
            obj = sprite(it.kind)
        obj = obj.copy()
        obj.thumbnail((px, px), Image.BILINEAR)
        _paste(base, obj, u, v + obj.size[1] * 0.35)
    if shadow_seed is not None:
        rng = np.random.default_rng(shadow_seed)
        m = Image.new("L", (W, H), 0)
        y0 = int(rng.integers(HORIZON + 20, H - 40))
        pts = [(int(rng.integers(-50, W // 2)), y0), (int(rng.integers(W // 2, W + 50)), y0 + int(rng.integers(-30, 30))),
               (int(rng.integers(W // 2, W + 50)), y0 + int(rng.integers(30, 120))),
               (int(rng.integers(-50, W // 2)), y0 + int(rng.integers(30, 120)))]
        from PIL import ImageDraw
        ImageDraw.Draw(m).polygon(pts, fill=int(rng.integers(70, 140)))
        m = m.filter(ImageFilter.GaussianBlur(float(rng.uniform(3, 12))))
        base = Image.composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), base, m)
    img = np.asarray(base.convert("RGB"), np.float32) * light
    if dust > 0:
        img = np.asarray(Image.fromarray(img.astype(np.uint8)).filter(ImageFilter.GaussianBlur(4 * dust)), np.float32)
        img = img * (1 - 0.5 * dust) + 200 * 0.5 * dust
    return np.clip(img, 0, 255).astype(np.uint8)


def _person_box(distance_m: float, x_m: float, who: int):
    names = sprites("person")
    p = sprite(names[who % len(names)]).copy()
    h = int(F * 1.7 / distance_m)
    p = p.resize((max(4, int(p.size[0] * h / p.size[1])), h), Image.BILINEAR)
    feet = HORIZON + F * CAM_H / distance_m
    cx = W / 2 + F * x_m / distance_m
    return p, cx, feet


def person_view(distance_m: float | None, x_m: float = 0.0, who: int = 0, bg: str = BACKGROUNDS[0]) -> np.ndarray:
    """Body camera view with a person standing on the deck ``distance_m`` ahead (None = nobody)."""
    base = Image.fromarray(background(bg)).convert("RGBA")
    if distance_m is not None:
        p, cx, feet = _person_box(distance_m, x_m, who)
        _paste(base, p, cx, feet, shadow=False)
    return np.asarray(base.convert("RGB"))


def person_range(distance_m: float | None, x_m: float = 0.0, who: int = 0) -> np.ndarray:
    """The depth camera's ground distance per pixel for the same view: the deck plane below the horizon, far
    structure (30 m) above it, and the person (a flat board ``distance_m`` away; the sprite is a rectangle)."""
    from detection.rig_vision import floor_range
    r = floor_range(camera_model(), H, W)
    r = np.where(np.isfinite(r), r, 30.0).astype(np.float32)
    if distance_m is not None:
        p, cx, feet = _person_box(distance_m, x_m, who)
        x0, y0 = int(cx - p.size[0] / 2), int(feet - p.size[1])
        r[max(0, y0):max(0, min(H, y0 + p.size[1])), max(0, x0):max(0, min(W, x0 + p.size[0]))] = distance_m
    return r

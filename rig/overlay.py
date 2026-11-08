import math
from functools import lru_cache

import numpy as np

from rig.camera import anchor

PANEL = (38, 38, 36, 204)
CREAM = (245, 244, 238, 255)
DIM = (245, 244, 238, 170)
TONES = {"act": (217, 119, 87), "warn": (227, 179, 65), "stop": (248, 81, 73), "ok": (63, 185, 80)}
FONTS = {
    False: ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"),
    True: ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"),
}


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False):
    from PIL import ImageFont
    for path in FONTS[bold]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _w(d, text, f) -> float:
    return d.textlength(text, font=f)


def subtitle(d, cap: dict, cx: float, top: float, s: float) -> float:
    fb, fr = font(round(22 * s), True), font(round(22 * s))
    lw, rw = _w(d, cap["lead"], fb), _w(d, cap["rest"], fr)
    px, h = 22 * s, 46 * s
    x0 = cx - (lw + rw) / 2 - px
    d.rounded_rectangle([x0, top, cx + (lw + rw) / 2 + px, top + h], radius=h / 2, fill=PANEL)
    y = top + h / 2
    d.text((x0 + px, y), cap["lead"], font=fb, fill=TONES[cap["tone"]] + (255,), anchor="lm")
    d.text((x0 + px + lw, y), cap["rest"], font=fr, fill=CREAM, anchor="lm")
    return top + h


def label(d, r: dict, a: dict, W: int, H: int, s: float) -> None:
    fn, fc = font(round(15 * s), True), font(round(14 * s))
    cap = r["caption"]
    tw = max(_w(d, r["name"], fn) + 18 * s, _w(d, cap["text"], fc))
    pad, h = 12 * s, 50 * s
    cw = tw + 2 * pad
    alpha = 0.45 if a["mode"] == "occluded" else 1.0
    fade = lambda c: c[:3] + (round(c[3] * alpha),)
    ax, ay = a["x"], a["y"]
    if a["mode"] == "edge":
        bx = min(max(ax - math.cos(a["angle"]) * (cw / 2 + 14 * s), cw / 2 + 6), W - cw / 2 - 6)
        by = min(max(ay - math.sin(a["angle"]) * (h / 2 + 14 * s), h / 2 + 6), H - h / 2 - 6)
        x0, y0 = bx - cw / 2, by - h / 2
        ux, uy = math.cos(a["angle"]), math.sin(a["angle"])
        k = 9 * s
        d.polygon([(ax + ux * k, ay + uy * k), (ax - uy * k * 0.7, ay + ux * k * 0.7), (ax + uy * k * 0.7, ay - ux * k * 0.7)],
                  fill=TONES[cap["tone"]] + (255,))
    else:
        x0, y0 = min(max(ax - cw / 2, 6), W - cw - 6), ay - h - 18 * s
        d.line([(ax, y0 + h), (ax, ay)], fill=fade(DIM), width=max(1, round(1.5 * s)))
        d.ellipse([ax - 3 * s, ay - 3 * s, ax + 3 * s, ay + 3 * s], fill=fade(CREAM))
    d.rounded_rectangle([x0, y0, x0 + cw, y0 + h], radius=10 * s, fill=fade(PANEL))
    dot = 4 * s
    cy = y0 + pad + 7 * s
    d.ellipse([x0 + pad, cy - dot, x0 + pad + 2 * dot, cy + dot], fill=fade(TONES[cap["tone"]] + (255,)))
    d.text((x0 + pad + 18 * s, cy), r["name"], font=fn, fill=fade(CREAM), anchor="lm")
    d.text((x0 + pad, y0 + h - pad - 6 * s), cap["text"], font=fc, fill=fade(DIM), anchor="lm")


def burn(rgb: np.ndarray, robots: list[dict], camera: dict | None = None, top: float | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw
    if not robots:
        return rgb
    im = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3])).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    W, H = im.size
    s = H / 720
    y = 18 * s if top is None else top
    placed = [r for r in robots if camera is not None and r.get("world_pos") is not None] if len(robots) > 1 else []
    if placed:
        sx, sy = W / camera["w"], H / camera["h"]
        for r in placed:
            a = anchor(camera, r["world_pos"], margin=24, occluded=bool(r.get("occluded")))
            if a["mode"] != "behind":
                label(d, r, {**a, "x": a["x"] * sx, "y": a["y"] * sy}, W, H, s)
    else:
        for r in robots:
            cap = r["caption"]
            if len(robots) > 1:
                cap = {**cap, "lead": f"{r['name']} · {cap['lead']}"}
            y = subtitle(d, cap, W / 2, y, s) + 8 * s
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))

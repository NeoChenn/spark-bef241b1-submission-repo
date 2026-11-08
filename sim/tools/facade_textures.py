from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topside_frames import kelvin_rgb

OUT = Path(__file__).resolve().parents[1] / "assets" / "generated" / "facade"
T = 256
rng = np.random.default_rng(5)


def blur(a, r):
    img = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255


def room(kelvin, level, kind):
    tint = np.array(kelvin_rgb(kelvin), dtype=np.float32)
    y, x = np.mgrid[0:T, 0:T].astype(np.float32) / T
    light = level * (0.55 + 0.45 * np.exp(-((x - 0.5 + rng.uniform(-0.2, 0.2)) ** 2) / 0.12 - (y - 0.15) ** 2 / 0.08))
    img = light[..., None] * tint * rng.uniform(0.75, 1.0, 3).astype(np.float32) ** 0.3
    ceiling = y < 0.14
    img[ceiling] *= 0.55
    if kind == "tube":
        x0 = rng.uniform(0.2, 0.45)
        img[(y > 0.05) & (y < 0.1) & (x > x0) & (x < x0 + 0.4)] = np.minimum(tint * 1.6 * level + 0.2, 1.0)
    floor = y > 0.78
    img[floor] *= 0.35
    for _ in range(rng.integers(2, 5)):
        x0, w = rng.uniform(0, 0.85), rng.uniform(0.12, 0.4)
        top, shade = rng.uniform(0.45, 0.75), rng.uniform(0.08, 0.35)
        img[(x > x0) & (x < x0 + w) & (y > top)] *= shade
    for _ in range(rng.integers(0, 3)):
        x0, y0 = rng.uniform(0.05, 0.8), rng.uniform(0.25, 0.5)
        img[(x > x0) & (x < x0 + 0.12) & (y > y0) & (y < y0 + 0.16)] *= rng.uniform(0.4, 1.3)
    return img


def blinds(img, cover, tint, level):
    y = np.arange(T)[:, None] / T
    slat = ((np.arange(T) // 7) % 2 == 0)[:, None]
    mask = (y < cover)[..., None]
    lit = np.array(tint, dtype=np.float32) * level * np.where(slat, 0.85, 0.35)[..., None]
    return np.where(mask, lit * np.ones((T, T, 1), np.float32), img)


def curtains(img, open_frac, tint, level):
    x = np.arange(T)[None, :] / T
    folds = 0.65 + 0.35 * np.sin(x * 70 + np.sin(x * 9) * 2) ** 2
    cloth = np.array(tint, dtype=np.float32) * level * folds[..., None] * np.ones((T, 1, 1), np.float32)
    half = (1 - open_frac) / 2
    mask = ((x < half) | (x > 1 - half))[..., None]
    return np.where(mask, cloth, img)


def interiors():
    tiles = []
    for k in (2700, 3000, 2900):
        tiles.append(room(k, rng.uniform(0.6, 0.85), "lamp"))
    for k in (4200, 5200, 6500):
        tiles.append(room(k, rng.uniform(0.65, 0.9), "tube"))
    for k, cover in ((2800, 0.5), (4000, 1.0), (5000, 0.65), (3000, 1.0)):
        base = room(k, 0.7, "tube" if k > 3500 else "lamp")
        tiles.append(blinds(base, cover, kelvin_rgb(k), rng.uniform(0.45, 0.75)))
    tiles.append(curtains(room(2700, 0.7, "lamp"), 0.35, (1.0, 0.55, 0.25), 0.6))
    tiles.append(curtains(room(3000, 0.6, "lamp"), 0.0, (0.95, 0.75, 0.5), 0.45))
    dark = []
    for glow, level in (((0.3, 0.45, 0.9), 0.08), ((0, 0, 0), 0.0), ((1.0, 0.6, 0.3), 0.04), ((0.6, 0.7, 1.0), 0.02)):
        img = room(4000, 0.025, "lamp") * 0.6
        if level:
            y, x = np.mgrid[0:T, 0:T] / T
            g = np.exp(-((x - rng.uniform(0.3, 0.7)) ** 2 + (y - 0.55) ** 2) / 0.03) * level
            img = img + g[..., None] * np.array(glow, np.float32)
        dark.append(blinds(img, 0.4, (0.05, 0.05, 0.06), 1.0) if level == 0.04 else img)
    tiles += dark
    atlas = np.zeros((4 * T, 4 * T, 3), np.float32)
    for i, t in enumerate(tiles):
        r, c = divmod(i, 4)
        t = blur(t, 1.2) + rng.normal(0, 0.008, t.shape).astype(np.float32)
        atlas[r * T:(r + 1) * T, c * T:(c + 1) * T] = t
    Image.fromarray(np.clip(np.clip(atlas, 0, 1) ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)).save(OUT / "interiors.png")


def cladding():
    n = 1024
    px = n / 4.0
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    height = np.ones((n, n), np.float32)
    panel = 1.2
    seams = np.arange(0, 4.0 + 1e-6, 4.0 / round(4.0 / panel))
    for s in seams:
        d = np.abs(x / px - s)
        d = np.minimum(d, np.abs(x / px - s + 4.0))
        height -= 0.6 * np.exp(-(d * px / 2.2) ** 2)
    for zz in (0.0, 2.0):
        d = np.minimum(np.abs(y / px - zz), np.abs(y / px - zz - 4.0))
        height -= 0.35 * np.exp(-(d * px / 1.6) ** 2)
    ribs = 0.04 * np.sin(x / px * 2 * np.pi / 0.2) ** 8
    height += ribs
    tone = np.zeros((n, n), np.float32)
    edges = (seams * px).astype(int)
    for a, b in zip(edges, edges[1:]):
        tone[:, a:b] = rng.uniform(-0.04, 0.04)
    grain = blur(rng.random((n, n)).astype(np.float32), 6) - 0.5
    fine = rng.normal(0, 1, (n, n)).astype(np.float32)
    lum = 0.8 + tone + 0.12 * grain + 0.015 * fine - 0.35 * (1 - np.clip(height, 0, 1))
    weep = np.zeros((n, n), np.float32)
    for s in seams[:-1]:
        for _ in range(2):
            c = int(s * px + rng.uniform(-3, 3)) % n
            top = rng.integers(0, n)
            length = rng.integers(n // 8, n // 3)
            rows = (np.arange(length) + top) % n
            weep[rows, c] += np.linspace(1, 0, length)
    weep = blur(np.clip(weep, 0, 1), 3)
    rgb = lum[..., None] * np.array((0.98, 0.98, 0.95), np.float32)
    rust = np.array((0.36, 0.2, 0.1), np.float32)
    rgb = rgb * (1 - 0.55 * weep[..., None]) + rust * 0.55 * weep[..., None]
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).convert("RGB").save(OUT / "clad_diff.png")
    rough = 0.48 + 0.1 * grain + 0.2 * (1 - np.clip(height, 0, 1)) + 0.15 * weep
    Image.fromarray(np.clip(rough * 255, 0, 255).astype(np.uint8)).save(OUT / "clad_rough.png")
    gy, gx = np.gradient(height * 6.0 + grain * 0.6)
    nrm = np.stack((-gx, gy, np.ones_like(gx)), -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save(OUT / "clad_nor_gl.png")


def streaks():
    w, h = 256, 512
    atlas = np.zeros((h, 4 * w), np.float32)
    for v in range(4):
        m = np.zeros((h, w), np.float32)
        for _ in range(rng.integers(3, 8)):
            c = rng.uniform(0.15, 0.85) * w
            width = rng.uniform(3, 14)
            length = rng.uniform(0.35, 1.0) * h
            amp = rng.uniform(0.35, 1.0)
            yy = np.arange(h)[:, None]
            xx = np.arange(w)[None, :]
            wob = c + np.sin(yy / rng.uniform(20, 60)) * rng.uniform(1, 4)
            fade = np.clip(1 - yy / length, 0, 1) ** rng.uniform(0.6, 1.6)
            m += amp * np.exp(-((xx - wob) / (width * (0.6 + 0.6 * yy / h))) ** 2) * fade
        top = np.exp(-(np.arange(h) / 18.0))[:, None] * np.exp(-((np.arange(w) - w / 2) / (w * 0.35)) ** 2)[None, :]
        m += 0.6 * top
        m *= 0.7 + 0.6 * blur(rng.random((h, w)).astype(np.float32), 2)
        atlas[:, v * w:(v + 1) * w] = np.clip(m, 0, 1)
    Image.fromarray((atlas * 255).astype(np.uint8)).save(OUT / "streaks.png")


SIGNS = (
    ("MUSTER\nSTATION A", (0.05, 0.45, 0.2), (1, 1, 1)),
    ("LIFEBOAT\nNo. 1", (0.85, 0.35, 0.05), (1, 1, 1)),
    ("LIFEBOAT\nNo. 2", (0.85, 0.35, 0.05), (1, 1, 1)),
    ("TEMPORARY\nREFUGE", (0.08, 0.3, 0.6), (1, 1, 1)),
    ("EMERGENCY\nEXIT", (0.05, 0.45, 0.2), (1, 1, 1)),
    ("NO\nSMOKING", (0.75, 0.08, 0.06), (1, 1, 1)),
    ("LQ  DECK B", (0.9, 0.75, 0.05), (0.05, 0.05, 0.05)),
    ("FIRE\nPOINT", (0.75, 0.08, 0.06), (1, 1, 1)),
)


def signs():
    w, h = 512, 256
    atlas = Image.new("RGB", (2 * w, 4 * h))
    font = ImageFont.load_default(size=78)
    for i, (text, bg, fg) in enumerate(SIGNS):
        tile = Image.new("RGB", (w, h), tuple(int(c * 255) for c in bg))
        d = ImageDraw.Draw(tile)
        d.rectangle((8, 8, w - 9, h - 9), outline=tuple(int(c * 255) for c in fg), width=6)
        d.multiline_text((w / 2, h / 2), text, font=font, fill=tuple(int(c * 255) for c in fg), anchor="mm",
                         align="center", spacing=6)
        arr = np.asarray(tile, np.float32) / 255
        arr *= 0.85 + 0.15 * blur(rng.random((h, w)).astype(np.float32), 8)[..., None]
        r, c = divmod(i, 2)
        atlas.paste(Image.fromarray((arr * 255).astype(np.uint8)), (c * w, r * h))
    atlas.save(OUT / "signs.png")


def net():
    n = 512
    pitch = 32
    y, x = np.mgrid[0:n, 0:n]
    d = np.minimum(np.abs((x + pitch // 2) % pitch - pitch // 2), np.abs((y + pitch // 2) % pitch - pitch // 2))
    m = np.clip(1.6 - d / 1.4, 0, 1)
    Image.fromarray((m * 255).astype(np.uint8)).save(OUT / "net.png")


def louvre():
    n = 256
    y = np.arange(n)[:, None] / n
    slat = (0.5 + 0.5 * np.cos(y * 2 * np.pi * 12)) ** 3
    img = np.repeat(0.15 + 0.5 * slat, n, axis=1)
    Image.fromarray((img * 255).astype(np.uint8)).save(OUT / "louvre.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    interiors()
    cladding()
    streaks()
    signs()
    net()
    louvre()
    print("FACADE_TEXTURES", sorted(p.name for p in OUT.iterdir()))

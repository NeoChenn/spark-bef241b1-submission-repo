from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greebles import ATLAS, ATLAS_GRID, TEXTURES

RED = (196, 30, 28)
WHITE = (238, 236, 228)
BLACK = (22, 22, 22)
BLUE = (16, 78, 160)
GREEN = (18, 128, 70)
YELLOW = (240, 188, 20)
LABELS = {
    "crude": ("CRUDE OIL", (92, 58, 30), WHITE),
    "water": ("PRODUCED WATER", GREEN, WHITE),
    "gas": ("FUEL GAS", YELLOW, BLACK),
    "foam": ("FIRE WATER", RED, WHITE),
}


def font(size):
    return ImageFont.load_default(size=size)


def centred(draw, xy, text, size, fill, stroke=0):
    f = font(size)
    box = draw.textbbox((0, 0), text, font=f, stroke_width=stroke)
    draw.text((xy[0] - (box[2] - box[0]) / 2 - box[0], xy[1] - (box[3] - box[1]) / 2 - box[1]), text, font=f, fill=fill,
              stroke_width=stroke, stroke_fill=fill)


def weather(img: Image.Image, seed: int, amount: float = 1.0) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = img.size
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    noise = Image.fromarray((rng.random((h // 8, w // 8)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    noise = np.asarray(noise.filter(ImageFilter.GaussianBlur(6)), np.float32)[..., None] / 255.0
    fine = rng.random((h, w, 1)).astype(np.float32)
    grime = 1.0 - amount * (0.28 * noise + 0.08 * fine)
    a = a * grime
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) / (0.12 * min(w, h))
    a *= (0.72 + 0.28 * np.clip(edge, 0, 1))[..., None] ** amount
    streaks = np.zeros((h, w), np.float32)
    for _ in range(int(14 * amount)):
        x = rng.integers(0, w)
        top = rng.integers(0, h // 2)
        length = rng.integers(h // 4, h)
        width = rng.integers(2, 9)
        streaks[top:top + length, max(0, x - width):x + width] += rng.uniform(0.15, 0.45)
    streaks = np.asarray(Image.fromarray((np.clip(streaks, 0, 1) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(3)), np.float32) / 255.0
    rust = np.array([0.36, 0.2, 0.09], np.float32)
    a = a * (1 - streaks[..., None] * 0.7) + rust * streaks[..., None] * 0.35
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for _ in range(int(40 * amount)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        ang = rng.uniform(0, math.pi)
        ln = rng.integers(10, 60)
        d.line((x, y, x + ln * math.cos(ang), y + ln * math.sin(ang)), fill=(170, 165, 155), width=1)
    return out


def no_smoking():
    img = Image.new("RGB", (512, 692), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((150, 228, 362, 262), fill=BLACK)
    d.rectangle((330, 228, 362, 262), fill=(200, 120, 40))
    for k in range(3):
        d.arc((330 + k * 10, 120 + k * 30, 380 + k * 10, 200 + k * 30), 200, 340, fill=(90, 90, 90), width=6)
    d.ellipse((76, 66, 436, 426), outline=RED, width=44)
    d.line((140, 130, 372, 362), fill=RED, width=44)
    centred(d, (256, 520), "NO SMOKING", 62, BLACK, 2)
    centred(d, (256, 608), "NO NAKED FLAMES", 44, BLACK, 1)
    d.rectangle((0, 0, 511, 691), outline=BLACK, width=10)
    return img, None


def ppe():
    img = Image.new("RGB", (512, 692), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 512, 110), fill=BLUE)
    centred(d, (256, 55), "MANDATORY", 54, WHITE, 2)
    for k, cx in enumerate((92, 256, 420)):
        d.ellipse((cx - 76, 150, cx + 76, 302), fill=BLUE)
        if k == 0:
            d.chord((cx - 48, 190, cx + 48, 280), 180, 360, fill=WHITE)
            d.rectangle((cx - 58, 232, cx + 58, 246), fill=WHITE)
        elif k == 1:
            d.arc((cx - 46, 182, cx + 46, 272), 180, 360, fill=WHITE, width=10)
            d.ellipse((cx - 56, 222, cx - 26, 272), fill=WHITE)
            d.ellipse((cx + 26, 222, cx + 56, 272), fill=WHITE)
        else:
            d.rounded_rectangle((cx - 56, 206, cx - 6, 248), 14, fill=WHITE)
            d.rounded_rectangle((cx + 6, 206, cx + 56, 248), 14, fill=WHITE)
            d.rectangle((cx - 8, 216, cx + 8, 224), fill=WHITE)
    centred(d, (256, 390), "HARD HAT, EAR AND", 40, BLACK, 1)
    centred(d, (256, 444), "EYE PROTECTION", 40, BLACK, 1)
    centred(d, (256, 498), "MUST BE WORN", 40, BLACK, 1)
    centred(d, (256, 600), "IN THIS AREA", 46, BLUE, 2)
    d.rectangle((0, 0, 511, 691), outline=BLUE, width=10)
    return img, None


def hazard_flammable():
    img = Image.new("RGB", (512, 512), WHITE)
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).polygon(((256, 4), (508, 256), (256, 508), (4, 256)), fill=255)
    d = ImageDraw.Draw(img)
    d.polygon(((256, 24), (488, 256), (256, 488), (24, 256)), fill=RED)
    d.polygon(((256, 44), (468, 256), (256, 468), (44, 256)), outline=WHITE, width=8)
    d.polygon(((256, 110), (300, 190), (322, 160), (340, 230), (330, 290), (182, 290), (172, 230), (210, 170),
               (226, 200)), fill=WHITE)
    d.rectangle((176, 300, 336, 314), fill=WHITE)
    centred(d, (256, 360), "FLAMMABLE GAS", 30, WHITE, 1)
    centred(d, (256, 420), "2", 60, WHITE, 2)
    return img, mask


def muster():
    img = Image.new("RGB", (512, 348), GREEN)
    d = ImageDraw.Draw(img)
    for k, cx in enumerate((90, 160, 230)):
        d.ellipse((cx - 18, 46, cx + 18, 82), fill=WHITE)
        d.rectangle((cx - 16, 86, cx + 16, 150), fill=WHITE)
        d.rectangle((cx - 16, 150, cx - 4, 196), fill=WHITE)
        d.rectangle((cx + 4, 150, cx + 16, 196), fill=WHITE)
    d.polygon(((300, 100), (400, 100), (400, 60), (470, 125), (400, 190), (400, 150), (300, 150)), fill=WHITE)
    centred(d, (256, 256), "MUSTER POINT", 56, WHITE, 2)
    centred(d, (256, 314), "LIFEBOAT STATION 2", 32, WHITE, 1)
    d.rectangle((0, 0, 511, 347), outline=WHITE, width=10)
    return img, None


def h2s():
    img = Image.new("RGB", (512, 435), YELLOW)
    d = ImageDraw.Draw(img)
    d.polygon(((256, 24), (420, 270), (92, 270)), fill=BLACK)
    d.polygon(((256, 64), (386, 254), (126, 254)), fill=YELLOW)
    d.rectangle((246, 120, 266, 200), fill=BLACK)
    d.ellipse((244, 214, 268, 238), fill=BLACK)
    centred(d, (256, 320), "DANGER  H2S", 52, BLACK, 2)
    centred(d, (256, 380), "TOXIC GAS MAY BE PRESENT", 28, BLACK, 1)
    d.rectangle((0, 0, 511, 434), outline=BLACK, width=10)
    return img, None


def dial(needle: float, seed: int):
    img = Image.new("RGB", (512, 512), (40, 40, 42))
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).ellipse((2, 2, 510, 510), fill=255)
    d = ImageDraw.Draw(img)
    d.ellipse((26, 26, 486, 486), fill=(232, 230, 222))
    c = 256
    for k in range(51):
        a = math.radians(225 - 270 * k / 50)
        r0 = 196 if k % 5 else 176
        d.line((c + r0 * math.cos(a), c - r0 * math.sin(a), c + 212 * math.cos(a), c - 212 * math.sin(a)),
               fill=BLACK, width=3 if k % 5 else 6)
        if k % 10 == 0:
            centred(d, (c + 146 * math.cos(a), c - 146 * math.sin(a)), str(k // 5 * 2 // 2), 34, BLACK, 1)
    d.arc((44, 44, 468, 468), -45, -9, fill=RED, width=14)
    centred(d, (c, 340), "bar", 34, BLACK, 1)
    a = math.radians(225 - 270 * needle)
    d.line((c - 40 * math.cos(a), c + 40 * math.sin(a), c + 190 * math.cos(a), c - 190 * math.sin(a)), fill=BLACK, width=9)
    d.ellipse((c - 22, c - 22, c + 22, c + 22), fill=BLACK)
    img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(2)), 0.3)
    return weather(img, seed, 0.6), mask


def label(name: str, direction: int):
    text, bg, fg = LABELS[name]
    img = Image.new("RGB", (1024, 512), bg)
    d = ImageDraw.Draw(img)
    for band in (0, 256):
        mid = band + 128
        d.rectangle((0, band + 20, 1023, band + 26), fill=fg)
        d.rectangle((0, band + 230, 1023, band + 236), fill=fg)
        centred(d, (512, mid), text, 76, fg, 2)
        for x in (90, 934):
            tip = x + 60 * direction
            d.polygon(((x - 60 * direction, mid - 22), (x, mid - 22), (x, mid - 48), (tip, mid), (x, mid + 48),
                       (x, mid + 22), (x - 60 * direction, mid + 22)), fill=fg)
    return img, None


def tag():
    img = Image.new("RGB", (512, 512), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 512, 170), fill=RED)
    centred(d, (256, 88), "DANGER", 96, WHITE, 3)
    centred(d, (256, 250), "DO NOT", 72, BLACK, 2)
    centred(d, (256, 330), "ENTER", 72, BLACK, 2)
    centred(d, (256, 420), "OPEN DRAIN - PTW 0417", 34, BLACK, 1)
    d.ellipse((236, 470, 276, 510), outline=BLACK, width=6)
    return img, None


def puddle_masks(size=1024):
    rng = np.random.default_rng(21)
    half = size // 2
    out = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:half, 0:half].astype(np.float32) / half - 0.5
    for q in range(4):
        field = np.zeros((half, half), np.float32)
        for _ in range(11):
            cx, cy = rng.uniform(-0.27, 0.27, 2)
            r = rng.uniform(0.05, 0.14)
            field += np.exp(-((xx - cx) ** 2 + ((yy - cy) * rng.uniform(0.7, 1.4)) ** 2) / (2 * r * r))
        noise = np.asarray(Image.fromarray((rng.random((20, 20)) * 255).astype(np.uint8)).resize((half, half), Image.BICUBIC),
                           np.float32) / 255.0
        field = field * (0.35 + 1.3 * noise)
        mask = np.clip((field - 0.55) / 0.35, 0, 1)
        mask *= np.clip((0.5 - np.maximum(np.abs(xx), np.abs(yy))) / 0.06, 0, 1)
        r0, c0 = (q // 2) * half, (q % 2) * half
        out[r0:r0 + half, c0:c0 + half] = mask
    img = Image.fromarray((out * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
    return img


def wet_sign():
    img = Image.new("RGB", (512, 768), YELLOW)
    d = ImageDraw.Draw(img)
    d.polygon(((256, 60), (430, 360), (82, 360)), fill=BLACK)
    d.polygon(((256, 110), (392, 340), (120, 340)), fill=YELLOW)
    d.ellipse((236, 150, 276, 190), fill=BLACK)
    d.polygon(((250, 195), (275, 200), (300, 260), (330, 300), (300, 305), (262, 262), (225, 330), (200, 320), (240, 250)), fill=BLACK)
    centred(d, (256, 470), "CAUTION", 84, BLACK, 3)
    centred(d, (256, 580), "WET FLOOR", 70, BLACK, 2)
    centred(d, (256, 680), "SLIP HAZARD", 40, BLACK, 1)
    return weather(img, 55, 0.7)


def id_board():
    w, h = 2048, 512
    img = Image.new("RGB", (w, h), (14, 30, 62))
    emit = Image.new("RGB", (w, h), (6, 10, 18))
    for im, fg, stripe in ((img, WHITE, YELLOW), (emit, (255, 250, 236), (150, 110, 10))):
        d = ImageDraw.Draw(im)
        for x in range(-h, w, 96):
            d.polygon(((x, 0), (x + 48, 0), (x + 48 - 40, 40), (x - 40, 40)), fill=stripe)
            d.polygon(((x, h - 40), (x + 48, h - 40), (x + 48 - 40, h), (x - 40, h)), fill=stripe)
        centred(d, (w * 0.36, h / 2 + 4), "SA-7", 300, fg, 8)
        centred(d, (w * 0.78, h / 2 - 50), "PLATFORM", 86, fg, 3)
        centred(d, (w * 0.78, h / 2 + 60), "SIERRA ALPHA", 70, fg, 2)
    return weather(img, 77, 0.5), emit


def build() -> dict:
    TEXTURES.mkdir(parents=True, exist_ok=True)
    items = {"no_smoking": no_smoking(), "ppe": ppe(), "hazard_flammable": hazard_flammable(), "muster": muster(),
             "h2s": h2s(), "dial_a": dial(0.38, 4), "dial_b": dial(0.62, 9), "tag": tag()}
    for name in LABELS:
        for direction in (1, -1):
            items[f"label_{name}_{'f' if direction > 0 else 'r'}"] = label(name, direction)
    out = {}
    for k, (name, (img, mask)) in enumerate(items.items()):
        if not name.startswith("dial"):
            img = weather(img, 31 + k, 1.0)
        path = TEXTURES / f"{name}.jpg"
        img.convert("RGB").save(path, quality=90)
        out[name] = path
        if mask is not None:
            mask.save(TEXTURES / f"{name}_mask.png")
    Image.new("L", (8, 8), 255).save(TEXTURES / "solid_mask.png")
    puddle_masks().save(TEXTURES / "puddles_mask.png")
    wet_sign().save(TEXTURES / "wet_sign.jpg", quality=90)
    board, emit = id_board()
    board.save(TEXTURES / "id_board.jpg", quality=90)
    emit.save(TEXTURES / "id_board_emit.jpg", quality=90)
    cell = 512
    atlas = Image.new("RGB", (cell * ATLAS_GRID, cell * ATLAS_GRID), (128, 128, 128))
    amask = Image.new("L", atlas.size, 255)
    for name, (col, row, w, h) in ATLAS.items():
        box = (col * cell, row * cell)
        atlas.paste(Image.open(out[name]).convert("RGB").resize((w * cell, h * cell), Image.LANCZOS), box)
        mask = TEXTURES / f"{name}_mask.png"
        if mask.exists():
            amask.paste(Image.open(mask).convert("L").resize((w * cell, h * cell), Image.LANCZOS), box)
    atlas.save(TEXTURES / "atlas.jpg", quality=92)
    amask.save(TEXTURES / "atlas_mask.png")
    return out


if __name__ == "__main__":
    for name, path in build().items():
        print(name, path)

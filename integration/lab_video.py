from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
ALERT = ("BACK_OFF", "ESCALATE", "HOLD", "PAUSE")
TEAL, AMBER, RED, WHITE = (80, 220, 210), (255, 190, 60), (255, 80, 60), (232, 234, 236)


def _font(name: str, size: int):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


class FrameComposer:
    def __init__(self, size=(1280, 720), limit: float = 40.0, cap: float = 38.0, window_s: float = 12.0):
        self.W, self.H = size
        self.limit, self.cap, self.window = limit, cap, window_s
        self.big, self.mid, self.small = _font("DejaVuSans-Bold.ttf", 24), _font("DejaVuSans.ttf", 17), _font("DejaVuSans.ttf", 14)
        self.mono = _font("DejaVuSansMono-Bold.ttf", 30)

    def _state_colour(self, state: str):
        if state.startswith("BACK_OFF") or state.startswith("ESCALATE"):
            return RED
        if any(state.startswith(k) for k in ALERT):
            return AMBER
        return TEAL

    def compose(self, wide: np.ndarray, close: np.ndarray | None, state: str, lines: list[str], force: float,
                history: list[tuple[float, float]], t: float, speed: float = 1.0) -> np.ndarray:
        im = Image.fromarray(np.ascontiguousarray(wide[..., :3])).resize((self.W, self.H))
        d = ImageDraw.Draw(im, "RGBA")
        if close is not None:
            w, h = self.W * 2 // 5, self.H * 2 // 5
            x0, y0 = self.W - w - 16, 16
            cl = Image.fromarray(np.ascontiguousarray(close[..., :3])).resize((w, h))
            d.rectangle([x0 - 2, y0 - 2, x0 + w + 1, y0 + h + 1], outline=(255, 255, 255, 90))
            im.paste(cl, (x0, y0))
            d.rectangle([x0, y0 + h - 22, x0 + 150, y0 + h], fill=(10, 12, 16, 170))
            d.text((x0 + 8, y0 + h - 19), "BRUSH CLOSE-UP", fill=WHITE, font=self.small)
        col = self._state_colour(state)
        d.rounded_rectangle([16, 16, 560, 58], radius=10, fill=(10, 12, 16, 165))
        d.ellipse([30, 30, 44, 44], fill=col)
        d.text((54, 24), state, fill=col, font=self.big)
        tl = f"t {t:6.1f} s" + (f"   x{speed:.0f}" if speed > 1.5 else "")
        d.text((420, 30), tl, fill=WHITE, font=self.mid)
        y = self.H - 24 - 22 * len(lines)
        d.rounded_rectangle([16, y - 12, 700, self.H - 16], radius=10, fill=(10, 12, 16, 165))
        for k, line in enumerate(lines):
            d.text((30, y + 22 * k), line[:80], fill=WHITE, font=self.mid)
        self._force_panel(d, force, history, t)
        return np.asarray(im)

    def _force_panel(self, d, force: float, history, t: float) -> None:
        pw, ph = 420, 170
        x0, y0 = self.W - pw - 16, self.H - ph - 16
        d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=10, fill=(10, 12, 16, 175))
        col = RED if force > self.limit else (AMBER if force > self.cap else TEAL)
        d.text((x0 + 14, y0 + 10), "BRUSH CONTACT FORCE (sensor)", fill=(200, 204, 208), font=self.small)
        d.text((x0 + 14, y0 + 30), f"{force:5.1f} N", fill=col, font=self.mono)
        gx0, gy0, gx1, gy1 = x0 + 14, y0 + 74, x0 + pw - 14, y0 + ph - 12
        fmax = 70.0
        fy = lambda f: gy1 - (min(f, fmax) / fmax) * (gy1 - gy0)
        d.rectangle([gx0, gy0, gx1, gy1], outline=(255, 255, 255, 30))
        for level, c in ((self.limit, RED), (self.cap, AMBER)):
            yy = fy(level)
            for xx in range(gx0, gx1, 8):
                d.line([xx, yy, min(xx + 4, gx1), yy], fill=c + (200,), width=1)
        d.text((gx1 - 70, fy(self.limit) - 16), f"limit {self.limit:.0f} N", fill=RED, font=self.small)
        d.text((gx0 + 4, fy(self.cap) + 2), f"cap {self.cap:.0f} N", fill=AMBER, font=self.small)
        pts = [(gx1 - (t - ti) / self.window * (gx1 - gx0), fy(fi)) for ti, fi in history if t - ti <= self.window]
        if len(pts) > 1:
            d.line(pts, fill=TEAL + (255,), width=2)

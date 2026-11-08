import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.plume.colormap import colorbar

BG = (38, 38, 36)
FG = (245, 244, 238)
DIM = (184, 182, 174)
ACCENT = (217, 119, 87)
OK, WARN, STOP = (63, 185, 80), (227, 179, 65), (248, 81, 73)
SIZE = (480, 270)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def level(max_lel: float) -> tuple[str, tuple]:
    if max_lel >= 50:
        return "STOP", STOP
    if max_lel >= 10:
        return "WARN", WARN
    return "OK", OK


def compose(rgb, stats: dict, flow: dict, wind: dict, title: str = "GAS PLUME", size=SIZE) -> Image.Image:
    im = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8)).convert("RGB").resize(size)
    w, h = im.size
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f9, f11, f13 = _font(9), _font(11), _font(13)
    d.rectangle([0, 0, w, 22], fill=BG + (215,))
    d.text((8, 5), title, font=f11, fill=FG)
    tag, col = level(stats.get("max_lel_pct", 0.0))
    tw = d.textlength(tag, font=f11)
    d.rounded_rectangle([w - tw - 20, 4, w - 6, 19], 4, fill=col + (60,))
    d.text((w - tw - 13, 5), tag, font=f11, fill=col)
    d.text((8 + d.textlength(title, font=f11) + 10, 6), f"t {stats.get('t', 0):.1f}s", font=f9, fill=DIM)
    bh, bw, bx, by = 118, 8, w - 40, 34
    d.rectangle([bx - 6, by - 8, w - 4, by + bh + 20], fill=BG + (200,))
    ov.paste(Image.fromarray(colorbar(bw, bh)).convert("RGBA"), (bx, by))
    for pct in (0, 10, 50, 100):
        y = by + round((1 - pct / 100) * (bh - 1))
        d.line([bx + bw, y, bx + bw + 3, y], fill=FG)
        d.text((bx + bw + 5, y - 5), f"{pct}", font=f9, fill=FG)
    d.text((bx - 3, by + bh + 6), "%LEL", font=f9, fill=DIM)
    d.rectangle([0, h - 58, w, h], fill=BG + (215,))
    rows = [
        ("Leak", f"{flow.get('kg_s', 0) * 1000:.1f} g/s   {flow.get('sm3_h', 0):.0f} Sm3/h", FG),
        ("Cloud", f">10% LEL {stats.get('vol_10lel_m3', 0):.2f} m3   >50% LEL {stats.get('vol_50lel_m3', 0):.2f} m3", FG),
        ("Wind", f"{wind.get('speed_m_s', 0):.1f} m/s  to {wind.get('heading_deg', 0):.0f} deg   peak {stats.get('max_lel_pct', 0):.0f}% LEL", DIM),
    ]
    for k, (label, text, c) in enumerate(rows):
        y = h - 54 + k * 17
        d.text((8, y + 1), label.upper(), font=f9, fill=ACCENT)
        d.text((54, y), text, font=f13 if k == 0 else f11, fill=c)
    im = im.convert("RGBA")
    im.alpha_composite(ov)
    return im.convert("RGB")


def jpeg_b64(im: Image.Image, quality: int = 78) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

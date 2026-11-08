import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BAR = (20, 22, 26, 200)
FG = (245, 244, 238)
DIM = (184, 182, 174)
ACCENT = (217, 119, 87)
STOP = (248, 81, 73)
DEPTH = np.array([[255, 236, 160], [250, 150, 60], [200, 60, 90], [90, 40, 140], [20, 30, 90]], np.float32)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def encode(im: Image.Image, caption: str, seq: int, quality: int = 70) -> dict:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return {"jpg": base64.b64encode(buf.getvalue()).decode(), "w": im.size[0], "h": im.size[1], "seq": int(seq),
            "caption": caption}


def label(im: Image.Image, title: str, sub: str = "", color=FG) -> Image.Image:
    im = im.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle([0, 0, im.size[0], 15], fill=BAR)
    d.text((5, 2), title, font=font(10), fill=color)
    if sub:
        f = font(9)
        d.text((im.size[0] - d.textlength(sub, font=f) - 5, 3), sub, font=f, fill=DIM)
    im.alpha_composite(ov)
    return im.convert("RGB")


def depth_rgb(depth, near: float = 0.3, far: float = 25.0) -> Image.Image:
    z = np.asarray(depth, np.float32)
    ok = np.isfinite(z) & (z > 0)
    s = np.clip((np.log(np.where(ok, z, far)) - np.log(near)) / (np.log(far) - np.log(near)), 0, 1) * (len(DEPTH) - 1)
    i = np.minimum(s.astype(int), len(DEPTH) - 2)
    f = (s - i)[..., None]
    rgb = DEPTH[i] * (1 - f) + DEPTH[i + 1] * f
    rgb[~ok] = (8, 10, 14)
    return Image.fromarray(rgb.astype(np.uint8))


def sparkline(im: Image.Image, values, box, lo: float, hi: float, ref=None, color=ACCENT) -> Image.Image:
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(im)
    d.rectangle(box, fill=(20, 22, 26))
    if ref is not None:
        y = y1 - (ref - lo) / (hi - lo) * (y1 - y0)
        d.line([x0, y, x1, y], fill=STOP)
    v = list(values)
    if len(v) > 1:
        pts = [(x0 + k * (x1 - x0) / (len(v) - 1), y1 - (min(max(a, lo), hi) - lo) / (hi - lo) * (y1 - y0)) for k, a in enumerate(v)]
        d.line(pts, fill=color, width=1)
    return im


def draw_boxes(im: Image.Image, bs, scale: float = 1.0, color=STOP) -> Image.Image:
    d = ImageDraw.Draw(im)
    f = font(9)
    for x0, y0, x1, y1, s in bs:
        r = [x0 * scale, y0 * scale, x1 * scale - 1, y1 * scale - 1]
        d.rectangle(r, outline=color, width=2)
        d.text((r[0] + 2, r[1] + 1), f"rust {s:.2f}", font=f, fill=color)
    return im

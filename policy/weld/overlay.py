from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 320, 240
TOP = 150
PX_MM_X = 2.0
PX_MM_Y = 4.0
SPAN_Y = 17.0
TONE = {"INSPECT": (90, 200, 250), "GRIND": (250, 200, 80), "WELD": (255, 140, 40), "INSPECT_WELD": (90, 200, 250),
        "HOT_WORK_HOLD": (255, 70, 60), "DONE": (80, 220, 120), "ESCALATE_TO_HUMAN": (255, 70, 60)}
_tex: dict[int, np.ndarray] = {}


def font(size: int = 10):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def texture(seam) -> np.ndarray:
    k = id(seam)
    if k not in _tex:
        rng = np.random.default_rng(0)
        c = seam.cfg
        n = rng.normal(0, 1, (c.nx, c.ny))
        n = 0.6 * n + 0.4 * np.roll(n, 1, 0)
        streak = np.sin(seam.ys[None, :] * 3.1 + rng.normal(0, 0.3, (c.nx, 1))) * 0.5
        _tex.clear()
        _tex[k] = (n * 7 + streak * 4).astype(np.float32)
    return _tex[k]


def surface(job, xc: float) -> np.ndarray:
    s, c = job.seam, job.seam.cfg
    px = np.arange(W)
    py = np.arange(TOP)
    xw = xc + (px - W / 2) / PX_MM_X
    yw = SPAN_Y - py / PX_MM_Y
    i = np.clip(np.floor(xw / c.dx_mm).astype(int), 0, c.nx - 1)
    j = np.clip(np.round((yw + c.half_w_mm) / c.dy_mm).astype(int), 0, c.ny - 1)
    inside = ((xw >= 0) & (xw < c.length_mm))[None, :]
    I, J = np.meshgrid(i, j)
    tex = texture(s)[I, J]
    yy = s.ys[J]
    h = -c.groove_depth_mm * np.clip(1 - np.abs(yy - s.center[I]) / c.groove_half_mm, 0, 1)
    base = np.stack([112 + 22 * h + tex, 117 + 22 * h + tex, 124 + 20 * h + tex], -1)
    g = s.ground[I, J]
    base[g] = np.stack([168 + tex[g] * 1.5, 172 + tex[g] * 1.5, 178 + tex[g] * 1.5], -1)
    b = s.bead[I, J]
    rip = np.sin(xw[None, :].repeat(TOP, 0) * 2.4 + np.abs(yy - s.center[I]) * 0.6)
    bead = np.stack([128 + 18 * rip, 104 + 14 * rip, 84 + 10 * rip], -1)
    tx = job.view.get("x", xc)
    if job.state == "WELD":
        d = (tx - xw)[None, :].repeat(TOP, 0)
        glow = np.clip(1 - d / 18.0, 0, 1) * (d >= -1) * np.clip((s.temp_c - 40) / 120, 0.3, 1)
        bead = bead * (1 - glow[..., None]) + np.array([255, 170, 60]) * glow[..., None]
    base[b] = bead[b]
    oc = s.open_crack()[I, J]
    base[oc] = (225, 30, 30)
    base[~inside.repeat(TOP, 0)] = (18, 20, 24)
    return np.clip(base, 0, 255).astype(np.uint8)


def to_px(x, y, xc):
    return W / 2 + (x - xc) * PX_MM_X, (SPAN_Y - y) * PX_MM_Y


def poly(d: ImageDraw.ImageDraw, pts, xc, fill, width=1):
    q = [to_px(x, y, xc) for x, y in pts if abs(x - xc) < W / PX_MM_X]
    if len(q) > 1:
        d.line(q, fill=fill, width=width)


def render(job) -> np.ndarray:
    v = job.view
    xc = float(v.get("x", job.seam.cfg.length_mm / 2))
    im = Image.new("RGB", (W, H), (14, 16, 20))
    im.paste(Image.fromarray(surface(job, xc)), (0, 0))
    d = ImageDraw.Draw(im)
    cs = sorted(job.centers.items())
    poly(d, cs[::3], xc, (235, 235, 235))
    eddy_strip(d, job, xc)
    poly(d, job.paths["GRIND"], xc, (250, 210, 90))
    poly(d, job.paths["WELD"], xc, (40, 230, 255), 2)
    st = job.state
    tone = TONE.get(st, (200, 200, 200))
    if "y" in v:
        tx, ty = to_px(xc, v["y"], xc)
        d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], outline=(255, 235, 60), width=2)
        if st == "WELD" and v.get("profile_y") is not None:
            lx = tx + 12 * PX_MM_X
            d.line([(lx, 0), (lx, TOP)], fill=(60, 255, 120))
    f = font(10)
    d.rectangle([0, 0, W, 13], fill=(0, 0, 0))
    d.text((4, 1), f"WELD  {st}", fill=tone, font=f)
    d.text((W - 130, 1), f"t {job.t:5.1f}s  x {xc:5.1f}mm", fill=(200, 200, 200), font=f)
    if st in ("HOT_WORK_HOLD", "ESCALATE_TO_HUMAN"):
        msg = "HOT WORK BLOCKED" if st == "HOT_WORK_HOLD" else "ESCALATED TO HUMAN"
        d.rectangle([40, 100, W - 40, 138], fill=(90, 10, 10), outline=(255, 70, 60), width=2)
        d.text((W / 2, 111), msg, fill=(255, 255, 255), font=font(13), anchor="mm")
        d.text((W / 2, 128), f"gas {job.ev.get('lel_pct', 0)}% LEL  (limit 10%)", fill=(255, 190, 180), font=f,
               anchor="mm")
    panel(d, job, v, f)
    return np.asarray(im)


def eddy_strip(d, job, xc):
    if not job.scan:
        return
    y0 = TOP - 7
    d.rectangle([0, y0, W, TOP], fill=(10, 12, 16))
    for x, s in job.scan[::2]:
        px = W / 2 + (x - xc) * PX_MM_X
        if 0 <= px < W:
            col = (255, 40, 40) if s > 0.5 else (int(40 + 200 * s), int(90 - 40 * s), 120)
            d.line([(px, y0 + 1), (px, TOP - 1)], fill=col, width=2)


def panel(d, job, v, f):
    d.line([(0, TOP), (W, TOP)], fill=(60, 64, 72))
    x0, x1, y0, y1 = 4, 196, TOP + 14, H - 6
    d.text((x0, TOP + 2), "laser profile (mm)", fill=(140, 150, 160), font=f)
    d.rectangle([x0, y0, x1, y1], outline=(50, 54, 60))
    ys, z = v.get("profile_y"), v.get("profile_z")
    zt, zb = 1.5, -3.5

    def zp(zz):
        return y0 + (zt - zz) / (zt - zb) * (y1 - y0)
    d.line([(x0, zp(0)), (x1, zp(0))], fill=(45, 48, 54))
    if ys is not None and z is not None:
        yr = float(v["y"])

        def yp(yy):
            return x0 + (yy - yr + 16) / 32 * (x1 - x0)
        pts = [(yp(a), zp(b)) for a, b in zip(ys, z) if np.isfinite(b)]
        for p in pts:
            d.point(p, fill=(60, 255, 120))
        if len(pts) > 1:
            d.line(pts, fill=(40, 170, 90))
        d.line([(yp(yr), y0), (yp(yr), y1)], fill=(255, 235, 60))
        if v.get("center_est") is not None:
            cx = yp(v["center_est"])
            d.line([(cx, y0), (cx, y1)], fill=(255, 80, 200))
    ev = v.get("ev") or job.ev
    first = ("F", ev.get("force"), "N") if job.state == "GRIND" else ("HI", ev.get("heat_kj_mm"), "kJ/mm")
    rows = [first, ("I", ev.get("current_a"), "A"), ("U", ev.get("voltage_v"), "V"),
            ("v", ev.get("speed_mm_s"), "mm/s"), ("T", ev.get("temp_c"), "C"), ("err", ev.get("track_err_mm"), "mm"),
            ("crack", ev.get("crack_score"), ""), ("LEL", ev.get("lel_pct"), "%")]
    yy = TOP + 3
    for k, val, u in rows:
        hot = (k == "LEL" and val is not None and val > 10) or (k == "crack" and val is not None and val > 0.5)
        d.text((204, yy), f"{k:>5} {'-' if val is None else val} {u}", fill=(255, 90, 80) if hot else (200, 206, 214),
               font=f)
        yy += 11


def caption(job) -> str:
    r = job.records[-1] if job.records else None
    return f"Weld {job.state}: {r.reason}" if r else "Weld idle"


def thumb(job, seq: int = 0, img: np.ndarray | None = None) -> dict:
    img = render(job) if img is None else img
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=78)
    return {"jpg": base64.b64encode(buf.getvalue()).decode(), "w": W, "h": H, "caption": caption(job), "seq": seq}

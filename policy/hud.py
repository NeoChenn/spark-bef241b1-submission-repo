from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from policy.night_look import grade

FORCE_CAP, FORCE_LIMIT, FORCE_MAX = 38.0, 40.0, 50.0
GREY = (138, 137, 130)
LIME = (196, 214, 74)
GREEN = (74, 222, 128)
AMBER = (245, 165, 36)
RED = (239, 68, 68)
CORAL = (217, 119, 87)
BLUE = (125, 178, 240)
CREAM = (245, 244, 238)
MUTED = (163, 162, 154)
CARD = (38, 38, 36)
TRACK = (66, 65, 61)
RUST = (196, 98, 52)
STAGES = ("SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT")
STATE_COLORS = {"PATROL": BLUE, "APPROACH": BLUE, "SPRAY": BLUE, "SOAK": BLUE, "SCRUB": GREEN, "RE-PLAN": AMBER,
                "VERIFY": BLUE, "PAINT": CORAL, "DONE": GREEN, "BACK_OFF": RED, "ESCALATE": RED, "HOLD": AMBER}
STAGE_OF = {"APPROACH": "SCRUB", "RE-PLAN": "SCRUB", "BACK_OFF": "SCRUB"}


def lerp_color(a, b, t: float):
    t = min(max(float(t), 0.0), 1.0)
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def force_color(f: float):
    if f <= 10.0:
        return GREY
    if f < 18.0:
        return lerp_color(GREY, LIME, (f - 10.0) / 8.0)
    if f < 25.0:
        return lerp_color(LIME, GREEN, (f - 18.0) / 7.0)
    if f <= 34.0:
        return GREEN
    if f < FORCE_CAP:
        return lerp_color(GREEN, AMBER, (f - 34.0) / (FORCE_CAP - 34.0))
    if f <= FORCE_LIMIT:
        return lerp_color(AMBER, RED, (f - FORCE_CAP) / (FORCE_LIMIT - FORCE_CAP + 1.0))
    return RED


def force_zone(f: float) -> str:
    if f < 5.0:
        return "no contact"
    if f < 22.0:
        return "light"
    if f <= 36.0:
        return "effective"
    if f <= FORCE_LIMIT:
        return "near cap"
    return "over limit"


def gauge_x(f: float, x0: float, w: float, fmax: float = FORCE_MAX) -> float:
    return x0 + w * min(max(f / fmax, 0.0), 1.0)


def parse_state(s: str):
    key, _, cap = s.strip().partition("  ")
    return key.strip(), cap.strip()


def state_color(key: str):
    return STATE_COLORS.get(key, STATE_COLORS["PATROL"])


def pipeline(key: str):
    if key == "DONE":
        return [(s, "done") for s in STAGES]
    cur = STAGE_OF.get(key, key)
    if cur not in STAGES:
        return [(s, "todo") for s in STAGES]
    i = STAGES.index(cur)
    return [(s, "done" if k < i else "active" if k == i else "todo") for k, s in enumerate(STAGES)]


def sparkline_points(values, x0, y0, w, h, vmax):
    v = list(values)
    if not v:
        return []
    if len(v) == 1:
        v = v * 2
    n = len(v) - 1
    return [(int(round(x0 + w * k / n)), int(round(y0 + h - h * min(max(x / vmax, 0.0), 1.0))))
            for k, x in enumerate(v)]


def project(p, eye, look, focal, w, h, v_aperture=15.2908):
    p, e, l = (np.asarray(v, float) for v in (p, eye, look))
    f = (l - e) / np.linalg.norm(l - e)
    r = np.cross(f, (0.0, 0.0, 1.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = p - e
    z = d @ f
    if z <= 1e-6:
        return None
    k = focal / v_aperture * h
    return float(w / 2 + k * (d @ r) / z), float(h / 2 - k * (d @ u) / z)


def _ring(base, xy, force):
    from PIL import Image, ImageDraw, ImageFilter

    if xy is None or force < 2.0:
        return
    c = force_color(force)
    R = 26
    im = Image.new("RGBA", (2 * R * S + 40, 2 * R * S + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    o = 20
    d.ellipse([o, o, o + 2 * R * S, o + 2 * R * S], outline=c + (230,), width=3 * S)
    glow = im.filter(ImageFilter.GaussianBlur(6))
    im = Image.alpha_composite(glow, im).resize((im.width // S, im.height // S), Image.LANCZOS)
    base.alpha_composite(im, (int(xy[0] - im.width / 2), int(xy[1] - im.height / 2)))


FONT_DIRS = ("/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
             "/usr/share/fonts/truetype/inter", "/usr/share/fonts/truetype/dejavu")
SANS = ("Inter-Variable.ttf", "Inter.ttc", "SFNS.ttf", "DejaVuSans.ttf")
MONO = ("SFNSMono.ttf", "JetBrainsMono-Regular.ttf", "DejaVuSansMono.ttf")
_FONTS: dict = {}


def font(kind: str, size: int, weight: str = "Regular"):
    from PIL import ImageFont

    key = (kind, size, weight)
    if key in _FONTS:
        return _FONTS[key]
    for name in SANS if kind == "sans" else MONO:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                f = ImageFont.truetype(p, size)
                try:
                    f.set_variation_by_name(weight)
                except Exception:
                    pass
                _FONTS[key] = f
                return f
    _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


S = 2
CARD_W, CARD_H, MARGIN, PAD = 380, 246, 24, 18
_SHADOW: dict = {}


def _shadow(w, h, r):
    from PIL import Image, ImageDraw, ImageFilter

    if (w, h) not in _SHADOW:
        m = Image.new("L", (w + 80, h + 80), 0)
        ImageDraw.Draw(m).rounded_rectangle([40, 48, 40 + w, 48 + h], radius=r, fill=120)
        _SHADOW[(w, h)] = m.filter(ImageFilter.GaussianBlur(18))
    return _SHADOW[(w, h)]


def _card(state, frac, force, history, t, policy):
    from PIL import Image, ImageDraw

    W, H = CARD_W * S, CARD_H * S
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=14 * S, fill=CARD + (232,), outline=(255, 255, 255, 22), width=S)
    key, cap = parse_state(state)
    sc = state_color(key)
    x, y, iw = PAD * S, PAD * S, (CARD_W - 2 * PAD) * S

    f_chip = font("sans", 12 * S, "Bold")
    label = key.replace("_", " ")
    tw = d.textlength(label, font=f_chip)
    d.rounded_rectangle([x, y, x + tw + 34 * S, y + 24 * S], radius=12 * S, fill=sc + (46,))
    d.ellipse([x + 10 * S, y + 8 * S, x + 18 * S, y + 16 * S], fill=sc)
    d.text((x + 24 * S, y + 12 * S), label, font=f_chip, fill=sc, anchor="lm")
    d.text((x + iw, y + 12 * S), f"t {t:4.1f} s", font=font("mono", 12 * S), fill=MUTED, anchor="rm")
    y += 34 * S
    d.text((x, y), cap[:1].upper() + cap[1:], font=font("sans", 15 * S, "Semibold"), fill=CREAM)
    d.text((x + iw, y + 3 * S), policy, font=font("sans", 11 * S), fill=MUTED, anchor="ra")
    y += 30 * S

    steps = pipeline(key)
    step = iw / len(steps)
    cy = y + 5 * S
    d.line([x + step / 2, cy, x + iw - step / 2, cy], fill=TRACK, width=2 * S)
    for k, (name, st) in enumerate(steps):
        cx = x + step * (k + 0.5)
        if st == "done":
            d.line([x + step / 2, cy, cx, cy], fill=CORAL, width=2 * S)
    for k, (name, st) in enumerate(steps):
        cx = x + step * (k + 0.5)
        r = 5 * S
        if st == "done":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CORAL)
        elif st == "active":
            d.ellipse([cx - r - 3 * S, cy - r - 3 * S, cx + r + 3 * S, cy + r + 3 * S], fill=sc + (60,))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=sc)
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CARD, outline=TRACK, width=2 * S)
        col = CREAM if st == "active" else MUTED if st == "done" else (110, 109, 104)
        d.text((cx, cy + 10 * S), name, font=font("sans", 9 * S, "Semibold"), fill=col, anchor="ma")
    y += 40 * S

    f_lab, f_num = font("sans", 10 * S, "Semibold"), font("mono", 14 * S)
    d.text((x, y + 8 * S), "RUST REMAINING", font=f_lab, fill=MUTED, anchor="lm")
    d.text((x + iw, y + 8 * S), f"{100 * frac:5.1f} %", font=f_num, fill=CREAM, anchor="rm")
    y += 20 * S
    d.rounded_rectangle([x, y, x + iw, y + 6 * S], radius=3 * S, fill=TRACK)
    fw = iw * min(max(frac, 0.0), 1.0)
    if fw > 6 * S:
        d.rounded_rectangle([x, y, x + fw, y + 6 * S], radius=3 * S, fill=RUST if frac > 0.02 else GREEN)
    y += 18 * S

    fc = force_color(force)
    d.text((x, y + 8 * S), "CONTACT FORCE", font=f_lab, fill=MUTED, anchor="lm")
    num = f"{force:4.1f} N"
    d.text((x + iw, y + 8 * S), num, font=f_num, fill=fc, anchor="rm")
    zx = x + iw - d.textlength(num, font=f_num) - 8 * S
    d.text((zx, y + 8 * S), force_zone(force), font=font("sans", 10 * S), fill=fc, anchor="rm")
    y += 20 * S
    gh = 8 * S
    d.rounded_rectangle([x, y, x + iw, y + gh], radius=4 * S, fill=TRACK)
    d.rectangle([gauge_x(22, x, iw), y, gauge_x(36, x, iw), y + gh], fill=GREEN + (40,))
    gx = gauge_x(force, x, iw)
    if gx - x > gh:
        d.rounded_rectangle([x, y, gx, y + gh], radius=4 * S, fill=fc)
    for fv, col, txt in ((FORCE_CAP, AMBER, "38 cap"), (FORCE_LIMIT, RED, "40 limit")):
        tx = gauge_x(fv, x, iw)
        d.line([tx, y - 3 * S, tx, y + gh + 3 * S], fill=col, width=2 * S)
    d.text((gauge_x(FORCE_CAP, x, iw) - 3 * S, y + gh + 4 * S), "38 cap", font=font("sans", 9 * S), fill=AMBER,
           anchor="ra")
    d.text((gauge_x(FORCE_LIMIT, x, iw) + 3 * S, y + gh + 4 * S), "40 limit", font=font("sans", 9 * S), fill=RED)
    y += gh + 18 * S

    sh = 22 * S
    ly = y + sh - sh * FORCE_LIMIT / FORCE_MAX
    for sx in range(int(x), int(x + iw), 8 * S):
        d.line([sx, ly, sx + 4 * S, ly], fill=RED + (110,), width=S)
    pts = sparkline_points(history, x, y, iw, sh, FORCE_MAX)
    if len(pts) > 1:
        d.polygon(pts + [(pts[-1][0], y + sh), (pts[0][0], y + sh)], fill=fc + (40,))
        d.line(pts, fill=fc, width=2 * S, joint="curve")
        px, py = pts[-1]
        d.ellipse([px - 3 * S, py - 3 * S, px + 3 * S, py + 3 * S], fill=fc)
    return im.resize((CARD_W, CARD_H), Image.LANCZOS)


def draw_card(frame, state, frac, force, history, t, policy="PPO policy · Isaac Lab", brush_px=None):
    from PIL import Image

    base = Image.fromarray(np.ascontiguousarray(frame[..., :3])).convert("RGBA")
    if brush_px is not None and 0 <= brush_px[0] < base.width and 0 <= brush_px[1] < base.height:
        _ring(base, brush_px, force)
    x0, y0 = MARGIN, base.height - MARGIN - CARD_H
    sh = _shadow(CARD_W, CARD_H, 14)
    shade = Image.new("RGBA", sh.size, (0, 0, 0, 255))
    shade.putalpha(sh)
    base.alpha_composite(shade, (x0 - 40, y0 - 40))
    base.alpha_composite(_card(state, frac, force, history, t, policy), (x0, y0))
    return np.asarray(base.convert("RGB"))


def _frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def _writer(path, w, h, fps, crf):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def composite(raw, log_path, sinks, keys=(), key_stem=None, spark_s=3.0):
    from PIL import Image

    meta = json.load(open(log_path))
    log, fps = meta["log"], meta["fps"]
    w, h = meta.get("size", (1280, 720))
    n_spark = int(spark_s * fps)
    want = {int(q * (len(log) - 1)): q for q in keys}
    forces = [e["force"] for e in log]
    for i, fr in enumerate(_frames(raw, w, h)):
        if i >= len(log):
            break
        e = log[i]
        bp = project(e["brush"], e["eye"], e["look"], e["focal"], w, h, e.get("v_aperture", 15.2908)) if "brush" in e else None
        img = draw_card(grade(fr), e["state"], e["frac"], e["force"], forces[max(0, i - n_spark):i + 1], e["t"],
                        brush_px=bp)
        for s in sinks:
            s.write(img.tobytes())
        if i in want and key_stem:
            Image.fromarray(img).save(f"{key_stem}_{int(want[i] * 100):02d}.png")
    return len(log)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--shots", default="push,close,medium")
    p.add_argument("--prefix", default="look")
    p.add_argument("--crf", type=int, default=23)
    a = p.parse_args(argv)
    shots = a.shots.split(",")
    meta = json.load(open(os.path.join(a.dir, f"raw_{shots[0]}.json")))
    w, h = meta.get("size", (1280, 720))
    combo = _writer(os.path.join(a.dir, f"{a.prefix}_combined.mp4"), w, h, meta["fps"], a.crf)
    for s in shots:
        out = os.path.join(a.dir, f"{a.prefix}_{s}.mp4")
        single = _writer(out, w, h, meta["fps"], a.crf)
        n = composite(os.path.join(a.dir, f"raw_{s}.mp4"), os.path.join(a.dir, f"raw_{s}.json"),
                      [single.stdin, combo.stdin], keys=(0.1, 0.35, 0.6, 0.9),
                      key_stem=os.path.join(a.dir, f"key_{s}"))
        single.stdin.close()
        single.wait()
        print(f"{s}: {n} frames -> {out}", flush=True)
    combo.stdin.close()
    combo.wait()


if __name__ == "__main__":
    main()

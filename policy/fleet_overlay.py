from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ACCENT = (46, 196, 182)
WARN = (227, 179, 65)
STOP = (248, 81, 73)
OKG = (63, 185, 80)
GREY = (142, 142, 150)
LABEL = (255, 255, 255, 230)
LABEL2 = (235, 235, 245, 153)
LABEL3 = (235, 235, 245, 92)
PANEL = (14, 14, 18, 200)
HAIR = (255, 255, 255, 22)
LED = {"green": OKG, "amber": WARN, "red": STOP, "grey": GREY, "off": GREY}
FORCE_LIMIT_N = 40.0

UNITS = (("unit1", "UNIT 1", "Scrubber"), ("unit2", "UNIT 2", "Fixer"))
VERDICT_RANK = ("LEAK_CONFIRMED", "LEAK_UNSEEN_BY_FIXED", "PIPE_SENSOR_FAULT", "UNVERIFIED", "OK")
VERDICT_TONE = {"LEAK_CONFIRMED": WARN, "LEAK_UNSEEN_BY_FIXED": WARN, "UNVERIFIED": WARN,
                "PIPE_SENSOR_FAULT": GREY, "OK": ACCENT}
STOP_WORDS = ("HOLD", "RETREAT", "STOP", "E-STOP", "ESTOP", "HALT")
WARN_WORDS = ("PAUSED", "WAIT", "FAULT", "LOW")
IDLE_WORDS = ("DOCK", "IDLE", "DONE", "PARKED", "STANDBY")
CAPTION_TONE = {"warn": WARN, "stop": STOP}
COMMAND_TONE = {"hold": STOP, "retreat": STOP, "log_sensor_fault": WARN}

FONT_DIRS = [os.environ.get("FLEET_FONTS", ""), str(Path.home() / ".cache/fleet_overlay/fonts"),
             str(Path(__file__).resolve().parents[1] / "rig/web/static/fonts")]
FONT_FILES = {"sans": ("Inter.ttf", "Inter-Variable.ttf", "InterVariable.ttf"),
              "mono": ("JetBrainsMono.ttf", "JetBrainsMono-Variable.ttf")}
FONT_FALLBACK = {"sans": ("/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                 "mono": ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")}
FONT_URLS = {"Inter.ttf": "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf",
             "JetBrainsMono.ttf": "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf"}


def fetch_fonts(dest: Optional[str] = None) -> Path:
    d = Path(dest or Path.home() / ".cache/fleet_overlay/fonts")
    d.mkdir(parents=True, exist_ok=True)
    for name, url in FONT_URLS.items():
        if not (d / name).exists():
            urllib.request.urlretrieve(url, d / name)
    font.cache_clear()
    return d


@lru_cache(maxsize=256)
def font(kind: str, size: int, weight: int = 400):
    for base in filter(None, FONT_DIRS):
        for name in FONT_FILES[kind]:
            p = Path(base) / name
            if p.exists():
                f = ImageFont.truetype(str(p), size)
                try:
                    axes = f.get_variation_axes()
                    f.set_variation_by_axes([min(max(weight if b"eight" in a["name"] else
                                                     min(max(size, a["minimum"]), a["maximum"]),
                                                     a["minimum"]), a["maximum"]) for a in axes])
                except OSError:
                    pass
                return f
    for p in FONT_FALLBACK[kind]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def robot_tone(r: dict) -> tuple:
    t = tone_of_state(str(r.get("state") or ""))
    cap = r.get("caption")
    ct = CAPTION_TONE.get(str(cap.get("tone") or "")) if isinstance(cap, dict) else None
    return ct if ct is not None and t in (ACCENT, GREY) else t


def tone_of_state(state: str) -> tuple:
    s = (state or "").upper()
    if any(w in s for w in STOP_WORDS):
        return STOP
    if any(w in s for w in WARN_WORDS):
        return WARN
    if any(w in s for w in IDLE_WORDS):
        return GREY
    return ACCENT


def rgba(c, a=255):
    return tuple(c[:3]) + (a,)


def fit(d: ImageDraw.ImageDraw, text: str, f, width: float) -> str:
    text = str(text or "")
    if d.textlength(text, font=f) <= width:
        return text
    while text and d.textlength(text + "…", font=f) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def num(v, fmt="{:.1f}", none="--"):
    try:
        return none if v is None or not np.isfinite(float(v)) else fmt.format(float(v))
    except (TypeError, ValueError):
        return none


def pct(v):
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if v <= 1.0 else v


class Canvas:
    def __init__(self, img: Image.Image):
        self.base = img
        self.W, self.H = img.size
        self.s = min(self.W / 1920.0, self.H / 1080.0) ** 0.75
        self.d = ImageDraw.Draw(img, "RGBA")
        self.rects: list[tuple] = []

    def u(self, v: float) -> int:
        return int(round(v * self.s))

    def f(self, kind: str, size: float, weight: int = 400):
        return font(kind, max(8, self.u(size)), weight)

    def panel(self, box, radius=16, fill=PANEL):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        r = self.u(radius)
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), r, fill=255)
        region = self.base.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(self.u(18)))
        self.base.paste(region, (x0, y0), mask)
        self.d.rounded_rectangle((x0, y0, x1 - 1, y1 - 1), r, fill=fill, outline=HAIR, width=max(1, self.u(1)))
        self.rects.append((x0, y0, x1, y1))

    def chip(self, x_right, y_mid, text, tone, size=13, anchor_right=True):
        f = self.f("mono", size, 600)
        w = self.d.textlength(text, font=f) + self.u(20)
        h = self.u(size + 11)
        x0 = x_right - w if anchor_right else x_right
        self.d.rounded_rectangle((x0, y_mid - h / 2, x0 + w, y_mid + h / 2), h / 2, fill=rgba(tone, 50))
        self.d.text((x0 + w / 2, y_mid), text, font=f, fill=rgba(tone), anchor="mm")
        return x0, w

    def dot(self, x, y, r, tone, glow=False):
        if glow:
            g = self.u(r * 2.2)
            self.d.ellipse((x - g, y - g, x + g, y + g), fill=rgba(tone, 50))
        rr = self.u(r)
        self.d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=rgba(tone))

    def text(self, xy, text, kind="sans", size=15, weight=400, fill=LABEL, anchor="ls"):
        self.d.text(xy, str(text), font=self.f(kind, size, weight), fill=fill, anchor=anchor)

    def width(self, text, kind="sans", size=15, weight=400):
        return self.d.textlength(str(text), font=self.f(kind, size, weight))


def caption_line(c: Canvas, x, y, w, cap, tone=None):
    if isinstance(cap, dict):
        lead, rest = str(cap.get("lead") or ""), str(cap.get("rest") or "")
        if not lead and not rest:
            lead = str(cap.get("text") or "")
    else:
        lead, rest = "", str(cap or "")
    fb, fr = c.f("sans", 14, 600), c.f("sans", 14)
    lead = fit(c.d, lead, fb, w)
    if lead:
        c.d.text((x, y), lead, font=fb, fill=rgba(tone, 240) if tone else LABEL, anchor="lm")
        x, w = x + c.d.textlength(lead, font=fb), w - c.d.textlength(lead, font=fb)
    if rest and w > c.u(24):
        c.d.text((x, y), fit(c.d, rest, fr, w), font=fr, fill=LABEL2, anchor="lm")


def robot_card(c: Canvas, x, y, w, rid, tag, name, r: dict, scrubber: bool) -> float:
    u = c.u
    pad = u(18)
    h = u(176 if scrubber else 136)
    c.panel((x, y, x + w, y + h))
    state = str(r.get("state") or "--").upper().replace("_", " ")
    tone = robot_tone(r)
    cy = y + pad + u(9)
    c.text((x + pad, cy), tag, "mono", 12, 600, LABEL3, "lm")
    c.text((x + pad + c.width(tag, "mono", 12, 600) + u(9), cy), name, "sans", 17, 600, LABEL, "lm")
    cx0, _ = c.chip(x + w - pad, cy, state, tone, 12)
    caption_line(c, x + pad, cy + u(28), w - 2 * pad, r.get("caption"), tone if tone in (WARN, STOP) else None)
    my = cy + u(70)
    cells = [("BATT", num(pct(r.get("battery")), "{:.0f}"), "%")] if r.get("battery") is not None else []
    cells.append(("GAS", num(r.get("lel")), "%LEL"))
    if scrubber:
        cells.append(("RUST", num(pct(r.get("rust")), "{:.0f}"), "% left"))
    cw = (w - 2 * pad) / 3
    for i, (k, v, unit) in enumerate(cells):
        x0 = x + pad + i * cw
        c.text((x0, my - u(14)), k, "mono", 11, 600, LABEL3, "lm")
        vt = WARN if k == "GAS" and v != "--" and float(v) >= 5 else (255, 255, 255)
        if k == "BATT" and v != "--" and float(v) < 20:
            vt = WARN
        c.text((x0, my + u(8)), v, "mono", 20, 500, rgba(vt, 235), "lm")
        c.text((x0 + c.width(v, "mono", 20, 500) + u(4), my + u(10)), unit, "sans", 11, 400, LABEL3, "lm")
    if scrubber:
        sparkline(c, x + pad, my + u(30), w - 2 * pad, u(28), r)
    return y + h


def sparkline(c: Canvas, x, y, w, h, r: dict):
    u = c.u
    hist = r.get("force_hist")
    if hist is None and isinstance(r.get("force_n"), (list, tuple)):
        hist = r["force_n"]
    hist = [float(v) for v in (hist or []) if v is not None]
    cur = hist[-1] if hist else r.get("force_n")
    label_w = u(92)
    c.text((x, y + h / 2 - u(6)), "BRUSH", "mono", 11, 600, LABEL3, "lm")
    over = cur is not None and float(cur) > FORCE_LIMIT_N
    c.text((x, y + h / 2 + u(9)), num(cur, "{:.0f}"), "mono", 15, 500, rgba(WARN if over else (255, 255, 255), 235), "lm")
    c.text((x + c.width(num(cur, "{:.0f}"), "mono", 15, 500) + u(3), y + h / 2 + u(10)), "N", "sans", 11, 400,
           LABEL3, "lm")
    gx0, gx1 = x + label_w, x + w
    top = max([FORCE_LIMIT_N * 1.35] + hist)

    def py(v):
        return y + h - (v / top) * h

    ly = py(FORCE_LIMIT_N)
    dash = u(5)
    for xx in range(int(gx0), int(gx1), dash * 2):
        c.d.line((xx, ly, min(xx + dash, gx1), ly), fill=LABEL3, width=max(1, u(1)))
    c.text((gx1, ly - u(3)), "40 N", "mono", 11, 500, LABEL3, "rs")
    if len(hist) >= 2:
        pts = [(gx0 + (gx1 - gx0) * i / (len(hist) - 1), py(v)) for i, v in enumerate(hist)]
        poly = pts + [(gx1, y + h), (gx0, y + h)]
        c.d.polygon(poly, fill=rgba(ACCENT, 34))
        for (a, va), (b, vb) in zip(zip(pts, hist), zip(pts[1:], hist[1:])):
            col = WARN if max(va, vb) > FORCE_LIMIT_N else ACCENT
            c.d.line((a, b), fill=rgba(col), width=max(1, u(2)))
        c.dot(pts[-1][0], pts[-1][1], 2.5, WARN if over else ACCENT)


def findings_by_sensor(header: dict) -> dict:
    fs = header.get("findings") or header.get("fusion") or {}
    if isinstance(fs, list):
        fs = {f.get("sensor"): f for f in fs if isinstance(f, dict)}
    return fs


def headline(header: dict, fs: dict):
    verdicts = dict((header.get("argon") or {}).get("verdicts") or {})
    verdicts.update({k: v.get("kind") for k, v in fs.items() if v.get("kind")})
    if not verdicts:
        return None, None
    sensor, kind = min(verdicts.items(), key=lambda kv: (VERDICT_RANK.index(kv[1]) if kv[1] in VERDICT_RANK
                                                         else len(VERDICT_RANK), kv[0]))
    return sensor, kind


def gas_panel(c: Canvas, x, y, w, header: dict) -> float:
    u = c.u
    dets = header.get("detectors") or []
    fs = findings_by_sensor(header)
    pad = u(18)
    row = u(38)
    tickets = (header.get("argon") or {}).get("tickets") or []
    h = pad + u(18) + u(30) + u(12) + row * max(1, len(dets)) + u(6) + (u(40) if tickets else 0)
    c.panel((x, y, x + w, y + h))
    cy = y + pad + u(9)
    c.text((x + pad, cy), "GAS", "mono", 12, 600, LABEL3, "lm")
    c.text((x + pad + c.width("GAS", "mono", 12, 600) + u(9), cy), "Fire & Gas", "sans", 17, 600, LABEL, "lm")
    sensor, kind = headline(header, fs)
    if kind:
        hazard = any(f.get("hazard") for f in fs.values())
        tone = STOP if hazard else VERDICT_TONE.get(kind, GREY)
        c.chip(x + w - pad, cy, kind.replace("_", " "), tone, 11)
    hy = cy + u(30)
    col_sniff = x + w * 0.56
    c.text((x + pad + u(22), hy), "FIXED", "mono", 11, 600, LABEL3, "lm")
    c.text((col_sniff, hy), "NEAREST SNIFFER", "mono", 11, 600, LABEL3, "lm")
    ry = hy + u(12) + row / 2
    for dd in dets:
        did = str(dd.get("id", "?"))
        lel = dd.get("lel")
        tone = LED.get(str(dd.get("led") or "grey"), GREY)
        c.d.line((x + pad, ry - row / 2, x + w - pad, ry - row / 2), fill=HAIR, width=max(1, u(1)))
        c.dot(x + pad + u(6), ry, 4.5, tone, glow=dd.get("led") in ("amber", "red"))
        c.text((x + pad + u(22), ry), did, "mono", 14, 500, LABEL2, "lm")
        vx = x + pad + u(22) + c.width("G-F0", "mono", 14, 500) + u(16)
        v = num(lel)
        vcol = tone if dd.get("led") in ("amber", "red", "grey") else (255, 255, 255)
        c.text((vx, ry), v, "mono", 18, 500, rgba(vcol, 235), "lm")
        c.text((vx + c.width(v, "mono", 18, 500) + u(3), ry + u(2)), "%LEL", "sans", 11, 400, LABEL3, "lm")
        f = fs.get(did) or {}
        if f.get("robot") and f.get("robot_lel") is not None:
            rid = str(f["robot"]).replace("unit", "U")
            sv = num(f.get("robot_lel"))
            c.text((col_sniff, ry), rid, "mono", 12, 600, LABEL3, "lm")
            sx = col_sniff + c.width("U0", "mono", 12, 600) + u(8)
            c.text((sx, ry), sv, "mono", 15, 500, LABEL, "lm")
            dx = sx + c.width(sv, "mono", 15, 500) + u(8)
            c.text((dx, ry + u(1)), f"@ {num(f.get('robot_dist'))} m", "sans", 12, 400, LABEL3, "lm")
        else:
            c.text((col_sniff, ry), "--", "mono", 14, 400, LABEL3, "lm")
        ry += row
    if tickets:
        c.d.line((x + pad, ry - row / 2, x + w - pad, ry - row / 2), fill=HAIR, width=max(1, u(1)))
        tcy = ry - row / 2 + u(22)
        x0, tw = c.chip(x + pad, tcy, "TICKET " + str(tickets[-1]), WARN, 11, anchor_right=False)
        c.text((x0 + tw + u(10), tcy), "maintenance only, no repair", "sans", 13, 400, LABEL2, "lm")
    return y + h


def decision(argon: dict) -> tuple[str, str]:
    cmd = argon.get("last_command") or {}
    line = argon.get("line") or ""
    reason = str(cmd.get("reason") or "")
    if not line:
        if " | " in reason:
            line, rest = reason.split(" | ", 1)
            return line, rest
        line = reason
    for pre in ("Argon unavailable -> ", "edge rule "):
        if line.startswith(pre):
            line = line[len(pre):]
    head, sep, tail = line.partition(": ")
    if sep and head.replace("_", "").isupper():
        return head.replace("_", " ").capitalize(), tail.replace("%LEL", " %LEL").replace("  ", " ")
    return line[:1].upper() + line[1:], ""


def argon_strip(c: Canvas, header: dict):
    u = c.u
    a = header.get("argon") or {}
    if not a:
        return
    w = min(c.W - 2 * u(32) - 2 * u(400) - 2 * u(24), u(1080))
    w = max(w, u(880))
    h = u(84)
    x = (c.W - w) / 2
    y = c.H - u(32) - h
    cut = str(a.get("link", "")).upper() == "CUT"
    if cut:
        msg = "LINK CUT"
        sub = "Argon cloud offline · on-rig edge rules in command"
        bw = c.width(msg, "mono", 14, 700) + c.width(sub, "sans", 15) + u(70)
        bh = u(40)
        bx, by = (c.W - bw) / 2, u(32)
        c.panel((bx, by, bx + bw, by + bh), radius=20, fill=(44, 33, 8, 210))
        c.dot(bx + u(20), by + bh / 2, 4.5, WARN, glow=True)
        c.text((bx + u(36), by + bh / 2), msg, "mono", 14, 700, rgba(WARN), "lm")
        c.text((bx + u(36) + c.width(msg, "mono", 14, 700) + u(14), by + bh / 2), sub, "sans", 15, 400, LABEL, "lm")
    c.panel((x, y, x + w, y + h), radius=18)
    pad = u(20)
    cmd = a.get("last_command") or {}
    src = str(cmd.get("source") or a.get("source") or ("edge" if cut else "")).lower()
    edge = src == "edge" or (cut and not cmd)
    cy1, cy2 = y + h / 2 - u(13), y + h / 2 + u(14)
    c.text((x + pad, cy1), "ARGON", "mono", 14, 700, rgba(ACCENT), "lm")
    if edge:
        stext, stone = "EDGE", WARN
        sdetail = "on-rig rules"
    else:
        stext, stone = "CLOUD", ACCENT
        model = str(a.get("model") or "gemini-4-argon").replace(" (replay)", "")
        lat = a.get("latency_ms")
        sdetail = model + (f"  {float(lat) / 1000:.1f} s" if lat is not None else "")
        if a.get("pending"):
            sdetail = model + "  thinking…"
    c.text((x + pad, cy2), stext, "mono", 11, 700, rgba(stone), "lm")
    c.text((x + pad + c.width(stext, "mono", 11, 700) + u(8), cy2), sdetail, "mono", 11, 400, LABEL2, "lm")
    left_w = max(c.width("ARGON", "mono", 14, 700), c.width(stext, "mono", 11, 700) + u(8) +
                 c.width(sdetail, "mono", 11, 400)) + pad + u(20)
    sep = x + left_w
    c.d.line((sep, y + u(16), sep, y + h - u(16)), fill=HAIR, width=max(1, u(1)))
    cr = x + w - pad
    if cmd.get("kind"):
        kind = str(cmd["kind"])
        robot = str(cmd.get("robot") or "").upper().replace("UNIT", "UNIT ")
        target = str(cmd.get("target") or "")
        label = kind.upper().replace("_", " ") + "  →  " + " ".join(p for p in (robot, target) if p)
        x0, _ = c.chip(cr, y + h / 2, label, COMMAND_TONE.get(kind, ACCENT), 13)
        cr = x0 - u(20)
    line, rest = decision(a)
    tx = sep + u(20)
    tw = cr - tx
    if a.get("pending") and not line:
        line = "Weighing the fixed detector against the sniffer…"
    c.text((tx, cy1 if rest else y + h / 2), fit(c.d, line or "Monitoring", c.f("sans", 16, 500), tw), "sans", 16,
           500, LABEL, "lm")
    if rest:
        c.text((tx, cy2), fit(c.d, rest, c.f("sans", 13), tw), "sans", 13, 400, LABEL2, "lm")


def label_points(labels) -> list[tuple]:
    out = []
    for key, v in (labels or {}).items():
        if v is None:
            continue
        if isinstance(v, dict):
            if v.get("visible") is False:
                continue
            v = v.get("xy") or v.get("px")
            if v is None:
                continue
        if len(v) < 2 or not all(np.isfinite(float(t)) for t in v[:2]):
            continue
        out.append((str(key), float(v[0]), float(v[1])))
    return out


def floating_labels(c: Canvas, header: dict, labels):
    u = c.u
    robots = header.get("robots") or {}
    names = {rid: (tag, name) for rid, tag, name in UNITS}
    dets = {str(d.get("id")): d for d in header.get("detectors") or []}
    for key, px, py in label_points(labels):
        if not (0 <= px < c.W and 0 <= py < c.H):
            continue
        if key in names:
            tag, name = names[key]
            r = robots.get(key) or {}
            title, sub = f"{tag} · {name}", str(r.get("state") or "").upper().replace("_", " ")
            tone = robot_tone(r)
        elif key in dets:
            dd = dets[key]
            title, sub = key, f"{num(dd.get('lel'))} %LEL"
            tone = LED.get(str(dd.get("led") or "grey"), GREY)
        else:
            title, sub, tone = key, "", ACCENT
        fb, fs = c.f("sans", 13, 600), c.f("mono", 11, 500)
        tw = max(c.d.textlength(title, font=fb) + u(14), c.d.textlength(sub, font=fs) if sub else 0)
        bw, bh = tw + u(22), u(44 if sub else 28)
        stem = u(18)
        best = None
        for dy in (-1, 1):
            bx = min(max(px - bw / 2, u(8)), c.W - bw - u(8))
            by = py - stem - bh if dy < 0 else py + stem
            by = min(max(by, u(8)), c.H - bh - u(8))
            box = (bx, by, bx + bw, by + bh)
            if not any(box[0] < r[2] and box[2] > r[0] and box[1] < r[3] and box[3] > r[1] for r in c.rects):
                best = (box, dy)
                break
        if best is None:
            continue
        (bx, by, bx2, by2), dy = best
        c.d.line((px, py, px, by2 if dy < 0 else by), fill=(255, 255, 255, 120), width=max(1, u(1)))
        c.dot(px, py, 3, (255, 255, 255))
        c.d.rounded_rectangle((bx, by, bx2, by2), u(10), fill=(16, 16, 20, 190), outline=HAIR)
        ty = by + (u(15) if sub else bh / 2)
        c.dot(bx + u(14), ty, 3.5, tone)
        c.d.text((bx + u(25), ty), title, font=fb, fill=LABEL, anchor="lm")
        if sub:
            c.d.text((bx + u(11), by + u(31)), sub, font=fs, fill=rgba(tone, 230), anchor="lm")
        c.rects.append((bx, by, bx2, by2))


def normalize(header: Optional[dict]) -> dict:
    header = dict(header or {})
    robots = header.get("robots") or {}
    if isinstance(robots, list):
        robots = {str(r.get("id")): r for r in robots if isinstance(r, dict)}
    header["robots"] = robots
    a = header.get("argon")
    if a and header.get("source") and not (a.get("last_command") or {}).get("source"):
        header["argon"] = {**a, "source": str(header["source"]).lower()}
    return header


def with_force_history(rows: list[dict], window_s: float = 8.0, n: int = 64) -> list[dict]:
    trail: list[tuple] = []
    out = []
    for r in rows:
        r = normalize(r)
        u1 = r["robots"].get("unit1")
        if isinstance(u1, dict) and "force_hist" not in u1 and isinstance(u1.get("force_n"), (int, float)):
            t = float(r.get("t", 0.0))
            trail = [p for p in trail if p[0] > t - window_s] + [(t, float(u1["force_n"]))]
            r["robots"] = {**r["robots"], "unit1": {**u1, "force_hist": [v for _, v in trail][-n:]}}
        out.append(r)
    return out


def draw(frame_rgb, header: dict, labels: Optional[dict] = None):
    as_array = isinstance(frame_rgb, np.ndarray)
    img = Image.fromarray(np.asarray(frame_rgb, np.uint8)[..., :3]) if as_array else frame_rgb.convert("RGB")
    img = img.copy()
    c = Canvas(img)
    header = normalize(header)
    robots = header.get("robots") or {}
    edge, gap, cw = c.u(32), c.u(12), c.u(380)
    y = edge
    for rid, tag, name in UNITS:
        if rid in robots:
            y = robot_card(c, edge, y, cw, rid, tag, name, robots[rid] or {}, rid == "unit1") + gap
    if header.get("detectors"):
        gas_panel(c, c.W - edge - c.u(400), edge, c.u(400), header)
    argon_strip(c, header)
    floating_labels(c, header, labels)
    return np.asarray(c.base) if as_array else c.base


def load_headers(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            if isinstance(r, dict) and ("robots" in r or "argon" in r or "detectors" in r):
                rows.append(r)
    rows.sort(key=lambda r: float(r.get("t", 0.0)))
    return with_force_history(rows)


def header_at(rows: list[dict], t: float) -> dict:
    cur = {}
    for r in rows:
        if float(r.get("t", 0.0)) > t:
            break
        cur = r
    return cur


def _det(i, lel, led_, did=None):
    return {"id": did or f"G-F{i}", "flange": f"F{i}", "lel": lel, "led": led_}


def _cmd(kind, robot, target, source, reason, finding="", t=0.0):
    return {"kind": kind, "robot": robot, "target": target, "source": source, "reason": reason, "confidence": 0.9,
            "finding": finding, "t": t}


def _argon(cmd=None, link="UP", status="ok", pending=False, held=False, verdicts=None, tickets=(), lat=None,
           spark=()):
    return {"model": "gemini-4-argon", "link": link, "status": status, "pending": pending, "held": held,
            "spark": list(spark), "verdicts": verdicts or {}, "tickets": list(tickets), "last_command": cmd,
            "latency_ms": lat, "cost_usd": 0.0012}


def _force(seed, n=48, mean=34.0, spike=False):
    rng = np.random.default_rng(seed)
    v = mean + 3.5 * np.sin(np.linspace(0, 7, n)) + rng.normal(0, 1.4, n)
    if spike:
        v[-9:-5] += 9.0
    return [round(float(x), 1) for x in v]


def sample_beats() -> list[dict]:
    u1 = lambda state, cap, rust, gas=0.3, force=True, spike=False: {
        "state": state, "caption": cap, "battery": 0.82, "lel": gas, "rust": rust,
        "force_hist": _force(3, spike=spike) if force else [0.0] * 8}
    u2 = lambda state, cap, gas=0.1, batt=0.94: {"state": state, "caption": cap, "battery": batt, "lel": gas}
    fd = lambda sensor, kind, fixed, robot, rl, dist, hazard=False: {
        "kind": kind, "sensor": sensor, "flange": sensor[2:], "fixed_lel": fixed, "robot_lel": rl, "robot_dist": dist,
        "robot": robot, "hazard": hazard, "reason": ""}
    return [
        {"beat": "1_scrub", "t": 4.0,
         "robots": {"unit1": u1("SCRUB", "Scrubbing rust on the main pipe, pass 2 of 3", 0.58),
                    "unit2": u2("DOCKED", "Charging at the deck dock")},
         "detectors": [_det(1, 0.2, "green"), _det(2, 0.4, "green"), _det(3, 0.1, "green")],
         "findings": [fd("G-F1", "OK", 0.2, "unit1", 0.3, 7.9), fd("G-F2", "OK", 0.4, "unit1", 0.3, 4.1),
                      fd("G-F3", "OK", 0.1, "unit2", 0.1, 3.0)],
         "argon": _argon(_cmd("resume_rust", "unit1", "", "argon", "Deck clear, rust job running | All three "
                                                                    "detectors green and both sniffers clean."),
                         verdicts={"G-F1": "OK", "G-F2": "OK", "G-F3": "OK"}, lat=5400.0)},
        {"beat": "2_alarm", "t": 11.0,
         "robots": {"unit1": u1("SCRUB", "Scrubbing rust on the main pipe, pass 2 of 3", 0.47, 1.1, spike=True),
                    "unit2": u2("DOCKED", "Charging at the deck dock")},
         "detectors": [_det(1, 0.2, "green"), _det(2, 13.6, "red"), _det(3, 0.3, "green")],
         "findings": [fd("G-F1", "OK", 0.2, "unit1", 1.1, 7.9), fd("G-F2", "UNVERIFIED", 13.6, "unit1", 1.1, 4.1),
                      fd("G-F3", "OK", 0.3, "unit2", 0.1, 3.0)],
         "argon": _argon(None, status="pending", pending=True, verdicts={"G-F2": "UNVERIFIED"}, spark=["G-F2"])},
        {"beat": "3_hold_dispatch", "t": 17.0,
         "robots": {"unit1": u1("HOLD", "Spark rule: brush lifted, stepped back 2 m, rust job paused", 0.47, 1.0,
                                force=False),
                    "unit2": u2("WALK", "Dispatched to flange F2 to verify")},
         "detectors": [_det(1, 0.2, "green"), _det(2, 14.2, "red"), _det(3, 0.3, "green")],
         "findings": [fd("G-F1", "OK", 0.2, "unit1", 1.0, 8.2), fd("G-F2", "UNVERIFIED", 14.2, "unit1", 1.0, 4.4),
                      fd("G-F3", "OK", 0.3, "unit2", 0.2, 5.6)],
         "argon": _argon(_cmd("go_check", "unit2", "F2", "argon",
                              "G-F2 alarms, Unit 1 too far to confirm: send Unit 2 | G-F2 reads 14.2 %LEL while "
                              "Unit 1's sniffer reads 1.0 %LEL at 4.4 m, outside the 2 m confirm radius.",
                              "UNVERIFIED"),
                         held=True, verdicts={"G-F2": "UNVERIFIED"}, lat=6020.0, spark=["G-F2"])},
        {"beat": "4_confirm_fix", "t": 29.0,
         "robots": {"unit1": u1("HOLD", "Holding clear of F2, rust job paused at 47 %", 0.47, 0.9, force=False),
                    "unit2": u2("TORQUE", "Re-torquing the loose bolt at F2, 3 of 4", 11.8, 0.88)},
         "detectors": [_det(1, 0.2, "green"), _det(2, 12.9, "red"), _det(3, 0.3, "green")],
         "findings": [fd("G-F1", "OK", 0.2, "unit1", 0.9, 8.2), fd("G-F2", "LEAK_CONFIRMED", 12.9, "unit2", 11.8, 0.7),
                      fd("G-F3", "OK", 0.3, "unit2", 0.4, 6.1)],
         "argon": _argon(_cmd("fix_leak", "unit2", "F2", "argon",
                              "Leak confirmed at F2: re-torque the loose bolt | G-F2 12.9 %LEL and Unit 2 11.8 "
                              "%LEL at 0.7 m agree.", "LEAK_CONFIRMED"),
                         held=True, verdicts={"G-F2": "LEAK_CONFIRMED"}, lat=4870.0)},
        {"beat": "5_resume", "t": 44.0,
         "robots": {"unit1": u1("SCRUB", "Resumed: final pass, then paint", 0.12),
                    "unit2": u2("RETURN", "Bolt torqued, returning to dock", 0.3, 0.81)},
         "detectors": [_det(1, 0.2, "green"), _det(2, 0.9, "green"), _det(3, 0.3, "green")],
         "findings": [fd("G-F1", "OK", 0.2, "unit1", 0.3, 7.9), fd("G-F2", "OK", 0.9, "unit2", 0.8, 1.6),
                      fd("G-F3", "OK", 0.3, "unit2", 0.3, 4.0)],
         "argon": _argon(_cmd("resume_rust", "unit1", "", "argon",
                              "Zone clear: Unit 1 resumes the rust job | G-F2 back to 0.9 %LEL and Unit 2 reads "
                              "0.8 %LEL beside the flange."), verdicts={"G-F2": "OK"}, lat=5120.0)},
        {"beat": "6_sensor_fault", "t": 51.0,
         "robots": {"unit1": u1("PAINT", "Painting the scrubbed section", 0.0, force=False),
                    "unit2": u2("RETURN", "Passing F1 on the way to dock", 0.2, 0.79)},
         "detectors": [_det(1, 24.0, "grey"), _det(2, 0.6, "green"), _det(3, 0.2, "green")],
         "findings": [fd("G-F1", "PIPE_SENSOR_FAULT", 24.0, "unit2", 0.2, 0.9), fd("G-F2", "OK", 0.6, "unit1", 0.3, 5.0),
                      fd("G-F3", "OK", 0.2, "unit2", 0.2, 6.3)],
         "argon": _argon(_cmd("log_sensor_fault", "supv", "G-F1", "argon",
                              "G-F1 stuck high, Unit 2 reads clean beside it: ticket, no repair | G-F1 24.0 %LEL "
                              "but Unit 2 reads 0.2 %LEL at 0.9 m.", "PIPE_SENSOR_FAULT"),
                         verdicts={"G-F1": "PIPE_SENSOR_FAULT"}, tickets=["MT-001 G-F1"], lat=5580.0)},
        {"beat": "7_link_cut", "t": 56.0,
         "robots": {"unit1": u1("HOLD", "Edge rule hold: G-F2 amber within 6 m", 0.0, 0.8, force=False),
                    "unit2": u2("WALK", "Edge dispatch to F2", 0.3, 0.77)},
         "detectors": [_det(1, 0.3, "green"), _det(2, 11.2, "red"), _det(3, 0.2, "green")],
         "findings": [fd("G-F1", "OK", 0.3, "unit2", 0.3, 4.4), fd("G-F2", "UNVERIFIED", 11.2, "unit1", 0.8, 4.6),
                      fd("G-F3", "OK", 0.2, "unit2", 0.2, 6.0)],
         "argon": _argon(_cmd("go_check", "unit2", "F2", "edge",
                              "edge rule UNVERIFIED: G-F2 11.2%LEL, robot 0.8%LEL at 4.6 m: too far to confirm",
                              "UNVERIFIED"), link="CUT", status="cut", held=True,
                         verdicts={"G-F2": "UNVERIFIED"})},
    ]


SAMPLE_LABELS = {"unit1": (470 / 1280, 592 / 720), "unit2": (905 / 1280, 560 / 720), "G-F2": (640 / 1280, 622 / 720)}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default="sim/output/full_rig_on_deck.png")
    ap.add_argument("--headers", default="")
    ap.add_argument("--out", default="sim/output/fleet_overlay")
    ap.add_argument("--sizes", default="1280x720,1920x1080")
    ap.add_argument("--at", default="", help="comma-separated times to render from --headers")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.headers:
        hs = load_headers(a.headers)
        ts = [float(v) for v in a.at.split(",")] if a.at else list(np.arange(0.0, float(hs[-1]["t"]) + 1e-6, 5.0))
        rows = [header_at(hs, t) for t in ts]
    else:
        rows = sample_beats()
        (out / "fake_timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        rows = load_headers(out / "fake_timeline.jsonl")
    still = Image.open(a.still).convert("RGB")
    for size in a.sizes.split(","):
        W, H = (int(v) for v in size.split("x"))
        frame = np.asarray(still.resize((W, H), Image.LANCZOS))
        for r in rows:
            labels = {k: (x * W, y * H) for k, (x, y) in SAMPLE_LABELS.items()}
            name = r.get("beat") or f"t{float(r.get('t', 0)):06.2f}"
            p = out / f"{name}_{H}p.png"
            Image.fromarray(draw(frame, r, labels)).save(p)
            print(p)


if __name__ == "__main__":
    main()

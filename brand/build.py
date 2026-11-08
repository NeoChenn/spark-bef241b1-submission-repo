"""Spark brand kit generator.

One geometry, two outputs: every graphic is described once as a list of
primitives, then written as a hand-readable SVG and rendered to PNG with PIL
(no SVG rasteriser is needed). Re-run after changing tokens or layouts:

    C:\\isaacsim\\python.bat brand\\build.py
    C:\\isaacsim\\python.bat brand\\build.py caption "Dog-1 closes the valve" "DECK B  00:42" out.png

PNG text uses Windows stand-ins for the brand fonts (Bahnschrift for Space
Grotesk, Segoe UI for Inter, Consolas for JetBrains Mono). The SVGs name the
real brand fonts (Google Fonts, OFL). The SPARK wordmark and the mark are
pure paths, so they are identical in both.
"""
from __future__ import annotations

import json
import math
import os
import sys
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "logo")
VIDEO = os.path.join(HERE, "video")

# ---------------------------------------------------------------- tokens
T = {
    "bg": "#0B0D10",
    "surface": "#15181D",
    "line": "#262A31",
    "text": "#F3F5F8",
    "muted": "#9AA3AE",
    "accent": "#2EE6D6",
    "accent2": "#0FA3A3",
    "warning": "#FFB020",
    "stop": "#FF4D4F",
    "success": "#3DDC84",
    "dog1": "#2EE6D6",
    "dog2": "#FFB020",
    "drone": "#F3F5F8",
    "crawler": "#A78BFA",
}

FONT_STACK = {
    "display": "'Space Grotesk', 'Bahnschrift', 'Segoe UI', sans-serif",
    "body": "'Inter', 'Segoe UI', sans-serif",
    "mono": "'JetBrains Mono', 'Consolas', monospace",
}
GOOGLE_FONTS = ("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600"
                "&amp;family=Inter:wght@400;500;600&amp;family=JetBrains+Mono:wght@500;600&amp;display=swap")

# ---------------------------------------------------------------- geometry
def offset_polyline(pts, half):
    """Outline of a constant-width stroke along pts, mitred joins, butt caps."""
    n = len(pts)
    normals = []
    for i in range(n - 1):
        dx, dy = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        L = math.hypot(dx, dy)
        normals.append((-dy / L, dx / L))

    def side(s):
        out = []
        for i, p in enumerate(pts):
            if i == 0:
                nx, ny = normals[0]
                out.append((p[0] + s * half * nx, p[1] + s * half * ny))
            elif i == n - 1:
                nx, ny = normals[-1]
                out.append((p[0] + s * half * nx, p[1] + s * half * ny))
            else:
                a, b = normals[i - 1], normals[i]
                k = 1 + a[0] * b[0] + a[1] * b[1]
                mx, my = (a[0] + b[0]) / k, (a[1] + b[1]) / k
                out.append((p[0] + s * half * mx, p[1] + s * half * my))
        return out

    return side(1) + side(-1)[::-1]


# SPARK letters: monoline strokes on a 72 x 100 box, stroke 16, one 45-degree
# chamfer per letter and one shared diagonal (the R leg = the K arms).
STROKE, CH = 16, 16
H8 = STROKE / 2
GLYPHS = {
    "S": [[(72, 8), (8 + CH, 8), (8, 8 + CH), (8, 50), (64, 50), (64, 92 - CH), (64 - CH, 92), (0, 92)]],
    "P": [[(8, 100), (8, 8 + CH), (8 + CH, 8), (64, 8), (64, 50), (40, 50)], [(8, 50), (48, 50)]],
    "A": [[(8, 100), (8, 8 + CH), (8 + CH, 8), (64, 8), (64, 100)], [(8, 60), (64, 60)]],
    "R": [[(8, 100), (8, 8 + CH), (8 + CH, 8), (64, 8), (64, 50), (40, 50)], [(8, 50), (48, 50)],
          [(40, 50), (64, 74), (64, 100)]],
    "K": [[(8, 0), (8, 100)], [(8, 50), (40, 50), (64, 26), (64, 0)], [(40, 50), (64, 74), (64, 100)]],
}
GLYPH_W, GLYPH_GAP = 72, 22
WORD_W = 5 * GLYPH_W + 4 * GLYPH_GAP  # 448 units for 100 units of cap height

# The mark: a spark arc of straight segments (100-unit box, 62 wide).
BOLT = [(41, 0), (73, 0), (53, 40), (81, 40), (35, 100), (47, 58), (19, 58)]
BOLT_X0, BOLT_W = 19, 62


def wordmark(x, y, cap, fill):
    s = cap / 100.0
    items = []
    for i, ch in enumerate("SPARK"):
        ox = x + i * (GLYPH_W + GLYPH_GAP) * s
        for line in GLYPHS[ch]:
            poly = offset_polyline(line, H8)
            items.append(("poly", dict(pts=[(ox + px * s, y + py * s) for px, py in poly], fill=fill)))
    return items


def mark(x, y, size, fill):
    s = size / 100.0
    return [("poly", dict(pts=[(x + (px - BOLT_X0) * s, y + py * s) for px, py in BOLT], fill=fill))]


def mark_w(size):
    return BOLT_W * size / 100.0


def lockup(x, y, cap, text_fill, mark_fill):
    """Horizontal lockup. Mark is 1.3x cap height, centred on the caps."""
    msize = cap * 1.3
    gap = cap * 0.42
    items = mark(x, y - (msize - cap) / 2, msize, mark_fill)
    items += wordmark(x + mark_w(msize) + gap, y, cap, text_fill)
    return items


def lockup_w(cap):
    return mark_w(cap * 1.3) + cap * 0.42 + WORD_W * cap / 100.0


# ---------------------------------------------------------------- fonts (PNG side)
_fc = {}


def font(role, size, bold=False):
    key = (role, size, bold)
    if key in _fc:
        return _fc[key]
    if role == "display":
        f = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", size)
        f.set_variation_by_name(b"Bold" if bold else b"SemiBold")
    elif role == "body":
        f = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf", size)
    else:
        f = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf", size)
    _fc[key] = f
    return f


def measure(s, role, size, ls=0, bold=False):
    f = font(role, size, bold)
    return sum(f.getlength(c) for c in s) + ls * max(0, len(s) - 1) if ls else f.getlength(s)


def text(x, y, s, role, size, fill, anchor="start", ls=0, bold=False, op=1.0):
    return ("text", dict(x=x, y=y, s=s, role=role, size=size, fill=fill, anchor=anchor, ls=ls, bold=bold, op=op))


def rect(x, y, w, h, rx=0, fill=None, stroke=None, sw=1, op=1.0, dash=None):
    return ("rect", dict(x=x, y=y, w=w, h=h, rx=rx, fill=fill, stroke=stroke, sw=sw, op=op, dash=dash))


def line(x1, y1, x2, y2, stroke, sw=1, op=1.0):
    return ("line", dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, sw=sw, op=op))


def circle(cx, cy, r, fill=None, stroke=None, sw=1, op=1.0):
    return ("circle", dict(cx=cx, cy=cy, r=r, fill=fill, stroke=stroke, sw=sw, op=op))


# ---------------------------------------------------------------- SVG writer
def f2(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def to_svg(w, h, items, bg=None, fonts=True, title="Spark"):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{f2(w)}" height="{f2(h)}" viewBox="0 0 {f2(w)} {f2(h)}">',
           f"  <title>{escape(title)}</title>"]
    if fonts and any(k == "text" for k, _ in items):
        out.append(f"  <style>@import url('{GOOGLE_FONTS}');</style>")
    if bg:
        out.append(f'  <rect width="100%" height="100%" fill="{bg}"/>')
    polys = []

    def flush():
        if polys:
            fill = polys[0][1]
            # one subpath per stroke; strokes overlap, so each is its own path (no winding holes)
            out.append(f'  <g fill="{fill}">')
            for p, _ in polys:
                d = "M" + " L".join(f"{f2(px)} {f2(py)}" for px, py in p) + " Z"
                out.append(f'    <path d="{d}"/>')
            out.append("  </g>")
            polys.clear()

    for kind, a in items:
        if kind == "poly":
            if polys and polys[0][1] != a["fill"]:
                flush()
            polys.append((a["pts"], a["fill"]))
            continue
        flush()
        op = f' opacity="{f2(a["op"])}"' if a.get("op", 1) < 1 else ""
        if kind == "rect":
            fill = a["fill"] or "none"
            st = f' stroke="{a["stroke"]}" stroke-width="{f2(a["sw"])}"' if a["stroke"] else ""
            if a["dash"]:
                st += f' stroke-dasharray="{a["dash"][0]} {a["dash"][1]}"'
            rx = f' rx="{f2(a["rx"])}"' if a["rx"] else ""
            # hairlines sit on the pixel grid: inset strokes by half their width
            o = a["sw"] / 2 if a["stroke"] else 0
            out.append(f'  <rect x="{f2(a["x"] + o)}" y="{f2(a["y"] + o)}" width="{f2(a["w"] - 2 * o)}" '
                       f'height="{f2(a["h"] - 2 * o)}"{rx} fill="{fill}"{st}{op}/>')
        elif kind == "line":
            out.append(f'  <line x1="{f2(a["x1"])}" y1="{f2(a["y1"])}" x2="{f2(a["x2"])}" y2="{f2(a["y2"])}" '
                       f'stroke="{a["stroke"]}" stroke-width="{f2(a["sw"])}"{op}/>')
        elif kind == "circle":
            fill = a["fill"] or "none"
            st = f' stroke="{a["stroke"]}" stroke-width="{f2(a["sw"])}"' if a["stroke"] else ""
            out.append(f'  <circle cx="{f2(a["cx"])}" cy="{f2(a["cy"])}" r="{f2(a["r"])}" fill="{fill}"{st}{op}/>')
        elif kind == "text":
            weight = {"display": 600, "body": 600 if a["bold"] else 400, "mono": 600 if a["bold"] else 500}[a["role"]]
            ls = f' letter-spacing="{f2(a["ls"])}"' if a["ls"] else ""
            out.append(f'  <text x="{f2(a["x"])}" y="{f2(a["y"])}" font-family="{FONT_STACK[a["role"]]}" '
                       f'font-size="{a["size"]}" font-weight="{weight}" fill="{a["fill"]}" '
                       f'text-anchor="{a["anchor"]}"{ls}{op}>{escape(a["s"])}</text>')
    flush()
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- PNG renderer
SS = 3  # supersampling


def rgba(hexs, op=1.0):
    hexs = hexs.lstrip("#")
    return tuple(int(hexs[i:i + 2], 16) for i in (0, 2, 4)) + (int(round(255 * op)),)


def to_png(w, h, items, bg=None, path=None):
    W, H = int(round(w * SS)), int(round(h * SS))
    canvas = Image.new("RGBA", (W, H), rgba(bg) if bg else (0, 0, 0, 0))
    for kind, a in items:
        op = a.get("op", 1.0)
        direct = op >= 1 and kind != "text"  # opaque shapes go straight onto the canvas
        layer = canvas if direct else Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if kind == "poly":
            d.polygon([(px * SS, py * SS) for px, py in a["pts"]], fill=rgba(a["fill"]))
        elif kind == "rect":
            x0, y0, x1, y1 = a["x"] * SS, a["y"] * SS, (a["x"] + a["w"]) * SS - 1, (a["y"] + a["h"]) * SS - 1
            r = a["rx"] * SS
            if a["fill"]:
                d.rounded_rectangle((x0, y0, x1, y1), r, fill=rgba(a["fill"]))
            if a["stroke"]:
                sw = max(1, int(round(a["sw"] * SS)))
                if not a["dash"]:
                    d.rounded_rectangle((x0, y0, x1, y1), r, outline=rgba(a["stroke"]), width=sw)
                else:
                    _dashed_round_rect(d, x0, y0, x1, y1, r, rgba(a["stroke"]), sw, a["dash"])
        elif kind == "line":
            sw = max(1, int(round(a["sw"] * SS)))
            x1, y1, x2, y2 = a["x1"] * SS, a["y1"] * SS, a["x2"] * SS, a["y2"] * SS
            if x1 == x2:
                d.rectangle((x1 - sw / 2, min(y1, y2), x1 + sw / 2 - 1, max(y1, y2) - 1), fill=rgba(a["stroke"]))
            elif y1 == y2:
                d.rectangle((min(x1, x2), y1 - sw / 2, max(x1, x2) - 1, y1 + sw / 2 - 1), fill=rgba(a["stroke"]))
            else:
                d.line((x1, y1, x2, y2), fill=rgba(a["stroke"]), width=sw)
        elif kind == "circle":
            cx, cy, r = a["cx"] * SS, a["cy"] * SS, a["r"] * SS
            if a["fill"]:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba(a["fill"]))
            if a["stroke"]:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=rgba(a["stroke"]),
                          width=max(1, int(round(a["sw"] * SS))))
        elif kind == "text":
            f = font(a["role"], a["size"] * SS, a["bold"])
            ls = a["ls"] * SS
            s = a["s"]
            tw = sum(f.getlength(c) for c in s) + ls * (len(s) - 1)
            x = a["x"] * SS - {"start": 0, "middle": tw / 2, "end": tw}[a["anchor"]]
            y = a["y"] * SS
            if ls:
                for c in s:
                    d.text((x, y), c, font=f, fill=rgba(a["fill"]), anchor="ls")
                    x += f.getlength(c) + ls
            else:
                d.text((x, y), s, font=f, fill=rgba(a["fill"]), anchor="ls")
        if direct:
            continue
        if op < 1:
            alpha = layer.getchannel("A").point(lambda v: int(v * op))
            layer.putalpha(alpha)
        canvas.alpha_composite(layer)
    img = canvas.resize((int(round(w)), int(round(h))), Image.LANCZOS)
    if path:
        img.save(path)
    return img


def _dashed_round_rect(d, x0, y0, x1, y1, r, col, sw, dash):
    on, off = dash[0] * SS, dash[1] * SS
    for (ax, ay, bx, by) in ((x0 + r, y0, x1 - r, y0), (x0 + r, y1 - sw + 1, x1 - r, y1 - sw + 1),
                             (x0, y0 + r, x0, y1 - r), (x1 - sw + 1, y0 + r, x1 - sw + 1, y1 - r)):
        L = math.hypot(bx - ax, by - ay)
        t = 0
        while t < L:
            t2 = min(L, t + on)
            if ay == by:
                d.rectangle((ax + t, ay, ax + t2, ay + sw - 1), fill=col)
            else:
                d.rectangle((ax, ay + t, ax + sw - 1, ay + t2), fill=col)
            t = t2 + off
    for (bx0, by0, start) in ((x0, y0, 180), (x1 - 2 * r, y0, 270), (x1 - 2 * r, y1 - 2 * r, 0), (x0, y1 - 2 * r, 90)):
        d.arc((bx0, by0, bx0 + 2 * r, by0 + 2 * r), start, start + 90, fill=col, width=sw)


def emit(name, folder, w, h, items, bg=None, title="Spark", png=True):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".svg"), "w", encoding="utf-8") as fh:
        fh.write(to_svg(w, h, items, bg, title=title))
    if png:
        return to_png(w, h, items, bg, os.path.join(folder, name + ".png"))


# ---------------------------------------------------------------- logo files
def build_logos():
    # glyph only
    emit("spark-mark", LOGO, 62, 100, mark(0, 0, 100, T["accent"]), title="Spark mark", png=False)
    to_png(62 * 5.12, 512, mark(0, 0, 512, T["accent"]), path=os.path.join(LOGO, "spark-mark-512.png"))
    # wordmark, inherits colour from CSS (currentColor) when inlined
    emit("spark-wordmark", LOGO, WORD_W, 100, wordmark(0, 0, 100, "currentColor"), title="Spark wordmark", png=False)
    # horizontal lockup (neutral: currentColor text, spark teal mark)
    cap = 100
    msize = cap * 1.3
    pad = (msize - cap) / 2
    W = lockup_w(cap)
    emit("spark-lockup-horizontal", LOGO, W, msize,
         lockup(0, pad, cap, "currentColor", T["accent"]), title="Spark lockup", png=False)
    emit("spark-lockup-dark", LOGO, W, msize, lockup(0, pad, cap, T["text"], T["accent"]),
         title="Spark lockup for dark backgrounds", png=False)
    emit("spark-lockup-light", LOGO, W, msize, lockup(0, pad, cap, T["bg"], T["accent2"]),
         title="Spark lockup for light backgrounds", png=False)
    # PNG previews of the lockups at 1200 px wide, on their intended backgrounds plus transparent
    s = 1200 / W
    for nm, tf, mf, bgc in (("dark", T["text"], T["accent"], T["bg"]), ("light", T["bg"], T["accent2"], T["text"])):
        to_png(W * s, msize * s, lockup(0, pad * s, cap * s, tf, mf),
               path=os.path.join(LOGO, f"spark-lockup-{nm}.png"))
    # favicon: spark on a dark 10/32-radius tile
    fav = favicon_items(32)
    emit("spark-favicon", LOGO, 32, 32, fav, title="Spark favicon", png=False)
    for px in (16, 32, 180, 512):
        to_png(px, px, favicon_items(px), path=os.path.join(LOGO, f"spark-favicon-{px}.png"))


def favicon_items(px):
    k = px / 32.0
    size = 23 * k
    items = [rect(0, 0, px, px, rx=7 * k, fill=T["bg"])]
    if px >= 64:
        items.append(rect(0, 0, px, px, rx=7 * k, stroke=T["line"], sw=max(1, k / 2)))
    return items + mark((px - mark_w(size)) / 2, (px - size) / 2, size, T["accent"])


# ---------------------------------------------------------------- video graphics
VW, VH = 1920, 1080
M = 96  # safe margin


def frame_marks(op=1.0):
    """Registration ticks in the four corners plus footer meta - the card chrome."""
    items = []
    L = 28
    for (x, y, sx, sy) in ((M, M, 1, 1), (VW - M, M, -1, 1), (M, VH - M, 1, -1), (VW - M, VH - M, -1, -1)):
        items.append(line(x, y, x + sx * L, y, T["muted"], 1.5, 0.55 * op))
        items.append(line(x, y, x, y + sy * L, T["muted"], 1.5, 0.55 * op))
    return items


def card_base(index=None, total=7):
    items = frame_marks()
    items += mark(M + 48, M + 40, 40, T["accent"])
    items += wordmark(M + 48 + mark_w(40) + 14, M + 51, 18, T["muted"])
    items.append(text(VW - M - 48, M + 72, "OFFSHORE AUTONOMY", "mono", 20, T["muted"], "end", ls=5))
    if index is not None:
        items.append(text(VW - M - 48, VH - M - 48, f"{index:02d} / {total:02d}", "mono", 20, T["muted"], "end", ls=4))
    return items


ACTS = ["The problem", "Meet the fleet", "It sees", "It works", "Something goes wrong", "Safe offline", "Done"]
ACT_COL = {5: "warning", 7: "success"}


def build_cards():
    # title card
    cap = 120
    lw = lockup_w(cap)
    items = card_base()
    items += lockup((VW - lw) / 2, 420, cap, T["text"], T["accent"])
    items.append(line((VW - lw) / 2, 616, (VW + lw) / 2, 616, T["line"], 1))
    items.append(text(VW / 2, 690, "Autonomous robot fleet for offshore rigs", "body", 40, T["muted"], "middle"))
    emit("title_card", VIDEO, VW, VH, items, bg=T["bg"], title="Spark title card")

    # act cards
    for i, label in enumerate(ACTS, 1):
        col = T[ACT_COL.get(i, "accent")]
        x = M + 48
        items = card_base(i, len(ACTS))
        items.append(rect(x, 452, 6, 172, fill=col))
        items.append(text(x + 40, 492, f"ACT {i:02d}", "mono", 26, col, ls=6, bold=True))
        items.append(text(x + 36, 612, label, "display", 112, T["text"], ls=-1))
        items.append(line(x, 708, VW - M - 48, 708, T["line"], 1))
        # progress ticks: one per act, current one lit
        for k in range(len(ACTS)):
            tx = x + k * 64
            items.append(rect(tx, 738, 48, 4, rx=2, fill=col if k + 1 == i else T["line"]))
        slug = label.lower().replace(" ", "_")
        emit(f"act_{i:02d}_{slug}", VIDEO, VW, VH, items, bg=T["bg"], title=f"Act {i}: {label}")

    # end card
    items = card_base()
    items.append(text(VW / 2, 470, "Inspect. Repair. Isolate.", "display", 104, T["text"], "middle", ls=-1))
    items.append(text(VW / 2, 600, "Autonomously.", "display", 104, T["accent"], "middle", ls=-1))
    cap = 56
    lw = lockup_w(cap)
    items.append(line(VW / 2 - 240, 700, VW / 2 + 240, 700, T["line"], 1))
    items += lockup((VW - lw) / 2, 776, cap, T["text"], T["accent"])
    emit("end_card", VIDEO, VW, VH, items, bg=T["bg"], title="Spark end card")


def lower_third_items(title, sub, accent=None):
    accent = accent or T["accent"]
    tw = max(measure(title, "display", 44), measure(sub, "mono", 22, ls=2)) + 96
    w, h = max(640, tw), 132
    x, y = M, VH - M - h
    return [rect(x, y, w, h, rx=10, fill=T["surface"], op=0.92),
            rect(x, y, w, h, rx=10, stroke=T["line"], sw=1),
            rect(x + 24, y + 28, 4, h - 56, rx=2, fill=accent),
            text(x + 52, y + 66, title, "display", 44, T["text"]),
            text(x + 52, y + 104, sub, "mono", 22, T["muted"], ls=2)]


def tag_items(x, y, label, kind):
    """Small status tag. kind: 'policy' (solid: in control) or 'advisor' (dashed: advises only)."""
    h = 44
    tw = measure(label, "mono", 20, ls=2, bold=True)
    w = tw + 66
    if kind == "policy":
        col = T["accent"]
        return w, h, [rect(x, y, w, h, rx=10, fill=T["bg"], op=0.85),
                      rect(x, y, w, h, rx=10, fill=col, op=0.12),
                      rect(x, y, w, h, rx=10, stroke=col, sw=1.5),
                      rect(x + 20, y + h / 2 - 5, 10, 10, fill=col),
                      text(x + 46, y + 29, label, "mono", 20, col, ls=2, bold=True)]
    col = T["text"]
    return w, h, [rect(x, y, w, h, rx=10, fill=T["surface"], op=0.9),
                  rect(x, y, w, h, rx=10, stroke=T["muted"], sw=1.5, dash=(6, 5)),
                  circle(x + 25, y + h / 2, 5.5, stroke=col, sw=2),
                  text(x + 46, y + 29, label, "mono", 20, col, ls=2, bold=True)]


ROBOTS = [("Dog-1", "dog1", "QUADRUPED + ARM"), ("Dog-2", "dog2", "QUADRUPED + ARM"),
          ("Drone", "drone", "AERIAL"), ("Crawler", "crawler", "PIPE CRAWLER")]


def name_tag_items(x, y, name, colkey, role):
    col = T[colkey]
    h = 56
    w = 40 + measure(name, "body", 26, bold=True) + 18 + measure(role, "mono", 17, ls=2) + 24
    return w, h, [rect(x, y, w, h, rx=10, fill=T["surface"], op=0.92),
                  rect(x, y, w, h, rx=10, stroke=T["line"], sw=1),
                  rect(x + 18, y + h / 2 - 6, 12, 12, fill=col),
                  text(x + 40, y + 37, name, "body", 26, T["text"], bold=True),
                  text(x + 40 + measure(name, "body", 26, bold=True) + 18, y + 36, role, "mono", 17, T["muted"], ls=2)]


def robot_view_items(robot=None):
    items = []
    L, inset = 56, 48
    for (x, y, sx, sy) in ((inset, inset, 1, 1), (VW - inset, inset, -1, 1),
                           (inset, VH - inset, 1, -1), (VW - inset, VH - inset, -1, -1)):
        items.append(line(x, y, x + sx * L, y, T["text"], 3, 0.9))
        items.append(line(x, y, x, y + sy * L, T["text"], 3, 0.9))
    label = "ROBOT VIEW"
    x, y, h = 88, 84, 48
    lw = measure(label, "mono", 22, ls=4, bold=True)
    w = 52 + lw + 22
    if robot:
        w += measure(robot[0].upper(), "mono", 22, ls=4, bold=True) + 40
    items += [rect(x, y, w, h, rx=10, fill=T["bg"], op=0.78),
              rect(x, y, w, h, rx=10, stroke=T["line"], sw=1),
              circle(x + 27, y + h / 2, 6, fill=T["stop"]),
              text(x + 52, y + 32, label, "mono", 22, T["text"], ls=4, bold=True)]
    if robot:  # divider, then the robot name in its own colour
        name, colkey, _ = robot
        items += [line(x + 52 + lw + 22, y + 12, x + 52 + lw + 22, y + h - 12, T["line"], 1),
                  text(x + 52 + lw + 42, y + 32, name.upper(), "mono", 22, T[colkey], ls=4, bold=True)]
    return items


def build_overlays():
    emit("lower_third", VIDEO, VW, VH,
         lower_third_items("Dog-1 closes the isolation valve", "DECK B  ·  00:42  ·  TRAINED POLICY"),
         title="Spark lower third")
    emit("robot_view", VIDEO, VW, VH, robot_view_items(), title="Robot view label")
    for r in ROBOTS:
        emit("robot_view_" + r[1], VIDEO, VW, VH, robot_view_items(r), title=f"Robot view {r[0]}")
    for nm, label, kind in (("tag_trained_policy", "TRAINED POLICY", "policy"),
                            ("tag_gemini_advisor", "GEMINI ADVISOR", "advisor")):
        w, h, items = tag_items(0, 0, label, kind)
        emit(nm, VIDEO, w, h, items, title=label.title())
    for r in ROBOTS:
        w, h, items = name_tag_items(0, 0, *r)
        emit("nametag_" + r[1], VIDEO, w, h, items, title=f"{r[0]} name tag")


# ---------------------------------------------------------------- contact sheet
def footage(w, h):
    """Stand-in for camera footage behind overlays: a dull deck-steel gradient."""
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for yy in range(h):
        t = yy / h
        d.line((0, yy, w, yy), fill=(int(52 - 30 * t), int(62 - 34 * t), int(72 - 36 * t)))
    for k in range(0, w, w // 12):
        d.line((k, int(h * 0.62), k + w // 5, h), fill=(70, 80, 90), width=max(1, w // 400))
    return img.convert("RGBA")


def build_contact_sheet():
    SW = 2400
    pad = 80
    sheet = Image.new("RGBA", (SW, 3720), rgba(T["bg"]))
    d = ImageDraw.Draw(sheet)

    def label(x, y, s, col=T["muted"]):
        d.text((x, y), s, font=font("mono", 20, True), fill=rgba(col), anchor="ls")

    def hdr(y, s):
        label(pad, y, s.upper(), T["accent"])
        d.line((pad, y + 18, SW - pad, y + 18), fill=rgba(T["line"]), width=1)

    def paste(img, x, y, w=None):
        if w:
            img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
        sheet.alpha_composite(img, (int(x), int(y)))
        return img

    # header lockup
    cap = 64
    lk = to_png(lockup_w(cap) + 2, cap * 1.3, lockup(0, cap * 0.15, cap, T["text"], T["accent"]))
    paste(lk, pad, 70)
    d.text((SW - pad, 128), "Brand kit v1  ·  autonomous robot fleet for offshore rigs",
           font=font("body", 26), fill=rgba(T["muted"]), anchor="rs")

    # logos
    y = 250
    hdr(y, "Logo")
    y += 60
    tiles = [("Lockup on dark", T["bg"], T["text"], T["accent"]), ("Lockup on light", T["text"], T["bg"], T["accent2"])]
    tw_ = (SW - 2 * pad - 40) // 2
    for i, (nm, bgc, tf, mf) in enumerate(tiles):
        x = pad + i * (tw_ + 40)
        d.rounded_rectangle((x, y, x + tw_, y + 300), 10, fill=rgba(bgc), outline=rgba(T["line"]))
        c = 88
        lk = to_png(lockup_w(c) + 2, c * 1.3, lockup(0, c * 0.15, c, tf, mf))
        paste(lk, x + (tw_ - lk.width) / 2, y + (300 - lk.height) / 2)
        label(x, y + 340, nm)
    y += 400
    # mark, wordmark, favicon sizes
    d.rounded_rectangle((pad, y, pad + 300, y + 300), 10, fill=rgba(T["surface"]), outline=rgba(T["line"]))
    mk = to_png(mark_w(200) + 1, 200, mark(0, 0, 200, T["accent"]))
    paste(mk, pad + (300 - mk.width) / 2, y + 50)
    label(pad, y + 340, "Mark")
    x = pad + 340
    d.rounded_rectangle((x, y, x + 900, y + 300), 10, fill=rgba(T["surface"]), outline=rgba(T["line"]))
    wm = to_png(WORD_W * 1.2 + 1, 120, wordmark(0, 0, 120, T["text"]))
    paste(wm, x + (900 - wm.width) / 2, y + 90)
    label(x, y + 340, "Wordmark (paths, no font needed)")
    x += 940
    d.rounded_rectangle((x, y, SW - pad, y + 300), 10, fill=rgba(T["surface"]), outline=rgba(T["line"]))
    fx = x + 50
    for px in (16, 24, 32, 48, 64, 128):
        ic = to_png(px, px, favicon_items(px))
        paste(ic, fx, y + 150 - px / 2)
        d.text((fx + px / 2, y + 260), f"{px}", font=font("mono", 18), fill=rgba(T["muted"]), anchor="ms")
        fx += px + 56
    small = to_png(lockup_w(24) + 2, 32, lockup(0, 4, 24, T["text"], T["accent"]))
    paste(small, fx + 10, y + 134)
    d.text((fx + 10 + small.width / 2, y + 260), "24 px lockup", font=font("mono", 18),
           fill=rgba(T["muted"]), anchor="ms")
    label(x, y + 340, "Favicon / app tile and minimum sizes")
    y += 420

    # colours
    hdr(y, "Colour")
    y += 60
    sw_list = [("Background", "bg"), ("Surface", "surface"), ("Line", "line"), ("Text", "text"), ("Muted", "muted"),
               ("Spark teal", "accent"), ("Deep teal", "accent2"), ("Warning", "warning"), ("Stop", "stop"),
               ("Success", "success"), ("Dog-1", "dog1"), ("Dog-2", "dog2"), ("Drone", "drone"), ("Crawler", "crawler")]
    per = 7
    cw = (SW - 2 * pad - (per - 1) * 24) // per
    for i, (nm, k) in enumerate(sw_list):
        cx = pad + (i % per) * (cw + 24)
        cy = y + (i // per) * 230
        d.rounded_rectangle((cx, cy, cx + cw, cy + 130), 10, fill=rgba(T[k]), outline=rgba(T["line"]))
        d.text((cx, cy + 166), nm, font=font("body", 24, True), fill=rgba(T["text"]), anchor="ls")
        d.text((cx, cy + 196), T[k], font=font("mono", 20), fill=rgba(T["muted"]), anchor="ls")
    y += 480

    # type
    hdr(y, "Type")
    y += 70
    d.text((pad, y + 70), "Inspect. Repair. Isolate.", font=font("display", 76), fill=rgba(T["text"]), anchor="ls")
    d.text((pad + 1180, y + 30), "Space Grotesk 600  ·  headlines, cards", font=font("mono", 20),
           fill=rgba(T["muted"]), anchor="ls")
    d.text((pad, y + 140), "Dog-2 found a leak at flange F-14 and is closing the valve.",
           font=font("body", 34), fill=rgba(T["text"]), anchor="ls")
    d.text((pad + 1180, y + 110), "Inter 400/600  ·  UI and body", font=font("mono", 20),
           fill=rgba(T["muted"]), anchor="ls")
    d.text((pad, y + 200), "BATT 82%   CH4 0.4 %LEL   T+00:42.18", font=font("mono", 30, True),
           fill=rgba(T["accent"]), anchor="ls")
    d.text((pad + 1180, y + 190), "JetBrains Mono 500  ·  numbers, telemetry", font=font("mono", 20),
           fill=rgba(T["muted"]), anchor="ls")
    y += 280

    # video
    hdr(y, "Video graphics  (1920 x 1080)")
    y += 50
    cols = 4
    gw = (SW - 2 * pad - (cols - 1) * 30) // cols
    gh = gw * 9 // 16
    cards = ["title_card"] + [f"act_{i:02d}_{a.lower().replace(' ', '_')}" for i, a in enumerate(ACTS, 1)] + ["end_card"]
    bg_ft = footage(VW, VH)
    overlays = ["lower_third", "robot_view_dog1"]
    allv = cards + overlays
    for i, nm in enumerate(allv):
        cx = pad + (i % cols) * (gw + 30)
        cy = y + (i // cols) * (gh + 70)
        im = Image.open(os.path.join(VIDEO, nm + ".png")).convert("RGBA")
        if nm in overlays:
            base = bg_ft.copy()
            base.alpha_composite(im)
            im = base
        paste(im, cx, cy, gw)
        d.rectangle((cx, cy, cx + gw - 1, cy + gh - 1), outline=rgba(T["line"]))
        label(cx, cy + gh + 36, nm + ".png")
    y += ((len(allv) + cols - 1) // cols) * (gh + 70) + 30

    # tags and name tags at 1:1 on footage
    strip_h = 220
    strip = footage(SW - 2 * pad, strip_h)
    sx = 40
    for nm in ["nametag_dog1", "nametag_dog2", "nametag_drone", "nametag_crawler"]:
        im = Image.open(os.path.join(VIDEO, nm + ".png")).convert("RGBA")
        strip.alpha_composite(im, (sx, 40))
        sx += im.width + 28
    sx = 40
    for nm in ["tag_trained_policy", "tag_gemini_advisor"]:
        im = Image.open(os.path.join(VIDEO, nm + ".png")).convert("RGBA")
        strip.alpha_composite(im, (sx, 136))
        sx += im.width + 28
    sheet.alpha_composite(strip, (pad, y))
    d.rectangle((pad, y, SW - pad - 1, y + strip_h - 1), outline=rgba(T["line"]))
    label(pad, y + strip_h + 36, "Name tags and tags at 1:1 (solid = in control, dashed = advises only)")
    y += strip_h + 80
    sheet = sheet.crop((0, 0, SW, y))
    sheet.convert("RGB").save(os.path.join(HERE, "contact_sheet.png"))


# ---------------------------------------------------------------- tokens
def build_tokens():
    tokens = {
        "color": {
            "bg": T["bg"], "surface": T["surface"], "line": T["line"], "text": T["text"], "muted": T["muted"],
            "accent": T["accent"], "accent-2": T["accent2"], "warning": T["warning"], "stop": T["stop"],
            "success": T["success"],
            "robot": {"dog-1": T["dog1"], "dog-2": T["dog2"], "drone": T["drone"], "crawler": T["crawler"]},
        },
        "font": {"display": FONT_STACK["display"], "body": FONT_STACK["body"], "mono": FONT_STACK["mono"],
                 "google-fonts-url": GOOGLE_FONTS.replace("&amp;", "&")},
        "font-weight": {"regular": 400, "medium": 500, "semibold": 600},
        "type-scale-ui": {"display": "32px", "h1": "24px", "h2": "18px", "body": "14px", "small": "12px",
                          "mono-label": "11px"},
        "type-scale-video": {"display": "112px", "h1": "72px", "h2": "44px", "body": "32px", "caption": "22px"},
        "radius": {"panel": "10px", "chip": "10px", "inner": "6px"},
        "line": {"hairline": "1px"},
        "space": {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "24px", "6": "32px", "7": "48px",
                  "8": "64px"},
        "letter-spacing": {"mono-label": "0.12em", "display": "-0.01em"},
    }
    with open(os.path.join(HERE, "tokens.json"), "w", encoding="utf-8") as fh:
        json.dump(tokens, fh, indent=2)
        fh.write("\n")

    lines = ["/* Spark design tokens. Generated by brand/build.py - edit there, then re-run. */",
             f"@import url('{GOOGLE_FONTS.replace('&amp;', '&')}');", "", ":root {"]

    def walk(prefix, obj):
        for k, v in obj.items():
            if k == "google-fonts-url":
                continue
            name = f"{prefix}-{k}"
            if isinstance(v, dict):
                walk(name, v)
            else:
                lines.append(f"  --spark-{name}: {v};")

    for group, vals in tokens.items():
        walk(group, vals)
    lines.append("}")
    with open(os.path.join(HERE, "tokens.css"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "caption":
        out = sys.argv[4] if len(sys.argv) > 4 else "caption.png"
        to_png(VW, VH, lower_third_items(sys.argv[2], sys.argv[3]), path=out)
        print("wrote", out)
        sys.exit(0)
    build_tokens()
    build_logos()
    build_cards()
    build_overlays()
    build_contact_sheet()
    print("brand kit built in", HERE)

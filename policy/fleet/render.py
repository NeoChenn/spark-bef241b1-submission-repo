from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

BG = (13, 19, 27)
PANEL = (22, 31, 42)
WALK = (92, 108, 124)
NARROW = (201, 138, 58)
STAIRS = (150, 110, 220)
TEXT = (220, 228, 236)
MUTED = (130, 142, 155)
ROBOT_COLORS = [(64, 156, 255), (60, 200, 120), (255, 184, 48), (230, 90, 200), (90, 220, 230), (240, 240, 240)]
TASK_COLORS = {"gas": (255, 64, 64), "crack": (230, 80, 220), "rust": (214, 110, 40), "leak_check": (80, 200, 220),
               "inspect": (200, 210, 220)}
DECK_ORDER = ["top_deck", "middle_deck", "lower_deck"]
SCALE = 6.0
PAD = 10
HEAD = 34
FOOT = 46


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_png(map_json: dict, snap: dict, path: str) -> None:
    b = map_json["bounds"]
    pw = int((b["xmax"] - b["xmin"]) * SCALE)
    ph = int((b["ymax"] - b["ymin"]) * SCALE)
    W = PAD + 3 * (pw + PAD)
    H = HEAD + ph + FOOT + PAD
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    f10, f12, f14 = _font(10), _font(12), _font(14)
    zdeck = {k["name"]: k["z"] for k in map_json["decks"]}

    def deck_of(z):
        n = min(zdeck, key=lambda k: abs(zdeck[k] - z))
        return n if abs(zdeck[n] - z) < 1.5 else None

    def px(i, x, y):
        return (PAD + i * (pw + PAD) + (x - b["xmin"]) * SCALE, HEAD + (b["ymax"] - y) * SCALE)

    for i, name in enumerate(DECK_ORDER):
        x0 = PAD + i * (pw + PAD)
        d.rectangle([x0, HEAD, x0 + pw, HEAD + ph], fill=PANEL)
        d.text((x0 + 6, HEAD + ph - 16), f"{name.replace('_', ' ')}  z={zdeck[name]:g} m", fill=MUTED, font=f12)
    for w in map_json["walkways"]:
        a, c = w["a"], w["b"]
        if w["kind"] == "stairs":
            for p in (a, c):
                i = DECK_ORDER.index(deck_of(p[2]))
                x, y = px(i, p[0], p[1])
                d.rectangle([x - 4, y - 4, x + 4, y + 4], outline=STAIRS, width=2)
            continue
        i = DECK_ORDER.index(deck_of(a[2]))
        col, wd = (NARROW, 2) if w["narrow"] else (WALK, 4)
        d.line([px(i, a[0], a[1]), px(i, c[0], c[1])], fill=col, width=wd)
    ti = DECK_ORDER.index("top_deck")
    dx, dy = px(ti, *map_json["dock"]["xyz"][:2])
    d.rectangle([dx - 7, dy - 7, dx + 7, dy + 7], outline=(60, 220, 120), width=2)
    d.text((dx + 9, dy - 6), "DOCK", fill=(60, 220, 120), font=f10)
    for fl in map_json["flanges"]:
        x, y = px(ti, *fl["xyz"][:2])
        d.polygon([(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)], outline=(80, 200, 220))
        d.text((x + 6, y - 12), fl["id"], fill=(80, 200, 220), font=f10)
    for z in snap["zones"]:
        dk = deck_of(z["xyz"][2])
        if dk is None:
            continue
        i = DECK_ORDER.index(dk)
        x, y = px(i, *z["xyz"][:2])
        r = z["radius"] * SCALE
        if z["kind"] == "gas":
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 50, 50, 50), outline=(255, 70, 70), width=2)
        elif z["kind"] == "person":
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 60, 45), outline=(255, 220, 60), width=1)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 220, 60))
    for t in snap["tasks"]:
        dk = deck_of(t["work_xyz"][2])
        i = DECK_ORDER.index(dk)
        x, y = px(i, *t["xyz"][:2])
        col = TASK_COLORS[t["kind"]]
        r = 3 + (4 * t["severity"] if t["kind"] == "rust" else 1)
        if t["status"] in ("pending", "assigned", "active"):
            d.ellipse([x - r, y - r, x + r, y + r], fill=col + (230,))
        else:
            d.ellipse([x - r, y - r, x + r, y + r], outline=col + (120,), width=1)
    for k, rb in enumerate(snap["robots"]):
        col = ROBOT_COLORS[k % len(ROBOT_COLORS)]
        pts = [p for p in rb["path"]]
        for a, c in zip(pts, pts[1:]):
            da, dc = deck_of(a[2]), deck_of(c[2])
            if da and da == dc:
                i = DECK_ORDER.index(da)
                d.line([px(i, a[0], a[1]), px(i, c[0], c[1])], fill=col + (170,), width=2)
        p = rb["pose"]
        dk = deck_of(p["z"])
        if dk is None:
            dk = min(DECK_ORDER, key=lambda n: abs(zdeck[n] - p["z"]))
        i = DECK_ORDER.index(dk)
        x, y = px(i, p["x"], p["y"])
        if rb["state"] == "FAILED":
            d.line([x - 6, y - 6, x + 6, y + 6], fill=(255, 80, 80), width=3)
            d.line([x - 6, y + 6, x + 6, y - 6], fill=(255, 80, 80), width=3)
        else:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=(0, 0, 0), width=2)
        d.text((x + 8, y + 2), rb["id"], fill=col, font=f12)
    m = snap["metrics"]
    gas = [a for a in snap["alarms"] if a["outcome"] is None]
    head = f"t={snap['t']:.0f}s   jobs done {m['jobs_done']}  escalated {m['jobs_escalated']}  open {m['open_tasks']}"
    if gas:
        g = gas[-1]
        head += f"   GAS ALARM {g['flange']} -> {g['responder'] or 'unassigned'}"
    d.text((PAD, 9), head, fill=TEXT, font=f14)
    lx = W - PAD - 470
    for name, col in TASK_COLORS.items():
        d.ellipse([lx, 13, lx + 8, 21], fill=col)
        d.text((lx + 11, 10), name, fill=MUTED, font=f10)
        lx += 18 + 7 * len(name)
    d.rectangle([lx, 13, lx + 8, 21], outline=STAIRS, width=2)
    d.text((lx + 11, 10), "stairs", fill=MUTED, font=f10)
    d.line([lx + 54, 17, lx + 70, 17], fill=NARROW, width=2)
    d.text((lx + 73, 10), "narrow", fill=MUTED, font=f10)
    d.rectangle([0, HEAD + ph + 1, W, H], fill=BG)
    y0 = HEAD + ph + 10
    colw = (W - 2 * PAD) // max(len(snap["robots"]), 1)
    for k, rb in enumerate(snap["robots"]):
        col = ROBOT_COLORS[k % len(ROBOT_COLORS)]
        x0 = PAD + k * colw
        soc = rb["battery"]["soc"]
        d.rectangle([x0, y0, x0 + 60, y0 + 10], outline=MUTED)
        d.rectangle([x0 + 1, y0 + 1, x0 + 1 + int(58 * soc), y0 + 9],
                    fill=(60, 200, 120) if soc > 0.3 else (255, 120, 60))
        js = f" / {rb['job_state']}" if rb["job_state"] else ""
        dk = rb["deck"].replace("_deck", "")
        d.text((x0 + 66, y0 - 2), f"{rb['id']} {soc:.0%}  {rb['state']}{js}  [{dk}]", fill=col, font=f12)
        d.text((x0 + 66, y0 + 13), rb["task"] or "-", fill=MUTED, font=f10)
    img.save(path)

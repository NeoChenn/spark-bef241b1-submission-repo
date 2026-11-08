from __future__ import annotations

import hashlib
import math

import numpy as np

CLAD_DEPTH = 0.32
GLASS_DEPTH = 0.1
WINDOW_W, WINDOW_H, SILL = 1.3, 1.0, 1.05
DOOR_W, DOOR_H = 1.0, 2.15
BAY = 2.6
MARGIN = 0.9
TILES = 4
STYLES = (("cabin", 1.3, 1.0, 0.5), ("small", 0.85, 0.9, 0.2), ("office", 1.3, 1.0, 0.3))
OFFICE_W, OFFICE_H = 3.0, 1.3
TILE_KELVIN = (2700, 3000, 2900, 4200, 5200, 6500, 2800, 4000, 5000, 3000, 2700, 3000, 0, 0, 0, 0)
TILE_WEIGHTS = (
    ("warm", (0, 1, 2), 0.24),
    ("cool", (3, 4, 5), 0.2),
    ("blind", (6, 7, 8, 9), 0.2),
    ("curtain", (10, 11), 0.1),
    ("dark", (12, 13, 14, 15), 0.26),
)


def wall(name, axis, sign, plane, along, z, storeys, doors=(), windows=True, blank=0.15, depth=None, clad="clad",
         ribs=False):
    return {"name": name, "axis": axis, "sign": sign, "plane": plane, "along": along, "z": z, "storeys": storeys,
            "doors": tuple(doors), "windows": windows, "blank": blank, "depth": depth, "clad": clad, "ribs": ribs}


WALLS = (
    wall("lq_s_upper", "y", -1, -37.5, (-25.0, 0.3), (30.0, 41.2), 3, doors=((-19.8, 0), (-3.4, 0))),
    wall("lq_s_lower", "y", -1, -37.5, (-18.1, 0.3), (20.0, 30.0), 3),
    wall("lq_notch_w", "x", -1, -18.1, (-37.5, -27.5), (20.0, 30.0), 3, doors=((-30.2, 0),)),
    wall("lq_notch_s", "y", -1, -27.5, (-25.0, -18.1), (20.0, 30.0), 3),
    wall("lq_w_lower", "x", -1, -25.0, (-27.5, -22.9), (20.0, 30.0), 3),
    wall("lq_w_upper", "x", -1, -25.0, (-37.5, -22.9), (30.0, 41.2), 3, doors=((-25.6, 0),)),
    wall("lq_e_lower", "x", 1, 0.3, (-37.5, -22.9), (20.0, 30.0), 3, doors=((-24.6, 0),)),
    wall("lq_e_upper", "x", 1, 0.3, (-37.5, -22.9), (30.0, 41.2), 3, doors=((-29.9, 0),)),
    wall("lq_n_lower", "y", 1, -22.9, (-25.0, 0.3), (20.0, 30.0), 3, doors=((-2.0, 0),)),
    wall("lq_n_upper", "y", 1, -22.9, (-22.1, 0.3), (30.0, 41.2), 3, windows=False),
    wall("m40_s", "y", -1, 10.1, (-21.0, -11.4), (31.25, 41.2), 3, doors=((-13.0, 0),)),
    wall("m40_n", "y", 1, 21.0, (-21.0, -11.4), (31.25, 41.2), 3),
    wall("m40_w", "x", -1, -21.0, (10.1, 21.0), (31.25, 41.2), 3),
    wall("m40_e", "x", 1, -11.4, (10.1, 21.0), (31.25, 41.2), 3),
    wall("m71_w", "x", -1, -5.8, (-17.9, -8.8), (31.25, 37.2), 2),
    wall("m71_s", "y", -1, -17.9, (-5.8, -1.1), (31.25, 37.2), 2, doors=((-3.4, 0),)),
    wall("m71_e1", "x", 1, -1.1, (-17.9, -12.3), (31.25, 37.2), 2),
    wall("m71_s2", "y", -1, -12.3, (-1.1, 4.0), (31.25, 37.2), 2),
    wall("m71_e2", "x", 1, 4.0, (-12.3, -8.8), (31.25, 37.2), 2),
    wall("m71_n", "y", 1, -8.8, (-5.8, 4.0), (31.25, 37.2), 2),
    wall("annex_s", "y", -1, -37.0, (25.5, 37.0), (31.25, 40.0), 1, doors=((26.6, 0),), windows=False, depth=0.12,
         clad="clad_grey", ribs=True),
    wall("annex_e", "x", 1, 37.0, (-37.0, -18.0), (31.25, 40.0), 1, doors=((-21.0, 0),), windows=False, depth=0.22,
         clad="clad_grey", ribs=True),
    wall("annex_w", "x", -1, 25.5, (-37.0, -18.0), (31.25, 40.0), 1, windows=False, depth=0.1, clad="clad_grey",
         ribs=True),
)
ANNEX_LADDER = ((37.5, -33.0, 31.25), 8.75)
KEEP_CLEAR = ((27.4, 36.1, 35.1, 38.3),)


def unit(key: str) -> float:
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big") / 2**32


def pick_tile(key: str, dark: float = 1.0) -> int:
    r = unit(key)
    weights = [(tiles, wt * (dark if name == "dark" else 1.0)) for name, tiles, wt in TILE_WEIGHTS]
    total = sum(wt for _, wt in weights)
    acc = 0.0
    for tiles, weight in weights:
        acc += weight / total
        if r < acc:
            return tiles[int(unit(key + "t") * len(tiles))]
    return TILE_WEIGHTS[-1][1][-1]


def tile_uv(tile: int, tiles: int = TILES) -> tuple[float, float, float, float]:
    col, row = tile % tiles, tile // tiles
    pad = 0.5 / 256
    return (col / tiles + pad, 1 - (row + 1) / tiles + pad, (col + 1) / tiles - pad, 1 - row / tiles - pad)


def bays(a0: float, a1: float, bay: float = BAY, margin: float = MARGIN) -> list[float]:
    count = int((a1 - a0 - 2 * margin) // bay)
    if count < 1:
        return []
    start = (a0 + a1) / 2 - (count - 1) * bay / 2
    return [round(start + i * bay, 4) for i in range(count)]


def storey_height(w) -> float:
    return (w["z"][1] - w["z"][0]) / w["storeys"]


def openings(w) -> list[dict]:
    h = storey_height(w)
    out = []
    for s in range(w["storeys"]):
        floor = w["z"][0] + s * h
        here = [a for a, k in w["doors"] if k == s]
        for a in here:
            out.append({"kind": "door", "along": a, "z0": floor + 0.05, "w": DOOR_W, "h": DOOR_H, "storey": s})
        if not w["windows"]:
            continue
        row = f"{w['name']}:{s}"
        style = storey_style(row)
        sill = floor + SILL + (unit(row + "sill") - 0.5) * 0.2
        dark = 0.4 + 2.2 * unit(row + "dark")
        centres = bays(*w["along"])
        skip = set()
        for i, a in enumerate(centres):
            if i in skip or any(abs(a - d) < (WINDOW_W + DOOR_W) / 2 + 0.3 for d in here):
                continue
            key = f"{row}:{i}"
            if unit(key + "b") < w["blank"]:
                out.append({"kind": "blank", "along": a, "z0": sill, "w": WINDOW_W, "h": WINDOW_H, "storey": s, "key": key})
                continue
            nxt = centres[i + 1] if i + 1 < len(centres) else None
            if style[0] == "office" and nxt is not None and unit(key + "o") < 0.6 and not any(
                    abs(nxt - d) < (OFFICE_W + DOOR_W) / 2 + 0.3 for d in here):
                skip.add(i + 1)
                out.append({"kind": "window", "along": (a + nxt) / 2, "z0": sill - 0.1, "w": OFFICE_W, "h": OFFICE_H,
                            "storey": s, "tile": pick_tile(key, dark * 0.6), "key": key})
                continue
            _, ww, wh, _ = style
            out.append({"kind": "window", "along": a, "z0": sill, "w": ww, "h": wh, "storey": s,
                        "tile": pick_tile(key, dark), "key": key})
    return out


def storey_style(row: str):
    r = unit(row + "style")
    acc = 0.0
    for style in STYLES:
        acc += style[3]
        if r < acc:
            return style
    return STYLES[0]


def lit(o) -> bool:
    return o["kind"] == "window" and TILE_KELVIN[o["tile"]] > 0


def holes(items) -> list[tuple[float, float, float, float]]:
    return [(o["along"] - o["w"] / 2, o["z0"], o["along"] + o["w"] / 2, o["z0"] + o["h"])
            for o in items if o["kind"] in ("window", "door")]


def subtract(rect, cuts) -> list[tuple[float, float, float, float]]:
    a0, z0, a1, z1 = rect
    xs = sorted({a0, a1, *[c for h in cuts for c in (h[0], h[2]) if a0 < c < a1]})
    zs = sorted({z0, z1, *[c for h in cuts for c in (h[1], h[3]) if z0 < c < z1]})
    out = []
    for zl, zh in zip(zs, zs[1:]):
        row = []
        for xl, xh in zip(xs, xs[1:]):
            cx, cz = (xl + xh) / 2, (zl + zh) / 2
            if any(h[0] < cx < h[2] and h[1] < cz < h[3] for h in cuts):
                continue
            if row and abs(row[-1][2] - xl) < 1e-9:
                row[-1] = (row[-1][0], zl, xh, zh)
            else:
                row.append((xl, zl, xh, zh))
        out += row
    return out


def to_world(w, along: float, z: float, depth: float) -> tuple[float, float, float]:
    off = w["plane"] + w["sign"] * depth
    return (along, off, z) if w["axis"] == "y" else (off, along, z)


def outward(w) -> tuple[float, float, float]:
    n = [0.0, 0.0, 0.0]
    n[0 if w["axis"] == "x" else 1] = float(w["sign"])
    return tuple(n)


def quad(w, a0, z0, a1, z1, depth) -> list[tuple[float, float, float]]:
    pts = [to_world(w, a0, z0, depth), to_world(w, a1, z0, depth), to_world(w, a1, z1, depth),
           to_world(w, a0, z1, depth)]
    return orient(pts, outward(w))


def orient(pts, normal):
    p = np.asarray(pts, dtype=float)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    return pts if float(np.dot(n, normal)) > 0 else pts[::-1]


def face_normal(pts) -> tuple[float, float, float]:
    p = np.asarray(pts, dtype=float)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    return tuple(float(c) for c in n / max(np.linalg.norm(n), 1e-12))


def reveal(w, hole, d_front, d_back) -> list[list[tuple[float, float, float]]]:
    a0, z0, a1, z1 = hole
    faces = []
    for (pa, pz), (qa, qz), inward in (
        ((a0, z0), (a1, z0), (0, 1)), ((a1, z0), (a1, z1), (-1, 0)),
        ((a1, z1), (a0, z1), (0, -1)), ((a0, z1), (a0, z0), (1, 0)),
    ):
        pts = [to_world(w, pa, pz, d_front), to_world(w, qa, qz, d_front), to_world(w, qa, qz, d_back),
               to_world(w, pa, pz, d_back)]
        mid = np.mean(np.asarray(pts), axis=0)
        target = np.asarray(to_world(w, (a0 + a1) / 2, (z0 + z1) / 2, (d_front + d_back) / 2)) - mid
        faces.append(orient(pts, target))
    return faces


def box_faces(lo, hi) -> list[list[tuple[float, float, float]]]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7))
    return [[c[i] for i in f] for f in idx]


def wall_box(w, a0, z0, a1, z1, d0, d1):
    p, q = to_world(w, a0, z0, d0), to_world(w, a1, z1, d1)
    return tuple(min(a, b) for a, b in zip(p, q)), tuple(max(a, b) for a, b in zip(p, q))


def streak_height(o, w) -> float:
    room = o["z0"] - w["z"][0] - 0.15
    if o["storey"] > 0:
        room = min(room, storey_height(w) - OFFICE_H - 0.45)
    return max(0.0, min(room, 0.9 + 1.6 * unit(o["key"] + "s")))


def helideck_lights(polygon, spacing: float = 3.0) -> list[tuple[float, float]]:
    out = []
    for p, q in zip(polygon, list(polygon[1:]) + [polygon[0]]):
        n = max(1, math.ceil(math.dist(p, q) / spacing - 1e-9))
        out += [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n)]
    return out

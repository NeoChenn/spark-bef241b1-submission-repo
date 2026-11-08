from __future__ import annotations

import math

import numpy as np


def leg_axes(points) -> list[dict]:
    points = np.asarray(points, float)
    middle = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
    legs = []
    for east in (False, True):
        for north in (False, True):
            leg = points[((points[:, 0] > middle[0]) == east) & ((points[:, 1] > middle[1]) == north)]
            if not len(leg):
                continue
            top_z, bottom_z = leg[:, 2].max(), leg[:, 2].min()
            top = leg[leg[:, 2] > top_z - 0.5, :2]
            bottom = leg[leg[:, 2] < bottom_z + 0.5, :2]
            tc, bc = (top.min(0) + top.max(0)) / 2, (bottom.min(0) + bottom.max(0)) / 2
            legs.append({"top": np.array((tc[0], tc[1], top_z)), "bottom": np.array((bc[0], bc[1], bottom_z)),
                         "radius": float((top.max(0) - top.min(0)).max()) / 2})
    return legs


def axis_at(leg, z: float) -> np.ndarray:
    top, bottom = leg["top"], leg["bottom"]
    t = (top[2] - z) / (top[2] - bottom[2])
    return top[:2] + t * (bottom[:2] - top[:2])


def edge_profile(n: int, base: float, amp: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.linspace(0.0, math.tau, n + 1)
    wave = sum(rng.uniform(0.4, 1.0) / k * np.sin(k * a + rng.uniform(0, math.tau)) for k in (1, 2, 3, 5, 8, 13))
    wave = wave / max(np.abs(wave).max(), 1e-9)
    wave[-1] = wave[0]
    return base + amp * wave


def column_band(leg, sides: int, low, high, rows: int = 4, radius: float | None = None) -> dict:
    radius = leg["radius"] if radius is None else radius
    low = np.broadcast_to(np.asarray(low, float), (sides + 1,))
    high = np.broadcast_to(np.asarray(high, float), (sides + 1,))
    a = np.linspace(0.0, math.tau, sides + 1)
    f = np.linspace(0.0, 1.0, rows + 1)[:, None]
    z = low[None, :] + (high - low)[None, :] * f
    top, bottom = leg["top"], leg["bottom"]
    t = (top[2] - z) / (top[2] - bottom[2])
    cx = top[0] + t * (bottom[0] - top[0])
    cy = top[1] + t * (bottom[1] - top[1])
    ring = np.stack((np.cos(a), np.sin(a)), -1)
    points = np.stack((cx + radius * ring[None, :, 0], cy + radius * ring[None, :, 1], z), -1).reshape(-1, 3)
    normals = np.broadcast_to(np.concatenate((ring, np.zeros((sides + 1, 1))), 1), (rows + 1, sides + 1, 3)).reshape(-1, 3)
    uvs = np.stack(np.broadcast_arrays(a[None, :] * radius, z), -1).reshape(-1, 2)
    row = sides + 1
    i, j = np.meshgrid(np.arange(rows), np.arange(sides), indexing="ij")
    k = i * row + j
    quads = np.stack((k, k + 1, k + row + 1, k + row), -1).reshape(-1)
    return {"points": points, "normals": normals, "uvs": uvs, "counts": np.full(rows * sides, 4), "indices": quads}


def split_by_bands(p, q, cuts) -> list:
    p, q = tuple(float(c) for c in p), tuple(float(c) for c in q)
    lo, hi = (p, q) if p[2] <= q[2] else (q, p)
    zs = [lo[2]] + [c for c in cuts if lo[2] < c < hi[2]] + [hi[2]]
    out = []
    for z0, z1 in zip(zs, zs[1:]):
        def at(z):
            if hi[2] == lo[2]:
                return lo
            t = (z - lo[2]) / (hi[2] - lo[2])
            return tuple(lo[i] + t * (hi[i] - lo[i]) for i in range(3))
        band = sum(1 for c in cuts if c <= (z0 + z1) / 2)
        out.append((at(z0) if z0 != lo[2] else lo, at(z1) if z1 != hi[2] else hi, band))
    return out


def scanline(poly, c: float, axis: str = "x") -> list:
    pts = [(x, y) if axis == "x" else (y, x) for x, y in poly]
    hits = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if (y0 <= c < y1) or (y1 <= c < y0):
            hits.append(x0 + (c - y0) * (x1 - x0) / (y1 - y0))
    hits.sort()
    return [(hits[i], hits[i + 1]) for i in range(0, len(hits) - 1, 2)]


def clear_stations(a: float, b: float, spacing: float, avoid=(), clear: float = 1.3) -> list:
    count = max(1, round((b - a) / spacing))
    xs = [a + (b - a) * i / count for i in range(count + 1)]
    return [x for x in xs if all(abs(x - v) >= clear for v in avoid)]


def ring_mesh(r_in: float, r_out: float, height: float, sides: int) -> dict:
    a = np.linspace(0.0, math.tau, sides + 1)
    c, s = np.cos(a), np.sin(a)
    h = height / 2
    groups = (
        (r_out, r_out, -h, h, np.stack((c, s, 0 * c), -1)),
        (r_in, r_in, h, -h, np.stack((-c, -s, 0 * c), -1)),
        (r_in, r_out, h, h, np.stack((0 * c, 0 * c, 1 + 0 * c), -1)),
        (r_out, r_in, -h, -h, np.stack((0 * c, 0 * c, -1 + 0 * c), -1)),
    )
    points, normals, uvs, indices = [], [], [], []
    for ra, rb, za, zb, n in groups:
        base = sum(len(p) for p in points)
        pa = np.stack((ra * c, ra * s, np.full_like(c, za)), -1)
        pb = np.stack((rb * c, rb * s, np.full_like(c, zb)), -1)
        points.append(np.concatenate((pa, pb)))
        normals.append(np.concatenate((n, n)))
        uvs.append(np.concatenate((np.stack((a * ra, 0 * a), -1), np.stack((a * rb, 0 * a + 1), -1))))
        j = np.arange(sides)
        indices.append(np.stack((base + j, base + j + 1, base + sides + 2 + j, base + sides + 1 + j), -1).reshape(-1))
    return {"points": np.concatenate(points), "normals": np.concatenate(normals), "uvs": np.concatenate(uvs),
            "counts": np.full(4 * sides, 4), "indices": np.concatenate(indices)}


def _frame(angle_deg: float):
    t = math.radians(angle_deg)
    return np.array((math.cos(t), math.sin(t), 0.0)), np.array((-math.sin(t), math.cos(t), 0.0))


def _at(leg, z):
    c = axis_at(leg, z)
    return np.array((c[0], c[1], z))


def boat_landing(leg, angle_deg: float, deck_z: float = 4.75) -> dict:
    u, t = _frame(angle_deg)
    r = leg["radius"]
    d = r + 2.4
    tubes, rubber, rails = [], [], []
    for s in (-3.0, -1.0, 1.0, 3.0):
        tubes.append((_at(leg, -4.0) + u * d + t * s, _at(leg, deck_z - 0.1) + u * d + t * s, 0.25))
    for z in (-2.5, 0.0, 2.4):
        tubes.append((_at(leg, z) + u * d - t * 3.4, _at(leg, z) + u * d + t * 3.4, 0.2))
    for z in (-2.5, 2.4, deck_z - 0.25):
        for s in (-3.0, 3.0):
            depth = math.sqrt(r * r - s * s) * 0.97
            tubes.append((_at(leg, z) + u * depth + t * s, _at(leg, z) + u * d + t * s, 0.18))
    for s0, s1 in ((-3.0, -1.0), (3.0, 1.0)):
        tubes.append((_at(leg, -2.5) + u * d + t * s0, _at(leg, 2.4) + u * d + t * s1, 0.15))
    for s in (-1.0, 1.0):
        rubber.append((_at(leg, -1.6) + u * (d + 0.05) + t * s, _at(leg, 3.2) + u * (d + 0.05) + t * s, 0.36))
    for s in (-3.65, 3.65):
        a, b = _at(leg, deck_z) + u * (r + 0.2) + t * s, _at(leg, deck_z) + u * (d + 0.3) + t * s
        for h in (0.55, 1.1):
            rails.append((a + (0, 0, h), b + (0, 0, h), 0.03))
        for f in (0.0, 0.5, 1.0):
            p = a + (b - a) * f
            rails.append((p, p + (0, 0, 1.1), 0.03))
    deck = _at(leg, deck_z) + u * (r + 1.3)
    ul, tl = _frame(angle_deg + 21.0)
    rr = r + 0.25
    ladder, rungs, hoops = [], [], []
    top = leg["top"][2]
    for side in (-0.23, 0.23):
        ladder.append((_at(leg, deck_z) + ul * rr + tl * side, _at(leg, top) + ul * rr + tl * side, 0.035))
    z = deck_z + 0.3
    while z < top - 0.1:
        rungs.append((_at(leg, z) + ul * rr - tl * 0.23, _at(leg, z) + ul * rr + tl * 0.23, 0.02))
        z += 0.3
    z = deck_z + 2.4
    while z < top - 0.2:
        hoops.append((_at(leg, z) + ul * rr, angle_deg + 21.0))
        z += 0.9
    straps = []
    for k in (-0.42, 0.0, 0.42):
        off = ul * (0.42 if k == 0.0 else 0.0) + tl * k
        straps.append((_at(leg, deck_z + 2.4) + ul * rr + off, _at(leg, top - 0.2) + ul * rr + off, 0.025))
    return {"tubes": tubes, "rubber": rubber, "rails": rails, "deck": deck, "deck_size": (3.0, 7.4, 0.1),
            "deck_angle": angle_deg, "ladder": ladder + straps, "rungs": rungs, "hoops": hoops}


def risers(leg, specs, bottom: float = -25.0, spacing: float = 3.0, gap: float = 0.7) -> dict:
    r = leg["radius"]
    top = leg["top"][2]
    pipes, clamps = [], []
    for angle, pr in specs:
        u, _ = _frame(angle)
        d = r + gap + pr
        pipes.append((_at(leg, bottom) + u * d, _at(leg, top + 0.6) + u * d, pr))
        z = -3.0
        while z < top - 1.4:
            clamps.append({"center": _at(leg, z) + u * d, "pipe_r": pr, "angle": angle,
                           "bracket": _at(leg, z) + u * (r + (gap + pr) / 2 - 0.05), "length": gap + pr + 0.1})
            z += spacing
    return {"pipes": pipes, "clamps": clamps}


def braces(legs, z: float = 7.0, deck: float = 16.4, radius: float = 0.7, diagonal: float = 0.55) -> dict:
    horizontal, diag, cans = [], [], []
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            dx, dy = b["top"][:2] - a["top"][:2]
            if min(abs(dx), abs(dy)) > 1.0:
                continue
            ca, cb = _at(a, z), _at(b, z)
            u = (cb - ca) / np.linalg.norm(cb - ca)
            pa, pb = ca + u * a["radius"], cb - u * b["radius"]
            horizontal.append((pa, pb, radius))
            cans += [(pa - u * 0.3, pa + u * 1.8, radius * 1.3), (pb + u * 0.3, pb - u * 1.8, radius * 1.3)]
            mid = (_at(a, deck) + _at(b, deck)) / 2
            for c, leg in ((ca, a), (cb, b)):
                base = c + (0, 0, 1.4)
                w = mid - base
                w[2] = 0
                w /= np.linalg.norm(w)
                start = base + w * leg["radius"]
                diag.append((start, mid + (0, 0, -0.1), diagonal))
                v = (mid - start) / np.linalg.norm(mid - start)
                cans.append((start - v * 0.3, start + v * 1.6, diagonal * 1.3))
    return {"horizontal": horizontal, "diagonal": diag, "cans": cans}


def gussets(leg, count: int = 8, height: float = 2.4, reach: float = 2.0) -> list:
    out = []
    for k in range(count):
        angle = 360.0 * k / count + 22.5
        u, _ = _frame(angle)
        out.append((_at(leg, leg["top"][2] - height) + u * (leg["radius"] - 0.05), angle, (reach, 0.08, height)))
    return out


def anodes(leg, levels=(-1.7, -6.0), count: int = 6, offset: float = 0.0) -> list:
    out = []
    for n, z in enumerate(levels):
        for k in range(count):
            angle = 360.0 * (k + 0.5 * n) / count + offset
            u, _ = _frame(angle)
            out.append((_at(leg, z) + u * (leg["radius"] + 0.35), angle))
    return out


def seams(leg, z0: float, z1: float, pitch: float = 2.4, verticals: int = 12) -> dict:
    rings = []
    z = z0
    while z <= z1 + 1e-9:
        rings.append(_at(leg, z))
        z += pitch
    lines = []
    for k in range(verticals):
        u, _ = _frame(360.0 * k / verticals + 7.0)
        for n, (a, b) in enumerate(zip(rings, rings[1:])):
            u2 = _frame(360.0 * k / verticals + 7.0 + 180.0 / verticals)[0] if n % 2 else u
            lines.append((a + u2 * leg["radius"], b + u2 * leg["radius"]))
    return {"rings": rings, "lines": lines}


def walkway(p, q, z: float, width: float = 1.0, hang_to: float | None = None, spacing: float = 2.5) -> dict:
    p, q = np.array((p[0], p[1], z), float), np.array((q[0], q[1], z), float)
    d = q - p
    length = float(np.linalg.norm(d))
    u = d / length
    n = np.array((-u[1], u[0], 0.0))
    angle = math.degrees(math.atan2(u[1], u[0]))
    rails, hangers, posts = [], [], []
    for side in (-1, 1):
        edge = n * side * width / 2
        for h in (0.55, 1.1):
            rails.append((p + edge + (0, 0, h), q + edge + (0, 0, h)))
        for k in range(int(length // spacing) + 1):
            base = p + u * min(k * spacing, length) + edge
            posts.append((base, base + (0, 0, 1.1)))
            if hang_to is not None:
                hangers.append((base, base * (1, 1, 0) + (0, 0, hang_to)))
    return {"deck": ((p + q) / 2, angle, (length, width, 0.06)), "rails": rails, "posts": posts, "hangers": hangers}


def caged_ladder(base, z0: float, z1: float, angle_deg: float, half: float = 0.23, pitch: float = 0.3,
                 cage_from: float = 2.4, hoop_pitch: float = 0.9) -> dict:
    u, t = _frame(angle_deg)
    b = np.array((base[0], base[1], 0.0))
    rails = [(b + t * s + (0, 0, z0), b + t * s + (0, 0, z1 + 1.0)) for s in (-half, half)]
    rungs = [(b - t * half + (0, 0, z), b + t * half + (0, 0, z)) for z in np.arange(z0 + pitch, z1 - 1e-6, pitch)]
    hoops = [(b + (0, 0, z), angle_deg) for z in np.arange(z0 + cage_from, z1 + 1.0, hoop_pitch)]
    straps = []
    if hoops:
        top = z1 + 1.0
        for off in (u * 0.42, t * 0.42, -t * 0.42):
            straps.append((b + off + (0, 0, z0 + cage_from), b + off + (0, 0, top)))
    return {"rails": rails, "rungs": rungs, "hoops": hoops, "straps": straps}


def edge_detail(p, q, z0: float, height: float, outward, spacing: float = 2.2) -> dict:
    p, q = np.array((p[0], p[1]), float), np.array((q[0], q[1]), float)
    d = q - p
    length = float(np.linalg.norm(d))
    u = d / length
    out = np.asarray(outward, float) * 0.06
    angle = math.degrees(math.atan2(u[1], u[0]))
    ribs = [(np.append(p + u * f + out, z0 + height / 2), angle) for f in np.arange(1.0, length - 0.5, spacing)]
    strip = (np.append(p + out, z0 + height - 0.05), np.append(q + out, z0 + height - 0.05))
    return {"ribs": ribs, "strip": strip, "size": (0.1, 0.14, height)}

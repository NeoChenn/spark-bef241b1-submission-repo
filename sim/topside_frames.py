from __future__ import annotations

import math

import numpy as np

RAIL_TOP = 1.1
RAIL_KNEE = 0.55
TOE = 0.075


def stations(a: float, b: float, spacing: float) -> list[float]:
    count = max(1, math.ceil((b - a) / spacing - 1e-9))
    return [a + (b - a) * i / count for i in range(count)] + [b]


def segment_pose(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    d = q - p
    length = float(np.linalg.norm(d))
    u = d / length
    dot = u[0]
    if dot < -1 + 1e-12:
        quat = np.array((0.0, 0.0, 0.0, 1.0))
    else:
        quat = np.array((1.0 + dot, 0.0, -u[2], u[1]))
        quat /= np.linalg.norm(quat)
    return (p + q) / 2, quat, length


def frame(low, high, levels, bay=5.0, brace_faces=("-x", "+x", "-y", "+y")) -> dict:
    xs = stations(low[0], high[0], bay)
    ys = stations(low[1], high[1], bay)
    columns, beams, braces = [], [], []
    for z0, z1 in zip(levels, levels[1:]):
        columns += [((x, y, z0), (x, y, z1)) for x in xs for y in ys]
    for z in levels[1:]:
        beams += [((a, y, z), (b, y, z)) for y in ys for a, b in zip(xs, xs[1:])]
        beams += [((x, a, z), (x, b, z)) for x in xs for a, b in zip(ys, ys[1:])]
    faces = {
        "-y": (xs, lambda s, z: (s, low[1], z)),
        "+y": (xs, lambda s, z: (s, high[1], z)),
        "-x": (ys, lambda s, z: (low[0], s, z)),
        "+x": (ys, lambda s, z: (high[0], s, z)),
    }
    for name in brace_faces:
        line, point = faces[name]
        for k, (z0, z1) in enumerate(zip(levels, levels[1:])):
            for i, (a, b) in enumerate(zip(line, line[1:])):
                if (i + k) % 2:
                    continue
                if k % 2:
                    a, b = b, a
                braces.append((point(a, z0), point(b, z1)))
    return {"columns": columns, "beams": beams, "braces": braces}


def rail(p, q, z: float, spacing: float = 1.5) -> dict:
    (x0, y0), (x1, y1) = p, q
    rails = [((x0, y0, round(z + h, 4)), (x1, y1, round(z + h, 4))) for h in (RAIL_TOP, RAIL_KNEE)]
    count = max(1, math.ceil(math.dist(p, q) / spacing - 1e-9))
    posts = []
    for i in range(count + 1):
        t = i / count
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        posts.append(((x, y, z), (x, y, round(z + RAIL_TOP, 4))))
    toe = [((x0, y0, round(z + TOE, 4)), (x1, y1, round(z + TOE, 4)))]
    return {"rails": rails, "posts": posts, "toe": toe}


def stair_tower(origin, z0: float, z1: float, axis: str = "x", width: float = 1.0, going: float = 0.27,
                max_riser: float = 0.2, flight_rise: float = 3.75) -> dict:
    flights = max(1, math.ceil((z1 - z0) / flight_rise - 1e-9))
    rise = (z1 - z0) / flights
    count = math.ceil(rise / max_riser - 1e-9)
    riser = rise / count
    run = going * (count - 1)
    landing = 1.2
    length = run + 2 * landing
    ox, oy = origin
    steps, stringers, landings = [], [], []

    def place(u, v, z):
        return (ox + u, oy + v, z) if axis == "x" else (ox + v, oy + u, z)

    def size(du, dv, dz):
        return (du, dv, dz) if axis == "x" else (dv, du, dz)

    for f in range(flights):
        base = z0 + f * rise
        forward = f % 2 == 0
        v = width / 2 if forward else width * 1.5 + 0.1
        start = landing
        for s in range(1, count):
            u = start + (s - 0.5) * going if forward else start + run - (s - 0.5) * going
            steps.append((place(u, v, base + s * riser - 0.025), size(going, width, 0.05)))
        u0, u1 = (start, start + run) if forward else (start + run, start)
        for edge in (v - width / 2, v + width / 2):
            stringers.append((place(u0, edge, base), place(u1, edge, base + rise)))
        far = length - landing / 2 if forward else landing / 2
        landings.append((place(far, width + 0.05, base + rise - 0.03), size(landing, 2 * width + 0.1, 0.06)))
    low = place(0, 0, z0)
    high = place(length, 2 * width + 0.1, z1)
    box_low = (min(low[0], high[0]), min(low[1], high[1]), z0)
    box_high = (max(low[0], high[0]), max(low[1], high[1]), z1)
    levels = [z0 + rise * i for i in range(flights + 1)]
    return {
        "steps": steps,
        "stringers": stringers,
        "landings": landings,
        "frame": frame(box_low, box_high, levels, bay=length),
        "low": box_low,
        "high": box_high,
    }


def ladder(base, height: float, facing: str = "x", width: float = 0.45, pitch: float = 0.3) -> dict:
    x, y, z = base
    half = width / 2
    offsets = ((0, -half), (0, half)) if facing == "x" else ((-half, 0), (half, 0))
    rails = [((x + dx, y + dy, z), (x + dx, y + dy, z + height + 1.0)) for dx, dy in offsets]
    rungs = []
    level = z + pitch
    while level < z + height - 1e-6:
        (ax, ay), (bx, by) = offsets
        rungs.append(((x + ax, y + ay, round(level, 4)), (x + bx, y + by, round(level, 4))))
        level += pitch
    return {"rails": rails, "rungs": rungs}


def pipe_rack(start, end, z0: float, tiers=(24.0, 26.0), width: float = 3.0, pipes_per_tier: int = 5,
              portal: float = 6.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    (x0, y0), (x1, y1) = start, end
    along_x = abs(x1 - x0) >= abs(y1 - y0)
    length = math.dist(start, end)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -uy, ux
    columns, beams, pipes, valves, trays = [], [], [], [], []
    for t in stations(0.0, length, portal):
        cx, cy = x0 + ux * t, y0 + uy * t
        feet = [(cx + nx * s * width / 2, cy + ny * s * width / 2) for s in (-1, 1)]
        columns += [((fx, fy, z0), (fx, fy, tiers[-1])) for fx, fy in feet]
        beams += [((feet[0][0], feet[0][1], z), (feet[1][0], feet[1][1], z)) for z in tiers]
    for s in (-1, 1):
        for z in tiers:
            ox, oy = nx * s * width / 2, ny * s * width / 2
            beams.append(((x0 + ox, y0 + oy, z), (x1 + ox, y1 + oy, z)))
    for z in tiers:
        offsets = np.linspace(-width / 2 + 0.35, width / 2 - 0.35, pipes_per_tier)
        for k, o in enumerate(offsets):
            radius = float(rng.choice((0.08, 0.11, 0.15, 0.2)))
            lift = z + 0.12 + radius
            a = (x0 + nx * o, y0 + ny * o, lift)
            b = (x1 + nx * o, y1 + ny * o, lift)
            pipes.append(((a, b), radius))
            for t in rng.uniform(0.15, 0.85, size=1 + k % 2):
                valves.append(((a[0] + ux * length * t, a[1] + uy * length * t, lift), radius))
    tray_z = tiers[-1] + 0.9
    for o in (-width / 4, width / 4):
        trays.append(((x0 + nx * o, y0 + ny * o, tray_z), (x1 + nx * o, y1 + ny * o, tray_z)))
    return {"columns": columns, "beams": beams, "pipes": pipes, "valves": valves, "trays": trays, "along_x": along_x}


def clear_of(box, zones) -> bool:
    (ax0, ay0), (ax1, ay1) = box
    for (bx0, by0), (bx1, by1) in zones:
        if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
            return False
    return True


def kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    rgb = np.clip((r, g, b), 0, 255)
    rgb = rgb / rgb.max()
    return tuple(float(c) for c in rgb)


def horizon_gain(elevation_deg, floor: float = 0.05, top: float = 25.0):
    u = np.clip(np.asarray(elevation_deg, dtype=float) / top, 0.0, 1.0)
    smooth = u * u * (3 - 2 * u)
    return floor + (1 - floor) * smooth


ENCLOSED = (((-25.2, -37.7, 18.0), (0.5, -22.7, 41.4)),)
FASCIA_HEIGHT = 1.5


def keep_facade(low, high, enclosed=ENCLOSED) -> bool:
    if high[2] - low[2] < FASCIA_HEIGHT:
        return True
    return any(all(a[i] <= low[i] and high[i] <= b[i] for i in range(3)) for a, b in enclosed)

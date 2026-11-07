"""Materials, lights, props and decals that dress full_rig.usda as a working offshore rig. Pure numpy/PIL."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

DECK_TOPS = (20.0, 31.25, 42.5)
SLAB = 1.25
HELIDECK_TOP = 51.25
HELIDECK_CENTRE = (-22.5, -22.5)
HELIDECK_HALF = 14.5
HELIDECK_RADIUS = 14.43
TDPM_RADIUS = 9.0
H_BARS = ((-26.25, -27.5, -23.75, -17.5), (-21.25, -27.5, -18.75, -17.5), (-23.75, -23.75, -21.25, -21.25))
MESH_TILES = {"deck": 2.5, "concrete": 6.0}


@dataclass(frozen=True)
class Mat:
    texture: str
    metres: float
    tint: tuple[float, float, float] | None = None
    metallic: float | None = None


MATERIALS = {
    "rail_yellow": Mat("Paint006", 2.0, (0.95, 0.78, 0.3), 0.0),
    "crane_yellow": Mat("PaintedMetal009", 3.0, (1.0, 0.85, 0.45)),
    "derrick_white": Mat("PaintedMetal010", 2.0, (0.9, 0.9, 0.88)),
    "pipe_orange": Mat("PaintedMetal008", 1.5),
    "tank_white": Mat("Paint004", 2.0, (0.8, 0.81, 0.79), 0.0),
    "vessel_blue": Mat("PaintedMetal014", 2.0, (0.55, 0.7, 0.85)),
    "module_white": Mat("Paint004", 2.0, (0.82, 0.83, 0.8), 0.0),
    "module_blue": Mat("Paint004", 2.0, (0.2, 0.33, 0.45), 0.0),
    "module_grey": Mat("Paint004", 2.0, (0.45, 0.48, 0.49), 0.0),
    "steel_grey": Mat("PaintedMetal014", 3.0, (0.8, 0.82, 0.84)),
    "deck_plate": Mat("Asphalt031", 2.0, (0.42, 0.47, 0.44), 0.0),
    "deck_edge": Mat("MetalPlates010", 3.0, (0.85, 0.85, 0.83)),
    "underside": Mat("Metal053C", 3.0, (0.7, 0.7, 0.7)),
    "grating": Mat("MetalWalkway004", 0.75),
    "concrete_leg": Mat("Concrete034", 3.0, (0.62, 0.62, 0.6), 0.0),
    "helideck": Mat("Asphalt031", 2.0, None, 0.0),
    "container_green": Mat("container_side", 1.0),
    "container_blue": Mat("container_side", 1.0, (0.55, 0.75, 1.0)),
}

MESH_SUFFIXES = (
    ("_rail", "rail_yellow"), ("crane_crane", "crane_yellow"), ("derrick_derrick", "derrick_white"),
    ("_pipe", "pipe_orange"), ("_tank", "tank_white"), ("_vessel", "vessel_blue"),
    ("_module_white", "module_white"), ("_module_blue", "module_blue"), ("_module_grey", "module_grey"),
    ("_concrete", "concrete_leg"), ("_deck", "deck_plate"), ("helideck_helideck", "deck_edge"),
)


def mesh_tile(name: str) -> float:
    return MESH_TILES.get(name.rsplit("_", 1)[-1], 3.0)


def uv_scale(tile: float, texture_metres: float) -> float:
    return tile / texture_metres


def mesh_material(name: str) -> str:
    return next((mat for suffix, mat in MESH_SUFFIXES if name.endswith(suffix)), "steel_grey")


def _near(values: np.ndarray, levels, tol: float = 0.03) -> np.ndarray:
    return np.any(np.abs(values[:, None] - np.asarray(levels)[None, :]) < tol, axis=1)


def classify_faces(name: str, tris: np.ndarray) -> np.ndarray:
    """Material key per triangle of one rig mesh, (n, 3, 3) in metres."""
    z = tris[:, :, 2]
    zmin, zmax, zc = z.min(1), z.max(1), z.mean(1)
    flat = (zmax - zmin) < 0.02
    out = np.full(len(tris), mesh_material(name), dtype=object)
    undersides = [t - SLAB for t in DECK_TOPS]
    if name == "upper_structure_frame":
        out[flat & _near(zc, [DECK_TOPS[2]])] = "deck_plate"
        out[flat & _near(zc, [undersides[2]])] = "underside"
        out[~flat & (zmin > undersides[2] - 0.05) & (zmax < DECK_TOPS[2] + 0.05)] = "deck_edge"
    elif name == "legs_and_lower_deck_deck":
        out[flat & _near(zc, [DECK_TOPS[0]])] = "grating"
        out[flat & _near(zc, undersides)] = "underside"
    elif name == "legs_and_lower_deck_frame":
        for top in DECK_TOPS:
            out[(zmin > top - SLAB - 0.05) & (zmax < top + 0.05)] = "deck_edge"
    elif name == "helideck_helideck":
        out[flat & (np.abs(zc - HELIDECK_TOP) < 0.1)] = "helideck"
        out[flat & _near(zc, [HELIDECK_TOP - SLAB])] = "underside"
    return out


def helideck_pixel(x: float, y: float, px: int) -> tuple[int, int]:
    x0, y1 = HELIDECK_CENTRE[0] - HELIDECK_HALF, HELIDECK_CENTRE[1] + HELIDECK_HALF
    size = 2 * HELIDECK_HALF
    return int((y1 - y) / size * px), int((x - x0) / size * px)


def helideck_markings(px: int = 2048) -> Image.Image:
    size = 2 * HELIDECK_HALF
    cx, cy = HELIDECK_CENTRE
    coords = (np.arange(px) + 0.5) / px * size
    x = cx - HELIDECK_HALF + coords[None, :]
    y = cy + HELIDECK_HALF - coords[:, None]
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    img = np.empty((px, px, 3), np.float32)
    img[:] = (0.12, 0.3, 0.17)
    angle = np.arctan2(dy, dx)
    sector = np.mod(angle, np.pi / 3) - np.pi / 6
    edge = HELIDECK_RADIUS * math.cos(math.pi / 6) / np.maximum(np.cos(sector), 1e-6)
    img[(r > edge - 0.6) & (r < edge - 0.3)] = (0.9, 0.9, 0.88)
    img[(r > TDPM_RADIUS - 1.0) & (r < TDPM_RADIUS)] = (0.95, 0.78, 0.1)
    for x0, y0, x1, y1 in H_BARS:
        img[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = (0.92, 0.92, 0.9)
    return Image.fromarray((img * 255).astype(np.uint8))


def helideck_uv_transform(tile: float) -> tuple[float, tuple[float, float]]:
    """Scale and offset that turn box-projected st (world / tile) into marking-image uv."""
    size = 2 * HELIDECK_HALF
    x0, y0 = HELIDECK_CENTRE[0] - HELIDECK_HALF, HELIDECK_CENTRE[1] - HELIDECK_HALF
    return tile / size, (-x0 / size, -y0 / size)


def streak_mask(width: int = 512, height: int = 1024, seed: int = 0, count: int = 140) -> Image.Image:
    """Rust run-off: vertical streaks hanging from the top edge, white = rust."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), np.float32)
    rows = np.arange(height)[:, None] / height
    mask += np.clip(0.55 - rows * 4.0, 0, 1) * rng.uniform(0.6, 1.0, (1, width))
    for _ in range(count):
        x = int(rng.integers(width))
        w = int(rng.integers(1, 6))
        length = rng.uniform(0.15, 0.95)
        strength = rng.uniform(0.3, 1.0)
        fade = np.clip(1 - rows / length, 0, 1) ** 1.5 * strength
        mask[:, x:x + w] = np.maximum(mask[:, x:x + w], fade)
    img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(1.5))


def aim_matrix(pos, target) -> list[list[float]]:
    """Row-major USD transform at ``pos`` whose local -Z looks at ``target``."""
    p, t = np.asarray(pos, float), np.asarray(target, float)
    z = p - t
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return [[*x, 0.0], [*y, 0.0], [*z, 0.0], [*p, 1.0]]


def surface_points(tris: np.ndarray, spacing: float = 0.15, seed: int = 0) -> np.ndarray:
    """Points covering triangle surfaces and edges about ``spacing`` apart."""
    rng = np.random.default_rng(seed)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    k = np.ceil(area / spacing**2).astype(int) + 1
    idx = np.repeat(np.arange(len(tris)), k)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    inner = a[idx] + u[:, None] * (b - a)[idx] + v[:, None] * (c - a)[idx]
    edges = [inner]
    for p, q in ((a, b), (b, c), (c, a)):
        n = np.ceil(np.linalg.norm(q - p, axis=1) / spacing).astype(int) + 1
        j = np.repeat(np.arange(len(tris)), n)
        s = np.concatenate([np.linspace(0, 1, m) for m in n])
        edges.append(p[j] + s[:, None] * (q - p)[j])
    return np.concatenate(edges)


def box_blocked(points: np.ndarray, lo, hi) -> bool:
    inside = np.all((points >= np.asarray(lo)) & (points <= np.asarray(hi)), axis=1)
    return bool(inside.any())


def boxes_overlap(a, b) -> bool:
    return all(a[0][i] < b[1][i] and b[0][i] < a[1][i] for i in range(3))


@dataclass(frozen=True)
class Light:
    name: str
    kind: str
    pos: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    target: tuple[float, float, float] | None = None
    size: tuple[float, float] = (0.2, 0.2)
    cone: float | None = None
    fixture: str = ""


SODIUM = (1.0, 0.62, 0.28)
LED = (1.0, 0.86, 0.68)
RED = (1.0, 0.06, 0.03)
FLOOD_TARGET = (5.0, 0.0, DECK_TOPS[2])


def lights() -> list[Light]:
    out = []
    for i, (x, y) in enumerate((x, y) for x in (-17.0, -6.0, 6.0, 17.0) for y in (-33.0, -18.0, -3.0, 12.0)):
        out.append(Light(f"lower_ceiling_{i}", "rect", (x, y, DECK_TOPS[1] - SLAB - 0.12), SODIUM, 3.0e5,
                         (x, y, 0.0), (1.2, 0.35), fixture="ceiling"))
    middle = [(x, y) for x in (-17.0, -4.0, 9.0, 22.0, 33.0) for y in (-21.0, -6.0, 8.0, 21.0)]
    for i, (x, y) in enumerate(p for p in middle if p != (-17.0, 21.0)):
        out.append(Light(f"middle_ceiling_{i}", "rect", (x, y, DECK_TOPS[2] - SLAB - 0.12), LED, 2.5e5,
                         (x, y, 0.0), (1.2, 0.35), fixture="ceiling"))
    floods = [(-26.8, 26.6), (36.8, 26.6), (36.8, -9.5), (-3.0, 26.6), (11.6, -26.8), (36.8, 10.0)]
    for i, (x, y) in enumerate(floods):
        out.append(Light(f"top_flood_{i}", "rect", (x, y, DECK_TOPS[2] + 7.0), LED, 2.0e6, FLOOD_TARGET,
                         (0.6, 0.4), cone=55.0, fixture="flood"))
    out.append(Light("laydown_flood", "rect", (36.8, -38.4, DECK_TOPS[1] + 7.0), SODIUM, 1.5e6,
                     (30.0, -28.0, DECK_TOPS[1]), (0.6, 0.4), cone=55.0, fixture="flood"))
    hx, hy = HELIDECK_CENTRE
    for i, a in enumerate((30.0, 150.0, 270.0)):
        x = hx + (HELIDECK_RADIUS + 0.4) * math.cos(math.radians(a))
        y = hy + (HELIDECK_RADIUS + 0.4) * math.sin(math.radians(a))
        out.append(Light(f"helideck_flood_{i}", "rect", (x, y, HELIDECK_TOP + 0.5), (1.0, 0.95, 0.85), 4.0e5,
                         (hx, hy, HELIDECK_TOP), (0.5, 0.2), cone=70.0, fixture="flood"))
    for name, pos in (("derrick", (20.0, 20.0, 81.7)), ("crane", (34.6, -4.5, 51.6)),
                      ("living_quarters", (-11.8, 20.6, 41.6)), ("derrick_mid", (15.2, 15.2, 62.0))):
        out.append(Light(f"aviation_{name}", "sphere", pos, RED, 6.0e5, size=(0.2, 0.2), fixture="beacon"))
    return out


def helideck_lamps() -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Inset perimeter (green) and TD/PM circle (yellow) lamps as (position, colour)."""
    hx, hy = HELIDECK_CENTRE
    lamps = []
    corners = [(hx + HELIDECK_RADIUS * math.cos(math.radians(60 * k)), hy + HELIDECK_RADIUS * math.sin(math.radians(60 * k)))
               for k in range(7)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        for s in np.linspace(0, 1, 6, endpoint=False):
            lamps.append(((x0 + s * (x1 - x0), y0 + s * (y1 - y0), HELIDECK_TOP + 0.05), (0.1, 1.0, 0.25)))
    for k in range(16):
        a = 2 * math.pi * k / 16
        lamps.append(((hx + (TDPM_RADIUS + 0.2) * math.cos(a), hy + (TDPM_RADIUS + 0.2) * math.sin(a),
                       HELIDECK_TOP + 0.05), (1.0, 0.8, 0.1)))
    return lamps


@dataclass(frozen=True)
class Prop:
    name: str
    asset: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    rot_z: float = 0.0
    on_floor: bool = True

    def box(self):
        sx, sy, sz = self.size
        if round(self.rot_z / 90) % 2:
            sx, sy = sy, sx
        x, y, z = self.pos
        return (x - sx / 2, y - sy / 2, z), (x + sx / 2, y + sy / 2, z + sz)


CONTAINER = (6.06, 2.44, 2.59)
IBC = (1.05, 1.26, 1.41)
DRUM = (0.64, 0.66, 0.87)
LPG = (0.4, 0.4, 1.3)

WALKWAYS = (
    ((-25.0, -1.5, DECK_TOPS[0]), (25.0, 1.5, DECK_TOPS[0] + 2.2)),
    ((-25.0, -24.5, DECK_TOPS[0]), (25.0, -21.5, DECK_TOPS[0] + 2.2)),
    ((-25.0, 1.0, DECK_TOPS[1]), (37.5, 4.0, DECK_TOPS[1] + 2.2)),
    ((24.0, -39.0, DECK_TOPS[1]), (26.0, 25.0, DECK_TOPS[1] + 2.2)),
    ((-27.5, 1.0, DECK_TOPS[2]), (37.5, 3.5, DECK_TOPS[2] + 2.2)),
    ((-2.0, -27.5, DECK_TOPS[2]), (1.0, 27.5, DECK_TOPS[2] + 2.2)),
)


def _row(prefix, asset, size, start, step, n, rot=0.0):
    x, y, z = start
    return [Prop(f"{prefix}_{i}", asset, (x + step[0] * i, y + step[1] * i, z), size, rot) for i in range(n)]


def props() -> list[Prop]:
    low, mid, top = DECK_TOPS
    out = [
        Prop("container_0", "container", (29.2, -33.0, mid), CONTAINER, 90.0),
        Prop("container_1", "container", (32.4, -33.0, mid), CONTAINER, 90.0),
        Prop("container_2", "container", (29.2, -33.0, mid + CONTAINER[2]), CONTAINER, 90.0, on_floor=False),
        Prop("container_3", "container", (-19.0, 23.6, top), CONTAINER),
        Prop("container_4", "container", (-19.0, 20.6, top), CONTAINER),
        Prop("generator_0", "portable_generator", (-8.0, 23.5, top), (1.2, 0.8, 0.9)),
        Prop("welding_cart_0", "portable_welding_cart", (-10.5, 23.5, top), (1.0, 0.6, 1.2)),
    ]
    out += _row("ibc_mid", "nv_ibc_tank", IBC, (28.0, -22.5, mid), (1.4, 0, 0), 4)
    out += _row("drum_mid", "nv_steel_drum", DRUM, (35.6, -24.0, mid), (0, 0.8, 0), 4)
    out += _row("lpg_mid", "small_lpg_tank", LPG, (35.6, -18.5, mid), (0, 0.55, 0), 5)
    out += _row("ibc_low", "nv_ibc_tank", IBC, (-21.0, -12.0, low), (0, 1.5, 0), 3)
    out += _row("drum_low", "nv_steel_drum", DRUM, (-23.0, -12.0, low), (0, 0.8, 0), 5)
    out += _row("drum_low_b", "Barrel_01", DRUM, (-23.9, -12.0, low), (0, 0.8, 0), 4)
    out += _row("lpg_top", "small_lpg_tank", LPG, (-26.2, 6.0, top), (0, 0.55, 0), 6)
    out += _row("ibc_top", "nv_ibc_tank", IBC, (-12.0, 9.0, top), (1.4, 0, 0), 3)
    out += [
        Prop("toolchest_0", "metal_tool_chest", (16.5, -14.0, low), (0.7, 0.5, 1.0)),
        Prop("toolcart_0", "tool_cart", (16.5, -16.0, low), (0.8, 0.5, 1.0)),
        Prop("rack_0", "worn_metal_rack", (-22.5, 8.0, low), (1.8, 0.6, 2.0), 90.0),
        Prop("eyewash_0", "nv_eyewash_shower", (-24.5, -16.0, low), (1.13, 0.38, 2.32), 90.0),
        Prop("fire_cabinet_0", "nv_fire_cabinet", (36.9, -12.0, mid), (1.22, 0.4, 1.46), 90.0),
        Prop("hydrant_0", "fire_hydrant", (-26.6, 12.5, top), (0.87, 0.32, 0.8), 90.0),
    ]
    out += lifebuoys()
    return out


def lifebuoys() -> list[Prop]:
    top = DECK_TOPS[2] + 0.35
    spots = [((x, 27.0), 0.0) for x in (-20.0, 4.0, 28.0)] + [((37.1, y), 90.0) for y in (-4.0, 18.0)]
    spots += [((-27.1, y), 90.0) for y in (16.0,)] + [((x, -27.2), 0.0) for x in (6.0,)]
    return [Prop(f"lifebuoy_{i}", "lifebuoy", (x, y, top), (0.84, 0.16, 0.84), rot, on_floor=False)
            for i, ((x, y), rot) in enumerate(spots)]


@dataclass(frozen=True)
class Window:
    pos: tuple[float, float, float]
    normal: tuple[float, float]
    lit: bool


LIVING_QUARTERS = (-21.0, 10.1, 31.25, -11.4, 21.0, 41.2)


def windows(seed: int = 3) -> list[Window]:
    rng = np.random.default_rng(seed)
    x0, y0, z0, x1, y1, _ = LIVING_QUARTERS
    faces = [((x0 - 0.03, None), (-1, 0)), ((x1 + 0.03, None), (1, 0)), ((None, y1 + 0.03), (0, 1)), ((None, y0 - 0.03), (0, -1))]
    out = []
    for (fx, fy), normal in faces:
        lo, hi = (y0, y1) if fx is not None else (x0, x1)
        for z in (z0 + 1.7, z0 + 4.7, z0 + 7.7):
            for s in np.arange(lo + 1.4, hi - 1.0, 2.4):
                pos = (fx, float(s), z) if fx is not None else (float(s), fy, z)
                out.append(Window(pos, normal, bool(rng.random() < 0.7)))
    return out


SIGNS = (
    ("e007", (-16.2, 9.6, DECK_TOPS[1] + 1.8), (0, -1)),
    ("m014", (-13.0, 9.6, DECK_TOPS[1] + 1.8), (0, -1)),
    ("p002", (-21.5, 14.0, DECK_TOPS[1] + 1.8), (-1, 0)),
    ("e003", (-21.5, 17.0, DECK_TOPS[1] + 1.8), (-1, 0)),
    ("w021", (-0.95, -16.0, DECK_TOPS[0] + 1.8), (1, 0)),
    ("w012", (4.05, -12.0, DECK_TOPS[0] + 1.8), (1, 0)),
)


FLARE_BASE = (36.0, 26.0, DECK_TOPS[2] + 1.0)
FLARE_DIR = (0.62, 0.62, 0.48)
FLARE_LENGTH = 32.0


def flare_boom(width: float = 1.6, bays: int = 12) -> tuple[list[tuple[tuple, tuple, float]], tuple[float, float, float]]:
    """Triangular lattice boom as (start, end, radius) members, and the flame position past its tip."""
    d = np.asarray(FLARE_DIR, float)
    d /= np.linalg.norm(d)
    side = np.cross(d, (0.0, 0.0, 1.0))
    side /= np.linalg.norm(side)
    up = np.cross(side, d)
    base = np.asarray(FLARE_BASE, float)
    offsets = [width / 2 * side, -width / 2 * side, width * 0.8 * up]
    stations = [base + d * FLARE_LENGTH * i / bays for i in range(bays + 1)]
    members = []
    for k, off in enumerate(offsets):
        members.append((tuple(stations[0] + off), tuple(stations[-1] + off * 0.4), 0.12))
    for i in range(bays + 1):
        taper = 1 - 0.6 * i / bays
        ring = [stations[i] + off * taper for off in offsets]
        for k in range(3):
            members.append((tuple(ring[k]), tuple(ring[(k + 1) % 3]), 0.05))
        if i < bays:
            nxt = [stations[i + 1] + off * (1 - 0.6 * (i + 1) / bays) for off in offsets]
            for k in range(3):
                members.append((tuple(ring[k]), tuple(nxt[(k + 1) % 3]), 0.05))
    flame = base + d * (FLARE_LENGTH + 2.5)
    return members, tuple(float(v) for v in flame)

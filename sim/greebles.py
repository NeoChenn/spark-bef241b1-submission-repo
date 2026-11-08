from __future__ import annotations

import math
import os
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
LAYER = SIM_DIR / "output" / "greebles.usda"
TEXTURES = SIM_DIR / "assets" / "generated" / "greebles"
ROOT = "/World/Greebles"
ZONE_ROOT = f"{ROOT}/Zone"
OFFLINE = f"{ROOT}/Offline"
ZONE_T = (18.75, -5.75, 42.5)
WORK = ((-2.5, 2.5), (-2.2, 2.2))
DAMAGE = (-0.4933, 0.095, 1.2199)
MAIN = {"y": 0.0, "z": 1.1, "r": 0.15, "x": (-7.6, 7.6)}

LINES = {
    "main": (0.0, 1.1, 0.15, "green_metal_rust"),
    "grey": (-3.0, 0.7, 0.1, "pipe_grey"),
    "orange": (-3.5, 0.7, 0.07, "pipe_orange"),
    "rusty": (3.2, 0.8, 0.12, "rusty_painted_metal"),
}
EXISTING_FLANGES = {"main": (-7.6, -3.0, 3.0, 7.6), "grey": (-7.6, 7.6), "orange": (-7.6, 7.6), "rusty": (-7.6, 7.6)}
NEW_FLANGES = {"grey": (-4.2, 0.9, 4.8), "orange": (-1.9, 2.9, 5.6), "rusty": (-3.9, 1.3, 4.7)}
SUPPORTS = {"main": (-6.0, -3.4, 3.4, 6.0), "grey": (-6.5, -2.5, 2.5, 6.5), "orange": (-6.5, -2.5, 2.5, 6.5),
            "rusty": (-6.5, -2.5, 2.5, 6.5)}
WELDS = {"main": (-5.6, -2.1, 1.5, 5.3), "grey": (-5.6, -2.05, 3.1), "orange": (-4.6, -0.6, 2.25), "rusty": (-5.2, -1.6, 2.9)}
LABELS = (
    ("main", 2.05, "crude", 1), ("main", -4.3, "crude", 1), ("grey", -0.25, "water", -1), ("grey", 3.6, "water", -1),
    ("orange", 0.4, "gas", 1), ("orange", -5.2, "gas", 1), ("rusty", -1.0, "foam", -1), ("rusty", 3.0, "foam", -1),
)
GAUGES = (
    ("grey", -1.75, "dial_a", 1), ("rusty", 2.15, "dial_b", 1), ("main", -3.75, "dial_b", 1), ("main", 4.05, "dial_a", 1),
    ("orange", 1.25, "dial_a", 1),
)
HANDWHEELS = (("orange", 1.75, 0.34), ("rusty", -2.0, 0.42), ("grey", -3.3, 0.36), ("rusty", 5.6, 0.4))
TRANSMITTERS = ((3.85, 0.62, "main", 3.65), (-3.95, -0.62, "main", -3.7))
TUBING_Y = (-2.62, -2.67, -2.72)
TUBING_X = (-6.8, 6.8)
COLUMNS = ((-7.0, -4.4), (7.0, -4.4), (-7.0, 4.4), (0.0, 4.4), (7.0, 4.4))
TRAYS = ((-4.4, -6.75, 6.75), (4.4, -6.75, -0.3), (4.4, 0.3, 6.75))
TRAY_Z = 3.72
JBOXES = (
    ((0.0, 4.4), "-y", 1.45, "jb"), ((0.0, 4.4), "+y", 1.35, "esd"), ((-7.0, 4.4), "+x", 1.4, "jb"),
    ((7.0, 4.4), "-x", 1.4, "esd"), ((-7.0, -4.4), "+x", 1.45, "jb"), ((7.0, -4.4), "-x", 1.3, "socket"),
    ((0.0, 4.4), "+x", 1.2, "switch"), ((7.0, -4.4), "+y", 1.55, "jb"),
)
SIGNS = (
    ((0.0, 4.4), "-y", 2.05, "no_smoking", 0.34, 0.46), ((0.0, 4.4), "+y", 2.0, "ppe", 0.34, 0.46),
    ((0.0, 4.4), "+x", 1.85, "hazard_flammable", 0.3, 0.3), ((-7.0, 4.4), "+y", 1.95, "muster", 0.5, 0.34),
    ((7.0, 4.4), "+y", 1.9, "h2s", 0.4, 0.34), ((-7.0, -4.4), "+y", 1.95, "hazard_flammable", 0.3, 0.3),
    ((7.0, -4.4), "+y", 2.05, "no_smoking", 0.34, 0.46), ((7.0, 4.4), "-y", 1.95, "ppe", 0.34, 0.46),
    ((-7.0, 4.4), "-y", 2.0, "h2s", 0.4, 0.34),
)
DRAINS = ((-1.0, -2.9), (-4.6, 2.55), (5.2, -2.55), (-1.6, 2.6))
DEBRIS = (
    ("rag", (-1.75, -2.88), 30.0), ("rag", (4.3, 2.55), 110.0), ("bucket", (2.05, -2.55), 0.0),
    ("cable_coil", (-2.4, 2.75), 0.0), ("puddle", (-0.3, -2.95), 15.0), ("nuts", (-0.6, -2.84), 0.0), ("spanner", (-1.38, -2.82), 65.0), ("puddle", (3.9, -1.0), 70.0),
    ("hose_coil", (-5.4, -2.45), 0.0), ("chain", (4.8, 2.9), 20.0), ("gloves", (2.6, -2.35), 140.0),
)
STRIPES = tuple(c for c in COLUMNS)
ALLOWED_IN_WORK = ("weld", "label")
ATLAS_GRID = 4
ATLAS_PAD = 3 / 2048
ATLAS = {
    "no_smoking": (0, 0, 1, 1), "ppe": (1, 0, 1, 1), "hazard_flammable": (2, 0, 1, 1), "muster": (3, 0, 1, 1),
    "h2s": (0, 1, 1, 1), "dial_a": (1, 1, 1, 1), "dial_b": (2, 1, 1, 1), "label_crude_f": (0, 2, 2, 1),
    "label_water_r": (2, 2, 2, 1), "label_gas_f": (0, 3, 2, 1), "label_foam_r": (2, 3, 2, 1),
}


def label_key(name: str, direction: int) -> str:
    return f"label_{name}_{'f' if direction > 0 else 'r'}"


def atlas_uv(name: str, u: float, v: float) -> tuple[float, float]:
    col, row, w, h = ATLAS[name]
    u0, u1 = col / ATLAS_GRID + ATLAS_PAD, (col + w) / ATLAS_GRID - ATLAS_PAD
    v0, v1 = 1 - (row + h) / ATLAS_GRID + ATLAS_PAD, 1 - row / ATLAS_GRID - ATLAS_PAD
    return u0 + (u1 - u0) * u, v0 + (v1 - v0) * v


def world(p):
    return tuple(a + b for a, b in zip(p, ZONE_T)) + tuple(p[3:])


def bolt_ring(radius: float, count: int, phase: float = 0.5) -> list[tuple[float, float]]:
    return [(radius * math.cos(2 * math.pi * (k + phase) / count), radius * math.sin(2 * math.pi * (k + phase) / count))
            for k in range(count)]


def bolt_count(pipe_radius: float) -> int:
    return 4 if pipe_radius < 0.06 else 8 if pipe_radius < 0.16 else 12


def flange_dims(pipe_radius: float) -> dict:
    outer = pipe_radius * 1.45
    return {"outer": outer, "pcd": (pipe_radius + outer) / 2 + 0.004, "bolt_r": max(0.006, pipe_radius * 0.075),
            "nut_r": max(0.011, pipe_radius * 0.13), "half_width": 0.05, "count": bolt_count(pipe_radius)}


def stations(a: float, b: float, spacing: float, margin: float = 0.0) -> list[float]:
    a, b = a + margin, b - margin
    if b <= a:
        return []
    count = max(1, round((b - a) / spacing))
    return [a + (b - a) * i / count for i in range(count + 1)]


def facing(face: str) -> tuple[float, float]:
    return {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}[face]


def on_column(column, face: str, standoff: float = 0.125) -> tuple[float, float]:
    nx, ny = facing(face)
    return column[0] + nx * standoff, column[1] + ny * standoff


def in_work_zone(x: float, y: float, margin: float = 0.0) -> bool:
    (x0, x1), (y0, y1) = WORK
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


def near_damage(x: float, y: float, z: float, radius: float = 0.9) -> bool:
    return math.dist((x, y, z), DAMAGE) < radius


def footprints() -> list[tuple[str, float, float, float]]:
    items = []
    for line, xs in NEW_FLANGES.items():
        y, z, _, _ = LINES[line]
        items += [("flange", x, y, z) for x in xs]
    for line, xs in WELDS.items():
        y, z, _, _ = LINES[line]
        items += [("weld", x, y, z) for x in xs]
    for line, x, _, _ in LABELS:
        y, z, _, _ = LINES[line]
        items.append(("label", x, y, z))
    for line, x, _, _ in GAUGES:
        y, z, r, _ = LINES[line]
        items.append(("gauge", x, y, z + r))
    for line, x, _ in HANDWHEELS:
        y, z, r, _ = LINES[line]
        items.append(("handwheel", x, y, z + r))
    items += [("transmitter", x, y, 0.0) for x, y, _, _ in TRANSMITTERS]
    items += [("tubing", x, y, 0.1) for x in TUBING_X for y in TUBING_Y]
    items += [("drain", x, y, 0.0) for x, y in DRAINS]
    items += [(kind, x, y, 0.0) for kind, (x, y), _ in DEBRIS]
    return items


def layout_issues() -> list:
    issues = []
    for kind, x, y, z in footprints():
        if kind not in ALLOWED_IN_WORK and in_work_zone(x, y, 0.1):
            issues.append((kind, x, y, z))
        elif near_damage(x, y, z):
            issues.append((kind, x, y, z))
    return issues


def current_quality() -> str:
    return os.environ.get("RIG_QUALITY", "live")


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_GREEBLES", "1") == "0":
        return False
    root = stage.GetRootLayer()
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    if not any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        root.subLayerPaths.insert(0, target)
    return True


def apply(stage, quality: str | None = None) -> dict:
    if not attach(stage):
        return {}
    offline = (quality or current_quality()) == "offline"
    zone = stage.GetPrimAtPath("/World/RepairZone").IsValid()
    for path, active in ((OFFLINE, offline), (ZONE_ROOT, zone), (f"{OFFLINE}/Zone", zone and offline)):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(active)
    return {"offline": offline, "zone": zone}

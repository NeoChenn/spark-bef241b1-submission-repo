from __future__ import annotations

import os
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
LAYER = SIM_DIR / "output" / "assets_hero.usda"
ROOT = "/World/AssetsHero"
OFFLINE = f"{ROOT}/Offline"
DECK = 42.5
WORK = (16.25, 21.25, -7.95, -3.55)
ROUTE = [(-21.0, 20.5), (-10.0, 20.5), (-10.0, 11.0), (-1.0, 11.0), (-1.0, -4.5), (24.0, -4.5), (24.0, -9.0),
         (9.0, -9.0), (9.0, -4.5), (-1.0, -4.5)]
EYES = [(25.25, 3.25), (30.0, -2.0), (21.61, 1.17), (19.64, -2.93), (36.74, 8.96), (19.03, -4.13)]

PLACE = [
    ("fuel_pump_skid", 13.4, -7.05, 90.0, 1.5),
    ("nv_tripod_light", 11.6, -7.3, 300.0, 1.0),
    ("nv_tripod_light", 22.7, -7.3, 230.0, 1.0),
    ("fuel_gate_valve", 13.3, -11.45, 180.0, 1.0),
    ("fuel_gate_valve", 14.4, -11.45, 180.0, 1.0),
    ("nv_air_compressor_a01", 10.9, -12.0, 0.0, 1.5),
    ("fuel_pump_skid", 8.6, -14.6, 0.0, 1.5),
    ("fuel_pump_skid", 10.0, -14.6, 0.0, 1.5),
    ("nv_ibc_steel_a01", 11.9, -14.2, 6.0, 1.0),
    ("nv_ibc_steel_a01", 11.9, -15.55, -3.0, 1.0),
    ("nv_stillage_d01", 9.4, -17.6, 90.0, 1.0),
    ("fuel_cable_spool", 11.8, -17.4, 20.0, 1.0),
    ("nv_pipe_bundle_b08", 10.6, -21.6, 0.0, 1.0),
    ("nv_hopper_blue_a04", 8.2, -20.2, 0.0, 1.0),
    ("nv_scaffold_a01", 2.0, -12.0, 90.0, 1.0),
    ("nv_hose_reel_exhaust", 1.2, -14.4, 90.0, 1.0),
]
WALL = [
    ("fuel_junction_box", 11.75, -10.15, "+y", 1.25),
    ("fuel_junction_box", 25.75, -10.15, "+y", 1.25),
    ("nv_first_aid_a01", 11.75, -1.35, "-y", 1.1),
    ("nv_extinguisher_a01", 25.75, -1.35, "-y", 0.9),
]
HUNG = [
    ("lifebuoy", 9.0, -27.35, 0.0, 1.05),
    ("lifebuoy", 2.0, -27.35, 0.0, 1.05),
    ("lifebuoy", 37.35, -6.0, 90.0, 1.05),
    ("lifebuoy", 37.35, 4.0, 90.0, 1.05),
    ("exterior_aircon_unit", 5.0, -27.2, 0.0, 0.35),
]
COLUMN_HALF = 0.125
FACING = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}
HIDE = []


def footprints() -> list:
    return [(x, y) for _, x, y, *_ in PLACE + WALL + HUNG]


def inside(point, rect, margin=0.0) -> bool:
    x0, x1, y0, y1 = rect
    return x0 - margin <= point[0] <= x1 + margin and y0 - margin <= point[1] <= y1 + margin


def layout_issues(radius: float = 1.0) -> list:
    from dressing_geometry import blocked

    issues = [p for p in footprints() if inside(p, WORK, margin=radius)]
    issues += [(x, y) for _, x, y, *_ in PLACE if blocked((x, y), ROUTE, [], EYES, corridor=1.3, eye_radius=1.2)]
    return issues


def wall_pose(x: float, y: float, face: str) -> tuple:
    sign = 1.0 if face[0] == "+" else -1.0
    dx, dy = (sign * COLUMN_HALF, 0.0) if face[1] == "x" else (0.0, sign * COLUMN_HALF)
    return x + dx, y + dy, FACING[face]


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_ASSETS_HERO", "1") == "0":
        return False
    root = stage.GetRootLayer()
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    if not any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        root.subLayerPaths.insert(0, target)
    offline = stage.GetPrimAtPath(OFFLINE)
    if offline.IsValid():
        offline.SetActive(os.environ.get("RIG_QUALITY", "live") == "offline")
    return True

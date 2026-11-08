from __future__ import annotations

import os
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
LAYER = Path(os.environ.get("RIG_ASSETS_HERO_LAYER", SIM_DIR / "output" / "assets_hero.usda"))
ROOT = "/World/AssetsHero"
OFFLINE = f"{ROOT}/Offline"
REMOTE = f"{ROOT}/Remote"
PROBE = ("https://omniverse-content-production.s3-us-west-2.amazonaws."
         "com/Assets/Isaac/6.2/Isaac/SimReady/Industrial/Warehouse/Containers/Ibc_Tank_Steel_Stainless_A01/sm_ibcTank_steel_stainless_a01_01.usd")
DECK = 42.5
WORK = (16.25, 21.25, -7.95, -3.55)
ROUTE = [(-21.0, 20.5), (-10.0, 20.5), (-10.0, 11.0), (-1.0, 11.0), (-1.0, -4.5), (24.0, -4.5), (24.0, -9.0),
         (9.0, -9.0), (9.0, -4.5), (-1.0, -4.5)]
EYES = [(25.25, 3.25), (30.0, -2.0), (21.61, 1.17), (19.64, -2.93), (36.74, 8.96), (19.03, -4.13)]

PLACE = [
    ("fuel_pump_skid", 13.4, -7.05, 90.0, 1.5),
    ("nv_tripod_light", 22.7, -7.3, 230.0, 1.0),
    ("fuel_gate_valve", 13.3, -11.45, 180.0, 1.0),
    ("fuel_gate_valve", 14.4, -11.45, 180.0, 1.0),
    ("fuel_pump_skid", 8.6, -14.6, 0.0, 1.5),
    ("fuel_pump_skid", 10.0, -14.6, 0.0, 1.5),
    ("nv_ibc_steel_a01", 11.9, -14.2, 6.0, 1.0),
    ("nv_ibc_steel_a01", 11.9, -15.55, -3.0, 1.0),
    ("nv_stillage_d01", 9.4, -17.6, 90.0, 1.0),
    ("fuel_cable_spool", 11.8, -17.4, 20.0, 1.0),
    ("nv_pipe_bundle_b08", 10.6, -21.6, 0.0, 1.0),
    ("nv_hopper_blue_a04", 8.2, -20.2, 0.0, 1.0),
    ("nv_scaffold_a01", 2.0, -12.0, 90.0, 1.0),
]
BEHIND_PIPE = (15.4, 18.0, -7.9, -6.6)
REPAIR_BG = [
    ("fuel_gate_valve", 16.0, -6.92, 90.0, 1.0),
    ("Barrel_02", 16.3, -7.55, 0.0, 1.0),
    ("metal_jerrycan", 16.85, -7.62, 75.0, 1.0),
    ("metal_toolbox", 17.45, -7.6, 15.0, 1.0),
]
DENSE = [
    ("small_lpg_tank", 24.2, 0.7, 0.0, 0.0),
    ("small_lpg_tank", 24.65, 0.72, 35.0, 0.0),
    ("small_lpg_tank", 25.1, 0.68, 80.0, 0.0),
    ("small_lpg_tank", 24.25, 1.15, 120.0, 0.0),
    ("small_lpg_tank", 24.7, 1.17, 160.0, 0.0),
    ("small_lpg_tank", 25.15, 1.13, 200.0, 0.0),
    ("worn_metal_rack", 24.7, 1.85, 0.0, 0.0),
    ("metal_tool_chest", 27.0, 1.2, 180.0, 0.0),
    ("metal_toolbox", 27.1, 1.2, 170.0, 0.83),
    ("hand_truck", 28.0, 1.35, 200.0, 0.0),
    ("fuel_cable_spool", 28.95, 1.0, 10.0, 0.0),
    ("fuel_steel_drum", 30.4, 0.7, 0.0, 0.0),
    ("fuel_steel_drum", 31.0, 0.75, 50.0, 0.0),
    ("fuel_steel_drum", 30.7, 1.28, 100.0, 0.0),
    ("fuel_steel_drum", 31.3, 1.32, 150.0, 0.0),
    ("fuel_euro_pallet", 32.45, 1.0, 0.0, 0.0),
    ("Barrel_02", 32.15, 1.0, 0.0, 0.144),
    ("Barrel_02", 32.72, 1.02, 60.0, 0.144),
    ("Barrel_01", 33.5, 0.65, 20.0, 0.0),
    ("fuel_euro_pallet", 35.6, 1.0, 90.0, 0.0),
    ("wooden_crate_02", 35.6, 1.0, 90.0, 0.144),
    ("plastic_crate_01", 35.55, 0.75, 0.0, 0.6),
    ("plastic_crate_01", 35.6, 1.25, 10.0, 0.6),
    ("fuel_fire_hydrant", 35.6, -3.4, 200.0, 0.0),
    ("korean_fire_extinguisher_01", 35.4, -4.3, 190.0, 0.0),
    ("korean_fire_extinguisher_01", 35.8, -4.4, 170.0, 0.0),
    ("fuel_extinguisher_cabinet", 35.7, -5.2, 180.0, 0.0),
    ("fuel_gate_valve", 28.0, -9.5, 180.0, 0.0),
    ("fuel_gate_valve", 30.5, -9.5, 180.0, 0.0),
    ("fuel_gate_valve", 33.0, -9.5, 180.0, 0.0),
    ("fuel_pump_skid", 29.2, -11.25, 90.0, 0.0),
    ("fuel_pump_skid", 32.2, -11.25, 90.0, 0.0),
    ("fuel_steel_drum", 34.6, -11.2, 0.0, 0.0),
    ("fuel_steel_drum", 35.2, -11.25, 40.0, 0.0),
    ("industrial_pastic_container", 35.9, -11.2, 90.0, 0.0),
]
DENSE_SCALE = {"fuel_pump_skid": 1.5}
DENSE_OFFLINE = [
    ("nv_ibc_steel_a01", 34.4, 1.1, 0.0, 0.0),
    ("nv_eyewash_shower", 35.9, -6.6, 90.0, 0.0),
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
    ("lifebuoy", 37.35, -1.0, 90.0, 1.05),
    ("exterior_aircon_unit", 5.0, -27.2, 0.0, 0.35),
]
COLUMN_HALF = 0.125
FACING = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}
HIDE = []


def footprints() -> list:
    return [(x, y) for _, x, y, *_ in PLACE + WALL + HUNG + REPAIR_BG + DENSE + DENSE_OFFLINE]


def inside(point, rect, margin=0.0) -> bool:
    x0, x1, y0, y1 = rect
    return x0 - margin <= point[0] <= x1 + margin and y0 - margin <= point[1] <= y1 + margin


def layout_issues(radius: float = 1.0) -> list:
    from dressing_geometry import blocked

    issues = [p for p in footprints() if inside(p, WORK, margin=radius) and not inside(p, BEHIND_PIPE)]
    issues += [(x, y) for _, x, y, *_ in PLACE + DENSE + DENSE_OFFLINE if blocked((x, y), ROUTE, [], EYES, corridor=1.3, eye_radius=1.2)]
    return issues


def wall_pose(x: float, y: float, face: str) -> tuple:
    sign = 1.0 if face[0] == "+" else -1.0
    dx, dy = (sign * COLUMN_HALF, 0.0) if face[1] == "x" else (0.0, sign * COLUMN_HALF)
    return x + dx, y + dy, FACING[face]


def remote_ok(timeout: float = 2.0) -> bool:
    import urllib.request

    default = "1" if os.environ.get("RIG_QUALITY", "live") == "offline" else "0"
    if os.environ.get("RIG_ASSETS_REMOTE", default) == "0":
        return False
    try:
        urllib.request.urlopen(urllib.request.Request(PROBE, method="HEAD"), timeout=timeout)
        return True
    except Exception:
        return False


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_ASSETS_HERO", "1") == "0":
        return False
    root = stage.GetRootLayer()
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    if not any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        root.subLayerPaths.insert(0, target)
    remote = stage.GetPrimAtPath(REMOTE)
    if remote.IsValid():
        remote.SetActive(remote_ok())
    offline = stage.GetPrimAtPath(OFFLINE)
    if offline.IsValid():
        offline.SetActive(os.environ.get("RIG_QUALITY", "live") == "offline")
    return True

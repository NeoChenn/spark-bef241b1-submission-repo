from __future__ import annotations

import numpy as np

from policy.fleet.fleet import Fleet
from policy.fleet.rigmap import load_rig_map

RUST_SPOTS = [("rust_top_pipe", [-1.8, 4.0, 43.0], 0.5, 0.4),
              ("rust_mid_east", [12.0, -6.8, 32.0], 0.6, 0.6),
              ("rust_mid_west", [-21.8, 8.0, 32.0], 0.4, 0.6),
              ("rust_low_splash", [-10.0, -4.8, 21.0], 0.7, 0.95),
              ("rust_low_east", [20.8, 6.0, 21.0], 0.5, 0.9)]
ROBOTS = [("R1", [-12.0, 9.4, 42.5], 0.95), ("R2", [9.0, -4.5, 42.5], 0.70), ("R3", [0.0, 4.0, 31.25], 0.36)]
EVENTS = [(90.0, "new_rust", {"poi": "rust_low_north", "xyz": [10.0, 12.8, 21.0], "severity": 0.9, "exposure": 1.0}),
          (205.0, "person", {"path": [[20.0, -4.0, 20.0], [20.0, 12.0, 20.0], [1.0, 12.0, 20.0]], "speed": 0.6,
                             "radius": 3.0, "duration": 90.0}),
          (240.0, "gas_alarm", {"flange": "F5", "scenario": "loose"}),
          (330.0, "robot_failure", {"robot": "R2"}),
          (420.0, "crack", {"flange": "F3"}),
          (520.0, "gas_alarm", {"flange": "F2", "scenario": "loose"})]
INSPECT_DEADLINE_S = 900.0
CHECK_DEADLINE_S = 1500.0


def nearest_node(rm, xyz) -> int:
    d = [float(np.linalg.norm(p - np.asarray(xyz))) for p in rm.pos]
    return int(np.argmin(d))


def build_default(n_robots: int = 3, jobs=None, rm=None) -> Fleet:
    rm = rm or load_rig_map()
    f = Fleet(rm, jobs)
    for i in range(n_robots):
        rid, xyz, soc = ROBOTS[i] if i < len(ROBOTS) else (f"R{i + 1}", ROBOTS[0][1], 0.9)
        f.add_robot(rid, nearest_node(rm, xyz), soc)
    for p in rm.inspection:
        if p["type"] == "gauge":
            f.add_task("inspect", p["id"], deadline=INSPECT_DEADLINE_S)
    for fid in rm.flanges:
        f.add_task("leak_check", fid, deadline=CHECK_DEADLINE_S)
    for poi, xyz, sev, exp in RUST_SPOTS:
        f.add_task("rust", poi, xyz, severity=sev, exposure=exp)
    ids = {r.id for r in f.robots}
    for t, kind, data in EVENTS:
        if kind != "robot_failure" or data["robot"] in ids:
            f.schedule(t, kind, **data)
    return f


JUDGES_ROBOTS = [("R1", [-12.0, 9.4, 42.5], 0.9), ("R2", [-21.0, 4.0, 31.25], 0.9), ("R3", [10.0, 12.0, 20.0], 0.9)]
JUDGES_RUST = [("rust_mid_west", [-21.8, 8.0, 32.0], 0.4, 0.6), ("rust_mid_east", [12.0, -6.8, 32.0], 0.6, 0.6),
               ("rust_low_east", [20.8, 6.0, 21.0], 0.5, 0.9), ("rust_low_splash", [-10.0, -4.8, 21.0], 0.7, 0.95)]
JUDGES_WELD = ("weld_riser_east", [10.0, -8.0, 42.5], "crack")
JUDGES_EVENTS = [(6.0, "drop_obstacle", {"kind": "crate", "at": "R1"}),
                 (55.0, "set_weather", {"level": "storm", "wind_dir": 90.0}),
                 (200.0, "set_weather", {"level": "calm"}),
                 (200.0, "clear", {"id": "all"})]


def build_judges(n_robots: int = 3, jobs=None, rm=None, disturb: bool = True) -> Fleet:
    rm = rm or load_rig_map()
    f = Fleet(rm, jobs)
    for rid, xyz, soc in JUDGES_ROBOTS[:n_robots]:
        f.add_robot(rid, nearest_node(rm, xyz), soc)
    f.add_task("leak_check", "F5", deadline=CHECK_DEADLINE_S)
    for poi, xyz, sev, exp in JUDGES_RUST:
        f.add_task("rust", poi, xyz, severity=sev, exposure=exp)
    f.add_task("weld", JUDGES_WELD[0], JUDGES_WELD[1], scenario=JUDGES_WELD[2])
    for t, kind, data in JUDGES_EVENTS if disturb else []:
        f.schedule(t, kind, **data)
    return f


SCENARIOS = {"default": build_default, "judges": build_judges}

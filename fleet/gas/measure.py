"""Measure gas_survey localization on the seeded scenarios (the same ones the tests use).

    python -m fleet.gas.measure [n_seeds]          # offline (no Gemini)
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

# name -> (source xy, wind m/s, process flow direction, expected upstream valve)
SCENARIOS = {
    "north_rack_east_wind": ((-12.0, 23.0), (2.0, 0.5), (1.0, 0.0), "valve_03"),
    "south_rack_ne_wind": ((22.0, -8.0), (-1.0, 1.0), (1.0, 0.0), "valve_05"),
    "west_walkway_north_wind": ((-19.5, 15.0), (0.0, 1.5), (0.0, 1.0), None),
    "south_junction_calm": ((9.0, -6.5), (0.0, 0.0), None, None),
}
START = (-21.0, 20.5, 42.5)


def run_one(rig, src, wind, flow, seed):
    from fleet.contract import Job, Level, Robot, RobotKind, World
    from fleet.gas.report import Gemini
    from fleet.jobs.gas_valves import GasSurveyJob
    w = World(assets=copy.deepcopy(rig.assets), state={"seed": seed, "wind": list(wind),
              "gas_leaks": [{"id": "leak_01", "position": [*src, 43.3], "rate_g_s": 3.5, "t0": 0.0,
                             "source": "scenario", "isolated": False, "flow_dir": flow}]})
    r = Robot("q1", RobotKind.QUADRUPED, {"gas_sensor", "arm"}, START, deck=START[2])
    res = GasSurveyJob(gemini=Gemini(enabled=False)).run(
        Job("survey-1", "gas_survey", "top_deck", START, Level.AUTO, {"gas_sensor"}), r, w)
    f = res.findings[0]
    return float(np.hypot(f.position[0] - src[0], f.position[1] - src[1])), f.evidence


def main(n: int = 20) -> None:
    from fleet.contract import World
    os.environ["FLEET_GEMINI"] = "0"
    rig = World.from_rig_meta()
    allerr = []
    for name, (src, wind, flow, _) in SCENARIOS.items():
        rows = [run_one(rig, src, wind, flow, s) for s in range(n)]
        errs = [e for e, _ in rows]
        ev = [v for _, v in rows]
        allerr += errs
        cov = sum(e <= v["radius_m"] for e, v in rows)
        print(f"{name:26s} err mean {np.mean(errs):.2f} max {np.max(errs):.2f} | radius first "
              f"{np.mean([v['radius_history_m'][0] for v in ev]):.2f} final {np.mean([v['radius_m'] for v in ev]):.2f}"
              f" | covered {cov}/{n} | samples {np.mean([v['n_samples'] for v in ev]):.0f} | time "
              f"{np.mean([v['time_s'] for v in ev]):.0f}s | peak {max(v['peak_lel'] for v in ev):.3f}")
    print(f"ALL err mean {np.mean(allerr):.2f} median {np.median(allerr):.2f} p90 {np.percentile(allerr, 90):.2f} "
          f"max {np.max(allerr):.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)

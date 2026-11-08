import argparse
import math

import numpy as np

from policy.gasbot.mock import build, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--wind-dir", type=float, default=20.0)
    ap.add_argument("--wind-speed", type=float, default=3.0)
    ap.add_argument("--hole-mm", type=float, default=2.0)
    ap.add_argument("--bar", type=float, default=10.0)
    ap.add_argument("--start", type=float, nargs=2, default=[6.6, 1.25])
    a = ap.parse_args(argv)
    from policy.gasbot.plume import deck_solver_field
    orch, bridge, truth = build(a.seed, "loose", fid="M3", wind_speed=a.wind_speed,
                                wind_dir=math.radians(a.wind_dir), h2s_ratio=2e-4, start=a.start)
    bridge.field = deck_solver_field(bridge.leak_world.flanges["M3"].geom, "M3", bridge.wind, a.wind_speed,
                                     a.wind_dir, a.hole_mm, a.bar)
    run(orch, bridge, 150)
    for r in orch.records:
        print(f"{r.t:6.1f} {r.frm:>14} -> {r.to:<14} {r.reason[:110]}")
    pts = np.array([[x, y, z] for x in np.arange(-2, 9, 1.0) for y in (0.8, 1.25, 2.0) for z in (0.45, 0.95)])
    c = bridge.field.sample(pts)
    print("max ppm on corridor grid", round(float(c.max())), "at", pts[int(np.argmax(c))], truth)


if __name__ == "__main__":
    main()

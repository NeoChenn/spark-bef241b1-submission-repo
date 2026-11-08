import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sim.plume.solver import Plume, Wind


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--flange", default="F5")
    a.add_argument("--hole-mm", type=float, default=3.0)
    a.add_argument("--bar", type=float, default=20.0)
    a.add_argument("--h2s", type=float, default=0.0)
    a.add_argument("--wind", type=float, default=5.0)
    a.add_argument("--heading", type=float, default=90.0)
    a.add_argument("--storm", type=float, default=0.0)
    a.add_argument("--dx", type=float, default=0.2)
    a.add_argument("--shape", default="96,80,48")
    a.add_argument("--seconds", type=float, default=30.0)
    a.add_argument("--dt", type=float, default=1 / 24)
    a.add_argument("--snap-every", type=float, default=1.0)
    a.add_argument("--out", default="runs/plume/F5")
    a.add_argument("--no-obstacles", action="store_true")
    args = a.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(s) for s in args.shape.split(","))
    t0 = time.perf_counter()
    p = Plume.for_flange(args.flange, args.hole_mm, args.bar, args.h2s, Wind(args.wind, args.heading, args.storm),
                         dx=args.dx, shape=shape, obstacles=not args.no_obstacles)
    init_s = time.perf_counter() - t0
    rows, snaps, snap_t = [], [], []
    next_snap = 0.0
    while p.t < args.seconds - 1e-9:
        t1 = time.perf_counter()
        s = p.step(args.dt)
        s["step_ms"] = (time.perf_counter() - t1) * 1e3
        s["substeps"] = p.substeps
        rows.append(s)
        if p.t >= next_snap - 1e-9:
            snaps.append(p.field().fraction.astype(np.float16))
            snap_t.append(p.t)
            next_snap += args.snap_every
    keys = list(rows[0])
    series = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez_compressed(out / "run.npz", snaps=np.stack(snaps), snap_t=np.array(snap_t), solid=p.solid_np,
                        velocity=p.velocity().astype(np.float16), origin=p.origin, dx=p.dx, source=p.source,
                        jet_dir=p.jet_dir, lel=p.lel, **{f"s_{k}": v for k, v in series.items()})
    summary = {"args": vars(args), "device": str(p.device), "cells": int(np.prod(shape)), "init_s": round(init_s, 2),
               "flow": p.flow_rate(), "wind": p.wind.to_json(p.t), "final": rows[-1],
               "step_ms_median": float(np.median(series["step_ms"][5:])), "substeps_median": float(np.median(series["substeps"])),
               "realtime_factor": float(args.dt * 1e3 / np.median(series["step_ms"][5:]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

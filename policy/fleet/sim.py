from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from policy.fleet.render import render_png
from policy.fleet.scenario import build_default


def write_sample(out: str, sample: str) -> None:
    src, dst = Path(out), Path(sample)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("fleet_map.json", "fleet_state.json", "summary.json", "fleet_map.png"):
        shutil.copyfile(src / name, dst / name)
    lines = (src / "fleet_log.jsonl").read_text().splitlines()
    (dst / "fleet_log_head.jsonl").write_text("\n".join(lines[:3] + lines[-1:]) + "\n")


def run(n_robots: int = 3, out: str = "runs/fleet", t_max: float = 2400.0, snap_t: float = 250.0, jobs=None):
    t0 = time.time()
    f = build_default(n_robots, jobs=jobs)
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    map_json = f.rm.to_json()
    keep = {}
    with open(out_p / "fleet_log.jsonl", "w") as log:
        log.write(json.dumps({"type": "map", **map_json}) + "\n")

        def tick(fl):
            fl.check_invariants()
            snap = fl.snapshot()
            log.write(json.dumps({"type": "tick", **snap}) + "\n")
            if "state" not in keep and fl.t >= snap_t:
                keep["state"] = snap
        f.run(t_max, on_tick=tick)
        summary = {**f.metrics(), "robots": n_robots, "wall_s": round(time.time() - t0, 2),
                   "orchestrator_jobs": {f"{k}/{s}": {"duration_s": tr.duration_s, "outcome": tr.outcome,
                                                      "decisions": tr.n_decisions}
                                         for (k, s), tr in sorted(f.jobs.cache.items())},
                   "events": f.event_log}
        log.write(json.dumps({"type": "summary", **summary}) + "\n")
    state = keep.get("state", f.snapshot())
    (out_p / "fleet_map.json").write_text(json.dumps(map_json, indent=1))
    (out_p / "fleet_state.json").write_text(json.dumps(state, indent=1))
    (out_p / "summary.json").write_text(json.dumps(summary, indent=1))
    render_png(map_json, state, str(out_p / "fleet_map.png"))
    render_png(map_json, f.snapshot(), str(out_p / "fleet_final.png"))
    return f, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--out", default="runs/fleet")
    ap.add_argument("--t-max", type=float, default=2400.0)
    ap.add_argument("--snap-t", type=float, default=250.0)
    ap.add_argument("--sample-dir", default="")
    a = ap.parse_args(argv)
    f, s = run(a.robots, a.out, a.t_max, a.snap_t)
    if a.sample_dir:
        write_sample(a.out, a.sample_dir)
    for e in s["events"]:
        if e["kind"] != "job_start":
            print(f"t={e['t']:7.1f}  {e['kind']:<16} {e['message']}")
    keys = ["t", "jobs_done", "jobs_escalated", "open_tasks", "gas_alarms", "gas_response_mean_s",
            "gas_response_max_s", "gas_resolve_mean_s", "idle_pct", "energy_used_wh", "energy_charged_wh",
            "distance_m", "preemptions", "robot_failures", "reservation_wait_s", "corridor_conflicts", "wall_s"]
    print("\n" + "  ".join(f"{k}={s[k]}" for k in keys))
    print("state %:", s["state_pct"], " jobs:", s["jobs_by_kind"])
    print(f"-> {a.out}/fleet_log.jsonl, summary.json, fleet_state.json, fleet_map.json, fleet_map.png")


if __name__ == "__main__":
    main()

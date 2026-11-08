from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from fleet.contract import REPO, Job, Level, Robot, RobotKind, World
from fleet.coordinator import OPEN, Coordinator
from fleet.demo import FMT_SKIP, GeminiExplainer, fmt, load_executors, render_safe, summarize
from fleet.rigplanner import RigPlanner
from fleet.risk import risk_priority
from fleet.sim import FleetSim, new_job, walk
from fleet.state import snapshot
from policy.fleet.rigmap import DECKS, DOCK_XY
from policy.fleet.scenario import CHECK_DEADLINE_S, EVENTS, INSPECT_DEADLINE_S, ROBOTS, RUST_SPOTS

CAPS = {"camera", "arm", "spray", "gas_sensor", "thermal"}
DOCK = (DOCK_XY[0], DOCK_XY[1], DECKS["top_deck"])
APPROVE_DELAY_S = 30.0
OUT = REPO / "runs/fleet/rigshift"


def rust_job(c: Coordinator, poi: str, xyz, severity: float, exposure: float, seed: int) -> Job:
    return new_job(c, "rust_repair", poi, xyz, Level.SUPV, priority=risk_priority("rust", severity, exposure),
                   est_duration_s=180.0, params={"scenario": {"seed": seed}, "standoff": 0.8, "severity": severity,
                                                 "exposure": exposure})


def build(n_robots: int = 3, offline: bool = True, timeline_path=None, states_path=None,
          approve_delay_s: float = APPROVE_DELAY_S, events=EVENTS):
    executors, loaded, stubbed = load_executors()
    planner = RigPlanner()
    rm = planner.rm
    world = World.from_rig_meta()
    world.state["flanges"] = {}
    robots = []
    for i in range(n_robots):
        rid, xyz, soc = ROBOTS[i] if i < len(ROBOTS) else (f"R{i + 1}", ROBOTS[0][1], 0.9)
        robots.append(Robot(rid, RobotKind.QUADRUPED, set(CAPS), tuple(float(v) for v in xyz), battery=soc))
    c = Coordinator(robots, world, executors, planner, docks={r.id: DOCK for r in robots},
                    explainer=None if offline else GeminiExplainer())
    sim = FleetSim(c, timeline_path=timeline_path, states_path=states_path)

    pending: dict[str, float] = {}
    original = c.on_event

    def watch(e):
        original(e)
        if e["event"] == "approval_requested":
            pending[e["job"]] = e["t"] + approve_delay_s
    c.on_event = watch

    def operator(s: FleetSim):
        for jid, due in list(pending.items()):
            if s.world.t >= due:
                pending.pop(jid)
                s.c.approve(jid, by="operator")
        s.at(s.world.t + 1, "operator", operator)
    sim.at(0, "operator", operator)

    for p in rm.inspection:
        if p["type"] == "gauge":
            c.submit(new_job(c, "gauge_read", p["id"], p["position_m"], Level.AUTO, priority=risk_priority("inspect"),
                             est_duration_s=40.0, params={"deadline": INSPECT_DEADLINE_S}))
    for fid, xyz in rm.flanges.items():
        c.submit(new_job(c, "flange_leak_check", fid, xyz, Level.AUTO, priority=risk_priority("leak_check"),
                         est_duration_s=20.0, params={"deadline": CHECK_DEADLINE_S}))
    for k, (poi, xyz, sev, exp) in enumerate(RUST_SPOTS):
        c.submit(rust_job(c, poi, xyz, sev, exp, k + 1))

    def supersede(s: FleetSim, fid: str):
        for jid, st in list(s.c.status.items()):
            j = s.c.jobs[jid]
            if j.kind == "flange_leak_check" and j.target == fid and st in OPEN:
                s.c.cancel(jid, by="coordinator", reason="superseded by an urgent flange job")

    def gas_alarm(fid: str, scenario: str):
        def fire(s: FleetSim):
            xyz = tuple(rm.flanges[fid])
            leak = {"id": f"leak_{fid}", "source": fid, "position": xyz, "rate_g_s": 3.5, "t0": s.world.t,
                    "reported": True}
            s.world.state["gas_leaks"].append(leak)
            s.world.state["flanges"][fid] = scenario
            s.c.emit("gas_leak_injected", None, None, position=list(xyz), source=fid, rate_g_s=3.5)
            supersede(s, fid)
            job = new_job(s.c, "flange_leak_check", fid, xyz, Level.AUTO, priority=risk_priority("gas"),
                          est_duration_s=20.0, params={"leak": scenario, "gas_response": True, "zone": f"flange:{fid}"})
            s.c.report(job, None, "gas_alarm", position=[round(v, 3) for v in xyz], source="fixed gas detector")
        return fire

    def crack(fid: str):
        def fire(s: FleetSim):
            s.world.state["flanges"][fid] = "crack"
            supersede(s, fid)
            job = new_job(s.c, "flange_leak_check", fid, tuple(rm.flanges[fid]), Level.AUTO,
                          priority=risk_priority("crack"), est_duration_s=20.0, params={"leak": "crack"})
            s.c.report(job, None, "crack_reported", flange=fid)
        return fire

    def new_rust(d: dict):
        def fire(s: FleetSim):
            s.c.report(rust_job(s.c, d["poi"], tuple(d["xyz"]), d["severity"], d["exposure"], 9), None, "new_rust",
                       poi=d["poi"], severity=d["severity"], exposure=d["exposure"])
        return fire

    def failure(rid: str):
        return lambda s: s.c.fail_robot(rid, "drive fault")

    ids = {r.id for r in robots}
    for t, kind, d in events:
        if kind == "gas_alarm":
            sim.at(t, kind, gas_alarm(d["flange"], d.get("scenario", "loose")))
        elif kind == "crack":
            sim.at(t, kind, crack(d["flange"]))
        elif kind == "new_rust":
            sim.at(t, kind, new_rust(d))
        elif kind == "robot_failure" and d["robot"] in ids:
            sim.at(t, kind, failure(d["robot"]))
        elif kind == "person":
            path = [tuple(p) for p in d["path"]]
            speed = d.get("speed", 0.8)
            walk_s = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(path, path[1:])) / speed
            sim.people_paths.append(walk(path, t0=t, speed=speed, linger_s=max(d.get("duration", 60.0) - walk_s, 0.0)))
    return sim, loaded, stubbed


def finished(sim: FleetSim) -> bool:
    c = sim.c
    return (not any(s in OPEN for s in c.status.values()) and not any(l for t, l, _ in sim.scripted if l != "operator")
            and all(rt.state in ("IDLE", "CHARGING", "FAILED", "DOCK_QUEUE") for rt in c.rt.values()))


def run(sim: FleetSim, t_max: float) -> None:
    while sim.world.t < t_max and not finished(sim):
        sim.step()
    sim.write()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="multi-deck rig shift on the unified fleet coordinator")
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--t-max", type=float, default=2400.0)
    ap.add_argument("--snap-t", type=float, default=250.0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample-dir", default="")
    a = ap.parse_args(argv)
    wall = time.perf_counter()
    out = Path(a.out)
    sim, loaded, stubbed = build(a.robots, a.offline, out / "timeline.jsonl", out / "fleet_states.jsonl")
    run(sim, a.t_max)
    c = sim.c
    if not a.quiet:
        for e in c.timeline:
            if e["event"] not in FMT_SKIP | {"job_queued", "assigned", "job_started"}:
                print(fmt(e))
    final = snapshot(c)
    m = final["metrics"]
    keep = next((x for x in sim.states if x["t"] >= a.snap_t), final)
    summary = {**m, "robots": a.robots, "counts": c.counts(), "jobs": summarize(c), "modules": loaded,
               "stubbed": stubbed, "wall_s": round(time.perf_counter() - wall, 2)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "fleet_state.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
    map_json = c.planner.to_json()
    (out / "fleet_map.json").write_text(json.dumps(map_json, indent=1), encoding="utf-8")
    from policy.fleet.render import render_png
    render_png(map_json, render_safe(keep), str(out / "fleet_map.png"))
    render_png(map_json, render_safe(final), str(out / "fleet_final.png"))
    if a.sample_dir:
        dst = Path(a.sample_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "fleet_state.json", "fleet_map.png"):
            (dst / name).write_bytes((out / name).read_bytes())
        ev = [json.dumps(l) for l in sim.lines if "event" in l and l["event"] != "segment_clear"]
        (dst / "timeline_events.jsonl").write_text("\n".join(ev) + "\n", encoding="utf-8")
    keys = ["t", "jobs_done", "jobs_escalated", "open_tasks", "gas_alarms", "gas_response_mean_s", "gas_resolve_mean_s",
            "idle_pct", "energy_used_wh", "energy_charged_wh", "distance_m", "preemptions", "robot_failures",
            "replans", "reservation_wait_s", "corridor_conflicts", "approvals_pending", "crew_jobs"]
    print("\n" + "  ".join(f"{k}={m[k]}" for k in keys))
    print("counts:", c.counts(), " wall", summary["wall_s"], "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

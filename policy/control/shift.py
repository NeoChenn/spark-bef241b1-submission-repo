from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from fleet.contract import Level, Robot, RobotKind, World
from fleet.coordinator import Coordinator
from fleet.demo import load_executors
from fleet.rigplanner import RigPlanner
from fleet.risk import risk_priority
from fleet.sim import FleetSim, new_job
from policy.control.agent import ArgonAgentBackend, OrchestrationAgent, ReplayAgentBackend
from policy.control.backend import ArgonBackend, ShiftReplayBackend
from policy.control.brain import MAINTENANCE, ArgonBrain, VerifySensor
from policy.control.centre import ControlCentre
from policy.control.envelope import Envelope
from policy.control.fleet_port import ARGON, FleetPort
from policy.control.rigsensors import RigSensors
from policy.control.schema import VERIFY
from policy.fleet.rigmap import DECKS, DOCK_XY

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/argon"
DAY = date(2026, 11, 8)
Q, C, A = RobotKind.QUADRUPED, RobotKind.CRAWLER, RobotKind.AERIAL
DOCK = (DOCK_XY[0], DOCK_XY[1], DECKS["top_deck"])
ROBOTS = [("Q1", Q, {"arm", "spray", "camera"}, (-12.0, 9.4, 42.5), 0.95, 1.0, DOCK),
          ("Q2", Q, {"camera", "thermal", "gas_sensor", "acoustic"}, (9.0, -4.5, 42.5), 0.70, 1.0, DOCK),
          ("S1", Q, {"camera", "gas_sensor", "thermal", "arm"}, (0.0, 4.0, 31.25), 0.90, 1.0, DOCK),
          ("S2", Q, {"camera", "gas_sensor"}, (-18.0, 20.0, 42.5), 0.31, 1.0, DOCK),
          ("A1", A, {"camera", "thermal", "gas_sensor"}, (-25.0, -25.0, 47.0), 0.60, 4.0, None),
          ("C1", C, {"camera", "ut_probe", "acoustic"}, (-3.0, -6.5, 42.5), 0.80, 0.3, None)]
FAULTS = [(60.0, "G-F1", "stuck", {"value": 0.0}),
          (165.0, "gauge_03", "dead", {}),
          (200.0, "T-FLARE", "stuck", {}),
          (285.0, "A-V101", "dead", {}),
          (345.0, "G-F6", "dead", {})]
PROCESS = [(250.0, "gauge_06", 1.0)]
LEAK = (330.0, "F5", 30.0)
LINK_CUT = (150.0, 240.0)
T_END = 800.0
SNAP_DT = 2.0
LABELS = {"sensor_fault": "FAULT", "fault_dispatch": "VERIFY", "verify_reading": "READING",
          "maintenance_raised": "MAINTENANCE", "fault_cleared": "CLEARED", "gas_alarm": "GAS ALARM",
          "gas_detected": "GAS", "leak_isolated": "ISOLATED", "agent_refused": "REFUSED", "approved": "APPROVED",
          "sensor_repaired": "REPAIRED", "link_cut": "LINK CUT", "argon_vetoed": "VETO", "gemini_call": "PLANNER",
          "link_restored": "LINK UP", "edge_rule": "EDGE", "argon_call": "ARGON"}
CONTROL_KEYS = ("mode", "online", "status", "banner", "model", "replay", "pending", "situation", "actions", "anomalies",
                "vetoed", "latency_ms", "tokens", "cost_today_usd", "calls_today", "gas", "health", "link", "agent")


def build(mode: str = "replay", out: Path = OUT, seed: int = 7, interval_s: float = 10.0, cap_usd: float = 1.0):
    executors, _, _ = load_executors()
    executors = {**executors, VERIFY: VerifySensor()}
    planner = RigPlanner()
    world = World.from_rig_meta()
    world.state["flanges"] = {}
    robots, docks = [], {}
    for rid, kind, caps, xyz, soc, speed, dock in ROBOTS:
        robots.append(Robot(rid, kind, set(caps), xyz, battery=soc, speed_mps=speed))
        docks[rid] = dock or xyz
    c = Coordinator(robots, world, executors, planner, docks=docks)
    out.mkdir(parents=True, exist_ok=True)
    sim = FleetSim(c, timeline_path=out / "timeline.jsonl")
    rm = planner.rm
    for p in rm.inspection:
        if p["type"] == "gauge":
            c.submit(new_job(c, "gauge_read", p["id"], p["position_m"], Level.AUTO, priority=risk_priority("inspect"),
                             est_duration_s=40.0, params={"deadline": 900.0, "use_gemini": False}))
    for fid, xyz in rm.flanges.items():
        c.submit(new_job(c, "flange_leak_check", fid, xyz, Level.AUTO, priority=risk_priority("leak_check"),
                         est_duration_s=20.0, params={"deadline": 1500.0}))
    c.submit(new_job(c, "aerial_survey", "crane_boom", (-28.0, -12.0, 58.0), Level.AUTO, priority=4,
                     est_duration_s=60.0, params={"standoff": 0.0}))
    c.submit(new_job(c, "crawler_ut", "separator_V101", (-8.0, -2.0, 44.5), Level.AUTO, priority=4,
                     est_duration_s=240.0, params={"standoff": 0.0}))
    rig = RigSensors(seed=seed)
    for t0, sid, mode_, kw in FAULTS:
        rig.inject(sid, mode_, t0, **kw)
    for t0, sid, delta in PROCESS:
        rig.process_step(sid, t0, delta)
    rig.leak(LEAK[1], LEAK[0], LEAK[2])

    def leak(s):
        s.world.state["flanges"][LEAK[1]] = "loose"
        s.world.state["gas_leaks"].append({"id": f"leak_{LEAK[1]}", "source": LEAK[1],
                                           "position": tuple(rm.flanges[LEAK[1]]), "rate_g_s": 3.5, "t0": s.world.t,
                                           "alarm_after_s": 1e9})
        s.c.emit("gas_leak_injected", None, None, position=list(rm.flanges[LEAK[1]]), source=LEAK[1])
    sim.at(LEAK[0], "leak", leak)
    port = FleetPort(c, rig.specs)
    backend = None if mode == "edge" else ArgonBackend() if mode == "live" else ShiftReplayBackend()
    agent_backend = None if mode == "edge" else ArgonAgentBackend() if mode == "live" else ReplayAgentBackend()
    (out / "agent_calls.jsonl").unlink(missing_ok=True)
    agent = OrchestrationAgent(port, agent_backend, Envelope(ARGON), log_path=out / "agent_calls.jsonl",
                               today=lambda: DAY)
    centre = ControlCentre(backend, log_path=out / "calls.jsonl", interval_s=interval_s, alarm_gap_s=1.5,
                           per_minute=12, cap_usd_day=cap_usd, sim_latency=True, today=lambda: DAY, timeline=port,
                           agent=agent)
    if centre.log_path is not None:
        centre.log_path.unlink(missing_ok=True)
    return ArgonBrain(sim, rig, centre, port)


def control_line(ctl: dict) -> dict:
    return {k: ctl.get(k) for k in CONTROL_KEYS}


def run(mode: str = "replay", out: Path = OUT, t_end: float = T_END, orders=(), **kw) -> dict:
    out = Path(out)
    brain = build(mode, out, **kw)
    wall, last = time.time(), None
    cut = False
    orders, pending = sorted(orders), []
    while brain.c.t < t_end:
        while orders and orders[0][0] <= brain.c.t:
            pending.append(brain.centre.intent(orders.pop(0)[1], now=brain.c.t, background=True))
        pending = [f for f in pending if not f.done()]
        if pending:
            time.sleep(0.002)
        want = LINK_CUT[0] <= brain.c.t < LINK_CUT[1]
        if want != cut:
            cut = want
            brain.centre.cut_link(cut)
        ctl = brain.step()
        if brain.c.t % SNAP_DT < 1e-6:
            line = control_line(ctl)
            if line != last:
                last = line
                brain.sim.lines.append({"t": round(brain.c.t, 2), "event": "argon_control", "job": None,
                                        "robot": None, "detail": line})
    brain.sim.write()
    if brain.centre.orders is not None:
        brain.centre.orders.shutdown(wait=True)
    calls = [e["detail"] for e in brain.port.timeline if e["event"] == "argon_call"]
    ok = [c for c in calls if c.get("status") == "ok"]
    summary = {"mode": mode, "model": getattr(brain.centre.backend, "name", None), "t_end": brain.c.t,
               "wall_s": round(time.time() - wall, 2), "argon_calls": len(calls), "argon_ok": len(ok),
               "latency_ms_mean": round(sum(c["latency_ms"] for c in ok) / len(ok)) if ok else None,
               "cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4),
               "events": {}, "jobs": [{k: j[k] for k in ("id", "kind", "target", "status", "requested_by", "robot")}
                                      for j in brain.port.jobs() if j["kind"] in (VERIFY, MAINTENANCE)]}
    for e in brain.port.timeline:
        summary["events"][e["event"]] = summary["events"].get(e["event"], 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"brain": brain, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Argon shift: sensor health, fault dispatch and the fleet planner.")
    ap.add_argument("--mode", choices=["replay", "live", "edge"], default="replay")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--interval", type=float, default=10.0, help="Argon call interval in sim seconds")
    ap.add_argument("--cap", type=float, default=1.0, help="USD cap for this run")
    ap.add_argument("--order", action="append", default=[], metavar="T:TEXT",
                    help="operator order for the mission planner at sim time T, e.g. '30:Inspect gauge_05'")
    a = ap.parse_args(argv)
    orders = [(float(o.split(":", 1)[0]), o.split(":", 1)[1]) for o in a.order]
    res = run(a.mode, Path(a.out), a.t_end, orders, interval_s=a.interval, cap_usd=a.cap)
    for e in res["brain"].port.timeline:
        if e["event"] in LABELS and e["event"] not in ("argon_call", "edge_rule"):
            d = e["detail"]
            print(f"t={e['t']:6.1f} {e['event']:<19} {e['robot'] or '':<4} {e['job'] or '':<9} "
                  f"{d.get('sensor', '')} {d.get('why', '') or d.get('status', '')}")
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("mode", "model", "argon_calls", "argon_ok", "latency_ms_mean", "cost_usd",
                                        "wall_s")}))
    print(f"-> {a.out}/timeline.jsonl, summary.json; HUD: /?replay=<that timeline.jsonl, relative to the repo>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

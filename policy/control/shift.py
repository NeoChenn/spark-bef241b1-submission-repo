from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from policy.control.agent import ArgonAgentBackend, OrchestrationAgent, ReplayAgentBackend
from policy.control.backend import ArgonBackend, ShiftReplayBackend
from policy.control.brain import ArgonBrain, register
from policy.control.centre import ControlCentre
from policy.control.envelope import Envelope
from policy.control.fleet_port import ARGON, FleetPort
from policy.control.rigsensors import RigSensors
from policy.fleet.fleet import Fleet
from policy.fleet.rigmap import load_rig_map
from policy.fleet.scenario import nearest_node

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/argon"
DAY = date(2026, 11, 8)
ROBOTS = [("R1", [-12.0, 9.4, 42.5], 0.95, ("camera", "thermal_cam")),
          ("R2", [9.0, -4.5, 42.5], 0.70, ("camera", "gas_sniffer", "ultrasonic_mic")),
          ("R3", [0.0, 4.0, 31.25], 0.90, ("camera", "gas_sniffer", "thermal_cam")),
          ("R4", [-18.0, 20.0, 42.5], 0.16, ("camera", "gas_sniffer"))]
FAULTS = [(60.0, "G-F1", "stuck", {"value": 0.0}),
          (165.0, "gauge_03", "dead", {}),
          (345.0, "G-F6", "dead", {})]
PROCESS = [(250.0, "gauge_06", 1.0)]
LEAK = (330.0, "F5", 30.0)
LINK_CUT = (150.0, 240.0)
T_END = 720.0
PLAYBACK = 12.0
SNAP_DT = 2.0
LABELS = {"sensor_fault": "FAULT", "fault_dispatch": "VERIFY", "verify_reading": "READING",
          "maintenance_raised": "MAINTENANCE", "fault_cleared": "CLEARED", "gas_alarm": "GAS ALARM",
          "leak_isolated": "ISOLATED", "agent_refused": "REFUSED", "approved": "APPROVED", "sensor_repaired": "REPAIRED",
          "link_cut": "LINK CUT", "argon_vetoed": "VETO", "gemini_call": "PLANNER", "link_restored": "LINK UP", "edge_rule": "EDGE", "argon_call": "ARGON"}


def build(mode: str = "replay", out: Path = OUT, seed: int = 7, interval_s: float = 10.0, cap_usd: float = 1.0):
    rm = load_rig_map()
    rig = RigSensors(seed=seed)
    register(rm, rig)
    f = Fleet(rm)
    for rid, xyz, soc, sensors in ROBOTS:
        f.add_robot(rid, nearest_node(rm, xyz), soc, sensors=sensors)
    for p in rm.inspection:
        if p["type"] == "gauge":
            f.add_task("inspect", p["id"], deadline=900.0)
    for fid in rm.flanges:
        f.add_task("leak_check", fid, deadline=1500.0)
    for t0, sid, mode_, kw in FAULTS:
        rig.inject(sid, mode_, t0, **kw)
    for t0, sid, delta in PROCESS:
        rig.process_step(sid, t0, delta)
    rig.leak(LEAK[1], LEAK[0], LEAK[2])
    out.mkdir(parents=True, exist_ok=True)
    port = FleetPort(f, rig.specs, timeline_path=out / "timeline.jsonl")
    backend = None if mode == "edge" else ArgonBackend() if mode == "live" else ShiftReplayBackend()
    planner = None if mode == "edge" else ArgonAgentBackend() if mode == "live" else ReplayAgentBackend()
    (out / "agent_calls.jsonl").unlink(missing_ok=True)
    agent = OrchestrationAgent(port, planner, Envelope(ARGON), log_path=out / "agent_calls.jsonl", today=lambda: DAY)
    centre = ControlCentre(backend, log_path=out / "calls.jsonl", interval_s=interval_s, alarm_gap_s=1.5,
                           per_minute=12, cap_usd_day=cap_usd, sim_latency=True, today=lambda: DAY, timeline=port,
                           agent=agent)
    if centre.log_path is not None:
        centre.log_path.unlink(missing_ok=True)
    return ArgonBrain(f, rig, centre, port)


def hud_events(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        lab = LABELS.get(e["event"])
        if lab is None or e["event"] == "argon_call":
            continue
        d = e["detail"]
        why = d.get("why") or ""
        if e["event"] == "sensor_fault":
            why = f"{d['sensor']} {d['status']}" + (" · hazard: treated as real gas" if d.get("hazard") else "")
        elif e["event"] == "verify_reading":
            why = f"{d['sensor']}: robot {d['robot_mean']} vs head {d['head_mean']} -> {d['why']}"
        elif e["event"] == "gemini_call":
            why = f"{d['status']} · {', '.join(d['decision']) or 'no change'} · {d['latency_ms']} ms"
        elif e["event"] in ("link_cut", "link_restored"):
            why = d.get("why", "")
        out.append({"t": e["t"], "from": e["robot"] or "Argon", "to": lab, "reason": why or e["event"]})
    return out


def run(mode: str = "replay", out: Path = OUT, t_end: float = T_END, orders=(), **kw) -> dict:
    out = Path(out)
    brain = build(mode, out, **kw)
    snaps, seen, wall = [], 0, time.time()
    cut = False
    orders, pending = sorted(orders), []
    while brain.f.t < t_end:
        while orders and orders[0][0] <= brain.f.t:
            pending.append(brain.centre.intent(orders.pop(0)[1], now=brain.f.t, background=True))
        pending = [f for f in pending if not f.done()]
        if pending:
            time.sleep(0.002)
        want = LINK_CUT[0] <= brain.f.t < LINK_CUT[1]
        if want != cut:
            cut = want
            brain.centre.cut_link(cut)
            brain.port.log("link_cut" if cut else "link_restored", None, None,
                           why="Argon link cut: edge rules and the fleet planner carry on" if cut
                           else "Argon link restored")
        ctl = brain.step()
        if brain.f.t % SNAP_DT < 1e-6:
            ev = brain.port.timeline[seen:]
            seen = len(brain.port.timeline)
            snaps.append({"t": round(brain.f.t, 1), "pt": round(brain.f.t / PLAYBACK, 3), "events": hud_events(ev),
                          "control": ctl})
    if brain.centre.orders is not None:
        brain.centre.orders.shutdown(wait=True)
    calls = [e["detail"] for e in brain.port.timeline if e["event"] == "argon_call"]
    ok = [c for c in calls if c.get("status") == "ok"]
    summary = {"mode": mode, "model": getattr(brain.centre.backend, "name", None), "t_end": brain.f.t,
               "wall_s": round(time.time() - wall, 2), "argon_calls": len(calls), "argon_ok": len(ok),
               "latency_ms_mean": round(sum(c["latency_ms"] for c in ok) / len(ok)) if ok else None,
               "cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4),
               "events": {}, "jobs": [{k: j[k] for k in ("id", "kind", "target", "status", "requested_by", "robot")}
                                      for j in brain.port.jobs.values()]}
    for e in brain.port.timeline:
        summary["events"][e["event"]] = summary["events"].get(e["event"], 0) + 1
    session = {"snapshots": snaps, "duration": snaps[-1]["pt"] if snaps else 0.0, "summary": summary}
    (out / "hud_session.json").write_text(json.dumps(session, separators=(",", ":"), default=str))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"brain": brain, "session": session, "summary": summary}


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
    print(f"-> {a.out}/timeline.jsonl, hud_session.json, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

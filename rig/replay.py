import json
from pathlib import Path

from rig.captions import robot_entry

NAMES = {"Q1": "Unit 1 · rust", "Q2": "Unit 2 · gas", "Q3": "Unit 3", "A1": "Drone A1", "C1": "Crawler C1", "R1": "ROV R1"}
STATE = {"MOVING": "TRANSIT", "WAITING_SEGMENT": "WAIT", "WORKING": "WORK", "PAUSED_PERSON": "PAUSED",
         "DOCKING": "TO_DOCK", "CHARGING": "CHARGING", "IDLE": "IDLE"}
TO = {"job_queued": "QUEUED", "approval_requested": "APPROVAL", "approved": "APPROVED", "rejected": "REJECTED",
      "assigned": "ASSIGNED", "job_started": "STARTED", "finding": "FINDING", "handed_to_crew": "CREW",
      "preempted": "PREEMPTED", "requeued": "REQUEUED", "priority_change": "REPLAN", "segment_wait": "WAIT",
      "segment_clear": "CLEAR", "person_pause": "PAUSE_PERSON", "person_clear": "RESUME", "low_battery": "LOW_BATTERY",
      "docked": "DOCKED", "charged": "CHARGED", "gas_leak_injected": "GAS_LEAK", "gas_detected": "GAS",
      "gas_alarm": "GAS", "duplicate_skipped": "SKIP", "cut_link": "EDGE", "sensor_fault": "FAULT",
      "fault_dispatch": "VERIFY", "verify_reading": "READING", "maintenance_raised": "MAINTENANCE",
      "fault_cleared": "CLEARED", "sensor_repaired": "REPAIRED", "leak_isolated": "ISOLATED", "argon_vetoed": "VETO",
      "agent_refused": "REFUSED", "link_cut": "LINK CUT", "link_restored": "LINK UP", "robot_held": "HOLD",
      "robot_resumed": "RESUME", "gemini_call": "PLANNER"}
QUIET = {"argon_call", "edge_rule", "escalation", "sensor_ok", "hold_expired"}
STATUS = {"job_queued": "QUEUED", "approval_requested": "WAITING_APPROVAL", "approved": "QUEUED", "rejected": "REJECTED",
          "assigned": "ASSIGNED", "job_started": "RUNNING", "handed_to_crew": "HANDED_TO_HUMAN", "preempted": "QUEUED",
          "requeued": "QUEUED"}
GAS_R, PERSON_R = 6.0, 3.0


def load_timeline(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return [x for x in out if isinstance(x, dict) and "t" in x]


def _reason(e: dict) -> str:
    d = e.get("detail") or {}
    if e["event"] == "person_pause":
        return f"person {float(d.get('distance_m', 0)):.1f} m away -> pause"
    if e["event"] == "sensor_fault":
        return f"{d.get('sensor')} {d.get('status')}" + (" · beside live gas: treated as gas" if d.get("hazard") else "")
    if e["event"] == "verify_reading":
        return f"{d.get('sensor')}: robot {d.get('robot_mean')} vs head {d.get('head_mean')} -> {d.get('why')}"
    for k in ("explanation", "summary", "reason", "why", "log", "source"):
        if d.get(k):
            return str(d[k])
    if "battery" in d:
        return f"battery {100 * float(d['battery']):.0f} %"
    if d.get("kind"):
        return " ".join(str(d[k]) for k in ("kind", "target") if d.get(k))
    return e["event"].replace("_", " ")


def _deck(z: float) -> str:
    return "air" if z > 50 else min((("lower_deck", 20.0), ("middle_deck", 31.25), ("top_deck", 42.5)),
                                    key=lambda d: abs(d[1] - z))[0]


def timeline_session(lines: list[dict], rate: float = 8.0) -> dict:
    frames: dict[float, list] = {}
    events: dict[float, list] = {}
    for x in lines:
        (events if "event" in x else frames).setdefault(float(x["t"]), []).append(x)
    jobs, approvals, battery, zones, last = {}, {}, {}, {}, {}
    gas, snaps, prev_orch = None, [], None
    times = sorted(frames)
    pending = sorted(events)
    k = 0
    for t in times:
        evs = []
        while k < len(pending) and pending[k] <= t:
            evs += events[pending[k]]
            k += 1
        decisions, control = [], None
        for e in evs:
            if e["event"] == "argon_control":
                control = e.get("detail")
                continue
            name, d, jid = e["event"], e.get("detail") or {}, e.get("job")
            if jid and name in ("job_queued",):
                jobs[jid] = {"id": jid, "kind": d.get("kind"), "target": d.get("target"), "level": d.get("level"),
                             "priority": d.get("priority"), "status": "QUEUED", "robot": None}
            if jid in jobs and name in STATUS:
                jobs[jid]["status"] = STATUS[name]
            if jid in jobs and name == "job_done":
                jobs[jid]["status"] = d.get("status", "DONE")
            if jid in jobs and name == "assigned":
                jobs[jid]["robot"] = e.get("robot")
            if name == "approval_requested":
                approvals[jid] = {"job": jid, "kind": d.get("kind"), "target": d.get("target"),
                                  "text": d.get("text") or d.get("explanation") or f"{d.get('kind')} at {d.get('target')} needs approval"}
            if name in ("approved", "rejected"):
                approvals.pop(jid, None)
            if "battery" in d and e.get("robot"):
                battery[e["robot"]] = float(d["battery"])
            if name in ("gas_detected", "gas_alarm", "gas_leak_injected") and d.get("position"):
                zones["gas"] = {"kind": "gas", "id": jid or "leak", "xyz": d["position"], "radius": GAS_R}
                if name != "gas_leak_injected":
                    gas = {"alarm": True, "max": None, "source": d.get("source"), "sensors": []}
            if name == "person_pause" and e.get("robot") in last:
                zones["person:" + e["robot"]] = {"kind": "person", "id": "person", "radius": PERSON_R,
                                                 "xyz": last[e["robot"]]["position"]}
            if name == "person_clear":
                zones.pop("person:" + str(e.get("robot")), None)
            if name == "job_done" and (d.get("kind") == "valve_isolate" or "iso" in str(jid)):
                gas, _ = None, zones.pop("gas", None)
            who = e.get("robot")
            if name in QUIET:
                continue
            decisions.append({"t": float(e["t"]), "from": NAMES.get(who, who) if who else "Fleet",
                              "to": TO.get(name, name.upper()), "reason": _reason(e), "job": jid})
        robots, fleet = [], []
        for r in frames[t]:
            last[r["robot"]] = r
            st = STATE.get(r.get("state"), r.get("state") or "IDLE")
            soc = battery.get(r["robot"])
            entry = robot_entry(r["robot"], NAMES.get(r["robot"], r["robot"]), st,
                                {"soc": soc, "job": (r.get("job") or "").split("-")[0] or None}, r.get("position"))
            entry.update(kind=r.get("kind"), yaw=r.get("yaw"), job=r.get("job"))
            what = (r.get("job") or "").split("-")[0].replace("_", " ")
            if what and st in ("WORK", "TRANSIT"):
                c = entry["caption"]
                c["rest"] = f" — {what}" if st == "WORK" else f" to the {what}"
                c["text"] = c["lead"] + c["rest"]
            robots.append(entry)
            x, y, z = r.get("position") or (0.0, 0.0, 0.0)
            fleet.append({"id": r["robot"], "name": entry["name"], "kind": r.get("kind"), "state": st,
                          "job_state": st, "task": r.get("job"), "deck": _deck(z),
                          "pose": {"x": x, "y": y, "z": z, "yaw": r.get("yaw") or 0.0},
                          "battery": {"soc": soc}})
        lead = next((r for r in robots if r.get("kind") == "Q"), robots[0])
        snap = {"t": t, "pt": round(t / rate, 3), "state": lead["state"], "caption": lead["caption"],
                "robots": robots, "events": decisions,
                "fleet": {"schema": "fleet_state.v1", "t": t, "robots": fleet, "zones": list(zones.values())}}
        orch = {"jobs": sorted(jobs.values(), key=lambda j: (j["priority"] or 9, j["id"])),
                "approvals": list(approvals.values())}
        key = json.dumps(orch, sort_keys=True)
        if key != prev_orch:
            prev_orch = key
            snap["orchestrator"] = json.loads(key)
        if gas is not None:
            snap["gas"] = dict(gas)
        if control is not None:
            snap["control"] = control
        snaps.append(snap)
    return {"snapshots": snaps, "duration": snaps[-1]["pt"] if snaps else 0.0, "replay": True}

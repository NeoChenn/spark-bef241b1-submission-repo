from __future__ import annotations

import bisect
from collections import Counter
from typing import Optional

import numpy as np

from fleet.contract import JobStatus
from fleet.coordinator import FAILED_RADIUS_M, GAS_ZONE_M, PERSON_RADIUS_M, Coordinator, RobotRT
from policy.fleet.api import SCHEMA, corners, deck_at
from policy.fleet.hazards import weather_state

STATE_V1 = {"IDLE": "IDLE", "MOVING": "TRANSIT", "WAITING_SEGMENT": "WAIT", "WORKING": "WORK",
            "PAUSED_PERSON": "PAUSED", "DOCKING": "TO_DOCK", "CHARGING": "CHARGING", "DOCK_QUEUE": "DOCK_QUEUE",
            "FAILED": "FAILED"}
STATUS_V1 = {JobStatus.QUEUED: "pending", JobStatus.WAITING_APPROVAL: "waiting_approval",
             JobStatus.ASSIGNED: "assigned", JobStatus.RUNNING: "active", JobStatus.DONE: "done",
             JobStatus.FINDING: "done", JobStatus.HANDED_TO_HUMAN: "escalated", JobStatus.FAILED: "failed",
             JobStatus.CANCELLED: "cancelled"}
SKIP_EVENTS = {"segment_clear"}


def _xyz(p) -> list[float]:
    return [round(float(v), 2) for v in p]


def message(e: dict) -> str:
    d = {k: v for k, v in e["detail"].items() if k != "bids"}
    who = " ".join(x for x in (e.get("robot"), e.get("job")) if x)
    if "explanation" in d:
        return d["explanation"]
    if "summary" in d:
        return f"{who}: {d['summary']}"
    if "reason" in d:
        return f"{who} {e['event'].replace('_', ' ')}: {d['reason']}".strip()
    rest = ", ".join(f"{k}={v}" for k, v in d.items())
    return f"{who} {e['event'].replace('_', ' ')}" + (f" ({rest})" if rest else "")


def event_v1(e: dict) -> dict:
    out = {"t": e["t"], "kind": e["event"], "message": message(e)}
    if e.get("robot"):
        out["robot"] = e["robot"]
    if e.get("job"):
        out["task"] = e["job"]
    return out


def phase(rt: RobotRT, t: float) -> tuple[Optional[str], Optional[str]]:
    if rt.state != "WORKING" or rt.result is None or not rt.result.phases:
        return None, None
    ph = sorted(rt.result.phases, key=lambda p: p[0])
    i = bisect.bisect_right([p[0] for p in ph], t - rt.work_t0) - 1
    _, st, why = ph[max(i, 0)]
    return st, why


def robot_json(c: Coordinator, rt: RobotRT) -> dict:
    r = rt.robot
    p = c.power(r)
    state = STATE_V1.get(rt.state, rt.state)
    if state == "CHARGING" and r.battery >= 0.999:
        state = "DOCKED"
    js, why = phase(rt, c.t)
    legs = rt.remaining
    pts = [np.asarray(r.position, float)] + [np.asarray(l.end, float) for l in legs]
    eta = sum(l.time(r.speed_mps) for l in legs) - (legs[0].time(r.speed_mps) * rt.leg_s / max(legs[0].length, 1e-9)
                                                     if legs else 0.0)
    return {"id": r.id, "kind": r.kind.value, "state": state, "coordinator_state": rt.state, "job_state": js,
            "job_reason": why, "task": rt.job, "job_kind": c.jobs[rt.job].kind if rt.job else None,
            "pose": {"x": round(float(r.position[0]), 2), "y": round(float(r.position[1]), 2),
                     "z": round(float(r.position[2]), 2), "yaw": round(float(rt.yaw), 3)},
            "deck": {"A": "air", "R": "subsea"}.get(r.kind.value) or deck_at(float(r.position[2])),
            "battery": {"soc": round(r.battery, 4), "wh": round(r.battery * p.capacity_wh, 2),
                        "used_wh": round(rt.used_wh, 2), "charged_wh": round(rt.charged_wh, 2)},
            "path": corners(pts), "prev_path": None, "reroute_t": None,
            "eta_s": round(max(eta, 0.0), 1), "dist_m": round(rt.dist_m, 1),
            "capabilities": sorted(r.capabilities)}


def hazards_json(c: Coordinator) -> dict:
    hz, rm = getattr(c.planner, "hazards", None), getattr(c.planner, "rm", None)
    if hz is not None and rm is not None:
        return hz.to_json(rm)
    return {"obstacles": [], "closures": [], "weather": weather_state(), "closed": [], "slow": []}


def task_json(c: Coordinator, jid: str) -> dict:
    j = c.jobs[jid]
    s = c.status[jid]
    res = c.results.get(jid)
    rid = c.assigned_to.get(jid) or (res.robot_id if res else None)
    rt = c.rt.get(rid) if rid else None
    progress = (c.t - rt.work_t0) if rt is not None and rt.job == jid and rt.state == "WORKING" else 0.0
    duration = (res.finished_t - res.started_t) if res else (rt.result.finished_t - rt.result.started_t
                                                            if rt is not None and rt.job == jid and rt.result else j.est_duration_s)
    if res is not None:
        progress = duration
    work = rt.route.legs[-1].end if rt is not None and rt.job == jid and rt.route and rt.route.legs else j.position
    return {"id": jid, "kind": j.kind, "poi": j.target, "xyz": _xyz(j.position), "work_xyz": _xyz(work),
            "deck": deck_at(float(work[2])), "priority": j.priority, "status": STATUS_V1[s],
            "coordinator_status": s.value, "level": j.level.value, "robot": rid,
            "severity": float(j.params.get("severity", 0.0)), "exposure": float(j.params.get("exposure", 0.0)),
            "deadline": j.params.get("deadline"), "progress": round(progress, 1), "duration": round(duration, 1),
            "outcome": (res.status.value if res else None), "parent": j.parent,
            "findings": [f.kind for f in res.findings] if res else []}


def zones_json(c: Coordinator) -> list[dict]:
    z = [{"kind": "gas", "id": l.get("id"), "xyz": _xyz(l["position"]), "radius": GAS_ZONE_M} for l in c.leaks()]
    z += [{"kind": "person", "id": f"person{i}", "xyz": _xyz(p), "radius": PERSON_RADIUS_M}
          for i, p in enumerate(c.world.people)]
    z += [{"kind": "failed_robot", "id": rt.robot.id, "xyz": _xyz(rt.robot.position), "radius": FAILED_RADIUS_M}
          for rt in c.rt.values() if rt.state == "FAILED"]
    return z


RESOLVED = {"valve_isolated", "valve_closed", "leak_isolated", "flange_retorqued"}


def alarms_json(c: Coordinator) -> list[dict]:
    out = []
    for l in c.world.state.get("gas_leaks", []):
        lid = l.get("id")
        names = {lid, l.get("source")}

        def about(e):
            j = c.jobs.get(e["job"]) if e.get("job") else None
            return j is not None and (j.target in names or j.params.get("leak") == lid)
        rep = next((e for e in c.timeline if e["event"] in ("gas_detected", "gas_alarm") and about(e)), None)
        start = next((e for e in c.timeline if e["event"] == "job_started" and rep and e["job"] == rep["job"]), None)
        given = next((e for e in c.timeline if e["event"] == "assigned" and rep and e["job"] == rep["job"]), None)
        fix = next((e for e in c.timeline if e["event"] == "finding" and e["detail"].get("kind") in RESOLVED
                    and about(e)), None)
        t0 = rep["t"] if rep else None
        out.append({"id": lid, "flange": l.get("source"), "source": l.get("source"), "t": l.get("t0"),
                    "reported_t": t0, "responder": given["robot"] if given else None,
                    "response_s": round(start["t"] - t0, 2) if start and t0 is not None else None,
                    "resolved_s": round(fix["t"] - t0, 2) if fix and t0 is not None else None,
                    "outcome": "isolated" if l.get("isolated") else ("reported" if rep else "undetected")})
    return out


def metrics(c: Coordinator) -> dict:
    kinds = sorted({j.kind for j in c.jobs.values()})
    by = {k: Counter(STATUS_V1[c.status[j]] for j in c.jobs if c.jobs[j].kind == k) for k in kinds}
    al = alarms_json(c)
    resp = [a["response_s"] for a in al if a["response_s"] is not None]
    reso = [a["resolved_s"] for a in al if a["resolved_s"] is not None]
    tot = Counter()
    for rt in c.rt.values():
        tot.update({k: v for k, v in rt.time_in.items() if k != "FAILED"})
    live = sum(tot.values()) or 1.0
    idle = sum(v for k, v in tot.items() if k in ("IDLE", "DOCK_QUEUE"))
    late = [j for j, job in c.jobs.items() if job.params.get("deadline") is not None and c.results.get(j)
            and c.results[j].finished_t > job.params["deadline"]]
    seg = Counter(rt.holding for rt in c.rt.values() if rt.holding and rt.robot.kind.value == "Q")
    return {"t": round(c.t, 1),
            "jobs_done": sum(v["done"] for v in by.values()),
            "jobs_escalated": sum(v["escalated"] for v in by.values()),
            "open_tasks": sum(1 for s in c.status.values() if STATUS_V1[s] in ("pending", "waiting_approval", "assigned", "active")),
            "jobs_by_kind": {k: dict(v) for k, v in by.items()},
            "gas_alarms": len(al),
            "gas_alarms_escalated": sum(1 for a in al if a["outcome"] == "undetected"),
            "gas_response_mean_s": round(float(np.mean(resp)), 1) if resp else None,
            "gas_response_max_s": round(float(np.max(resp)), 1) if resp else None,
            "gas_resolve_mean_s": round(float(np.mean(reso)), 1) if reso else None,
            "idle_pct": round(100.0 * idle / live, 1),
            "state_pct": {STATE_V1.get(k, k): round(100.0 * v / live, 1) for k, v in sorted(tot.items())},
            "energy_used_wh": round(sum(rt.used_wh for rt in c.rt.values()), 1),
            "energy_charged_wh": round(sum(rt.charged_wh for rt in c.rt.values()), 1),
            "energy_by_robot_wh": {rt.robot.id: round(rt.used_wh, 1) for rt in c.rt.values()},
            "distance_m": round(sum(rt.dist_m for rt in c.rt.values()), 1),
            "preemptions": c.preemptions, "robot_failures": c.failures, "replans": c.replans,
            "reservation_wait_s": round(c.wait_s, 1), "corridor_conflicts": sum(v - 1 for v in seg.values() if v > 1),
            "late_tasks": late, "approvals_pending": len(c.approvals), "crew_jobs": len(c.crew),
            "used_gemini": c.used_gemini, "reroutes": 0, "reroute_extra_m": 0.0, "reroute_extra_s": 0.0,
            "blocked_tasks": 0, "weather": hazards_json(c)["weather"]["level"]}


def approvals_json(c: Coordinator) -> list[dict]:
    return [{"job": j, "kind": c.jobs[j].kind, "target": c.jobs[j].target, "priority": c.jobs[j].priority,
             "xyz": _xyz(c.jobs[j].position), "requests": c.requests.get(j, [])} for j in c.approvals]


class Snapshotter:
    def __init__(self, c: Coordinator):
        self.c = c
        self.seen = 0

    def __call__(self) -> dict:
        c = self.c
        new = [e for e in c.timeline[self.seen:] if e["event"] not in SKIP_EVENTS]
        self.seen = len(c.timeline)
        recent = [event_v1(e) for e in c.timeline if e["event"] not in SKIP_EVENTS][-8:]
        return {"schema": SCHEMA, "t": round(c.t, 2),
                "robots": [robot_json(c, rt) for rt in c.rt.values()],
                "tasks": [task_json(c, j) for j in c.jobs],
                "zones": zones_json(c), "alarms": alarms_json(c), "hazards": hazards_json(c),
                "events": [event_v1(e) for e in new], "recent_events": recent,
                "assignments": [{"robot": e["robot"], "task": e["job"], "why": f"cost {e['detail'].get('cost_s')} s, "
                                 f"route {e['detail'].get('route_m')} m"} for e in new if e["event"] == "assigned"],
                "metrics": metrics(c), "approvals": approvals_json(c), "crew": list(c.crew)}


def snapshot(c: Coordinator) -> dict:
    return Snapshotter(c)()

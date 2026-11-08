from __future__ import annotations

from collections import Counter

import numpy as np

from policy.fleet.rigmap import DECKS
from policy.fleet.tasks import OPEN, priority

SCHEMA = "fleet_state.v1"
PREV_KEEP_S = 45.0


def _xyz(p) -> list[float]:
    return [round(float(v), 2) for v in p]


def corners(pts) -> list[list[float]]:
    out = [pts[0]]
    for a, b in zip(pts[1:], pts[2:]):
        u, v = a - out[-1], b - a
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-6 and nv > 1e-6 and float(np.dot(u, v)) / (nu * nv) < 0.999:
            out.append(a)
    if len(pts) > 1:
        out.append(pts[-1])
    return [_xyz(p) for p in out]


def deck_at(z: float) -> str:
    d = min(DECKS, key=lambda k: abs(DECKS[k] - z))
    return d if abs(DECKS[d] - z) < 0.5 else "stairs"


def robot_json(f, r) -> dict:
    js, why = f.job_state(r)
    path = corners([r.pos] + [f.rm.pos[lg.v] for lg in r.legs])
    b = r.battery
    return {"id": r.id, "state": r.state, "job_state": js, "job_reason": why,
            "task": r.task.id if r.task else None,
            "pose": {"x": round(float(r.pos[0]), 2), "y": round(float(r.pos[1]), 2), "z": round(float(r.pos[2]), 2),
                     "yaw": round(float(r.yaw), 3)},
            "deck": deck_at(float(r.pos[2])),
            "battery": {"soc": round(b.soc, 4), "wh": round(b.wh, 2), "used_wh": round(b.used_wh, 2),
                        "charged_wh": round(b.charged_wh, 2)},
            "path": path,
            "prev_path": r.prev_path if r.reroute_t is not None and f.t - r.reroute_t <= PREV_KEEP_S else None,
            "reroute_t": r.reroute_t, "eta_s": round(max(r.legs[-1].t1 - f.t, 0.0), 1) if r.legs else 0.0,
            "dist_m": round(r.dist_m, 1)}


def task_json(f, t) -> dict:
    p = f.rm.poi_pos.get(t.poi, f.rm.pos[t.node])
    return {"id": t.id, "kind": t.kind, "poi": t.poi, "xyz": _xyz(p), "work_xyz": _xyz(f.rm.pos[t.node]),
            "deck": deck_at(float(f.rm.pos[t.node][2])), "priority": round(priority(t, f.t), 2), "status": t.status,
            "robot": t.robot, "severity": t.severity, "exposure": t.exposure, "deadline": t.deadline,
            "progress": round(t.progress, 1), "duration": round(t.duration, 1), "outcome": t.outcome}


def zones_json(f) -> list[dict]:
    z = [{"kind": "gas", "id": t.id, "xyz": _xyz(f.rm.flanges[t.poi]), "radius": f.gas_zone_m}
         for t in f.tasks.values() if t.kind == "gas" and t.status in OPEN]
    z += [{"kind": "person", "id": f"person@{p['until']:.0f}", "xyz": _xyz(p["xyz"]), "radius": p["radius"]}
          for p in f.persons]
    z += [{"kind": "failed_robot", "id": r.id, "xyz": _xyz(r.pos), "radius": 1.0} for r in f.robots if r.failed]
    return z


def metrics(f) -> dict:
    kinds = sorted({t.kind for t in f.tasks.values()})
    by = {k: Counter(t.status for t in f.tasks.values() if t.kind == k) for k in kinds}
    resp = [a["response_s"] for a in f.alarms.values() if a["response_s"] is not None]
    reso = [a["resolved_s"] for a in f.alarms.values() if a["resolved_s"] is not None]
    tot = Counter()
    for r in f.robots:
        tot.update({k: v for k, v in r.time_in.items() if k != "FAILED"})
    live = sum(tot.values()) or 1.0
    idle = sum(v for k, v in tot.items() if k in ("IDLE", "DOCKED", "DOCK_QUEUE"))
    late = [t.id for t in f.tasks.values() if t.deadline is not None and t.finished is not None
            and t.finished > t.deadline]
    return {"t": round(f.t, 1),
            "jobs_done": sum(c["done"] for c in by.values()),
            "jobs_escalated": sum(c["escalated"] for c in by.values()),
            "open_tasks": sum(1 for t in f.tasks.values() if t.status in OPEN),
            "jobs_by_kind": {k: dict(c) for k, c in by.items()},
            "gas_alarms": len(f.alarms),
            "gas_alarms_escalated": sum(1 for a in f.alarms.values() if a["outcome"] == "escalated"),
            "gas_response_mean_s": round(float(np.mean(resp)), 1) if resp else None,
            "gas_response_max_s": round(float(np.max(resp)), 1) if resp else None,
            "gas_resolve_mean_s": round(float(np.mean(reso)), 1) if reso else None,
            "idle_pct": round(100.0 * idle / live, 1),
            "state_pct": {k: round(100.0 * v / live, 1) for k, v in sorted(tot.items())},
            "energy_used_wh": round(sum(r.battery.used_wh for r in f.robots), 1),
            "energy_charged_wh": round(sum(r.battery.charged_wh for r in f.robots), 1),
            "energy_by_robot_wh": {r.id: round(r.battery.used_wh, 1) for r in f.robots},
            "distance_m": round(sum(r.dist_m for r in f.robots), 1),
            "preemptions": f.preemptions, "robot_failures": f.failures, "replans": f.replans,
            "reservation_wait_s": round(f.wait_s, 1), "corridor_conflicts": f.conflicts, "late_tasks": late,
            "reroutes": len(f.reroutes),
            "reroute_extra_m": round(sum(x["extra_m"] or 0.0 for x in f.reroutes), 1),
            "reroute_extra_s": round(sum(x["extra_s"] or 0.0 for x in f.reroutes), 1),
            "blocked_tasks": sum(1 for t in f.tasks.values() if t.outcome == "blocked_by_hazard"),
            "weather": f.hz.weather["level"]}


def snapshot(f) -> dict:
    return {"schema": SCHEMA, "t": round(f.t, 2),
            "robots": [robot_json(f, r) for r in f.robots],
            "tasks": [task_json(f, t) for t in f.tasks.values()],
            "zones": zones_json(f),
            "hazards": f.hz.to_json(f.rm),
            "alarms": [{"id": k, **{a: (round(v, 2) if isinstance(v, float) else v) for a, v in al.items()}}
                       for k, al in f.alarms.items()],
            "events": f.tick_events, "recent_events": f.event_log[-8:], "assignments": f.tick_assign,
            "metrics": metrics(f)}

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from policy.control.envelope import ROBOT_FUNCTIONS, Envelope
from policy.control.schema import EDGE_KINDS, VERIFY
from policy.fleet.tasks import KIND_BASE

ARGON = "argon"
SENSOR_FOR = {"gas": "gas_sniffer", "thermal": "thermal_cam", "acoustic": "ultrasonic_mic", "pressure": "camera"}
FLEET_STATUS = {"pending": "QUEUED", "assigned": "ASSIGNED", "active": "RUNNING", "done": "DONE",
                "escalated": "HANDED_TO_HUMAN", "superseded": "CANCELLED", "cancelled": "CANCELLED"}
OPEN = ("QUEUED", "WAITING_APPROVAL", "ASSIGNED", "RUNNING")
BOOST_PER_LEVEL = 20.0
HOLD_TTL_S = 90.0
ALIASES = {"inspection": "inspect", "rust_inspect": "inspect", "gas_survey": "leak_check", "rust_repair": "rust"}


class FleetPort:
    def __init__(self, fleet, sensors: Optional[dict] = None, envelope: Optional[Envelope] = None,
                 timeline_path=None, agent: str = ARGON):
        self.f = fleet
        self.sensors = sensors or {}
        self.envelope = envelope or Envelope(agent, ROBOT_FUNCTIONS)
        self.agent = agent
        self.jobs: dict[str, dict] = {}
        self.timeline: list[dict] = []
        self.path = Path(timeline_path) if timeline_path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        self.seq = 0
        self.esd = {"active": False, "zone": None}
        self.active_edge: set[str] = set()
        self.holds: dict[str, float] = {}

    def log(self, event: str, job=None, robot=None, **detail) -> dict:
        e = {"t": round(float(self.f.t), 2), "event": event, "job": job, "robot": robot, "detail": detail}
        self.timeline.append(e)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(e, default=str) + "\n")
        return e

    def robot(self, rid: str):
        return next((r for r in self.f.robots if r.id == rid), None)

    def person_pauses(self) -> dict:
        return {r.id: None for r in self.f.robots if r.paused and not r.held and self.f._person_near(r)}

    def sync(self) -> None:
        for j in self.jobs.values():
            t = self.f.tasks.get(j.get("task") or "")
            if t is not None and j["status"] in OPEN:
                j["status"], j["robot"] = FLEET_STATUS.get(t.status, j["status"]), t.robot

    def fleet_state(self) -> dict:
        self.sync()
        robots = [{"id": r.id, "state": r.state, "battery": round(r.battery.soc, 3),
                   "position": [round(float(v), 2) for v in r.pos], "job": r.task.id if r.task else None,
                   "sensors": sorted(r.sensors), "held": r.held} for r in self.f.robots]
        return {"t": round(float(self.f.t), 2), "robots": robots, "jobs": [dict(j) for j in self.jobs.values()],
                "approvals": [j["id"] for j in self.jobs.values() if j["status"] == "WAITING_APPROVAL"],
                "esd": dict(self.esd), "person_pauses": self.person_pauses()}

    def job(self, job_id):
        self.sync()
        j = self.jobs.get(job_id)
        return dict(j) if j else None

    def open_job(self, kind: str, target: str) -> Optional[dict]:
        self.sync()
        return next((j for j in self.jobs.values() if j["kind"] == kind and j["target"] == target
                     and j["status"] in OPEN), None)

    def _task(self, j: dict) -> None:
        p = j["params"]
        kw = {}
        if j["kind"] == VERIFY:
            s = self.sensors[j["target"]]
            kw = {"xyz": list(s.pos), "needs": frozenset({SENSOR_FOR[s.kind]}), "pin": p.get("robot") or None}
        t = self.f.add_task(j["kind"], j["target"], **kw)
        t.priority_boost = (3 - j["priority"]) * BOOST_PER_LEVEL
        j["task"], j["status"] = t.id, "QUEUED"

    def plannable(self, kind: str, target: str) -> bool:
        return kind in KIND_BASE and (target in self.f.rm.poi_node or target in self.sensors)

    def queue(self, kind, target, level="AUTO", priority=3, params=None, by="operator", why="") -> str:
        kind = ALIASES.get(kind, kind)
        dup = self.open_job(kind, target)
        if dup:
            return dup["id"]
        self.seq += 1
        jid = f"{kind[:4].upper()}-{self.seq:03d}"
        j = {"id": jid, "kind": kind, "target": target, "level": level, "priority": int(priority), "status": "QUEUED",
             "requested_by": by, "why": why, "params": dict(params or {}), "task": None, "robot": None,
             "zone": getattr(self.sensors.get(target), "zone", "")}
        self.jobs[jid] = j
        if level == "SUPV":
            j["status"] = "WAITING_APPROVAL"
        elif level == "HUMAN" or not self.plannable(kind, target):
            j["status"] = "HANDED_TO_HUMAN"
        else:
            self._task(j)
        self.log("job_queued", jid, j["params"].get("robot"), kind=kind, target=target, level=level,
                 priority=int(priority), by=by, why=why)
        if j["status"] == "WAITING_APPROVAL":
            self.log("approval_requested", jid, None, kind=kind, target=target, by=by)
        return jid

    def cancel(self, job_id, by, why) -> bool:
        j = self.jobs.get(job_id)
        if not j or j["status"] not in ("QUEUED", "WAITING_APPROVAL"):
            return False
        t = self.f.tasks.get(j.get("task") or "")
        if t is not None:
            self.f._drop(t, "cancelled")
        j["status"] = "CANCELLED"
        self.log("cancelled", job_id, None, by=by, why=why)
        return True

    def approve_request(self, job_id, by, why) -> bool:
        j = self.jobs.get(job_id)
        if not j or j["status"] != "WAITING_APPROVAL":
            return False
        if self.plannable(j["kind"], j["target"]):
            self._task(j)
        else:
            j["status"] = "HANDED_TO_HUMAN"
        self.log("approved", job_id, None, by=by, why=why)
        return True

    def reprioritise(self, job_id, priority, by, why) -> bool:
        j = self.jobs.get(job_id)
        if not j or j["status"] not in OPEN:
            return False
        old, j["priority"] = j["priority"], int(priority)
        t = self.f.tasks.get(j.get("task") or "")
        if t is not None:
            t.priority_boost = (3 - int(priority)) * BOOST_PER_LEVEL
        self.log("reprioritised", job_id, None, by=by, why=why, old=old, new=int(priority))
        return True

    def hold(self, rid, by, why) -> bool:
        return self.f.hold(rid, f"{why} ({by})")

    def hold_refusal(self, rid) -> Optional[str]:
        r = self.robot(rid)
        if r is not None and r.task is not None and r.task.kind == "gas":
            return f"{rid} is on the gas response {r.task.id}; only the operator can stop a safety job"
        if r is not None and r.battery.soc < r.battery.return_frac:
            return (f"{rid} battery {r.battery.soc:.0%} is below the {r.battery.return_frac:.0%} return level; "
                    "holding it could strand it, the planner sends it to the dock")
        return None

    def resume(self, rid, by, why) -> bool:
        return self.f.resume(rid, f"{why} ({by})")

    def finish(self, job_id, status: str, **detail) -> None:
        j = self.jobs[job_id]
        j["status"] = status
        self.log("job_closed", job_id, j.get("robot"), status=status, **detail)

    def request(self, name: str, args: dict) -> dict:
        return self.envelope.call(self, name, args)

    def apply(self, actions: list[dict], health: dict) -> list[dict]:
        out, live = [], set()
        for a in actions:
            if not a.get("auto"):
                continue
            kind, rid, tgt = a["kind"], a.get("robot", ""), a.get("target", "")
            if kind in EDGE_KINDS or kind == "escalate":
                live.add(a["id"])
                if a["id"] not in self.active_edge:
                    self.log("edge_rule" if a["source"] == "edge" else "escalation", None, rid or None,
                             kind=kind, target=tgt, rule=a.get("rule"), source=a["source"], why=a["rationale"])
                continue
            if a.get("purpose") == VERIFY:
                h = health.get(tgt)
                if h is None or h.job or self.open_job(VERIFY, tgt):
                    continue
                res = self.request("queue_job", {"kind": VERIFY, "target": tgt, "level": "AUTO",
                                                 "priority": 1 if h.hazard else 2,
                                                 "params": {"robot": rid, "sensor_kind": h.kind},
                                                 "why": a["rationale"]})
                if res["ok"]:
                    jid = res["result"]["job_id"]
                    self.log("fault_dispatch", jid, rid, sensor=tgt, status=h.status, source=a["source"],
                             confidence=a["confidence"], why=a["rationale"])
                out.append(res)
            elif kind in ("hold", "resume"):
                if kind == "hold" and rid in self.holds:
                    self.holds[rid] = self.f.t
                if kind == "hold" and self.robot(rid) is not None and self.robot(rid).held:
                    continue
                if kind == "resume" and self.robot(rid) is not None and not self.robot(rid).held:
                    continue
                bad = self.hold_refusal(rid) if kind == "hold" else None
                if bad:
                    if a["id"] not in self.active_edge:
                        self.log("agent_refused", None, rid, call="hold_robot", why=bad)
                    live.add(a["id"])
                    out.append({"name": "hold_robot", "ok": False, "refused": bad})
                    continue
                res = self.request(f"{kind}_robot", {"robot": rid, "why": a["rationale"]})
                if res["ok"] and kind == "hold":
                    self.holds[rid] = self.f.t
                elif res["ok"]:
                    self.holds.pop(rid, None)
                out.append(res)
            elif kind == "dispatch" and tgt in self.jobs:
                t = self.f.tasks.get(self.jobs[tgt].get("task") or "")
                if t is not None and t.status == "pending" and t.pin != rid:
                    t.pin = rid
                    self.log("assigned_by_argon", tgt, rid, why=a["rationale"])
            elif kind not in ("dispatch", "second_view", "evacuate"):
                out.append(self.request(kind, {"robot": rid, "target": tgt, "why": a["rationale"]}))
        self.active_edge = live
        for rid, t in list(self.holds.items()):
            if self.f.t - t > HOLD_TTL_S:
                self.holds.pop(rid)
                self.f.resume(rid, f"Argon hold not renewed for {HOLD_TTL_S:.0f} s")
                self.log("hold_expired", None, rid, ttl_s=HOLD_TTL_S)
        return out

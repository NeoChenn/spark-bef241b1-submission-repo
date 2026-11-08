from __future__ import annotations

from typing import Optional

from fleet.contract import Job, JobStatus, Level, RobotKind
from fleet.coordinator import GAS_RESPONSE, IDLE_DOCK_BATTERY, OPEN
from policy.control.envelope import ROBOT_FUNCTIONS, Envelope
from policy.control.schema import EDGE_KINDS, VERIFY

ARGON = "argon"
SENSOR_FOR = {"gas": "gas_sensor", "thermal": "thermal", "acoustic": "acoustic", "pressure": "camera"}
VERIFY_S = 15.0
HOLD_TTL_S = 90.0


class FleetPort:
    def __init__(self, coordinator, sensors: Optional[dict] = None, envelope: Optional[Envelope] = None,
                 agent: str = ARGON):
        self.c = coordinator
        self.sensors = sensors or {}
        self.envelope = envelope or Envelope(agent, ROBOT_FUNCTIONS)
        self.agent = agent
        self.esd = {"active": False, "zone": None}
        self.active_edge: set[str] = set()
        self.holds: dict[str, float] = {}

    @property
    def t(self) -> float:
        return float(self.c.t)

    @property
    def timeline(self) -> list[dict]:
        return self.c.timeline

    def log(self, event: str, job=None, robot=None, **detail) -> dict:
        return self.c.emit(event, job, robot, **detail)

    def robot(self, rid: str):
        return self.c.rt.get(rid)

    def _status(self, jid) -> str:
        s = self.c.status.get(jid)
        return getattr(s, "value", s)

    def job(self, job_id):
        j = self.c.jobs.get(job_id)
        if j is None:
            return None
        return {"id": job_id, "kind": j.kind, "target": j.target, "level": j.level.value, "priority": j.priority,
                "status": self._status(job_id), "requested_by": j.params.get("requested_by", "coordinator"),
                "why": j.params.get("why", ""), "params": dict(j.params), "robot": self.c.assigned_to.get(job_id),
                "zone": getattr(self.sensors.get(j.target), "zone", j.params.get("zone", ""))}

    def jobs(self) -> list[dict]:
        return [self.job(j) for j in self.c.jobs]

    def open_job(self, kind: str, target: str) -> Optional[dict]:
        return next((self.job(j) for j, x in self.c.jobs.items() if x.kind == kind and x.target == target
                     and self.c.status.get(j) in OPEN), None)

    def fleet_state(self) -> dict:
        robots = [{"id": rid, "state": rt.state, "battery": round(float(rt.robot.battery), 3),
                   "position": [round(float(v), 2) for v in rt.robot.position], "job": rt.job,
                   "capabilities": sorted(rt.robot.capabilities)} for rid, rt in self.c.rt.items()]
        return {"t": round(self.t, 2), "robots": robots, "jobs": self.jobs(), "approvals": list(self.c.approvals),
                "crew": list(self.c.crew), "esd": dict(self.esd),
                "person_pauses": {r["id"]: None for r in robots if r["state"] == "PAUSED_PERSON"}}

    def queue(self, kind, target, level="AUTO", priority=3, params=None, by="operator", why="") -> str:
        params = dict(params or {})
        needs = set(params.pop("needs", ()))
        s = self.sensors.get(target)
        if s is not None:
            pos = s.pos
        else:
            asset = self.c.world.assets.get(target) or {}
            pos = params.pop("position", None) or asset.get("position_m")
            if pos is None:
                pos, level = (0.0, 0.0, 0.0), "HUMAN"
                params["note"] = f"{target} has no position on the rig: routed to the crew"
        kinds = set()
        if kind == VERIFY and s is not None:
            needs.add(SENSOR_FOR[s.kind])
            kinds = {RobotKind(k) for k in s.reach}
        job = Job(self.c.new_id(kind), kind, target, tuple(float(v) for v in pos), Level(level), needs=needs,
                  robot_kinds=kinds,
                  priority=int(priority), est_duration_s=VERIFY_S if kind == VERIFY else 60.0,
                  params={"standoff": 0.8, **params, "requested_by": by, "why": why})
        self.c.submit(job, allow_auto=False, by=by, note=why)
        if job.id in self.c.jobs:
            return job.id
        return self.open_job(kind, target)["id"]

    def cancel(self, job_id, by, why) -> bool:
        return self.c.cancel(job_id, by=by, reason=why)[0]

    def approve_request(self, job_id, by, why) -> bool:
        return self.c.approve(job_id, by=by)

    def reprioritise(self, job_id, priority, by, why) -> bool:
        return self.c.reprioritise(job_id, int(priority), by=by, why=why)[0]

    def hold(self, rid, by, why) -> bool:
        return self.c.hold(rid, by=by, why=why)

    def resume(self, rid, by, why) -> bool:
        return self.c.resume(rid, by=by, why=why)

    def hold_refusal(self, rid) -> Optional[str]:
        rt = self.robot(rid)
        if rt is None:
            return None
        job = self.c.jobs.get(rt.job) if rt.job else None
        if job is not None and (job.kind in GAS_RESPONSE or job.params.get("gas_response")):
            return f"{rid} is on the gas response {job.id}; only the operator can stop a safety job"
        if rt.robot.battery < IDLE_DOCK_BATTERY:
            return (f"{rid} battery {rt.robot.battery:.0%} is below the {IDLE_DOCK_BATTERY:.0%} return level; "
                    "holding it could strand it, the coordinator sends it to the dock")
        return None

    def request(self, name: str, args: dict) -> dict:
        return self.envelope.call(self, name, args)

    def to_crew(self, job_id: str, why: str) -> None:
        self.c._to_crew(job_id, why)

    def close(self, job_id: str, by: str, **detail) -> None:
        self.c.status[job_id] = JobStatus.DONE
        if job_id in self.c.crew:
            self.c.crew.remove(job_id)
        self.log("job_done", job_id, None, status="DONE", by=by, **detail)

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
                    self.log("fault_dispatch", res["result"]["job_id"], rid, sensor=tgt, status=h.status,
                             source=a["source"], confidence=a["confidence"], why=a["rationale"])
                out.append(res)
            elif kind in ("hold", "resume"):
                rt = self.robot(rid)
                if kind == "hold" and rid in self.holds:
                    self.holds[rid] = self.t
                if rt is None or (kind == "hold") == (rt.state == "HELD"):
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
                    self.holds[rid] = self.t
                elif res["ok"]:
                    self.holds.pop(rid, None)
                out.append(res)
            elif kind == "dispatch" and tgt in self.c.jobs:
                j = self.c.jobs[tgt]
                if self.c.status.get(tgt) == JobStatus.QUEUED and j.params.get("robot") != rid:
                    j.params["robot"] = rid
                    self.log("assigned_by_argon", tgt, rid, why=a["rationale"])
            elif kind not in ("dispatch", "second_view", "evacuate"):
                out.append(self.request(kind, {"robot": rid, "target": tgt, "why": a["rationale"]}))
        self.active_edge = live
        for rid, t in list(self.holds.items()):
            if self.t - t > HOLD_TTL_S:
                self.holds.pop(rid)
                self.c.resume(rid, by=self.agent, why=f"Argon hold not renewed for {HOLD_TTL_S:.0f} s")
                self.log("hold_expired", None, rid, ttl_s=HOLD_TTL_S)
        self.unpin()
        return out

    def unpin(self) -> None:
        for jid, j in self.c.jobs.items():
            rid = j.params.get("robot")
            if rid and self.c.status.get(jid) == JobStatus.QUEUED:
                rt = self.robot(rid)
                if rt is None or rt.state == "FAILED":
                    j.params.pop("robot")
                    self.log("unpinned", jid, rid, why=f"{rid} unavailable: any robot with {sorted(j.needs)}")

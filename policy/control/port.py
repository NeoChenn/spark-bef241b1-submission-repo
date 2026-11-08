from __future__ import annotations

from typing import Callable, Optional, Protocol

OPEN = ("QUEUED", "WAITING_APPROVAL", "ASSIGNED", "RUNNING")


class CoordinatorPort(Protocol):
    def fleet_state(self) -> dict: ...

    def job(self, job_id: str) -> Optional[dict]: ...

    def queue(self, kind: str, target: str, level: str = "AUTO", priority: int = 3, params: Optional[dict] = None,
              by: str = "operator", why: str = "") -> str: ...

    def cancel(self, job_id: str, by: str, why: str) -> bool: ...

    def approve_request(self, job_id: str, by: str, why: str) -> bool: ...

    def reprioritise(self, job_id: str, priority: int, by: str, why: str) -> bool: ...

    def log(self, event: str, job: Optional[str] = None, robot: Optional[str] = None, **detail) -> dict: ...


class MemoryCoordinator:
    def __init__(self, robots: list[dict], jobs=(), clock: Callable[[], float] = lambda: 0.0):
        self.robots = [dict(r) for r in robots]
        self.jobs: dict[str, dict] = {}
        for j in jobs:
            self.jobs[j["id"]] = {"params": {}, "why": "", "requested_by": "operator", "level": "AUTO", "priority": 3,
                                  "status": "QUEUED", **j}
        self.clock = clock
        self.esd = {"active": False, "zone": None}
        self.pauses: dict[str, float] = {}
        self.timeline: list[dict] = []
        self.seq = 0

    def log(self, event, job=None, robot=None, **detail) -> dict:
        e = {"t": round(float(self.clock()), 2), "event": event, "job": job, "robot": robot, "detail": detail}
        self.timeline.append(e)
        return e

    def fleet_state(self) -> dict:
        return {"t": round(float(self.clock()), 2), "robots": [dict(r) for r in self.robots],
                "jobs": [dict(j) for j in self.jobs.values()],
                "approvals": [j["id"] for j in self.jobs.values() if j["status"] == "WAITING_APPROVAL"],
                "esd": dict(self.esd), "person_pauses": dict(self.pauses)}

    def job(self, job_id):
        j = self.jobs.get(job_id)
        return dict(j) if j else None

    def queue(self, kind, target, level="AUTO", priority=3, params=None, by="operator", why="") -> str:
        dup = next((j["id"] for j in self.jobs.values() if j["kind"] == kind and j["target"] == target
                    and j["status"] in OPEN), None)
        if dup:
            return dup
        self.seq += 1
        jid = f"{kind[:4].upper()}-{self.seq:03d}"
        status = "WAITING_APPROVAL" if level == "SUPV" else "HANDED_TO_HUMAN" if level == "HUMAN" else "QUEUED"
        self.jobs[jid] = {"id": jid, "kind": kind, "target": target, "level": level, "priority": int(priority),
                          "status": status, "requested_by": by, "why": why, "params": dict(params or {})}
        self.log("job_queued", jid, None, kind=kind, target=target, level=level, priority=int(priority), by=by, why=why)
        if status == "WAITING_APPROVAL":
            self.log("approval_requested", jid, None, kind=kind, target=target, by=by)
        return jid

    def cancel(self, job_id, by, why) -> bool:
        j = self.jobs.get(job_id)
        if not j or j["status"] not in ("QUEUED", "WAITING_APPROVAL"):
            return False
        j["status"] = "CANCELLED"
        self.log("cancelled", job_id, None, by=by, why=why)
        return True

    def approve_request(self, job_id, by, why) -> bool:
        j = self.jobs.get(job_id)
        if not j or j["status"] != "WAITING_APPROVAL":
            return False
        j["status"] = "QUEUED"
        self.log("approved", job_id, None, by=by, why=why)
        return True

    def reprioritise(self, job_id, priority, by, why) -> bool:
        j = self.jobs.get(job_id)
        if not j or j["status"] not in OPEN:
            return False
        old, j["priority"] = j["priority"], int(priority)
        self.log("reprioritised", job_id, None, by=by, why=why, old=old, new=int(priority))
        return True


class ContractAdapter:
    def __init__(self, coord, contract=None):
        if contract is None:
            import fleet.contract as contract
        self.c, self.k = coord, contract

    def _status(self, jid) -> str:
        s = self.c.status.get(jid)
        return getattr(s, "value", s)

    def _job(self, jid) -> dict:
        j = self.c.jobs[jid]
        return {"id": jid, "kind": j.kind, "target": j.target, "level": getattr(j.level, "value", j.level),
                "priority": j.priority, "status": self._status(jid),
                "requested_by": j.params.get("requested_by", "coordinator"), "why": j.params.get("why", ""),
                "params": dict(j.params), "robot": self.c.assigned_to.get(jid), "parent": j.parent}

    def log(self, event, job=None, robot=None, **detail) -> dict:
        return self.c.emit(event, job, robot, **detail)

    def fleet_state(self) -> dict:
        robots = [{"id": rid, "kind": getattr(rt.robot.kind, "value", rt.robot.kind), "state": rt.state,
                   "battery": round(float(rt.robot.battery), 3),
                   "position": [round(float(v), 2) for v in rt.robot.position], "job": rt.job}
                  for rid, rt in self.c.rt.items()]
        st = getattr(self.c.world, "state", {}) or {}
        return {"t": round(float(self.c.t), 2), "robots": robots, "jobs": [self._job(j) for j in self.c.jobs],
                "approvals": list(self.c.approvals), "crew": list(getattr(self.c, "crew", [])),
                "esd": dict(st.get("esd") or {"active": False, "zone": None}),
                "person_pauses": {r["id"]: None for r in robots if r["state"] == "PAUSED_PERSON"}}

    def job(self, job_id):
        return self._job(job_id) if job_id in self.c.jobs else None

    def queue(self, kind, target, level="AUTO", priority=3, params=None, by="operator", why="") -> str:
        params = dict(params or {})
        jid = self.c.new_id(kind)
        asset = (getattr(self.c.world, "assets", {}) or {}).get(target) or {}
        pos = tuple(float(v) for v in (params.pop("position", None) or asset.get("position_m") or (0.0, 0.0, 0.0)))
        job = self.k.Job(id=jid, kind=kind, target=target, position=pos, level=self.k.Level(level),
                         priority=int(priority), params={**params, "requested_by": by, "why": why})
        self.c.submit(job, allow_auto=by == "operator", by=by, note=why)
        if jid in self.c.jobs:
            return jid
        return next((j for j, x in self.c.jobs.items() if x.kind == kind and x.target == target
                     and self._status(j) in OPEN), jid)

    def cancel(self, job_id, by, why) -> bool:
        return self.c.cancel(job_id, by=by, reason=why)[0]

    def approve_request(self, job_id, by, why) -> bool:
        ok = self.c.approve(job_id, by=by)
        if ok:
            self.c.emit("approval_reason", job_id, None, by=by, why=why)
        return ok

    def reprioritise(self, job_id, priority, by, why) -> bool:
        return self.c.reprioritise(job_id, int(priority), by=by, why=why)[0]

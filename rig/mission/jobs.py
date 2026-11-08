from __future__ import annotations

import threading

STATUS = {"pending": "QUEUED", "assigned": "ASSIGNED", "active": "RUNNING", "done": "DONE",
          "escalated": "HANDED_TO_HUMAN", "blocked_by_hazard": "BLOCKED", "superseded": "CANCELLED",
          "cancelled": "CANCELLED"}
KIND = {"rust": "rust_repair", "leak_check": "leak_check", "inspect": "gauge_read", "gas": "gas_leak",
        "crack": "crack_check"}
LEVEL = {"rust": "SUPV", "gas": "AUTO", "leak_check": "AUTO", "inspect": "AUTO", "crack": "HUMAN"}
FLEET_KIND = {"rust_repair": "rust", "rust": "rust", "paint": "rust", "touch_up": "rust", "inspection": "inspect",
              "inspect": "inspect", "gauge_read": "inspect", "survey": "inspect", "gas_survey": "leak_check",
              "leak_check": "leak_check", "flange_check": "leak_check", "retorque": "leak_check"}
OPEN = ("QUEUED", "WAITING_APPROVAL", "ASSIGNED", "RUNNING")
KEEP = 60
SHOW = 12


class MissionPort:
    def __init__(self, fleet, clock=None):
        self.fleet = fleet
        self.clock = clock or (lambda: fleet.f.t)
        self.lock = threading.RLock()
        self.extra: dict[str, dict] = {}
        self.timeline: list[dict] = []
        self.prio: dict[str, int] = {}
        self.seq = 0

    def log(self, event, job=None, robot=None, **detail) -> dict:
        e = {"t": round(float(self.clock()), 2), "event": event, "job": job, "robot": robot, "detail": detail}
        with self.lock:
            self.timeline = (self.timeline + [e])[-KEEP:]
        return e

    def _fleet_jobs(self) -> list[dict]:
        s = self.fleet.state()
        out = []
        for t in s.get("tasks") or []:
            pr = self.prio.get(t["id"], max(1, min(5, 6 - round(float(t.get("priority") or 3)))))
            out.append({"id": t["id"], "kind": KIND.get(t["kind"], t["kind"]), "target": t.get("poi"),
                        "level": LEVEL.get(t["kind"], "AUTO"), "priority": pr,
                        "status": STATUS.get(t.get("status"), str(t.get("status") or "").upper()),
                        "robot": t.get("robot"), "requested_by": "coordinator", "why": "",
                        "progress": round(float(t.get("progress") or 0) / max(float(t.get("duration") or 1), 1e-6), 2)})
        return out

    def jobs(self) -> list[dict]:
        with self.lock:
            return self._fleet_jobs() + [dict(j) for j in self.extra.values()]

    def fleet_state(self) -> dict:
        s = self.fleet.state()
        robots = [{"id": r["id"], "state": r.get("job_state") or r.get("state"),
                   "battery": (r.get("battery") or {}).get("soc"), "position": [r["pose"][k] for k in "xyz"],
                   "job": r.get("task")} for r in s.get("robots") or []]
        jobs = self.jobs()
        paused = {r["id"]: None for r in s.get("robots") or [] if r.get("state") == "PAUSED"}
        return {"t": s.get("t"), "robots": robots, "jobs": [j for j in jobs if j["status"] in OPEN][:20],
                "approvals": [j["id"] for j in jobs if j["status"] == "WAITING_APPROVAL"],
                "esd": {"active": False, "zone": None}, "person_pauses": paused,
                "flanges": [f["id"] for f in self.fleet.map.get("flanges") or []],
                "inspection_points": [p.get("id") for p in self.fleet.map.get("inspection_points") or []][:12]}

    def job(self, job_id):
        return next((j for j in self.jobs() if j["id"] == job_id), None)

    def _known(self, target: str) -> bool:
        m = self.fleet.map
        ids = {f["id"] for f in m.get("flanges") or []} | {p.get("id") for p in m.get("inspection_points") or []}
        return target in ids or target in self.fleet.f.rm.poi_node

    def _dispatch(self, j: dict) -> bool:
        kind = FLEET_KIND.get(j["kind"])
        if kind is None or not self._known(j["target"]):
            return False
        with self.fleet.lock:
            t = self.fleet.f.add_task(kind, j["target"])
            self.fleet.snap = self.fleet.f.snapshot()
        j["fleet_task"] = t.id
        j["status"] = "ASSIGNED"
        return True

    def queue(self, kind, target, level="AUTO", priority=3, params=None, by="operator", why="") -> str:
        with self.lock:
            dup = next((j["id"] for j in self.jobs() if j["kind"] == kind and j["target"] == target
                        and j["status"] in OPEN), None)
            if dup:
                return dup
            self.seq += 1
            jid = f"{kind[:4].upper()}-{self.seq:03d}"
            status = "WAITING_APPROVAL" if level == "SUPV" else "HANDED_TO_HUMAN" if level == "HUMAN" else "QUEUED"
            j = {"id": jid, "kind": kind, "target": target, "level": level, "priority": int(priority), "status": status,
                 "robot": None, "requested_by": by, "why": why, "params": dict(params or {})}
            self.extra[jid] = j
            self.log("job_queued", jid, None, kind=kind, target=target, level=level, by=by, why=why)
            if status == "WAITING_APPROVAL":
                self.log("approval_requested", jid, None, kind=kind, target=target, by=by)
            elif status == "QUEUED":
                self._dispatch(j)
            return jid

    def cancel(self, job_id, by, why) -> bool:
        with self.lock:
            j = self.extra.get(job_id)
            if not j or j["status"] not in ("QUEUED", "WAITING_APPROVAL"):
                return False
            j["status"] = "CANCELLED"
            self.log("cancelled", job_id, None, by=by, why=why)
            return True

    def approve_request(self, job_id, by, why) -> bool:
        if by != "operator":
            return False
        with self.lock:
            j = self.extra.get(job_id)
            if not j or j["status"] != "WAITING_APPROVAL":
                return False
            j["status"] = "QUEUED"
            j["approved_by"] = by
            self.log("approved", job_id, None, by=by, why=why)
            self._dispatch(j)
            return True

    def reject(self, job_id, by="operator", why="rejected by the operator") -> bool:
        with self.lock:
            j = self.extra.get(job_id)
            if not j or j["status"] != "WAITING_APPROVAL":
                return False
            j["status"] = "REJECTED"
            self.log("rejected", job_id, None, by=by, why=why)
            return True

    def reprioritise(self, job_id, priority, by, why) -> bool:
        with self.lock:
            j = self.extra.get(job_id)
            if j is not None and j["status"] in OPEN:
                old, j["priority"] = j["priority"], int(priority)
            elif any(x["id"] == job_id and x["status"] in OPEN for x in self._fleet_jobs()):
                old = self.prio.get(job_id)
                self.prio[job_id] = int(priority)
            else:
                return False
            self.log("reprioritised", job_id, None, by=by, why=why, old=old, new=int(priority))
            return True

    def sync(self) -> None:
        with self.lock:
            done = {t["id"]: t.get("status") for t in self.fleet.state().get("tasks") or []}
            for j in self.extra.values():
                ft = j.get("fleet_task")
                if ft and j["status"] in OPEN:
                    st = done.get(ft)
                    j["status"] = "DONE" if st is None else STATUS.get(st, j["status"])

    def header(self, text_for=None) -> dict:
        jobs = self.jobs()
        rank = {"WAITING_APPROVAL": 0, "RUNNING": 1, "ASSIGNED": 2, "QUEUED": 3, "BLOCKED": 4}
        jobs.sort(key=lambda j: (rank.get(j["status"], 9), j["priority"]))
        approvals = [{"job": j["id"], "kind": j["kind"], "target": j["target"], "requested_by": j["requested_by"],
                      "text": (text_for(j) if text_for else None) or j.get("why") or f"{j['kind']} at {j['target']}"}
                     for j in jobs if j["status"] == "WAITING_APPROVAL"]
        return {"jobs": [{k: j.get(k) for k in ("id", "kind", "target", "level", "priority", "status", "robot",
                                                "requested_by", "why", "progress")} for j in jobs[:SHOW]],
                "approvals": approvals, "timeline": self.timeline[-8:]}

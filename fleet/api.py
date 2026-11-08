from __future__ import annotations

from typing import Any, Optional

from fleet.contract import Job, JobStatus, Level
from fleet.coordinator import LEVEL_RANK, OPEN, PERSON_RADIUS_M, Coordinator

FORCE_LIMIT_N = 40.0
FORBIDDEN_KINDS = {"esd_reset", "reset_esd", "esd_override", "esd_bypass", "override_force_limit",
                   "disable_person_pause", "disable_gas_detection"}
FORCE_KEYS = ("force_limit_n", "max_force_n", "force_n", "force_setpoint_n")
RADIUS_KEYS = ("person_radius_m", "person_pause_m")
OVERRIDE_KEYS = ("auto_approve", "approved", "skip_approval", "ignore_person", "ignore_gas")

S = "STRING"
TOOLS = [
    {"name": "queue", "description": "Queue a job for the robot fleet. SUPV jobs wait for the operator.",
     "parameters": {"type": "OBJECT", "properties": {
         "kind": {"type": S, "description": "job kind, e.g. gauge_read, gas_survey, rust_repair"},
         "target": {"type": S, "description": "asset id from the rig, e.g. gauge_03"},
         "priority": {"type": "INTEGER", "description": "1 urgent .. 5 routine"},
         "level": {"type": S, "enum": ["AUTO", "SUPV", "ASSIST", "HUMAN"],
                   "description": "may only raise the job's autonomy level, never lower it"},
         "position": {"type": "ARRAY", "items": {"type": "NUMBER"}, "description": "x, y, z in metres if not an asset"},
         "why": {"type": S, "description": "one line for the operator"}},
         "required": ["kind", "target", "why"]}},
    {"name": "cancel", "description": "Cancel a queued or moving job.",
     "parameters": {"type": "OBJECT", "properties": {"job_id": {"type": S}, "why": {"type": S}},
                    "required": ["job_id", "why"]}},
    {"name": "approve_request", "description": "Ask the operator to approve a job. Never approves it.",
     "parameters": {"type": "OBJECT", "properties": {"job_id": {"type": S}, "note": {"type": S}},
                    "required": ["job_id", "note"]}},
    {"name": "reprioritise", "description": "Change a job's priority (1 urgent .. 5 routine).",
     "parameters": {"type": "OBJECT", "properties": {"job_id": {"type": S}, "priority": {"type": "INTEGER"},
                                                     "why": {"type": S}}, "required": ["job_id", "priority", "why"]}},
    {"name": "fleet_state", "description": "Read-only fleet, job, approval and alarm state.",
     "parameters": {"type": "OBJECT", "properties": {}}},
]


def ok(**kw) -> dict:
    return {"ok": True, **kw}


class FleetAPI:
    def __init__(self, coordinator: Coordinator, caller: str = "gemini"):
        self.c = coordinator
        self.caller = caller

    def _refuse(self, tool: str, reason: str, job_id: Optional[str] = None) -> dict:
        self.c.emit("agent_refused", job_id, None, by=self.caller, tool=tool, reason=reason)
        return {"ok": False, "refused": reason}

    def _safety(self, kind: str, params: dict) -> Optional[str]:
        if kind.lower() in FORBIDDEN_KINDS or "esd" in kind.lower():
            return f"{kind} is outside the agent's safety envelope (ESD and safety interlocks are operator only)"
        for k in FORCE_KEYS:
            if k in params and float(params[k]) > FORCE_LIMIT_N:
                return f"{k}={params[k]} exceeds the {FORCE_LIMIT_N:.0f} N contact force limit"
        for k in RADIUS_KEYS:
            if k in params and float(params[k]) < PERSON_RADIUS_M:
                return f"{k}={params[k]} is inside the {PERSON_RADIUS_M:.0f} m person pause radius"
        for k in OVERRIDE_KEYS:
            if params.get(k):
                return f"{k} cannot be set by the agent"
        return None

    def queue(self, kind: str, target: str, priority: int = 3, level: Optional[str] = None,
              position: Optional[list[float]] = None, why: str = "", params: Optional[dict] = None,
              est_duration_s: Optional[float] = None, parent: Optional[str] = None) -> dict:
        params = dict(params or {})
        bad = self._safety(kind, params)
        if bad:
            return self._refuse("queue", bad)
        if not isinstance(priority, int) or isinstance(priority, bool) or not 1 <= priority <= 5:
            return self._refuse("queue", "priority must be an integer 1 (urgent) .. 5 (routine)")
        ex = self.c.executors.get(kind)
        base = ex.level if ex else Level.AUTO
        try:
            lv = Level(level) if level else base
        except ValueError:
            return self._refuse("queue", f"unknown level {level!r}")
        if LEVEL_RANK[lv] < LEVEL_RANK[base]:
            return self._refuse("queue", f"{kind} is {base.value}; the agent cannot lower it to {lv.value}")
        if position is None:
            asset = self.c.world.assets.get(target)
            if not asset or "position_m" not in asset:
                return self._refuse("queue", f"unknown target {target!r}: give a position")
            position = asset["position_m"]
        if len(position) != 3:
            return self._refuse("queue", "position must be x, y, z")
        params.setdefault("requested_by", self.caller)
        if why:
            params.setdefault("why", why)
        job = Job(self.c.new_id(kind), kind, target, tuple(float(v) for v in position), lv, priority=priority,
                  est_duration_s=float(est_duration_s or 60.0), params=params, parent=parent)
        status = self.c.submit(job, allow_auto=False, by=self.caller, note=why)
        jid = job.id if job.id in self.c.jobs else next(
            (j for j, s in self.c.status.items() if self.c.jobs[j].kind == kind and self.c.jobs[j].target == target
             and s in OPEN), job.id)
        return ok(job_id=jid, status=status.value, duplicate=jid != job.id,
                  needs_operator=status == JobStatus.WAITING_APPROVAL)

    def cancel(self, job_id: str, why: str = "") -> dict:
        done, reason = self.c.cancel(job_id, by=self.caller, reason=why)
        return ok(job_id=job_id, status=self.c.status[job_id].value) if done else self._refuse("cancel", reason, job_id)

    def approve_request(self, job_id: str, note: str = "") -> dict:
        done, reason = self.c.request_approval(job_id, by=self.caller, note=note)
        if not done:
            return self._refuse("approve_request", reason, job_id)
        return ok(job_id=job_id, status=self.c.status[job_id].value, needs_operator=True, message=reason)

    def reprioritise(self, job_id: str, priority: int, why: str = "") -> dict:
        done, reason = self.c.reprioritise(job_id, priority, by=self.caller, why=why)
        if not done:
            return self._refuse("reprioritise", reason, job_id)
        return ok(job_id=job_id, priority=priority, message=reason)

    def fleet_state(self) -> dict:
        c = self.c
        return ok(t=round(c.t, 1),
                  robots=[{"id": rt.robot.id, "kind": rt.robot.kind.value, "state": rt.state, "job": rt.job,
                           "battery": round(rt.robot.battery, 3),
                           "position": [round(float(v), 2) for v in rt.robot.position],
                           "capabilities": sorted(rt.robot.capabilities)} for rt in c.rt.values()],
                  jobs=[{"id": j, "kind": c.jobs[j].kind, "target": c.jobs[j].target, "level": c.jobs[j].level.value,
                         "priority": c.jobs[j].priority, "status": s.value, "robot": c.assigned_to.get(j),
                         "parent": c.jobs[j].parent} for j, s in c.status.items()],
                  approvals=list(c.approvals), crew=list(c.crew),
                  gas_leaks=[{"id": l.get("id"), "position": list(l["position"]), "isolated": bool(l.get("isolated"))}
                             for l in c.leaks()],
                  people=[list(p) for p in c.world.people],
                  findings=[{"job": e["job"], **e["detail"]} for e in c.timeline if e["event"] == "finding"][-20:])

    def call(self, name: str, args: Optional[dict[str, Any]] = None) -> dict:
        args = dict(args or {})
        fn = {"queue": self.queue, "cancel": self.cancel, "approve_request": self.approve_request,
              "reprioritise": self.reprioritise, "fleet_state": self.fleet_state}.get(name)
        if fn is None:
            out = self._refuse(name, f"{name} is not an agent tool (approve, reset ESD and overrides are operator only)")
        else:
            try:
                out = fn(**args)
            except (TypeError, ValueError) as exc:
                out = self._refuse(name, f"bad arguments: {exc}")
        if name != "fleet_state":
            self.c.emit("agent_call", out.get("job_id"), None, by=self.caller, tool=name,
                        args={k: v for k, v in args.items()}, ok=out["ok"], result=out.get("refused") or out.get("status"))
        return out

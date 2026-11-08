from __future__ import annotations

import math
from typing import Optional

from fleet.api import FleetAPI
from fleet.coordinator import GAS_RESPONSE, IDLE_DOCK_BATTERY, OPEN, RESERVE, URGENT, Coordinator
from policy.control.envelope import MOTOR_REFUSAL, MOTOR_WORDS

ARGON = "argon"
RUST = {"brush", "spray"}
FIXER = {"gas_sensor", "gripper"}
NEEDS = {"rust_repair": RUST, "rust": RUST, "spray": RUST, "paint": RUST,
         "flange_leak_check": FIXER, "flange_retorque": FIXER, "gas_survey": FIXER, "leak_fix": FIXER,
         "valve_isolate": FIXER, "verify_sensor": FIXER}
PRIORITY = {"verify_sensor": 2, "flange_leak_check": 1, "flange_retorque": 1, "gas_survey": 1, "leak_fix": 1,
            "valve_isolate": 1, "rust_repair": 3, "rust": 3}
TICKET = "sensor_maintenance"
BUSY = {"FAILED": "robot failed", "HELD": "held", "PAUSED_PERSON": "paused for a person", "CHARGING": "charging",
        "DOCKING": "returning to dock", "DOCK_QUEUE": "waiting for a dock pad"}


class FleetAdapter:
    def __init__(self, coordinator: Coordinator, caller: str = ARGON, needs: Optional[dict] = None):
        self.c = coordinator
        self.api = FleetAPI(coordinator, caller)
        self.caller = caller
        self.needs = {**NEEDS, **(needs or {})}

    def log(self, event: str, job=None, robot=None, **detail) -> dict:
        return self.c.emit(event, job, robot, **detail)

    def needs_for(self, kind: str) -> set[str]:
        ex = self.c.executors.get(kind)
        return set(self.needs.get(kind, ())) | (set(ex.needs) if ex else set())

    def candidates(self, kind: str, position, standoff: float = 0.8, est_s: float = 60.0) -> list[dict]:
        need = self.needs_for(kind)
        goal = tuple(float(v) for v in position)
        out = []
        for rid, rt in self.c.rt.items():
            r = rt.robot
            route = self.c.planner.route(r, goal, standoff)
            eta = None if route.blocked else route.time(r.speed_mps)
            p = self.c.power(r)
            c = {"robot": rid, "kind": r.kind.value, "eta_s": None if eta is None else round(eta, 1),
                 "battery": round(float(r.battery), 3), "missing": sorted(need - set(r.capabilities)), "ok": False,
                 "why": ""}
            if eta is not None:
                end = route.legs[-1].end if route.legs else tuple(r.position)
                wh = route.energy_wh(p, r.speed_mps) + p.work_w * est_s / 3600.0 + self.c.to_dock_wh(r, end, rt.dock)
                c["need_wh"] = round(wh, 1)
                c["spare_wh"] = round((float(r.battery) - RESERVE) * p.capacity_wh, 1)
            job = self.c.jobs.get(rt.job) if rt.job else None
            if c["missing"]:
                c["why"] = "no " + ", ".join(c["missing"])
            elif rt.state in BUSY:
                c["why"] = BUSY[rt.state]
            elif job is not None and (job.priority <= URGENT or job.kind in GAS_RESPONSE
                                      or job.params.get("gas_response")):
                c["why"] = f"busy: {job.kind} {job.target}"
            elif eta is None:
                c["why"] = "unreachable"
            elif c["need_wh"] > c["spare_wh"]:
                c["why"] = f"battery: round trip {c['need_wh']:.0f} Wh > spare {c['spare_wh']:.0f} Wh"
            else:
                c["ok"] = True
            out.append(c)
        return sorted(out, key=lambda c: (not c["ok"], c["eta_s"] if c["eta_s"] is not None else math.inf))

    def assign(self, kind: str, target: str, position=None, priority: Optional[int] = None, why: str = "",
               params: Optional[dict] = None, robot: Optional[str] = None, est_s: float = 60.0) -> dict:
        bad = self._motor(kind, params or {})
        if bad:
            return self._refuse("assign", bad)
        if position is None:
            asset = self.c.world.assets.get(target) or {}
            position = asset.get("position_m")
        if position is None:
            return self._refuse("assign", f"unknown target {target!r}: give a position")
        cands = self.candidates(kind, position, (params or {}).get("standoff", 0.8), est_s)
        ok = [c for c in cands if c["ok"]]
        if robot is not None and robot not in {c["robot"] for c in ok}:
            why_not = next((c["why"] for c in cands if c["robot"] == robot), "unknown robot")
            return self._refuse("assign", f"{robot} cannot take {kind} at {target}: {why_not}", candidates=cands)
        pick = robot or (ok[0]["robot"] if ok else None)
        if pick is None:
            return self._refuse("assign", f"no capable robot with battery for {kind} at {target}", candidates=cands)
        res = self.api.queue(kind, target, priority or PRIORITY.get(kind, 3), position=list(position), why=why,
                             params={**(params or {}), "robot": pick}, est_duration_s=est_s)
        if not res["ok"]:
            return {**res, "candidates": cands}
        job = self.c.jobs[res["job_id"]]
        job.needs = set(job.needs) | self.needs_for(kind)
        self.log("fleet_assign", res["job_id"], pick, kind=kind, target=target, by=self.caller, why=why,
                 candidates=[{k: c[k] for k in ("robot", "eta_s", "ok", "why")} for c in cands])
        return {**res, "robot": pick, "candidates": cands}

    def ticket(self, sensor_id: str, why: str, position=None, evidence: Optional[dict] = None) -> dict:
        res = self.api.queue(TICKET, sensor_id, 2, level="SUPV", position=list(position) if position else None,
                             why=why, params={"evidence": dict(evidence or {})})
        if not res["ok"]:
            return res
        jid = res["job_id"]
        self.c.jobs[jid].needs = set(self.c.jobs[jid].needs) | {"crew"}
        inbox = self.api.approve_request(jid, note=why)
        self.log("sensor_ticket", jid, None, sensor=sensor_id, by=self.caller, why=why, evidence=evidence or {})
        return {**res, "inbox": inbox}

    def hold(self, robot: str, why: str = "") -> dict:
        rt = self.c.rt.get(robot)
        if rt is None:
            return self._refuse("hold", f"unknown robot {robot!r}")
        job = self.c.jobs.get(rt.job) if rt.job else None
        if job is not None and (job.kind in GAS_RESPONSE or job.params.get("gas_response")
                                or self.needs.get(job.kind) == FIXER):
            return self._refuse("hold", f"{robot} is on the gas response {job.id}; only the operator stops it")
        if rt.robot.battery < IDLE_DOCK_BATTERY:
            return self._refuse("hold", f"{robot} battery {rt.robot.battery:.0%} is below the return level")
        if not self.c.hold(robot, by=self.caller, why=why):
            return self._refuse("hold", f"{robot} cannot be held while {rt.state}")
        return {"ok": True, "robot": robot, "job": rt.job, "job_status": self._status(rt.job)}

    def resume(self, robot: str, why: str = "") -> dict:
        rt = self.c.rt.get(robot)
        if rt is None:
            return self._refuse("resume", f"unknown robot {robot!r}")
        if not self.c.resume(robot, by=self.caller, why=why):
            return self._refuse("resume", f"{robot} is not held ({rt.state})")
        return {"ok": True, "robot": robot, "state": rt.state, "job": rt.job}

    def call(self, name: str, args: Optional[dict] = None) -> dict:
        args = dict(args or {})
        bad = self._motor(name, args.get("params") or args)
        if bad:
            return self._refuse(name, bad)
        if name in ("assign", "ticket", "hold", "resume"):
            try:
                return getattr(self, name)(**args)
            except TypeError as exc:
                return self._refuse(name, f"bad arguments: {exc}")
        return self.api.call(name, args)

    def open_jobs(self, robot: Optional[str] = None) -> list[str]:
        return [j for j, s in self.c.status.items() if s in OPEN
                and (robot is None or self.c.jobs[j].params.get("robot") == robot)]

    def _status(self, jid) -> Optional[str]:
        s = self.c.status.get(jid) if jid else None
        return getattr(s, "value", s)

    def _motor(self, name: str, params: dict) -> Optional[str]:
        keys = [str(k).lower() for k in params] + [str(name).lower()]
        return MOTOR_REFUSAL if any(w in k for k in keys for w in MOTOR_WORDS) else None

    def _refuse(self, tool: str, reason: str, **extra) -> dict:
        self.c.emit("agent_refused", None, None, by=self.caller, tool=tool, reason=reason)
        return {"ok": False, "refused": reason, **extra}

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from fleet.contract import Job, JobResult, JobStatus, Level, RobotKind
from fleet.coordinator import GAS_RESPONSE, OPEN, RESERVE, URGENT
from fleet.risk import risk_priority
from fleet.sim import new_job
from policy.control.centre import ControlCentre, health_view
from policy.control.fleet_port import SENSOR_FOR, VERIFY_S, FleetPort
from policy.control.health import DRIFT_TOL, NOISE_FLOOR, NOISY_RATIO, STUCK_EPS, STUCK_RATIO, HealthMonitor
from policy.control.rigsensors import RigSensors, zone_of
from policy.control.schema import VERIFY
from policy.control.snapshot import Robot, SensorSnapshot
from policy.fleet.rigmap import DECKS

GAS_ALARM_LEL = 10.0
MAINTENANCE = "sensor_maintenance"
TRUST_S = 600.0
MAX_EVENTS = 12
STANDOFF_M = 0.8
CANDIDATE_TTL_S = 3.0
NOT_FREE = {"FAILED": "robot failed", "HELD": "held", "PAUSED_PERSON": "paused for a person",
            "CHARGING": "charging", "DOCKING": "returning to dock", "DOCK_QUEUE": "waiting for a dock pad"}
KIND_NAME = {RobotKind.QUADRUPED: "quadruped", RobotKind.AERIAL: "drone", RobotKind.CRAWLER: "crawler",
             RobotKind.ROV: "ROV"}


class VerifySensor:
    kind = VERIFY
    level = Level.AUTO
    needs: set = set()
    robot_kinds: set = set()

    def run(self, job: Job, robot, world) -> JobResult:
        return JobResult(job.id, robot.id, JobStatus.DONE, world.t, world.t + VERIFY_S,
                         log=[f"read {job.target} with the onboard {next(iter(job.needs), 'sensor')}"])


def deck_name(z: float) -> str:
    d = min(DECKS, key=lambda k: abs(DECKS[k] - z))
    return d if abs(DECKS[d] - z) < 2.0 else "stairs" if 20.0 < z < 43.0 else "air"


def compare(kind: str, robot: list[float], head: list[float]) -> dict:
    r = np.asarray(robot, float)
    h = np.asarray(head, float)
    out = {"robot_mean": round(float(r.mean()), 3) if r.size else None, "robot_std": round(float(r.std()), 4)
           if r.size else None, "head_mean": round(float(h.mean()), 3) if h.size else None,
           "head_std": round(float(h.std()), 4) if h.size else None, "n_robot": int(r.size), "n_head": int(h.size),
           "tol": DRIFT_TOL[kind]}
    if not r.size:
        return {**out, "agree": False, "why": "robot took no reading"}
    if not h.size:
        return {**out, "agree": False, "why": "head sent no data while the robot read it"}
    eps = STUCK_EPS[kind]
    bias = float(h.mean() - r.mean())
    out["bias"] = round(bias, 3)
    if abs(bias) > DRIFT_TOL[kind]:
        return {**out, "agree": False, "why": f"head is off by {bias:+.2f} (tol {DRIFT_TOL[kind]})"}
    if h.std() <= eps and r.std() > STUCK_RATIO * eps:
        return {**out, "agree": False, "why": "head is flat while the robot sees the signal move"}
    if h.std() > NOISY_RATIO * max(r.std(), NOISE_FLOOR[kind]):
        return {**out, "agree": False, "why": "head is far noisier than the robot's reading"}
    return {**out, "agree": True, "why": "robot and head agree"}


class ArgonBrain:
    def __init__(self, sim, rig: RigSensors, centre: ControlCentre, port: Optional[FleetPort] = None,
                 monitor: Optional[HealthMonitor] = None, operator_s: Optional[float] = 40.0, crew_s: float = 90.0):
        self.sim, self.c, self.rig, self.centre = sim, sim.c, rig, centre
        self.port = port or FleetPort(self.c, rig.specs)
        self.monitor = monitor or HealthMonitor()
        for sp in rig.specs.values():
            self.monitor.register(sp.id, sp.kind, sp.zone, sp.zone, sp.pos)
        self.operator_s, self.crew_s = operator_s, crew_s
        if centre.timeline is None:
            centre.timeline = self.port
        sim.on_gas = lambda s, rid, leak, event: self._raise_gas(leak["source"], event, robot=rid)
        self.probes: dict[str, dict] = {}
        self.statuses: dict[str, str] = {}
        self.alarmed: set[str] = set()
        self.isolated: set[str] = set()
        self.crew: dict[str, float] = {}
        self.vetoed: set[str] = set()
        self.cands: dict[str, tuple[float, list]] = {}
        self.last: dict = {}
        self.snap: Optional[SensorSnapshot] = None

    def zone(self, pos) -> str:
        z = zone_of(pos)
        if z:
            return z
        if any(math.dist(pos, rt.dock) < 2.0 for rt in self.c.rt.values()):
            return "dock"
        return deck_name(float(pos[2]))

    def robots(self) -> list[Robot]:
        out = []
        for rid, rt in self.c.rt.items():
            job = self.c.jobs.get(rt.job) if rt.job else None
            p = tuple(round(float(v), 2) for v in rt.robot.position)
            out.append(Robot(rid, state=rt.state, caption=f"{job.kind} {job.target}" if job else rt.state.lower(),
                             pose=p, battery=round(float(rt.robot.battery), 3), task=rt.job, zone=self.zone(p),
                             sensors=sorted(rt.robot.capabilities), deck=deck_name(p[2])))
        return out

    def candidates(self, sid: str) -> list[dict]:
        s = self.rig.specs[sid]
        need = SENSOR_FOR[s.kind]
        out = []
        for rid, rt in self.c.rt.items():
            r = rt.robot
            p = self.c.power(r)
            route = self.c.planner.route(r, s.pos, STANDOFF_M)
            eta = None if route.blocked else route.time(r.speed_mps)
            c = {"robot": rid, "kind": KIND_NAME.get(r.kind, r.kind.value), "sensor": need,
                 "eta_s": None if eta is None else round(eta, 1), "battery": round(float(r.battery), 3),
                 "ok": False, "why": ""}
            if eta is not None:
                end = route.legs[-1].end if route.legs else tuple(r.position)
                wh = (route.energy_wh(p, r.speed_mps) + p.work_w * VERIFY_S / 3600.0
                      + self.c.to_dock_wh(r, end, rt.dock))
                c["need_wh"] = round(wh, 1)
                c["spare_wh"] = round((float(r.battery) - RESERVE) * p.capacity_wh, 1)
            job = self.c.jobs.get(rt.job) if rt.job else None
            if need not in r.capabilities:
                c["why"] = f"no {need}"
            elif r.kind.value not in s.reach:
                c["why"] = f"a {c['kind']} cannot reach {sid} ({s.where})"
            elif rt.state in NOT_FREE:
                c["why"] = NOT_FREE[rt.state]
            elif job is not None and (job.priority <= URGENT or job.kind in GAS_RESPONSE or job.kind == VERIFY
                                      or job.params.get("gas_response")
                                      or (rt.state == "WORKING" and job.level == Level.SUPV)):
                c["why"] = f"busy: {job.kind} {job.target}"
            elif eta is None:
                c["why"] = "unreachable"
            elif c["need_wh"] > c["spare_wh"]:
                c["why"] = f"battery: round trip {c['need_wh']:.0f} Wh > spare {c['spare_wh']:.0f} Wh"
            else:
                c["ok"] = True
            out.append(c)
        return sorted(out, key=lambda c: (not c["ok"], c["eta_s"] if c["eta_s"] is not None else math.inf))

    def snapshot(self) -> SensorSnapshot:
        gas, fixed = self.rig.read(self.c.t)
        live = {s.value for s in OPEN}
        jobs = [{k: j[k] for k in ("id", "kind", "target", "zone", "robot", "status", "priority", "level")}
                for j in self.port.jobs() if j["status"] in live]
        events = [{"t": e["t"], "event": e["event"], "robot": e["robot"], "job": e["job"]}
                  for e in self.c.timeline[-MAX_EVENTS * 3:] if e["event"] not in ("argon_call", "argon_control",
                                                                                   "segment_wait", "segment_clear")]
        snap = SensorSnapshot(t=float(self.c.t), robots=self.robots(), gas=gas, fixed=fixed, jobs=jobs,
                              events=events[-MAX_EVENTS:], sources=["fleet_coordinator", "rig_sensors"])
        self.monitor.update(snap)
        for h in snap.health:
            if h.status != "OK" and h.id in self.rig.specs:
                t0, cands = self.cands.get(h.id, (-1e9, None))
                if self.c.t - t0 >= CANDIDATE_TTL_S:
                    t0, cands = self.c.t, self.candidates(h.id)
                    self.cands[h.id] = (t0, cands)
                h.candidates = cands
        return snap

    def _transitions(self, snap: SensorSnapshot) -> None:
        for h in snap.health:
            old = self.statuses.get(h.id, "OK")
            if h.status != old:
                if h.status == "OK":
                    self.port.log("sensor_ok", None, None, sensor=h.id, was=old)
                else:
                    self.port.log("sensor_fault", None, None, sensor=h.id, kind=h.kind, status=h.status, zone=h.zone,
                                  hazard=h.hazard, evidence=h.evidence,
                                  best=next((c["robot"] for c in h.candidates if c["ok"]), None))
            self.statuses[h.id] = h.status

    def _raise_gas(self, fid: str, event: str, robot: Optional[str] = None, **detail) -> None:
        if fid in self.alarmed:
            return
        self.alarmed.add(fid)
        xyz = tuple(self.c.planner.rm.flanges[fid])
        for leak in self.c.world.state.get("gas_leaks", []):
            if leak.get("source") == fid:
                leak["reported"] = True
        for jid, st in list(self.c.status.items()):
            j = self.c.jobs[jid]
            if j.kind == "flange_leak_check" and j.target == fid and st in OPEN and not j.params.get("gas_response"):
                self.c.cancel(jid, by="coordinator", reason="superseded by the gas response")
        job = new_job(self.c, "flange_leak_check", fid, xyz, Level.AUTO, priority=risk_priority("gas"),
                      est_duration_s=20.0, params={"leak": self.c.world.state.get("flanges", {}).get(fid, "loose"),
                                                   "gas_response": True, "zone": f"flange:{fid}"})
        self.c.report(job, robot, event, flange=fid, position=[round(v, 3) for v in xyz], **detail)

    def _gas(self, snap: SensorSnapshot) -> None:
        flanges = self.rig.flanges
        for g in snap.gas:
            spec = self.rig.specs.get(g.id)
            if g.lel < GAS_ALARM_LEL or spec is None:
                continue
            fid = spec.flange or min(flanges, key=lambda k: float(np.linalg.norm(flanges[k] - np.asarray(spec.pos))))
            if not spec.flange and any(zone_of(flanges[x]) == g.zone for x in self.alarmed):
                continue
            self._raise_gas(fid, "gas_alarm", source=f"fixed gas head {g.id}", sensor=g.id, lel=round(g.lel, 1),
                            why=f"{g.id} reads {g.lel:.0f}% LEL >= {GAS_ALARM_LEL:.0f}%: gas until proven otherwise")
        for leak in self.c.world.state.get("gas_leaks", []):
            fid = leak.get("source")
            if leak.get("isolated") and fid not in self.isolated:
                self.isolated.add(fid)
                self.rig.isolate(fid, self.c.t)
                self.port.log("leak_isolated", None, None, flange=fid,
                              why=f"{fid} re-torqued: leak isolated, gas decaying")

    def _verify(self, snap: SensorSnapshot) -> None:
        values = {g.id: g.lel for g in snap.gas} | {x.id: x.value for x in snap.fixed}
        health = {h.id: h for h in snap.health}
        for jid, j in list(self.c.jobs.items()):
            if j.kind != VERIFY:
                continue
            st = self.c.status.get(jid)
            sid = j.target
            p = self.probes.setdefault(jid, {"robot": [], "head": [], "done": False})
            if p["done"]:
                continue
            if st in OPEN:
                self.monitor.attach(sid, jid)
                if sid in health:
                    health[sid].job = jid
            rid = self.c.assigned_to.get(jid)
            rt = self.c.rt.get(rid) if rid else None
            if rt is not None and rt.job == jid and rt.state == "WORKING":
                p["robot"].append(self.rig.probe(sid, self.c.t))
                p["rid"] = rid
                if values.get(sid) is not None:
                    p["head"].append(values[sid])
            if st == JobStatus.DONE:
                p["done"] = True
                self._verdict(jid, sid, p, p.get("rid"))
            elif st in (JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.HANDED_TO_HUMAN):
                p["done"] = True
                self.monitor.attach(sid, None)

    def _verdict(self, jid: str, sid: str, p: dict, rid: Optional[str]) -> None:
        s = self.rig.specs[sid]
        res = compare(s.kind, p["robot"], p["head"])
        self.port.log("verify_reading", jid, rid, sensor=sid, **res)
        if s.kind == "gas" and res["robot_mean"] is not None and res["robot_mean"] >= GAS_ALARM_LEL:
            self.port.log("gas_confirmed", jid, rid, sensor=sid, lel=res["robot_mean"],
                          why="robot sniffer confirms real gas at the head")
        if res["agree"]:
            self.monitor.clear(sid, TRUST_S)
            self.port.log("fault_cleared", jid, rid, sensor=sid, why=res["why"])
            return
        mid = self.port.queue(MAINTENANCE, sid, "SUPV", 2,
                              params={"needs": ["crew"], "evidence": res, "verified_by": rid},
                              by=self.port.agent, why=f"{sid}: {res['why']}; replace or recalibrate the head")
        self.monitor.confirm(sid, mid)
        self.port.log("maintenance_raised", mid, rid, sensor=sid, why=res["why"])

    def _people(self) -> None:
        if self.operator_s is not None:
            for jid in list(self.c.approvals):
                req = (self.c.requests.get(jid) or [{"t": self.c.t}])[0]["t"]
                if self.c.t - req >= self.operator_s:
                    self.c.approve(jid, by="operator")
        for jid, j in list(self.c.jobs.items()):
            st = self.c.status.get(jid)
            if j.kind != MAINTENANCE:
                continue
            if st == JobStatus.QUEUED:
                self.port.to_crew(jid, f"approved: crew replaces or recalibrates {j.target}")
                self.crew[jid] = self.c.t
            elif st == JobStatus.HANDED_TO_HUMAN and self.c.t - self.crew.setdefault(jid, self.c.t) >= self.crew_s:
                self.rig.repair(j.target)
                self.monitor.clear(j.target)
                self.port.close(jid, "crew", sensor=j.target)
                self.port.log("sensor_repaired", jid, None, sensor=j.target, by="crew")

    def step(self) -> dict:
        with self.centre.lock:
            return self._step()

    def _step(self) -> dict:
        self.sim.step()
        snap = self.snapshot()
        self._verify(snap)
        self._transitions(snap)
        self._gas(snap)
        self._people()
        ctl = self.centre.tick(snap, now=float(self.c.t))
        self.port.apply(ctl["actions"], {h.id: h for h in snap.health})
        for v in ctl["vetoed"]:
            if v["id"] not in self.vetoed:
                self.port.log("argon_vetoed", None, v["robot"] or None, kind=v["kind"], target=v["target"],
                              purpose=v["purpose"], why=v["why"])
        self.vetoed = {v["id"] for v in ctl["vetoed"]}
        ctl["health"] = health_view(snap)
        ctl["fleet"] = [{"id": r.id, "state": r.state, "zone": r.zone, "battery": r.battery, "task": r.task,
                         "sensors": r.sensors, "pose": list(r.pose)} for r in snap.robots]
        self.snap, self.last = snap, ctl
        return ctl

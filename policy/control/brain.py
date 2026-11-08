from __future__ import annotations

import math
from typing import Optional

import numpy as np

from policy.control.centre import ControlCentre, health_view
from policy.control.fleet_port import SENSOR_FOR, FleetPort
from policy.control.health import DRIFT_TOL, NOISE_FLOOR, NOISY_RATIO, STUCK_EPS, STUCK_RATIO, HealthMonitor
from policy.control.rigsensors import RigSensors, zone_of
from policy.control.safety import LEL_CLEAR
from policy.control.schema import VERIFY
from policy.control.snapshot import Robot, SensorSnapshot
from policy.fleet.battery import Battery
from policy.fleet.jobs import SETUP_S, SIMPLE
from policy.fleet.rigmap import DECKS

GAS_ALARM_LEL = 10.0
MAINTENANCE = "sensor_maintenance"
BUSY_KINDS = ("gas", VERIFY)
TRUST_S = 600.0
MAX_EVENTS = 12


def deck_name(z: float) -> str:
    d = min(DECKS, key=lambda k: abs(DECKS[k] - z))
    return d if abs(DECKS[d] - z) < 2.0 else "stairs"


def register(rm, rig: RigSensors) -> None:
    for s in rig.specs.values():
        if s.id not in rm.poi_node and abs(s.pos[2] - DECKS["top_deck"]) < 3.0:
            rm.add_poi(s.id, s.pos, "top_deck")


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
    def __init__(self, fleet, rig: RigSensors, centre: ControlCentre, port: Optional[FleetPort] = None,
                 monitor: Optional[HealthMonitor] = None, operator_s: Optional[float] = 40.0, crew_s: float = 90.0):
        self.f, self.rig, self.centre = fleet, rig, centre
        self.port = port or FleetPort(fleet, rig.specs)
        self.monitor = monitor or HealthMonitor()
        self.operator_s, self.crew_s = operator_s, crew_s
        if centre.timeline is None:
            centre.timeline = self.port
        self.probes: dict[str, dict] = {}
        self.statuses: dict[str, str] = {}
        self.alarmed: set[str] = set()
        self.gas_tasks: dict[str, str] = {}
        self.isolated: set[str] = set()
        self.last: dict = {}
        self.snap: Optional[SensorSnapshot] = None

    def zone(self, pos) -> str:
        z = zone_of(pos)
        if z:
            return z
        if np.linalg.norm(np.asarray(pos)[:2] - self.f.rm.pos[self.f.rm.dock][:2]) < 2.0:
            return "dock"
        return deck_name(float(pos[2]))

    def robots(self) -> list[Robot]:
        out = []
        for r in self.f.robots:
            js, _ = self.f.job_state(r)
            out.append(Robot(r.id, state=r.state if js is None else js, caption=_caption(r, js),
                             pose=(round(float(r.pos[0]), 2), round(float(r.pos[1]), 2), round(float(r.pos[2]), 2)),
                             battery=round(r.battery.soc, 3), task=r.task.id if r.task else None,
                             zone=self.zone(r.pos), sensors=sorted(r.sensors), deck=deck_name(float(r.pos[2]))))
        return out

    def candidates(self, sid: str, cache: dict) -> list[dict]:
        s = self.rig.specs[sid]
        node = self.f.rm.poi_node.get(sid)
        need_sensor = SENSOR_FOR[s.kind]
        if "dock" not in cache:
            cache["dock"] = self.f._dock_costs()
        dd, ds = cache["dock"]
        work = Battery.work_energy(SETUP_S[VERIFY] + SIMPLE[VERIFY][0])
        out = []
        for r in self.f.robots:
            if r.id not in cache:
                a, t0 = self.f._origin(r)
                dist, _, stair = self.f.rm.dijkstra(a, self.f._blocked - {a})
                cache[r.id] = (dist, stair, t0 - self.f.t)
            dist, stair, lag = cache[r.id]
            eta = lag + dist[node] if node is not None else math.inf
            c = {"robot": r.id, "sensor": need_sensor, "eta_s": round(float(eta), 1) if math.isfinite(eta) else None,
                 "battery": round(r.battery.soc, 3), "ok": False, "why": ""}
            if math.isfinite(eta):
                need = Battery.path_energy(eta, stair[node]) + work + Battery.path_energy(dd[node], ds[node])
                c["need_wh"] = round(need, 1)
                c["spare_wh"] = round(r.battery.wh - r.battery.reserve_wh, 1)
            if need_sensor not in r.sensors:
                c["why"] = f"no {need_sensor}"
            elif r.failed:
                c["why"] = "robot failed"
            elif r.held or r.paused:
                c["why"] = "held" if r.held else "paused for a person"
            elif r.task is not None and (r.task.kind in BUSY_KINDS):
                c["why"] = f"busy: {r.task.kind}"
            elif not math.isfinite(eta):
                c["why"] = "unreachable"
            elif not r.battery.can_afford(c["need_wh"]):
                c["why"] = f"battery: round trip {c['need_wh']:.0f} Wh > spare {c['spare_wh']:.0f} Wh"
            else:
                c["ok"] = True
            out.append(c)
        return sorted(out, key=lambda c: (not c["ok"], c["eta_s"] if c["eta_s"] is not None else math.inf))

    def snapshot(self) -> SensorSnapshot:
        gas, fixed = self.rig.read(self.f.t)
        snap = SensorSnapshot(t=float(self.f.t), robots=self.robots(), gas=gas, fixed=fixed,
                              jobs=[{k: j[k] for k in ("id", "kind", "target", "zone", "robot", "status", "priority",
                                                       "level")} for j in self.port.fleet_state()["jobs"]
                                    if j["status"] not in ("DONE", "CANCELLED")],
                              events=[{"t": e["t"], "event": e["event"], "robot": e["robot"], "job": e["job"]}
                                      for e in self.port.timeline[-MAX_EVENTS:] if e["event"] != "argon_call"],
                              sources=["fleet_sim", "rig_sensors"])
        self.monitor.update(snap)
        cache: dict = {}
        for h in snap.health:
            if h.status != "OK" and h.id in self.rig.specs:
                h.candidates = self.candidates(h.id, cache)
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

    def _gas_alarms(self, snap: SensorSnapshot) -> None:
        flanges = self.rig.flanges
        for g in snap.gas:
            if g.lel < GAS_ALARM_LEL:
                continue
            spec = self.rig.specs.get(g.id)
            if spec is None:
                continue
            fid = spec.flange or min(flanges, key=lambda k: float(np.linalg.norm(flanges[k] - np.asarray(spec.pos))))
            if fid in self.alarmed or (not spec.flange and any(zone_of(flanges[x]) == g.zone for x in self.alarmed)):
                continue
            self.alarmed.add(fid)
            self.f.schedule(self.f.t, "gas_alarm", flange=fid, scenario="loose")
            self.port.log("gas_alarm", None, None, sensor=g.id, flange=fid, lel=g.lel,
                          why=f"{g.id} reads {g.lel:.0f}% LEL >= {GAS_ALARM_LEL:.0f}%: gas until proven otherwise")
        for t in self.f.tasks.values():
            if t.kind == "gas" and t.status == "done" and t.id not in self.gas_tasks:
                self.gas_tasks[t.id] = t.poi
                self.isolated.add(t.poi)
                self.rig.isolate(t.poi, self.f.t)
                self.port.log("leak_isolated", t.id, None, flange=t.poi,
                              why=f"gas response at {t.poi} done (re-torque): leak isolated")
        for fid in list(self.alarmed):
            if fid in self.isolated and all(g.lel < LEL_CLEAR for g in snap.gas if g.zone == zone_of(flanges[fid])):
                self.alarmed.discard(fid)
                self.isolated.discard(fid)

    def _verify(self, snap: SensorSnapshot) -> None:
        values = {g.id: g.lel for g in snap.gas} | {x.id: x.value for x in snap.fixed}
        health = {h.id: h for h in snap.health}
        for j in list(self.port.jobs.values()):
            if j["kind"] != VERIFY:
                continue
            sid = j["target"]
            if j["status"] in ("QUEUED", "ASSIGNED", "RUNNING"):
                self.monitor.attach(sid, j["id"])
                if sid in health:
                    health[sid].job = j["id"]
            t = self.f.tasks.get(j.get("task") or "")
            if t is None:
                continue
            p = self.probes.setdefault(j["id"], {"robot": [], "head": [], "done": False})
            r = next((x for x in self.f.robots if x.task is t), None)
            if t.status == "active" and r is not None and r.state == "WORK":
                p["robot"].append(self.rig.probe(sid, self.f.t))
                if values.get(sid) is not None:
                    p["head"].append(values[sid])
            if t.status == "done" and not p["done"]:
                p["done"] = True
                self._verdict(j, p, t.robot or j["params"].get("robot"))

    def _verdict(self, j: dict, p: dict, rid: Optional[str]) -> None:
        sid = j["target"]
        s = self.rig.specs[sid]
        res = compare(s.kind, p["robot"], p["head"])
        self.port.log("verify_reading", j["id"], rid, sensor=sid, **res)
        if s.kind == "gas" and res["robot_mean"] is not None and res["robot_mean"] >= GAS_ALARM_LEL:
            self.port.log("gas_confirmed", j["id"], rid, sensor=sid, lel=res["robot_mean"],
                          why="robot sniffer confirms real gas at the head")
        if res["agree"]:
            self.monitor.clear(sid, TRUST_S)
            self.port.finish(j["id"], "DONE", outcome="agree")
            self.port.log("fault_cleared", j["id"], rid, sensor=sid, why=res["why"])
            return
        mid = self.port.queue(MAINTENANCE, sid, "SUPV", 2, params={"evidence": res, "verified_by": rid},
                              by=self.port.agent, why=f"{sid}: {res['why']}; replace or recalibrate the head")
        self.monitor.confirm(sid, mid)
        self.port.finish(j["id"], "DONE", outcome="disagree", maintenance=mid)
        self.port.log("maintenance_raised", mid, rid, sensor=sid, why=res["why"])

    def _people(self) -> None:
        for j in list(self.port.jobs.values()):
            if j["status"] == "WAITING_APPROVAL" and self.operator_s is not None:
                req = next((e["t"] for e in reversed(self.port.timeline) if e["event"] == "approval_requested"
                            and e["job"] == j["id"]), self.f.t)
                if self.f.t - req >= self.operator_s:
                    self.port.approve_request(j["id"], "operator", f"operator approved {j['kind']} of {j['target']}")
                    j["handed_t"] = self.f.t
            elif j["kind"] == MAINTENANCE and j["status"] == "HANDED_TO_HUMAN":
                j.setdefault("handed_t", self.f.t)
                if self.f.t - j["handed_t"] >= self.crew_s:
                    self.rig.repair(j["target"])
                    self.monitor.clear(j["target"])
                    self.port.finish(j["id"], "DONE", by="crew")
                    self.port.log("sensor_repaired", j["id"], None, sensor=j["target"], by="crew")

    def step(self) -> dict:
        self.f.step()
        snap = self.snapshot()
        self._verify(snap)
        self._transitions(snap)
        self._gas_alarms(snap)
        self._people()
        ctl = self.centre.tick(snap, now=float(self.f.t))
        self.port.apply(ctl["actions"], {h.id: h for h in snap.health})
        ctl["health"] = health_view(snap)
        ctl["fleet"] = [{"id": r.id, "state": r.state, "zone": r.zone, "battery": r.battery, "task": r.task,
                         "sensors": r.sensors, "pose": list(r.pose)} for r in snap.robots]
        self.snap, self.last = snap, ctl
        return ctl


def _caption(r, js) -> str:
    if r.task is None:
        return r.state.lower()
    return f"{r.task.kind} {r.task.poi}" + (f" ({js.lower()})" if js else "")

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from policy.fleet.assign import INF, assign, pick_responder
from policy.fleet.battery import Battery
from policy.fleet.jobs import JobLibrary
from policy.fleet.reservations import Leg, Reservations, plan_legs
from policy.fleet.rigmap import DOCK_PADS, RigMap
from policy.fleet.tasks import OPEN, Task, priority

GAS_ZONE_M = 8.0
PERSON_RADIUS_M = 3.0
CREW_HOLD_S = 300.0
W_PRIO = 30.0
W_SOC = 60.0
DOCK_STATES = {"CHARGING", "DOCKED", "DOCK_QUEUE"}


@dataclass
class Robot:
    id: str
    node: int
    battery: Battery
    pos: np.ndarray
    state: str = "IDLE"
    task: Optional[Task] = None
    legs: list[Leg] = field(default_factory=list)
    goal: Optional[int] = None
    to_dock: bool = False
    low: bool = False
    paused: bool = False
    failed: bool = False
    stuck: bool = False
    yaw: float = 0.0
    dist_m: float = 0.0
    arrived_t: float = 0.0
    time_in: Counter = field(default_factory=Counter)
    sensors: frozenset = field(default_factory=frozenset)
    held: bool = False


@dataclass
class Event:
    t: float
    kind: str
    data: dict


def walk_along(path: list[np.ndarray], s: float) -> np.ndarray:
    for a, b in zip(path, path[1:]):
        L = float(np.linalg.norm(b - a))
        if s <= L:
            return a + (b - a) * (s / max(L, 1e-9))
        s -= L
    return path[-1].copy()


class Fleet:
    def __init__(self, rm: RigMap, jobs: Optional[JobLibrary] = None, dt: float = 1.0,
                 gas_zone_m: float = GAS_ZONE_M, person_radius_m: float = PERSON_RADIUS_M):
        self.rm, self.jobs, self.dt = rm, jobs or JobLibrary(), dt
        self.gas_zone_m, self.person_radius_m = gas_zone_m, person_radius_m
        self.t = 0.0
        self.robots: list[Robot] = []
        self.tasks: dict[str, Task] = {}
        self.pending_events: list[Event] = []
        self.event_log: list[dict] = []
        self.tick_events: list[dict] = []
        self.tick_assign: list[dict] = []
        self.res = Reservations()
        self.persons: list[dict] = []
        self.alarms: dict[str, dict] = {}
        self.conflicts = 0
        self.preemptions = 0
        self.replans = 0
        self.failures = 0
        self.wait_s = 0.0
        self._event_tick = True
        self._blocked: set[int] = set()

    def add_robot(self, rid: str, node: int, soc: float = 1.0, sensors=(), **battery) -> Robot:
        b = Battery(**battery)
        b.wh = soc * b.capacity_wh
        r = Robot(rid, node, b, self.rm.pos[node].copy(), sensors=frozenset(sensors))
        self.robots.append(r)
        return r

    def add_task(self, kind: str, poi: str, xyz=None, scenario: str = "clean", **kw) -> Task:
        node = self.rm.poi_node[poi] if poi in self.rm.poi_node else self.rm.add_poi(poi, xyz)
        tid, k = f"{kind}:{poi}", 2
        while tid in self.tasks:
            tid, k = f"{kind}:{poi}#{k}", k + 1
        tr = self.jobs.get(kind, scenario)
        task = Task(tid, kind, poi, node, self.t, scenario=scenario, duration=tr.total_s, **kw)
        self.tasks[tid] = task
        self._event_tick = True
        return task

    def schedule(self, t: float, kind: str, **data) -> None:
        self.pending_events.append(Event(float(t), kind, data))
        self.pending_events.sort(key=lambda e: e.t)

    def _log(self, kind: str, message: str, **extra) -> None:
        e = {"t": round(self.t, 2), "kind": kind, "message": message, **extra}
        self.event_log.append(e)
        self.tick_events.append(e)

    def _apply(self, ev: Event) -> None:
        d = ev.data
        self._event_tick = True
        if ev.kind == "new_rust":
            t = self.add_task("rust", d["poi"], d["xyz"], severity=d["severity"], exposure=d["exposure"])
            self._log("new_rust", f"new rust reported at {d['poi']} (severity {d['severity']:.1f}, "
                      f"exposure {d['exposure']:.1f}) -> queued", task=t.id)
        elif ev.kind == "gas_alarm":
            fid = d["flange"]
            self._supersede(fid)
            t = self.add_task("gas", fid, scenario=d.get("scenario", "loose"))
            self.alarms[t.id] = {"flange": fid, "t": self.t, "responder": None, "response_s": None,
                                 "resolved_s": None, "outcome": None}
            self._log("gas_alarm", f"GAS ALARM at flange {fid} -> pre-empt nearest robot, stop hot work within "
                      f"{self.gas_zone_m:.0f} m", task=t.id)
        elif ev.kind == "crack":
            self._supersede(d["flange"])
            t = self.add_task("crack", d["flange"], scenario="crack")
            self._log("crack", f"crack suspected at flange {d['flange']} -> queued above rust", task=t.id)
        elif ev.kind == "robot_failure":
            r = next(x for x in self.robots if x.id == d["robot"])
            self._fail(r)
        elif ev.kind == "person":
            way = [np.asarray(w, float) for w in (d["path"] if "path" in d else [d["xyz"]])]
            p = {"xyz": way[0].copy(), "path": way, "speed": d.get("speed", 0.8), "t0": self.t,
                 "radius": d.get("radius", self.person_radius_m), "until": self.t + d.get("duration", 60.0)}
            self.persons.append(p)
            self._log("person", f"person on deck at {np.round(p['xyz'], 1).tolist()} for "
                      f"{d.get('duration', 60.0):.0f} s -> keep {p['radius']:.0f} m clear")

    def _supersede(self, fid: str) -> None:
        for t in self.tasks.values():
            if t.kind == "leak_check" and t.poi == fid and t.status in OPEN:
                self._drop(t, "superseded")

    def _drop(self, task: Task, status: str) -> None:
        r = next((x for x in self.robots if x.task is task), None)
        task.status, task.robot = status, None
        if r is not None:
            r.task = None
            r.goal = None

    def _fail(self, r: Robot) -> None:
        self.failures += 1
        lg = self._mid(r)
        if lg is not None:
            r.node = min((lg.u, lg.v), key=lambda n: float(np.linalg.norm(r.pos - self.rm.pos[n])))
        r.pos = self.rm.pos[r.node].copy()
        r.legs, r.failed, r.state, r.goal = [], True, "FAILED", None
        self.res.release(r.id)
        name = r.task.id if r.task else "no task"
        if r.task is not None:
            self._release(r)
        self._log("robot_failure", f"{r.id} failed ({name}) -> task re-queued, robot marked as obstacle", robot=r.id)

    def _release(self, r: Robot, status: str = "pending") -> None:
        if r.task is not None:
            r.task.status, r.task.robot = status, None
            r.task = None
        r.goal = None

    def _blocked_nodes(self) -> set[int]:
        out = {r.node for r in self.robots if r.failed}
        for p in self.persons:
            for i, q in enumerate(self.rm.pos):
                if abs(q[2] - p["xyz"][2]) < 3.0 and np.linalg.norm(q[:2] - p["xyz"][:2]) < p["radius"]:
                    out.add(i)
        return out

    def _gas_zones(self) -> list[np.ndarray]:
        return [np.asarray(self.rm.flanges[t.poi]) for t in self.tasks.values() if t.kind == "gas"
                and (t.status in OPEN or (t.status == "escalated" and self.t < t.finished + CREW_HOLD_S))]

    def _in_gas(self, node: int) -> bool:
        p = self.rm.pos[node]
        return any(np.linalg.norm(p[:2] - z[:2]) < self.gas_zone_m and abs(p[2] - z[2]) < 5.0
                   for z in self._gas_zones())

    def _assignable(self, t: Task) -> bool:
        return (t.status == "pending" and t.node not in self._blocked
                and (t.kind == "gas" or not self._in_gas(t.node)))

    def _free(self, r: Robot) -> bool:
        return not r.failed and not r.paused and not r.held

    def hold(self, rid: str, why: str = "") -> bool:
        r = next((x for x in self.robots if x.id == rid), None)
        if r is None or r.failed or r.held:
            return False
        r.held = r.paused = True
        self._truncate(r)
        self._event_tick = True
        self._log("hold", f"{r.id} held: {why}" if why else f"{r.id} held", robot=r.id)
        return True

    def resume(self, rid: str, why: str = "") -> bool:
        r = next((x for x in self.robots if x.id == rid), None)
        if r is None or not r.held:
            return False
        r.held = False
        self._event_tick = True
        if not self._person_near(r):
            r.paused = False
            if r.goal is not None:
                self._go(r, r.goal)
        self._log("resume", f"{r.id} resumes: {why}" if why else f"{r.id} resumes", robot=r.id)
        return True

    def _assign_pinned(self) -> None:
        for t in [t for t in self.tasks.values() if t.pin and self._assignable(t)]:
            r = next((x for x in self.robots if x.id == t.pin), None)
            if r is None or r.failed:
                self._log("unpin", f"{t.id}: {t.pin} unavailable -> any robot with {sorted(t.needs)}", task=t.id)
                t.pin = None
                continue
            if not self._free(r) or (r.task is not None and (r.task.kind == "gas" or r.task.pin)):
                continue
            if r.task is not None:
                old = r.task
                old.preemptions += 1
                self.preemptions += 1
                self._release(r)
                self._log("preempt", f"{r.id} pre-empted from {old.id} -> {t.id}", robot=r.id, task=old.id)
            r.low = False
            self._give(r, t, f"pinned by Argon: {t.kind}")

    def _mid(self, r: Robot) -> Optional[Leg]:
        return next((lg for lg in r.legs if lg.t0 <= self.t < lg.t1), None)

    def _kept(self, r: Robot) -> list[Leg]:
        cur = self._mid(r)
        if cur is None:
            return []
        i = r.legs.index(cur)
        keep = [cur]
        while cur.corridor and i + 1 < len(r.legs) and r.legs[i + 1].corridor == cur.corridor:
            i += 1
            keep.append(r.legs[i])
        return keep

    def _origin(self, r: Robot) -> tuple[int, float]:
        keep = self._kept(r)
        return (keep[-1].v, keep[-1].t1) if keep else (r.node, self.t)

    def _truncate(self, r: Robot) -> tuple[int, float]:
        keep = self._kept(r)
        r.legs = keep
        self.res.release(r.id, after=self.t + 1e-6)
        return (keep[-1].v, keep[-1].t1) if keep else (r.node, self.t)

    def _go(self, r: Robot, goal: int) -> bool:
        a, t0 = self._truncate(r)
        r.goal = goal
        self.replans += 1
        if a == goal:
            r.stuck = False
            r.arrived_t = t0
            return True
        _, path = self.rm.route(a, goal, self._blocked)
        if not path:
            r.stuck = True
            return False
        r.stuck = False
        r.legs += plan_legs(self.rm, self.res, r.id, path, t0)
        return True

    def _give(self, r: Robot, task: Task, why: str) -> None:
        task.status, task.robot = "assigned", r.id
        r.task, r.to_dock = task, False
        self._go(r, task.node)
        self.tick_assign.append({"robot": r.id, "task": task.id, "why": why})

    def _send_dock(self, r: Robot, low: bool) -> None:
        self._release(r)
        r.to_dock, r.low = True, low or r.low
        if r.node != self.rm.dock or r.legs:
            self._go(r, self.rm.dock)
        else:
            r.goal = self.rm.dock

    def _perm_blocked(self) -> set[int]:
        return {r.node for r in self.robots if r.failed}

    def _dock_costs(self):
        d, _, s = self.rm.dijkstra(self.rm.dock, self._perm_blocked())
        return d, s

    def _need_wh(self, r: Robot, task: Task, dist, stair, dd, ds, lag: float) -> float:
        n = task.node
        travel = Battery.path_energy(dist[n] + lag, stair[n]) if math.isfinite(dist[n]) else INF
        back = Battery.path_energy(dd[n], ds[n]) if math.isfinite(dd[n]) else INF
        return travel + Battery.work_energy(max(task.duration - task.progress, 0.0)) + back

    def _plan_matrix(self, robots: list[Robot], tasks: list[Task]):
        dd, ds = self._dock_costs()
        eta = np.full((len(robots), len(tasks)), np.inf)
        need = np.full_like(eta, INF)
        for i, r in enumerate(robots):
            a, t0 = self._origin(r)
            dist, _, stair = self.rm.dijkstra(a, self._blocked - {a})
            lag = t0 - self.t
            for j, task in enumerate(tasks):
                eta[i, j] = lag + dist[task.node]
                need[i, j] = self._need_wh(r, task, dist, stair, dd, ds, lag)
        return eta, need

    def _escalate_cut_off(self) -> None:
        perm = self._perm_blocked()
        if not perm:
            return
        d, _, _ = self.rm.dijkstra(self.rm.dock, perm)
        for t in self.tasks.values():
            if t.status in OPEN and not math.isfinite(d[t.node]):
                r = next((x for x in self.robots if x.task is t), None)
                if r is not None:
                    self._release(r)
                t.status, t.outcome, t.finished = "escalated", "unreachable", self.t
                if t.id in self.alarms:
                    self.alarms[t.id]["outcome"] = "escalated"
                self._log("needs_human", f"{t.id} cut off by a failed robot -> escalate to human crew", task=t.id)

    def _preempt_gas(self) -> None:
        for gas in [t for t in self.tasks.values() if t.kind == "gas" and self._assignable(t)]:
            cands = [r for r in self.robots if self._free(r) and gas.needs <= r.sensors
                     and not (r.task is not None and r.task.kind == "gas")]
            if not cands:
                continue
            eta, need = self._plan_matrix(cands, [gas])
            ok = np.array([r.battery.can_afford(need[i, 0]) and math.isfinite(eta[i, 0]) for i, r in enumerate(cands)])
            i = pick_responder(eta[:, 0], ok)
            if i is None:
                continue
            r = cands[i]
            if r.task is not None:
                old = r.task
                old.preemptions += 1
                self.preemptions += 1
                self._release(r)
                self._log("preempt", f"{r.id} pre-empted from {old.id} -> gas alarm {gas.poi}", robot=r.id,
                          task=old.id)
            r.low = False
            self._give(r, gas, f"gas pre-emption, ETA {eta[i, 0]:.0f} s")
            self.alarms[gas.id]["responder"] = r.id

    def _suspend_gas_zone(self) -> None:
        for r in self.robots:
            if r.task is not None and r.task.kind != "gas" and self._in_gas(r.task.node):
                tid = r.task.id
                self._release(r)
                self._log("suspend", f"{r.id} stopped {tid}: inside gas zone -> re-queued until gas clears",
                          robot=r.id, task=tid)

    def _battery_checks(self, dd, ds) -> None:
        for r in self.robots:
            if r.failed or r.to_dock or r.task is None:
                continue
            a, _ = self._origin(r)
            if r.battery.must_return(Battery.path_energy(dd[a], ds[a]) + 1.0):
                tid = r.task.id
                self._send_dock(r, True)
                self._log("return_to_charge", f"{r.id} battery {r.battery.soc:.0%} -> abort {tid}, return to dock",
                          robot=r.id, task=tid)

    def _assign(self) -> None:
        if self._event_tick:
            for r in self.robots:
                if r.task is not None and r.task.kind != "gas" and not r.task.pin and r.task.status == "assigned":
                    self._release(r)
        robots = [r for r in self.robots if self._free(r) and r.task is None
                  and not (r.low and r.battery.soc < r.battery.resume_frac)]
        tasks = [t for t in self.tasks.values() if self._assignable(t) and not t.pin]
        if not robots or not tasks:
            return
        eta, need = self._plan_matrix(robots, tasks)
        prio = np.array([priority(t, self.t) for t in tasks])
        cost = np.full(eta.shape, INF)
        for i, r in enumerate(robots):
            for j in range(len(tasks)):
                if math.isfinite(eta[i, j]) and r.battery.can_afford(need[i, j]) and tasks[j].needs <= r.sensors:
                    cost[i, j] = eta[i, j] + W_SOC * (1.0 - r.battery.soc) - W_PRIO * prio[j]
        for i, j in assign(cost):
            r = robots[i]
            r.low = False
            self._give(r, tasks[j], f"hungarian: ETA {eta[i, j]:.0f} s, prio {prio[j]:.1f}, "
                       f"need {need[i, j]:.1f} Wh")

    def _advance(self, r: Robot) -> None:
        while r.legs and r.legs[0].t1 <= self.t + 1e-9:
            lg = r.legs.pop(0)
            r.dist_m += self.rm.adj[lg.u][lg.v].length
            r.node = lg.v
            r.arrived_t = lg.t1
        lg = r.legs[0] if r.legs else None
        if lg is not None and lg.t0 <= self.t:
            f = (self.t - lg.t0) / max(lg.t1 - lg.t0, 1e-9)
            a, b = self.rm.pos[lg.u], self.rm.pos[lg.v]
            r.pos = a + (b - a) * f
            if np.linalg.norm((b - a)[:2]) > 1e-6:
                r.yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        else:
            r.pos = self.rm.pos[r.node].copy()

    def _person_near(self, r: Robot) -> bool:
        return any(abs(r.pos[2] - p["xyz"][2]) < 3.0 and np.linalg.norm(r.pos[:2] - p["xyz"][:2]) < p["radius"]
                   for p in self.persons)

    def _safety(self, r: Robot) -> None:
        near = self._person_near(r)
        if near and not r.paused:
            if self._mid(r) is None:
                r.paused = True
                self._truncate(r)
                self._log("pause_person", f"{r.id} paused: person within {self.person_radius_m:.0f} m", robot=r.id)
            elif len(r.legs) > len(self._kept(r)):
                self._truncate(r)
        elif not near and r.paused and not r.held:
            r.paused = False
            self._log("resume", f"{r.id} resumes: person clear", robot=r.id)
            if r.goal is not None:
                self._go(r, r.goal)

    def _repair_paths(self, r: Robot) -> None:
        if r.paused or r.goal is None:
            return
        future = r.legs[len(self._kept(r)):]
        bad = any(lg.v in self._blocked and lg.v != r.goal for lg in future)
        if r.stuck or bad or (not r.legs and r.node != r.goal):
            self._go(r, r.goal)

    def _arrive_and_work(self, r: Robot) -> str:
        if r.failed:
            return "FAILED"
        if r.paused:
            return "PAUSED"
        if self._mid(r) is not None:
            return "TO_DOCK" if r.to_dock else "TRANSIT"
        if r.legs:
            self.wait_s += self.dt
            return "WAIT"
        if r.stuck:
            return "BLOCKED"
        if r.task is not None and r.node == r.task.node:
            task = r.task
            if task.arrived is None:
                task.arrived = r.arrived_t
                if task.id in self.alarms:
                    self.alarms[task.id]["response_s"] = round(task.arrived - self.alarms[task.id]["t"], 2)
            if task.status != "active":
                task.status = "active"
                self._log("job_start", f"{r.id} starts {task.id} ({task.kind}) -> orchestrator", robot=r.id,
                          task=task.id)
            return "WORK"
        if r.to_dock and r.node == self.rm.dock:
            pads = sum(1 for x in self.robots if x is not r and x.state in ("CHARGING", "DOCKED"))
            if r.state in ("CHARGING", "DOCKED") or pads < DOCK_PADS:
                return "DOCKED" if r.battery.soc >= 0.999 else "CHARGING"
            return "DOCK_QUEUE"
        return "IDLE"

    def _work(self, r: Robot) -> None:
        task = r.task
        task.progress += self.dt
        if task.progress >= task.duration:
            tr = self.jobs.get(task.kind, task.scenario)
            task.outcome = tr.outcome
            task.status = "escalated" if tr.outcome == "ESCALATE_TO_HUMAN" else "done"
            task.finished = self.t + self.dt
            if task.id in self.alarms:
                al = self.alarms[task.id]
                al["resolved_s"] = round(task.finished - al["t"], 2)
                al["outcome"] = task.status
            msg = (tr.alerts[-1]["message"] if tr.alerts else tr.segments[-1][2]).replace("F1", task.poi)
            self._log("job_done", f"{r.id} {task.status} {task.id}: {msg}", robot=r.id, task=task.id)
            self._release(r, task.status)

    def _energy(self, r: Robot) -> None:
        if r.failed:
            return
        if r.state in ("CHARGING", "DOCKED"):
            r.battery.charge(self.dt)
            return
        lg = self._mid(r)
        if lg is not None:
            r.battery.drain("stairs" if lg.kind == "stairs" else "walk", self.dt)
        else:
            r.battery.drain("work" if r.state == "WORK" else "idle", self.dt)

    def _corridor_occupancy(self) -> Counter:
        c = Counter()
        for r in self.robots:
            lg = self._mid(r)
            if lg is not None and lg.corridor:
                c[lg.corridor] += 1
        return c

    def check_invariants(self) -> None:
        assert all(v <= 1 for v in self._corridor_occupancy().values()), self._corridor_occupancy()
        assert all(r.battery.wh >= 0 for r in self.robots)
        for t in self.tasks.values():
            if t.status in ("assigned", "active"):
                assert sum(1 for r in self.robots if r.task is t) == 1, t.id

    def step(self) -> None:
        self.tick_events, self.tick_assign = [], []
        while self.pending_events and self.pending_events[0].t <= self.t + 1e-9:
            self._apply(self.pending_events.pop(0))
        gone = [p for p in self.persons if p["until"] <= self.t]
        for p in gone:
            self.persons.remove(p)
            self._event_tick = True
            self._log("person_left", "person left the deck -> clear zone")
        for p in self.persons:
            p["xyz"] = walk_along(p["path"], p["speed"] * (self.t - p["t0"]))
        self._blocked = self._blocked_nodes()
        self.res.prune(self.t - 60.0)
        for r in self.robots:
            if not r.failed:
                self._advance(r)
                self._safety(r)
        self._escalate_cut_off()
        self._preempt_gas()
        self._suspend_gas_zone()
        dd, ds = self._dock_costs()
        self._battery_checks(dd, ds)
        self._assign_pinned()
        self._assign()
        self._event_tick = False
        for r in self.robots:
            if not self._free(r) or r.task is not None:
                continue
            if not r.to_dock:
                low = r.battery.soc < r.battery.return_frac
                self._send_dock(r, low)
                if low:
                    self._log("return_to_charge", f"{r.id} battery {r.battery.soc:.0%} -> return to dock",
                              robot=r.id)
        for r in self.robots:
            if not r.failed:
                self._repair_paths(r)
        for r in self.robots:
            r.state = self._arrive_and_work(r)
            if r.state == "WORK":
                self._work(r)
                if r.task is None:
                    r.state = "IDLE"
            self._energy(r)
            r.time_in[r.state] += self.dt
        self.conflicts += sum(v - 1 for v in self._corridor_occupancy().values() if v > 1)
        self.t = round(self.t + self.dt, 6)

    def done(self) -> bool:
        return (not self.pending_events and not any(t.status in OPEN for t in self.tasks.values())
                and all(r.failed or r.state in DOCK_STATES for r in self.robots))

    def run(self, t_max: float = 3600.0, on_tick=None) -> None:
        while self.t < t_max:
            self.step()
            if on_tick is not None:
                on_tick(self)
            if self.done():
                break

    def job_state(self, r: Robot) -> tuple[Optional[str], Optional[str]]:
        if r.task is None or r.state != "WORK":
            return None, None
        return self.jobs.get(r.task.kind, r.task.scenario).state_at(r.task.progress)

    def metrics(self) -> dict:
        from policy.fleet.api import metrics
        return metrics(self)

    def snapshot(self) -> dict:
        from policy.fleet.api import snapshot
        return snapshot(self)

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from policy.fleet.api import corners
from policy.fleet.assign import INF, assign, pick_responder
from policy.fleet.battery import Battery
from policy.fleet.hazards import OBSTACLES, Hazards
from policy.fleet.jobs import JobLibrary
from policy.fleet.reservations import Leg, Reservations, plan_legs
from policy.fleet.rigmap import DECKS, DOCK_PADS, RigMap
from policy.fleet.tasks import OPEN, Task, priority

GAS_ZONE_M = 8.0
PERSON_RADIUS_M = 3.0
CREW_HOLD_S = 300.0
W_PRIO = 30.0
W_SOC = 60.0
DOCK_STATES = {"CHARGING", "DOCKED", "DOCK_QUEUE"}
HAZARD_EVENTS = ("drop_obstacle", "close_walkway", "close_stairs", "clear", "set_weather", "person_at")
BLOCKED = "blocked_by_hazard"
AHEAD_M = 7.0
PERSON_S = 120.0


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
    trail: list[Leg] = field(default_factory=list)
    prev_path: Optional[list] = None
    reroute_t: Optional[float] = None
    planned_t: float = -1.0
    popped: int = 0


@dataclass
class Event:
    t: float
    kind: str
    data: dict


def plen(q) -> float:
    return sum(float(np.linalg.norm(b - a)) for a, b in zip(q, q[1:]))


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
        self.hz = Hazards()
        self._hz_tick = False
        self._marks: dict[str, dict] = {}
        self.reroutes: list[dict] = []

    def add_robot(self, rid: str, node: int, soc: float = 1.0, **battery) -> Robot:
        b = Battery(**battery)
        b.wh = soc * b.capacity_wh
        r = Robot(rid, node, b, self.rm.pos[node].copy())
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

    def schedule(self, t: float, kind: str, /, **data) -> None:
        self.pending_events.append(Event(float(t), kind, data))
        self.pending_events.sort(key=lambda e: e.t)

    def _log(self, kind: str, message: str, **extra) -> None:
        e = {"t": round(self.t, 2), "kind": kind, "message": message, **extra}
        self.event_log.append(e)
        self.tick_events.append(e)

    def _apply(self, ev: Event) -> None:
        d = ev.data
        if ev.kind not in HAZARD_EVENTS:
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
        elif ev.kind in HAZARD_EVENTS:
            self._hazard(ev.kind, d)

    def event(self, name: str, /, **args) -> None:
        if name not in HAZARD_EVENTS:
            raise ValueError(f"unknown fleet event {name!r}; expected one of {HAZARD_EVENTS}")
        self._apply(Event(self.t, name, args))

    def _hazard(self, kind: str, d: dict) -> None:
        if kind == "person_at":
            if not any(k in d for k in ("xy", "xyz", "at")):
                return
            xyz = self._resolve_at(d.get("at", d.get("xyz", d.get("xy"))), d.get("deck"))
            self._mark("person")
            self._apply(Event(self.t, "person", {"xyz": xyz, "radius": d.get("radius", self.person_radius_m),
                                                 "duration": d.get("duration", PERSON_S)}))
            return
        if kind == "drop_obstacle":
            ob_kind = d.get("kind", "crate")
            if ob_kind not in OBSTACLES:
                raise ValueError(f"unknown obstacle {ob_kind!r}; expected one of {list(OBSTACLES)}")
            xyz = self._resolve_at(d.get("at"), d.get("deck"))
            h = self.hz.drop(ob_kind, xyz, d.get("at") if isinstance(d.get("at"), str) else None)
            what = {"block": "walkway blocked", "slow": "trip hazard, slow", "slip": "slippery, avoid stairs"}
            msg, cause = (f"{ob_kind} {h['id']} dropped at {np.round(xyz, 1).tolist()} -> {what[h['effect']]}",
                          f"{ob_kind} {h['id']}")
        elif kind in ("close_walkway", "close_stairs"):
            target = d.get("edge", d.get("id"))
            h = self.hz.close(self.rm, target, "stairs" if kind == "close_stairs" else None)
            msg, cause = f"{target} closed ({h['id']}) -> route around", f"{target} closed"
        elif kind == "clear":
            gone = self.hz.clear(d.get("id", "all"))
            h = {"id": ",".join(gone) or "none"}
            msg, cause = f"cleared {h['id']} -> routes restored", f"cleared {h['id']}"
        else:
            w = self.hz.set_weather(d.get("level", "calm"), d.get("wind_dir"), d.get("gust"))
            h = {"id": w["level"]}
            msg = "weather calm -> normal speed and sight" if w["level"] == "calm" else (f"weather {w['level']}: wind {w['wind']:.1f} toward {w['wind_dir']:.0f} deg, wet {w['wet']:.1f}, "
                   f"fog {w['fog']:.1f} -> speed x{w['speed']['walk']:.2f}, stairs x{w['speed']['stairs']:.2f}, "
                   f"sight {w['perception_range_m']:.0f} m" + (", windward walkways closed" if w["closes_outer"] else ""))
            cause = f"weather {w['level']}"
        self._mark(cause)
        self.hz.apply(self.rm)
        self._hz_tick = True
        self._log(kind, msg, hazard=h["id"])

    def _snap(self, xyz) -> list[float]:
        p = np.asarray(xyz, float)
        best = None
        for a, b, e in self.rm.edges():
            if e.kind == "stairs" or abs(self.rm.pos[a][2] - p[2]) > 1.5:
                continue
            pa, pb = self.rm.pos[a], self.rm.pos[b]
            dv = pb - pa
            s = float(np.clip(np.dot(p - pa, dv) / max(float(np.dot(dv, dv)), 1e-12), 0.0, 1.0))
            q = pa + s * dv
            dist = float(np.linalg.norm(q[:2] - p[:2]))
            if best is None or dist < best[0]:
                best = (dist, q)
        if best is None:
            raise ValueError(f"no walkway near {np.round(p, 1).tolist()}")
        return [round(float(v), 3) for v in best[1]]

    def _ahead(self, r: Robot, m: float) -> list[float]:
        for x in [r] + [o for o in self.robots if o is not r and not o.failed]:
            pts = [x.pos] + [self.rm.pos[lg.v] for lg in x.legs]
            L = plen(pts)
            if L >= 3.0:
                return [round(float(v), 3) for v in walk_along(pts, min(m, L - 2.0))]
        raise ValueError(f"no robot is moving right now; {r.id} has no route ahead to drop onto")

    def _resolve_at(self, at, deck=None) -> list[float]:
        if isinstance(at, dict):
            if "robot" in at:
                return self._ahead(self._robot(at["robot"]), float(at.get("ahead_m", AHEAD_M)))
            deck = at.get("deck", deck)
            at = at.get("edge", at.get("xyz", at.get("xy")))
        if isinstance(at, str):
            if any(r.id == at for r in self.robots):
                return self._ahead(self._robot(at), AHEAD_M)
            c = self.rm.coarse[self.hz.targets(self.rm, at)[0]]
            return [round(float(v), 3) for v in (np.array(c["a"]) + np.array(c["b"])) / 2]
        if at is None:
            raise ValueError("drop needs at: robot id, walkway id or xy/xyz")
        p = [float(v) for v in at]
        return self._snap(p if len(p) == 3 else [*p, DECKS[deck or "top_deck"]])

    def _robot(self, rid: str) -> Robot:
        r = next((x for x in self.robots if x.id == rid), None)
        if r is None:
            raise ValueError(f"unknown robot {rid!r}")
        return r

    def _mark(self, cause: str) -> None:
        for r in self.robots:
            if r.failed or not r.legs:
                continue
            m = self._marks.get(r.id)
            if m is not None:
                m["cause"] += f" + {cause}"
                continue
            self._marks[r.id] = {"nodes": [lg.v for lg in r.legs], "popped": r.popped, "end": r.legs[-1].t1,
                                 "goal": r.goal, "task": r.task.id if r.task else None, "cause": cause}

    def _log_reroutes(self) -> None:
        for rid, m in self._marks.items():
            r = self._robot(rid)
            old = m["nodes"][r.popped - m["popped"]:]
            new = [lg.v for lg in r.legs]
            if old == new or not old:
                continue
            op = [r.pos] + [self.rm.pos[n] for n in old]
            nw = [r.pos] + [self.rm.pos[n] for n in new]
            extra_m = round(plen(nw) - plen(op), 1) if new else None
            extra_s = round(r.legs[-1].t1 - m["end"], 1) if new else None
            task = r.task.id if r.task else None
            r.prev_path, r.reroute_t = corners(op), self.t
            rec = {"robot": rid, "cause": m["cause"], "old_path": r.prev_path, "new_path": corners(nw) if new else [],
                   "extra_m": extra_m, "extra_s": extra_s, "task": task, "retask": task != m["task"],
                   "backout": bool(r.legs and r.legs[0].back)}
            self.reroutes.append({"t": round(self.t, 2), **rec})
            if new:
                msg = (f"{rid} re-routes around {m['cause']}: {extra_m:+.0f} m, {extra_s:+.0f} s"
                       + (" (backs out of corridor)" if rec["backout"] else "") + (f" -> {task}" if rec["retask"] else ""))
            else:
                msg = f"{rid} has no route around {m['cause']} -> holds"
            self._log("reroute", msg, **rec)
        self._marks = {}

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
        if cur.back or not any(self.rm.is_closed(lg.u, lg.v) for lg in keep):
            return keep
        d = cur.t1 - cur.t0
        t0 = self.t - (cur.t1 - self.t)
        out = [Leg(cur.v, cur.u, t0, t0 + d, cur.kind, cur.corridor, True)]
        if cur.corridor:
            for lg in reversed(r.trail):
                if lg.corridor != cur.corridor:
                    break
                out.append(Leg(lg.v, lg.u, out[-1].t1, out[-1].t1 + lg.t1 - lg.t0, lg.kind, lg.corridor, True))
        return out

    def _origin(self, r: Robot) -> tuple[int, float]:
        keep = self._kept(r)
        return (keep[-1].v, keep[-1].t1) if keep else (r.node, self.t)

    def _truncate(self, r: Robot) -> tuple[int, float]:
        keep = self._kept(r)
        r.legs = keep
        self.res.release(r.id, after=self.t + 1e-6)
        if keep and keep[0].back and keep[0].corridor:
            for rid in self.res.claim(keep[0].corridor, self.t, keep[-1].t1, r.id):
                o = self._robot(rid)
                if o.goal is not None and not o.paused:
                    self._go(o, o.goal)
        return (keep[-1].v, keep[-1].t1) if keep else (r.node, self.t)

    def _go(self, r: Robot, goal: int) -> bool:
        a, t0 = self._truncate(r)
        r.goal = goal
        r.planned_t = self.t
        self.replans += 1
        if a == goal:
            r.stuck = False
            r.arrived_t = t0
            return True
        _, path = self.rm.route(a, goal, self._blocked)
        if not path and self.rm.trapped(a):
            esc = self.rm.escape(a)
            if esc:
                _, rest = self.rm.route(esc[-1], goal, self._blocked)
                path = esc + rest[1:] if rest else esc
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
        hz = self.hz.active()
        held = [t for t in self.tasks.values() if t.outcome == BLOCKED and t.status == "escalated"]
        if not perm and not hz and not held:
            return
        d, _, _ = self.rm.dijkstra(self.rm.dock, perm)
        for t in held:
            if math.isfinite(d[t.node]):
                t.status, t.outcome, t.finished = "pending", None, None
                self._event_tick = True
                self._log("reopen", f"{t.id} reachable again -> back in the queue", task=t.id)
        for t in self.tasks.values():
            if t.status in OPEN and not math.isfinite(d[t.node]):
                r = next((x for x in self.robots if x.task is t), None)
                if r is not None:
                    self._release(r)
                why = "a hazard" if hz else "a failed robot"
                t.status, t.outcome, t.finished = "escalated", BLOCKED if hz else "unreachable", self.t
                if t.id in self.alarms:
                    self.alarms[t.id]["outcome"] = "escalated"
                self._log("needs_human", f"{t.id} cut off by {why} -> escalate to human crew", task=t.id)

    def _preempt_gas(self) -> None:
        for gas in [t for t in self.tasks.values() if t.kind == "gas" and self._assignable(t)]:
            cands = [r for r in self.robots if not r.failed and not r.paused
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
            if not math.isfinite(dd[a]):
                continue
            if r.battery.must_return(Battery.path_energy(dd[a], ds[a]) + 1.0):
                tid = r.task.id
                self._send_dock(r, True)
                self._log("return_to_charge", f"{r.id} battery {r.battery.soc:.0%} -> abort {tid}, return to dock",
                          robot=r.id, task=tid)

    def _assign(self) -> None:
        if self._event_tick:
            for r in self.robots:
                if r.task is not None and r.task.kind != "gas" and r.task.status == "assigned":
                    self._release(r)
        robots = [r for r in self.robots if not r.failed and not r.paused and r.task is None
                  and not (r.low and r.battery.soc < r.battery.resume_frac)]
        tasks = [t for t in self.tasks.values() if self._assignable(t)]
        if not robots or not tasks:
            return
        eta, need = self._plan_matrix(robots, tasks)
        prio = np.array([priority(t, self.t) for t in tasks])
        cost = np.full(eta.shape, INF)
        for i, r in enumerate(robots):
            for j in range(len(tasks)):
                if math.isfinite(eta[i, j]) and r.battery.can_afford(need[i, j]):
                    cost[i, j] = eta[i, j] + W_SOC * (1.0 - r.battery.soc) - W_PRIO * prio[j]
        for i, j in assign(cost):
            r = robots[i]
            r.low = False
            self._give(r, tasks[j], f"hungarian: ETA {eta[i, j]:.0f} s, prio {prio[j]:.1f}, "
                       f"need {need[i, j]:.1f} Wh")

    def _advance(self, r: Robot) -> None:
        while r.legs and r.legs[0].t1 <= self.t + 1e-9:
            lg = r.legs.pop(0)
            r.popped += 1
            if lg.back or not lg.corridor:
                r.trail = []
            elif r.trail and r.trail[-1].corridor != lg.corridor:
                r.trail = [lg]
            else:
                r.trail.append(lg)
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
        elif not near and r.paused:
            r.paused = False
            self._log("resume", f"{r.id} resumes: person clear", robot=r.id)
            if r.goal is not None:
                self._go(r, r.goal)

    def _repair_paths(self, r: Robot) -> None:
        if r.paused or r.goal is None:
            return
        keep = self._kept(r)
        cur = self._mid(r)
        backing = bool(keep) and keep[0].back and not (cur is not None and cur.back)
        future = [] if backing else r.legs[len(keep):]
        bad = backing or any((lg.v in self._blocked and lg.v != r.goal) or self.rm.is_closed(lg.u, lg.v)
                             for lg in future)
        retime = self._hz_tick and r.planned_t != self.t and bool(r.legs)
        if r.stuck or bad or retime or (not r.legs and r.node != r.goal):
            was = r.stuck
            self._go(r, r.goal)
            if r.stuck and not was and self.hz.active() and r.task is not None and r.task.kind != "gas":
                tid = r.task.id
                self._release(r)
                self._log("blocked", f"{r.id} cannot reach {tid} -> hands it back to the queue", robot=r.id, task=tid)

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
        task.progress += self.dt / self.hz.work_slow(task.kind)
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
        if self.hz.stale(self.rm):
            self.hz.apply(self.rm)
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
        self._assign()
        self._event_tick = False
        for r in self.robots:
            if r.failed or r.paused or r.task is not None:
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
        self._log_reroutes()
        self._hz_tick = False
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

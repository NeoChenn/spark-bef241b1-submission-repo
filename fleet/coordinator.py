from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from fleet.contract import REGISTRY, Finding, Job, JobExecutor, JobResult, JobStatus, Level, Robot, RobotKind, World
from policy.fleet.assign import INF, assign as hungarian
from policy.fleet.battery import POWER_W

PERSON_RADIUS_M = 3.0
RESERVE = 0.15
IDLE_DOCK_BATTERY = 0.3
RESUME_BATTERY = 0.9
URGENT = 2
PREEMPTABLE = 3
BATTERY_PENALTY_S = 120.0
PRIORITY_STEP_S = 3600.0
DOCK_PADS = 2
GAS_ZONE_M = 8.0
FAILED_RADIUS_M = 1.0
CUTOFF_S = 120.0
DEADLINE_WINDOW_S = 600.0
DEADLINE_BOOST = 2.0
CUTOFF_EVERY_S = 10.0
GAS_RESPONSE = {"gas_survey", "valve_isolate", "leak_repair_prep", "gas", "flange_retorque"}
AGENTS = {"gemini", "agent", "planner", "orchestrator", "llm"}
ACTIVE = (JobStatus.ASSIGNED, JobStatus.RUNNING)
OPEN = (JobStatus.QUEUED, JobStatus.WAITING_APPROVAL) + ACTIVE
LEVEL_RANK = {Level.AUTO: 0, Level.SUPV: 1, Level.ASSIST: 2, Level.HUMAN: 3}

Vec = tuple[float, float, float]


@dataclass(frozen=True)
class Power:
    capacity_wh: float
    idle_w: float
    move_w: float
    stairs_w: float
    work_w: float
    charger_w: float

    def leg_w(self, kind: str) -> float:
        return self.stairs_w if kind == "stairs" else self.move_w


POWER = {
    RobotKind.QUADRUPED: Power(432.0, POWER_W["idle"], POWER_W["walk"], POWER_W["stairs"], POWER_W["work"], 1500.0),
    RobotKind.CRAWLER: Power(200.0, 7.0, 108.0, 108.0, 72.0, 600.0),
    RobotKind.AERIAL: Power(120.0, 9.0, 346.0, 346.0, 259.0, 700.0),
    RobotKind.ROV: Power(1000.0, 36.0, 720.0, 720.0, 540.0, 3000.0),
}


def dist(a: Iterable[float], b: Iterable[float]) -> float:
    return math.dist(tuple(a), tuple(b))


def is_agent(by: str) -> bool:
    return str(by).lower().split(":")[0] in AGENTS


@dataclass
class Leg:
    start: Vec
    end: Vec
    segment: Optional[str] = None
    speed: float = 1.0
    kind: str = "walk"
    node: Optional[int] = None

    @property
    def length(self) -> float:
        return dist(self.start, self.end)

    def time(self, speed_mps: float) -> float:
        return self.length / max(speed_mps * self.speed, 1e-6)


@dataclass
class Route:
    legs: list[Leg]
    blocked: bool = False

    @property
    def length(self) -> float:
        return sum(l.length for l in self.legs)

    def time(self, speed_mps: float) -> float:
        return sum(l.time(speed_mps) for l in self.legs)

    def energy_wh(self, p: Power, speed_mps: float) -> float:
        return sum(l.time(speed_mps) * p.leg_w(l.kind) for l in self.legs) / 3600.0


class StraightPlanner:
    def route(self, robot: Robot, goal: Vec, standoff: float = 0.0) -> Route:
        start = tuple(robot.position)
        d = dist(start, goal)
        if standoff and d > standoff:
            f = (d - standoff) / d
            goal = tuple(s + (g - s) * f for s, g in zip(start, goal))
        return Route([Leg(start, tuple(goal))])


def offline_explanation(ctx: dict) -> str:
    urgent, bumped = ctx.get("urgent", {}), ctx.get("preempted", {})
    if ctx.get("why") == "gas_zone":
        return (f"Gas at {urgent.get('target', '?')}: {ctx.get('robot')} stops {bumped.get('kind')} at "
                f"{bumped.get('target')} inside the {GAS_ZONE_M:.0f} m gas zone; it waits until the leak is isolated.")
    line = f"{urgent.get('kind', 'urgent job')} at {urgent.get('target', '?')} (priority {urgent.get('priority')})"
    if bumped:
        line += (f" takes {ctx.get('robot')} from {bumped.get('kind')} at {bumped.get('target')};"
                 " that job is back in the queue")
    else:
        line += f" jumps the queue and goes to {ctx.get('robot')}"
    return line + "."


@dataclass
class RobotRT:
    robot: Robot
    dock: Vec
    state: str = "IDLE"
    job: Optional[str] = None
    route: Optional[Route] = None
    leg_i: int = 0
    leg_s: float = 0.0
    holding: Optional[str] = None
    work_left: float = 0.0
    work_t0: float = 0.0
    result: Optional[JobResult] = None
    paused_from: Optional[str] = None
    yaw: float = 0.0
    wait_since: Optional[float] = None
    used_wh: float = 0.0
    charged_wh: float = 0.0
    dist_m: float = 0.0
    time_in: Counter = field(default_factory=Counter)

    @property
    def remaining(self) -> list[Leg]:
        return self.route.legs[self.leg_i:] if self.route else []


class Coordinator:
    def __init__(self, robots: list[Robot], world: World, executors: Optional[dict[str, JobExecutor]] = None,
                 planner=None, auto_approve: bool = False, docks: Optional[dict[str, Vec]] = None,
                 explainer: Optional[Callable[[dict], tuple[str, bool]]] = None,
                 on_event: Optional[Callable[[dict], None]] = None, dock_pads: int = DOCK_PADS):
        self.world = world
        self.executors = REGISTRY if executors is None else executors
        self.planner = planner or StraightPlanner()
        self.auto_approve = auto_approve
        self.explainer = explainer
        self.on_event = on_event
        self.dock_pads = dock_pads
        self.rt: dict[str, RobotRT] = {
            r.id: RobotRT(r, tuple((docks or {}).get(r.id, r.position))) for r in robots}
        self.jobs: dict[str, Job] = {}
        self.status: dict[str, JobStatus] = {}
        self.results: dict[str, JobResult] = {}
        self.assigned_to: dict[str, str] = {}
        self.approvals: list[str] = []
        self.requests: dict[str, list[dict]] = {}
        self.crew: list[str] = []
        self.timeline: list[dict] = []
        self.used_gemini = False
        self.preemptions = 0
        self.replans = 0
        self.failures = 0
        self.wait_s = 0.0
        self.cutoff: dict[str, float] = {}
        self._seq = 0

    @property
    def t(self) -> float:
        return self.world.t

    def emit(self, event: str, job: Optional[str] = None, robot: Optional[str] = None, **detail) -> dict:
        e = {"t": round(self.t, 2), "event": event, "job": job, "robot": robot, "detail": detail}
        self.timeline.append(e)
        if self.on_event:
            self.on_event(e)
        return e

    def new_id(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}-{self._seq:03d}"

    def power(self, r: Robot) -> Power:
        return POWER.get(r.kind, POWER[RobotKind.QUADRUPED])

    def executor_for(self, job: Job) -> Optional[JobExecutor]:
        return self.executors.get(job.kind)

    def submit(self, job: Job, allow_auto: bool = True, by: str = "coordinator", note: str = "") -> JobStatus:
        if job.id in self.jobs:
            return self.status[job.id]
        dup = next((j for j, s in self.status.items() if self.jobs[j].kind == job.kind
                    and self.jobs[j].target == job.target and s in OPEN), None)
        if dup:
            self.emit("duplicate_skipped", job.id, None, same_as=dup)
            return self.status[dup]
        job.created_t = job.created_t or self.t
        self.jobs[job.id] = job
        detail = {"kind": job.kind, "target": job.target, "level": job.level.value, "priority": job.priority}
        if job.parent:
            detail["parent"] = job.parent
        self.emit("job_queued", job.id, None, **detail)
        if job.level == Level.HUMAN:
            self._to_crew(job.id, "manual job: routed to crew")
        elif job.level == Level.SUPV:
            self._hold(job.id, by, note)
            if self.auto_approve and allow_auto:
                self.approve(job.id, by="auto")
        else:
            self.status[job.id] = JobStatus.QUEUED
        return self.status[job.id]

    def _hold(self, job_id: str, by: str, note: str):
        job = self.jobs[job_id]
        job.params.pop("approved", None)
        self.status[job_id] = JobStatus.WAITING_APPROVAL
        if job_id not in self.approvals:
            self.approvals.append(job_id)
        self.requests.setdefault(job_id, []).append({"t": round(self.t, 2), "by": by, "note": note})
        self.emit("approval_requested", job_id, None, kind=job.kind, target=job.target, by=by, note=note)

    def self_requested(self, job_id: str, by: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or not is_agent(by):
            return False
        who = {str(job.params.get("requested_by", ""))} | {r["by"] for r in self.requests.get(job_id, [])}
        return any(is_agent(w) for w in who)

    def approve(self, job_id: str, by: str = "operator") -> bool:
        if self.self_requested(job_id, by):
            self.emit("approval_refused", job_id, None, by=by, reason="an agent cannot approve a SUPV job it asked for")
            return False
        if self.status.get(job_id) != JobStatus.WAITING_APPROVAL:
            return False
        self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.QUEUED
        self.jobs[job_id].params["approved"] = True
        self.emit("approved", job_id, None, by=by)
        return True

    def reject(self, job_id: str, reason: str = "rejected by operator", by: str = "operator") -> bool:
        if is_agent(by) or self.status.get(job_id) != JobStatus.WAITING_APPROVAL:
            return False
        self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.FAILED
        self.emit("rejected", job_id, None, reason=reason)
        return True

    def request_approval(self, job_id: str, by: str = "operator", note: str = "") -> tuple[bool, str]:
        s = self.status.get(job_id)
        if s is None:
            return False, f"unknown job {job_id}"
        if s == JobStatus.WAITING_APPROVAL:
            self.requests.setdefault(job_id, []).append({"t": round(self.t, 2), "by": by, "note": note})
            self.emit("approval_note", job_id, None, by=by, note=note)
            return True, "already waiting for the operator; note attached"
        if s == JobStatus.RUNNING:
            return False, "job is already running; cancel it instead"
        if s not in (JobStatus.QUEUED, JobStatus.ASSIGNED):
            return False, f"job is {s.value}"
        if s == JobStatus.ASSIGNED:
            self._release(self.rt[self.assigned_to[job_id]], requeue=False, why="held for approval")
        job = self.jobs[job_id]
        if LEVEL_RANK[job.level] < LEVEL_RANK[Level.SUPV]:
            job.level = Level.SUPV
        self._hold(job_id, by, note)
        return True, "waiting for operator approval"

    def cancel(self, job_id: str, by: str = "operator", reason: str = "") -> tuple[bool, str]:
        s = self.status.get(job_id)
        if s is None:
            return False, f"unknown job {job_id}"
        if s not in OPEN:
            return False, f"job is already {s.value}"
        job = self.jobs[job_id]
        rid = self.assigned_to.get(job_id)
        if s == JobStatus.RUNNING and job.level == Level.SUPV:
            return False, "approved arm work in progress cannot be abandoned half done"
        if rid:
            self._release(self.rt[rid], requeue=False, why="cancelled")
        if job_id in self.approvals:
            self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.CANCELLED
        self.emit("cancelled", job_id, rid, by=by, reason=reason)
        return True, "cancelled"

    def reprioritise(self, job_id: str, priority: int, by: str = "operator", why: str = "") -> tuple[bool, str]:
        s = self.status.get(job_id)
        if s is None:
            return False, f"unknown job {job_id}"
        if s not in OPEN:
            return False, f"job is already {s.value}"
        if not isinstance(priority, int) or isinstance(priority, bool) or not 1 <= priority <= 5:
            return False, "priority must be an integer 1 (urgent) .. 5 (routine)"
        job = self.jobs[job_id]
        old = job.priority
        job.priority = priority
        self.emit("priority_change", job_id, self.assigned_to.get(job_id), by=by, old=old, new=priority,
                  explanation=why or f"{job.kind} at {job.target}: priority {old} -> {priority}",
                  used_gemini=is_agent(by))
        return True, f"priority {old} -> {priority}"

    def fail_robot(self, robot_id: str, reason: str = "robot fault") -> bool:
        rt = self.rt.get(robot_id)
        if rt is None or rt.state == "FAILED":
            return False
        jid = rt.job
        self._release(rt, requeue=True, why=f"robot failure: {reason}")
        rt.state = "FAILED"
        rt.holding = None
        self.failures += 1
        self.emit("robot_failure", jid, robot_id, reason=reason)
        return True

    def _to_crew(self, job_id: str, why: str, robot: Optional[str] = None):
        self.status[job_id] = JobStatus.HANDED_TO_HUMAN
        self.crew.append(job_id)
        self.emit("handed_to_crew", job_id, robot, reason=why)

    def requirements(self, job: Job) -> tuple[set[str], set[RobotKind]]:
        ex = self.executor_for(job)
        needs = set(job.needs) | (set(ex.needs) if ex else set())
        kinds = set(job.robot_kinds) or (set(ex.robot_kinds) if ex else set())
        return needs, kinds

    def eligible(self, job: Job, robot: Robot) -> bool:
        needs, kinds = self.requirements(job)
        return needs <= set(robot.capabilities) and (not kinds or robot.kind in kinds)

    def locks(self, job: Job) -> set[str]:
        keys = {f"asset:{job.target}"}
        if job.params.get("zone"):
            keys.add(f"zone:{job.params['zone']}")
        return keys

    def locked_by_other(self, job: Job, robot_id: str) -> Optional[str]:
        mine = self.locks(job)
        for jid, rid in self.assigned_to.items():
            if rid != robot_id and self.status.get(jid) in ACTIVE and jid != job.id and self.locks(self.jobs[jid]) & mine:
                return jid
        return None

    def to_dock_wh(self, r: Robot, frm: Vec, dock: Vec) -> float:
        p = self.power(r)
        if hasattr(self.planner, "dock_cost") and r.kind in (RobotKind.QUADRUPED, RobotKind.CRAWLER):
            t_all, t_st = self.planner.dock_cost(frm, dock, r.speed_mps)
            return (p.move_w * (t_all - t_st) + p.stairs_w * t_st) / 3600.0
        probe = Robot(r.id, r.kind, set(), tuple(frm), speed_mps=r.speed_mps)
        return self.planner.route(probe, dock, 0.0).energy_wh(p, r.speed_mps)

    def cost(self, job: Job, rt: RobotRT) -> Optional[float]:
        r = rt.robot
        route = self.planner.route(r, job.position, job.params.get("standoff", 1.0))
        if route.blocked:
            return None
        p = self.power(r)
        travel = route.time(r.speed_mps)
        end = route.legs[-1].end if route.legs else tuple(r.position)
        need = route.energy_wh(p, r.speed_mps) + p.work_w * job.est_duration_s / 3600.0 + self.to_dock_wh(r, end, rt.dock)
        if r.battery - need / p.capacity_wh < RESERVE:
            return None
        return travel + BATTERY_PENALTY_S * (1.0 - r.battery)

    def leaks(self) -> list[dict]:
        return [l for l in self.world.state.get("gas_leaks", []) if l.get("reported") and not l.get("isolated")]

    def gas_held(self, job: Job) -> bool:
        if job.kind in GAS_RESPONSE or job.params.get("gas_response"):
            return False
        return any(dist(job.position[:2], l["position"][:2]) < GAS_ZONE_M and abs(job.position[2] - l["position"][2]) < 5.0
                   for l in self.leaks())

    def effective_priority(self, job: Job) -> float:
        dl = job.params.get("deadline")
        if dl is None:
            return float(job.priority)
        urgency = min(1.0, max(0.0, 1.0 - (float(dl) - self.t) / DEADLINE_WINDOW_S))
        return max(1.0, job.priority - DEADLINE_BOOST * urgency)

    def queued(self) -> list[Job]:
        ids = [j for j, s in self.status.items() if s == JobStatus.QUEUED]
        return sorted((self.jobs[j] for j in ids), key=lambda j: (self.effective_priority(j), j.created_t, j.id))

    def assign(self):
        jobs = [j for j in self.queued() if not self.gas_held(j)]
        idle = [rt for rt in self.rt.values() if rt.state == "IDLE" and rt.job is None]
        if jobs and idle:
            base = np.full((len(idle), len(jobs)), INF)
            for i, rt in enumerate(idle):
                for k, job in enumerate(jobs):
                    if not self.eligible(job, rt.robot) or self.locked_by_other(job, rt.robot.id):
                        continue
                    c = self.cost(job, rt)
                    if c is not None:
                        base[i, k] = c
            prio = np.array([PRIORITY_STEP_S * (self.effective_priority(j) - 1) for j in jobs])
            cost = np.where(base < INF, base + prio[None, :], INF)
            taken: set[str] = set()
            for i, k in sorted(hungarian(cost), key=lambda p: (self.effective_priority(jobs[p[1]]), jobs[p[1]].created_t)):
                job, rt = jobs[k], idle[i]
                if self.locks(job) & taken or self.locked_by_other(job, rt.robot.id):
                    continue
                bids = [(float(base[x, k]), idle[x].robot.id) for x in range(len(idle)) if base[x, k] < INF]
                self._start(job, rt, float(base[i, k]), bids)
                taken |= self.locks(job)
        for job in jobs:
            if self.status[job.id] == JobStatus.QUEUED and job.priority <= URGENT:
                self._try_preempt(job)

    def _try_preempt(self, job: Job):
        cands = []
        for rt in self.rt.values():
            cur = self.jobs.get(rt.job) if rt.job else None
            if cur is None or cur.priority < PREEMPTABLE or cur.priority <= job.priority:
                continue
            if rt.state == "WORKING" and cur.level == Level.SUPV:
                continue
            if rt.state in ("DOCKING", "CHARGING", "DOCK_QUEUE", "FAILED") or not self.eligible(job, rt.robot):
                continue
            if self.locked_by_other(job, rt.robot.id):
                continue
            c = self.cost(job, rt)
            if c is not None:
                cands.append((c, rt.robot.id))
        if not cands:
            return
        c, rid = min(cands)
        rt = self.rt[rid]
        bumped = self.jobs[rt.job]
        self._release(rt, requeue=True, why=f"preempted by {job.id} (priority {job.priority})")
        self.preemptions += 1
        ctx = {"t": self.t, "robot": rid,
               "urgent": {"id": job.id, "kind": job.kind, "target": job.target, "priority": job.priority},
               "preempted": {"id": bumped.id, "kind": bumped.kind, "target": bumped.target,
                             "priority": bumped.priority}}
        self.explain(ctx)
        self._start(job, rt, c, cands)

    def explain(self, ctx: dict):
        line, used = offline_explanation(ctx), False
        if self.explainer:
            try:
                line, used = self.explainer(ctx)
            except Exception:
                line, used = offline_explanation(ctx), False
        self.used_gemini |= used
        self.emit("priority_change", ctx.get("urgent", {}).get("id"), ctx.get("robot"),
                  explanation=line, used_gemini=used)

    def _start(self, job: Job, rt: RobotRT, cost: float, bids):
        rt.job = job.id
        rt.robot.busy_with = job.id
        self.assigned_to[job.id] = rt.robot.id
        self.status[job.id] = JobStatus.ASSIGNED
        self._set_route(rt, self.planner.route(rt.robot, job.position, job.params.get("standoff", 1.0)))
        rt.state = "MOVING"
        self.emit("assigned", job.id, rt.robot.id, cost_s=round(cost, 1),
                  bids={rid: round(c, 1) for c, rid in bids}, route_m=round(rt.route.length, 1))

    def _set_route(self, rt: RobotRT, route: Route):
        rt.route, rt.leg_i, rt.leg_s = route, 0, 0.0

    def _release(self, rt: RobotRT, requeue: bool, why: str):
        if rt.job:
            jid = rt.job
            self.assigned_to.pop(jid, None)
            if requeue and self.status.get(jid) in ACTIVE:
                self.status[jid] = JobStatus.QUEUED
                self.emit("preempted" if "preempt" in why else "requeued", jid, rt.robot.id, reason=why)
        rt.job, rt.result, rt.route, rt.work_left = None, None, None, 0.0
        rt.robot.busy_with = None
        rt.state = "IDLE"

    def report(self, job: Job, robot: Optional[str] = None, event: str = "alarm", **detail) -> JobStatus:
        self.emit(event, job.id, robot, **detail)
        return self.submit(job)

    def tick(self, dt: float = 1.0):
        self._obstacles()
        self._safety()
        self._gas_zone()
        self._cutoffs()
        self.assign()
        for rt in self.rt.values():
            self._advance(rt, dt)
            rt.time_in[rt.state] += dt
        self.world.t += dt

    def obstacle_points(self) -> list[tuple[Vec, float]]:
        pts = [(tuple(rt.robot.position), FAILED_RADIUS_M) for rt in self.rt.values() if rt.state == "FAILED"]
        return pts + [(tuple(p), PERSON_RADIUS_M) for p in self.world.people]

    def _obstacles(self):
        if not hasattr(self.planner, "set_obstacles") or not self.planner.set_obstacles(self.obstacle_points()):
            return
        blocked = self.planner.blocked
        for rt in self.rt.values():
            if rt.state not in ("MOVING", "WAITING_SEGMENT", "DOCKING") or not rt.route:
                continue
            legs = rt.remaining
            cur = legs[0] if legs else None
            if cur is None or (cur.segment and rt.holding == cur.segment):
                continue
            if not any(l.node in blocked for l in legs[1:]):
                continue
            new = self.planner.route(rt.robot, legs[-1].end, 0.0)
            if new.blocked or not new.legs:
                continue
            old_m = sum(l.length for l in legs) - rt.leg_s
            rt.holding = None
            self._set_route(rt, new)
            self.replans += 1
            self.emit("replanned", rt.job, rt.robot.id, reason="obstacle on route", old_m=round(old_m, 1),
                      new_m=round(rt.route.length, 1))

    def _cutoffs(self):
        closed = getattr(getattr(self.planner, "rm", None), "closed", None)
        if self.t % CUTOFF_EVERY_S > 1e-6 or not (closed or any(rt.state == "FAILED" for rt in self.rt.values())):
            return
        live = [rt for rt in self.rt.values() if rt.state != "FAILED"]
        for job in self.queued():
            fleet = [rt for rt in live if self.eligible(job, rt.robot)]
            if not fleet or not all(self.planner.route(rt.robot, job.position, job.params.get("standoff", 1.0)).blocked
                                    for rt in fleet):
                self.cutoff.pop(job.id, None)
                continue
            since = self.cutoff.setdefault(job.id, self.t)
            if self.t - since >= CUTOFF_S:
                self.cutoff.pop(job.id)
                self._to_crew(job.id, f"no safe route for {CUTOFF_S:.0f} s (blocked by a failed robot or closure)")

    def _safety(self):
        for rt in self.rt.values():
            if rt.state == "FAILED":
                continue
            near = [p for p in self.world.people if dist(p, rt.robot.position) < PERSON_RADIUS_M]
            if near and rt.state != "PAUSED_PERSON":
                rt.paused_from = rt.state
                rt.state = "PAUSED_PERSON"
                self.emit("person_pause", rt.job, rt.robot.id, distance_m=round(min(
                    dist(p, rt.robot.position) for p in near), 2))
            elif not near and rt.state == "PAUSED_PERSON":
                rt.state = rt.paused_from or "IDLE"
                self.emit("person_clear", rt.job, rt.robot.id, resume=rt.state)
            if rt.state in ("DOCKING", "CHARGING", "DOCK_QUEUE", "PAUSED_PERSON"):
                continue
            r = rt.robot
            back = self.to_dock_wh(r, r.position, rt.dock) / self.power(r).capacity_wh
            low = r.battery - back < RESERVE
            idle_low = rt.state == "IDLE" and r.battery < IDLE_DOCK_BATTERY
            if low or idle_low:
                self.emit("low_battery", rt.job, r.id, battery=round(r.battery, 3), to_dock=round(back, 3),
                          reason="reserve needed to reach the dock" if low else "idle and low: charge while there is time")
                self._release(rt, requeue=True, why="low battery: robot returns to dock")
                self._go_dock(rt)

    def _gas_zone(self):
        if not self.leaks():
            return
        for rt in self.rt.values():
            job = self.jobs.get(rt.job) if rt.job else None
            if job is None or not self.gas_held(job) or rt.state == "PAUSED_PERSON":
                continue
            leak = min(self.leaks(), key=lambda l: dist(l["position"], job.position))
            self._release(rt, requeue=True, why="gas zone: work stopped until the leak is isolated")
            self.emit("gas_zone_hold", job.id, rt.robot.id, radius_m=GAS_ZONE_M, leak=leak.get("id"))
            self.explain({"t": self.t, "robot": rt.robot.id, "why": "gas_zone",
                          "urgent": {"id": leak.get("id"), "kind": "gas zone", "target": leak.get("source", leak.get("id")),
                                     "priority": 1},
                          "preempted": {"id": job.id, "kind": job.kind, "target": job.target, "priority": job.priority}})

    def _go_dock(self, rt: RobotRT):
        self._set_route(rt, self.planner.route(rt.robot, rt.dock, 0.0))
        rt.state = "DOCKING"

    def may_enter(self, rt: RobotRT, segment: str) -> bool:
        if rt.robot.kind != RobotKind.QUADRUPED:
            return True
        return not any(o is not rt and o.robot.kind == RobotKind.QUADRUPED and o.holding == segment
                       for o in self.rt.values())

    def pads_free(self, rt: RobotRT) -> bool:
        used = sum(1 for o in self.rt.values() if o is not rt and o.state == "CHARGING" and dist(o.dock, rt.dock) < 1.0)
        return used < self.dock_pads

    def _drain(self, rt: RobotRT, wh: float):
        r = rt.robot
        cap = self.power(r).capacity_wh
        wh = min(wh, r.battery * cap)
        r.battery = max(0.0, r.battery - wh / cap)
        rt.used_wh += wh

    def _advance(self, rt: RobotRT, dt: float):
        r = rt.robot
        p = self.power(r)
        if rt.state == "FAILED":
            return
        if rt.state == "PAUSED_PERSON":
            self._drain(rt, p.idle_w * dt / 3600.0)
            return
        if rt.state in ("MOVING", "WAITING_SEGMENT", "DOCKING"):
            arrived, wh, moved = self._move(rt, dt)
            self._drain(rt, wh)
            rt.dist_m += moved
            if rt.state == "WAITING_SEGMENT":
                self.wait_s += dt
            if arrived:
                if rt.state == "DOCKING":
                    rt.route = None
                    rt.state = "CHARGING" if self.pads_free(rt) else "DOCK_QUEUE"
                    self.emit("docked" if rt.state == "CHARGING" else "dock_queue", None, r.id,
                              battery=round(r.battery, 3))
                else:
                    self._arrive(rt)
        elif rt.state == "DOCK_QUEUE":
            self._drain(rt, p.idle_w * dt / 3600.0)
            if self.pads_free(rt):
                rt.state = "CHARGING"
                self.emit("docked", None, r.id, battery=round(r.battery, 3))
        elif rt.state == "WORKING":
            self._drain(rt, p.work_w * dt / 3600.0)
            rt.work_left -= dt
            if rt.work_left <= 0:
                self._finish(rt)
        elif rt.state == "CHARGING":
            wh = min((1.0 - r.battery) * p.capacity_wh, p.charger_w * dt / 3600.0)
            r.battery = min(1.0, r.battery + wh / p.capacity_wh)
            rt.charged_wh += wh
            if r.battery >= RESUME_BATTERY:
                rt.state = "IDLE"
                self.emit("charged", None, r.id, battery=round(r.battery, 3))
        else:
            self._drain(rt, p.idle_w * dt / 3600.0)

    def _move(self, rt: RobotRT, dt: float) -> tuple[bool, float, float]:
        legs = rt.route.legs if rt.route else []
        p = self.power(rt.robot)
        v = rt.robot.speed_mps
        was_waiting = rt.state == "WAITING_SEGMENT"
        left, wh, moved = dt, 0.0, 0.0
        while rt.leg_i < len(legs):
            leg = legs[rt.leg_i]
            if rt.leg_s == 0.0 and leg.segment and rt.holding != leg.segment:
                if not self.may_enter(rt, leg.segment):
                    if not was_waiting:
                        blocker = next(o.robot.id for o in self.rt.values() if o is not rt and o.holding == leg.segment)
                        self.emit("segment_wait", rt.job, rt.robot.id, segment=leg.segment, blocked_by=blocker)
                        rt.state = "WAITING_SEGMENT"
                    return False, wh + p.idle_w * left / 3600.0, moved
                rt.holding = leg.segment
            if rt.state == "WAITING_SEGMENT":
                rt.state = "MOVING"
                self.emit("segment_clear", rt.job, rt.robot.id, segment=leg.segment)
                was_waiting = False
            if left <= 1e-9:
                return False, wh, moved
            L = leg.length
            speed = max(v * leg.speed, 1e-6)
            step = min(speed * left, L - rt.leg_s)
            rt.leg_s += step
            left -= step / speed
            wh += step / speed * p.leg_w(leg.kind) / 3600.0
            moved += step
            if L > 1e-9:
                f = rt.leg_s / L
                rt.robot.position = tuple(a + (b - a) * f for a, b in zip(leg.start, leg.end))
                dx, dy = leg.end[0] - leg.start[0], leg.end[1] - leg.start[1]
                if abs(dx) + abs(dy) > 1e-6:
                    rt.yaw = math.atan2(dy, dx)
            if rt.leg_s >= L - 1e-9:
                rt.robot.position = tuple(leg.end)
                rt.leg_i += 1
                rt.leg_s = 0.0
                nxt = legs[rt.leg_i] if rt.leg_i < len(legs) else None
                if nxt is None or nxt.segment != rt.holding:
                    rt.holding = None
        rt.holding = None
        return True, wh, moved

    def _arrive(self, rt: RobotRT):
        job = self.jobs[rt.job]
        ex = self.executor_for(job)
        self.status[job.id] = JobStatus.RUNNING
        self.emit("job_started", job.id, rt.robot.id, kind=job.kind, target=job.target)
        if ex is None:
            rt.result = JobResult(job.id, rt.robot.id, JobStatus.HANDED_TO_HUMAN, self.t, self.t,
                                  log=[f"no executor for kind {job.kind!r}: routed to crew"])
        else:
            try:
                rt.result = ex.run(job, rt.robot, self.world)
            except Exception as exc:
                rt.result = JobResult(job.id, rt.robot.id, JobStatus.FAILED, self.t, self.t,
                                      log=[f"executor raised {type(exc).__name__}: {exc}"])
        span = rt.result.finished_t - rt.result.started_t
        rt.work_left = span if span > 0 else job.est_duration_s
        rt.work_t0 = self.t
        rt.state = "WORKING"

    def _finish(self, rt: RobotRT):
        job, res = self.jobs[rt.job], rt.result
        rid = rt.robot.id
        self.results[job.id] = res
        self.used_gemini |= bool(res.used_gemini)
        status = res.status
        if job.level == Level.ASSIST and status in (JobStatus.DONE, JobStatus.FINDING):
            status = JobStatus.HANDED_TO_HUMAN
        self.assigned_to.pop(job.id, None)
        rt.job, rt.result, rt.route = None, None, None
        rt.robot.busy_with = None
        rt.state = "IDLE"
        if status == JobStatus.HANDED_TO_HUMAN:
            self.status[job.id] = status
            self.crew.append(job.id)
            self.emit("handed_to_crew", job.id, rid, reason="; ".join(res.log[-1:]) or "robot prepared, crew fixes")
        else:
            self.status[job.id] = status
            self.emit("job_done", job.id, rid, status=status.value, findings=len(res.findings),
                      log=res.log[-1] if res.log else "")
        for f in res.findings:
            self._finding(job, rid, f)

    def _finding(self, job: Job, rid: str, f: Finding):
        detail = {"kind": f.kind, "severity": f.severity, "summary": f.summary}
        if f.position is not None:
            detail["position"] = [round(float(v), 3) for v in f.position]
        self.emit("finding", job.id, rid, **detail)
        for fj in f.follow_up:
            fj.parent = fj.parent or job.id
            fj.created_t = self.t
            self.submit(fj)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.status.values():
            out[s.value] = out.get(s.value, 0) + 1
        return out

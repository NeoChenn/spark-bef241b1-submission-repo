from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from fleet.contract import REGISTRY, Finding, Job, JobExecutor, JobResult, JobStatus, Level, Robot, RobotKind, World

PERSON_RADIUS_M = 3.0
LOW_BATTERY = 0.2
IDLE_DOCK_BATTERY = 0.3
RESUME_BATTERY = 0.9
URGENT = 2
PREEMPTABLE = 3
BATTERY_PENALTY_S = 120.0
ACTIVE = (JobStatus.ASSIGNED, JobStatus.RUNNING)

DRAIN = {
    RobotKind.QUADRUPED: (0.00030, 0.00020, 0.00002),
    RobotKind.CRAWLER: (0.00015, 0.00010, 0.00001),
    RobotKind.AERIAL: (0.00080, 0.00060, 0.00002),
    RobotKind.ROV: (0.00020, 0.00015, 0.00001),
}
CHARGE_PER_S = 0.004

Vec = tuple[float, float, float]


def dist(a: Iterable[float], b: Iterable[float]) -> float:
    return math.dist(tuple(a), tuple(b))


@dataclass
class Leg:
    start: Vec
    end: Vec
    segment: Optional[str] = None

    @property
    def length(self) -> float:
        return dist(self.start, self.end)


@dataclass
class Route:
    legs: list[Leg]

    @property
    def length(self) -> float:
        return sum(l.length for l in self.legs)


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
    result: Optional[JobResult] = None
    paused_from: Optional[str] = None
    yaw: float = 0.0
    wait_since: Optional[float] = None


class Coordinator:
    def __init__(self, robots: list[Robot], world: World, executors: Optional[dict[str, JobExecutor]] = None,
                 planner=None, auto_approve: bool = False, docks: Optional[dict[str, Vec]] = None,
                 explainer: Optional[Callable[[dict], tuple[str, bool]]] = None,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.world = world
        self.executors = REGISTRY if executors is None else executors
        self.planner = planner or StraightPlanner()
        self.auto_approve = auto_approve
        self.explainer = explainer
        self.on_event = on_event
        self.rt: dict[str, RobotRT] = {
            r.id: RobotRT(r, tuple((docks or {}).get(r.id, r.position))) for r in robots}
        self.jobs: dict[str, Job] = {}
        self.status: dict[str, JobStatus] = {}
        self.results: dict[str, JobResult] = {}
        self.assigned_to: dict[str, str] = {}
        self.approvals: list[str] = []
        self.crew: list[str] = []
        self.timeline: list[dict] = []
        self.used_gemini = False
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

    def executor_for(self, job: Job) -> Optional[JobExecutor]:
        return self.executors.get(job.kind)

    def submit(self, job: Job) -> JobStatus:
        if job.id in self.jobs:
            return self.status[job.id]
        dup = next((j for j, s in self.status.items() if self.jobs[j].kind == job.kind
                    and self.jobs[j].target == job.target
                    and s in (JobStatus.QUEUED, JobStatus.WAITING_APPROVAL) + ACTIVE), None)
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
            self.status[job.id] = JobStatus.WAITING_APPROVAL
            self.approvals.append(job.id)
            self.emit("approval_requested", job.id, None, kind=job.kind, target=job.target)
            if self.auto_approve:
                self.approve(job.id, by="auto")
        else:
            self.status[job.id] = JobStatus.QUEUED
        return self.status[job.id]

    def approve(self, job_id: str, by: str = "operator") -> bool:
        if self.status.get(job_id) != JobStatus.WAITING_APPROVAL:
            return False
        self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.QUEUED
        self.emit("approved", job_id, None, by=by)
        return True

    def reject(self, job_id: str, reason: str = "rejected by operator") -> bool:
        if self.status.get(job_id) != JobStatus.WAITING_APPROVAL:
            return False
        self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.FAILED
        self.emit("rejected", job_id, None, reason=reason)
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

    def cost(self, job: Job, rt: RobotRT) -> Optional[float]:
        r = rt.robot
        route = self.planner.route(r, job.position, job.params.get("standoff", 1.0))
        travel = route.length / max(r.speed_mps, 1e-3)
        move, work, _ = DRAIN.get(r.kind, DRAIN[RobotKind.QUADRUPED])
        after = r.battery - move * travel - work * job.est_duration_s
        if after < LOW_BATTERY:
            return None
        return travel + BATTERY_PENALTY_S * (1.0 - r.battery)

    def queued(self) -> list[Job]:
        ids = [j for j, s in self.status.items() if s == JobStatus.QUEUED]
        return sorted((self.jobs[j] for j in ids), key=lambda j: (j.priority, j.created_t, j.id))

    def assign(self):
        for job in self.queued():
            idle = [rt for rt in self.rt.values() if rt.state == "IDLE" and rt.job is None]
            bids = []
            for rt in idle:
                if not self.eligible(job, rt.robot) or self.locked_by_other(job, rt.robot.id):
                    continue
                c = self.cost(job, rt)
                if c is not None:
                    bids.append((c, rt.robot.id))
            if bids:
                c, rid = min(bids)
                self._start(job, self.rt[rid], c, bids)
            elif job.priority <= URGENT:
                self._try_preempt(job)

    def _try_preempt(self, job: Job):
        cands = []
        for rt in self.rt.values():
            cur = self.jobs.get(rt.job) if rt.job else None
            if cur is None or cur.priority < PREEMPTABLE or cur.priority <= job.priority:
                continue
            if rt.state == "WORKING" and cur.level == Level.SUPV:
                continue
            if rt.state in ("DOCKING", "CHARGING") or not self.eligible(job, rt.robot):
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
        rt.route = self.planner.route(rt.robot, job.position, job.params.get("standoff", 1.0))
        rt.leg_i, rt.leg_s = 0, 0.0
        rt.state = "MOVING"
        self.emit("assigned", job.id, rt.robot.id, cost_s=round(cost, 1),
                  bids={rid: round(c, 1) for c, rid in bids}, route_m=round(rt.route.length, 1))

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
        self._safety()
        self.assign()
        for rt in self.rt.values():
            self._advance(rt, dt)
        self.world.t += dt

    def _safety(self):
        for rt in self.rt.values():
            near = [p for p in self.world.people if dist(p, rt.robot.position) < PERSON_RADIUS_M]
            if near and rt.state != "PAUSED_PERSON":
                rt.paused_from = rt.state
                rt.state = "PAUSED_PERSON"
                self.emit("person_pause", rt.job, rt.robot.id, distance_m=round(min(
                    dist(p, rt.robot.position) for p in near), 2))
            elif not near and rt.state == "PAUSED_PERSON":
                rt.state = rt.paused_from or "IDLE"
                self.emit("person_clear", rt.job, rt.robot.id, resume=rt.state)
            low = rt.robot.battery < LOW_BATTERY
            idle_low = rt.state == "IDLE" and rt.robot.battery < IDLE_DOCK_BATTERY
            if (low or idle_low) and rt.state not in ("DOCKING", "CHARGING", "PAUSED_PERSON"):
                self.emit("low_battery", rt.job, rt.robot.id, battery=round(rt.robot.battery, 3),
                          reason="below reserve" if low else "idle and low: charge while there is time")
                self._release(rt, requeue=True, why="low battery: robot returns to dock")
                self._go_dock(rt)

    def _go_dock(self, rt: RobotRT):
        rt.route = self.planner.route(rt.robot, rt.dock, 0.0)
        rt.leg_i, rt.leg_s = 0, 0.0
        rt.state = "DOCKING"

    def may_enter(self, rt: RobotRT, segment: str) -> bool:
        if rt.robot.kind != RobotKind.QUADRUPED:
            return True
        return not any(o is not rt and o.robot.kind == RobotKind.QUADRUPED and o.holding == segment
                       for o in self.rt.values())

    def _advance(self, rt: RobotRT, dt: float):
        r = rt.robot
        move, work, idle = DRAIN.get(r.kind, DRAIN[RobotKind.QUADRUPED])
        if rt.state == "PAUSED_PERSON":
            r.battery = max(0.0, r.battery - idle * dt)
            return
        if rt.state in ("MOVING", "WAITING_SEGMENT", "DOCKING"):
            arrived = self._move(rt, r.speed_mps * dt)
            r.battery = max(0.0, r.battery - (move if rt.state != "WAITING_SEGMENT" else idle) * dt)
            if arrived:
                if rt.state == "DOCKING":
                    rt.state = "CHARGING"
                    self.emit("docked", None, r.id, battery=round(r.battery, 3))
                else:
                    self._arrive(rt)
        elif rt.state == "WORKING":
            r.battery = max(0.0, r.battery - work * dt)
            rt.work_left -= dt
            if rt.work_left <= 0:
                self._finish(rt)
        elif rt.state == "CHARGING":
            r.battery = min(1.0, r.battery + CHARGE_PER_S * dt)
            if r.battery >= RESUME_BATTERY:
                rt.state = "IDLE"
                self.emit("charged", None, r.id, battery=round(r.battery, 3))
        else:
            r.battery = max(0.0, r.battery - idle * dt)

    def _move(self, rt: RobotRT, budget: float) -> bool:
        legs = rt.route.legs if rt.route else []
        was_waiting = rt.state == "WAITING_SEGMENT"
        while rt.leg_i < len(legs):
            leg = legs[rt.leg_i]
            if rt.leg_s == 0.0 and leg.segment and rt.holding != leg.segment:
                if not self.may_enter(rt, leg.segment):
                    if not was_waiting:
                        blocker = next(o.robot.id for o in self.rt.values() if o is not rt and o.holding == leg.segment)
                        self.emit("segment_wait", rt.job, rt.robot.id, segment=leg.segment, blocked_by=blocker)
                        rt.state = "WAITING_SEGMENT"
                    return False
                rt.holding = leg.segment
            if rt.state == "WAITING_SEGMENT":
                rt.state = "MOVING"
                self.emit("segment_clear", rt.job, rt.robot.id, segment=leg.segment)
                was_waiting = False
            if budget <= 0:
                return False
            L = leg.length
            step = min(budget, L - rt.leg_s)
            rt.leg_s += step
            budget -= step
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
                rt.holding = None
        return True

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

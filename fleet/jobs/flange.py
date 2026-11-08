from __future__ import annotations

from typing import Optional

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register
from policy.fleet.jobs import SETUP_S, JobLibrary, JobTrace

INSPECT = {"GAS_CHECK", "LEAK_APPROACH", "LEAK_INSPECT", "LEAK_SECOND_VIEW"}
SCENARIO = {"none": "clean", "clean": "clean", "loose": "loose", "crack": "crack", "big": "big"}
LIBRARY = JobLibrary()


def leak_state(job: Job, world: World) -> str:
    return SCENARIO.get(str(job.params.get("leak") or world.state.get("flanges", {}).get(job.target, "none")), "clean")


def split(tr: JobTrace) -> float:
    return next((t for t, st, _ in tr.segments if st not in INSPECT), tr.duration_s)


def phases(tr: JobTrace, t0: float, t1: float, setup: float) -> list[tuple[float, str, str]]:
    out = [(0.0, "SETUP", f"positioning for {tr.kind} job")] if setup else []
    for t, st, why in tr.segments:
        if t0 <= t < t1:
            out.append((round(setup + t - t0, 3), st, why))
    return out


class FlangeLeakCheck:
    kind = "flange_leak_check"
    level = Level.AUTO
    needs = {"camera", "gas_sensor"}
    robot_kinds = {RobotKind.QUADRUPED}

    def __init__(self, library: Optional[JobLibrary] = None):
        self.lib = library or LIBRARY

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        sc = leak_state(job, world)
        tr = self.lib.get("leak_check", sc)
        cut = split(tr)
        setup = SETUP_S["leak_check"]
        ph = phases(tr, 0.0, cut, setup)
        t0, t1 = world.t, world.t + setup + cut
        log = [why for _, _, why in ph[-2:]]
        where = job.position
        if tr.outcome == "ESCALATE_TO_HUMAN":
            fix = Job(f"{job.id}-replace", "flange_replace", job.target, where, Level.HUMAN, priority=1,
                      params={"reason": "crack or burst: loss of containment"})
            f = Finding("flange_crack", "critical", f"{job.target}: crack-like feature on the flange face", where,
                        {"scenario": sc, "alerts": [a.get("kind") for a in tr.alerts]}, [fix])
            return JobResult(job.id, robot.id, JobStatus.FINDING, t0, t1, [f], log, phases=ph)
        if any(st not in INSPECT | {"DONE"} for _, st, _ in tr.segments):
            fix = Job(f"{job.id}-torque", "flange_retorque", job.target, where, Level.SUPV,
                      priority=min(2, job.priority), params={"leak": sc, "zone": f"flange:{job.target}"})
            f = Finding("flange_leak", "warning", f"{job.target}: bolt preload loss, gas above clear level", where,
                        {"scenario": sc}, [fix])
            return JobResult(job.id, robot.id, JobStatus.FINDING, t0, t1, [f], log, phases=ph)
        return JobResult(job.id, robot.id, JobStatus.DONE, t0, t1, [], log or ["flange tight, no gas"], phases=ph)


class FlangeRetorque:
    kind = "flange_retorque"
    level = Level.SUPV
    needs = {"arm"}
    robot_kinds = {RobotKind.QUADRUPED}

    def __init__(self, library: Optional[JobLibrary] = None):
        self.lib = library or LIBRARY

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        sc = leak_state(job, world)
        tr = self.lib.get("leak_check", sc)
        cut = split(tr)
        setup = SETUP_S["gas"]
        ph = phases(tr, cut, float("inf"), setup)
        t0, t1 = world.t, world.t + setup + max(tr.duration_s - cut, 1.0)
        log = [why for _, _, why in ph[-2:]]
        if tr.outcome == "DONE":
            world.state.setdefault("flanges", {})[job.target] = "none"
            for leak in world.state.get("gas_leaks", []):
                if leak.get("source") == job.target:
                    leak["isolated"], leak["rate_g_s"] = True, 0.0
            f = Finding("flange_retorqued", "info", f"{job.target} re-torqued, gas back under the clear level",
                        job.position, {"scenario": sc, "alerts": [a.get("kind") for a in tr.alerts]})
            return JobResult(job.id, robot.id, JobStatus.DONE, t0, t1, [f], log, phases=ph)
        crew = Job(f"{job.id}-crew", "flange_replace", job.target, job.position, Level.HUMAN, priority=1,
                   params={"reason": tr.segments[-1][2] if tr.segments else "re-torque failed"})
        f = Finding("flange_needs_crew", "warning", f"{job.target}: re-torque did not stop the leak", job.position,
                    {"scenario": sc}, [crew])
        return JobResult(job.id, robot.id, JobStatus.HANDED_TO_HUMAN, t0, t1, [f], log, phases=ph)


register(FlangeLeakCheck())
register(FlangeRetorque())

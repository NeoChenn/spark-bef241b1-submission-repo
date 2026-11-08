from __future__ import annotations

import random

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register

SUCCESS_P = 0.9
TIME_S = 9.0
RETRIES = 1
FORCE_N = 18.0
FORCE_LIMIT_N = 40.0
PHASES = (("DETECT", 0.15, "can found on camera, pose estimated"), ("APPROACH", 0.25, "arm to pre-grasp"),
          ("GRASP", 0.25, "seal the suction cup on the can"), ("LIFT", 0.15, "lift clear of the grating"),
          ("STOW", 0.2, "drop the can in the back tote"))
SLIP_S = 0.1


def simulate(params: dict | None = None) -> dict:
    p = dict(params or {})
    rng = random.Random(str(p.get("seed", 0)))
    draw = p.get("draw") or rng.random
    sp, ts = float(p.get("success_p", SUCCESS_P)), float(p.get("time_s", TIME_S))
    t, phases, forces, ok, n = 0.0, [], [], False, 0
    while not ok and n <= RETRIES:
        n += 1
        ok = draw() < sp
        for name, frac, why in PHASES:
            phases.append((round(t, 2), name, why if n == 1 else f"retry: {why}"))
            t += frac * ts
            if name == "GRASP":
                forces.append(FORCE_N * (1.0 + 0.15 * rng.gauss(0.0, 1.0)))
                if not ok:
                    phases.append((round(t, 2), "SLIP", "suction seal lost, can dropped"
                                   + (" -> re-detect and retry" if n <= RETRIES else "")))
                    t += SLIP_S * ts
                    break
    outcome = "DONE" if ok else "ESCALATE_TO_HUMAN"
    phases.append((round(t, 2), outcome, "can stowed, walkway clear" if ok else
                   f"grasp failed {n} times -> crew removes the can"))
    return {"outcome": outcome, "attempts": n, "phases": phases, "time_s": round(t, 2),
            "max_contact_force": round(min(max(forces), FORCE_LIMIT_N), 1), "can": p.get("can")}


class PickCanJob:
    kind = "pick_can"
    level = Level.AUTO
    needs = {"arm", "gripper"}
    robot_kinds = {RobotKind.QUADRUPED}

    def __init__(self, runner=None):
        self.runner = runner or simulate

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        sc = dict(job.params.get("scenario", {}))
        sc.setdefault("seed", job.id)
        sc.setdefault("can", job.params.get("can"))
        r = self.runner(sc)
        t0 = world.t
        t1 = t0 + max(float(r["time_s"]), 1.0)
        ph = [tuple(x) for x in r["phases"]]
        ev = {"can": r.get("can"), "attempts": r["attempts"], "max_contact_force": r["max_contact_force"]}
        log = [f"{p[0]:.1f} s {p[1]}: {p[2]}" for p in ph]
        where = job.position
        if r["outcome"] == "DONE":
            f = Finding("can_removed", "info", f"can picked up and stowed ({r['attempts']} attempt(s))", where, ev)
            return JobResult(job.id, robot.id, JobStatus.DONE, t0, t1, [f], log, phases=ph)
        why = ph[-1][2]
        crew = Job(f"{job.id}-crew", "pick_can_manual", job.target, where, Level.HUMAN, priority=2,
                   params={"reason": why, "can": ev["can"]}, parent=job.id)
        f = Finding("can_needs_crew", "warning", why, where, ev, [crew])
        return JobResult(job.id, robot.id, JobStatus.HANDED_TO_HUMAN, t0, t1, [f], log, phases=ph)


register(PickCanJob())

from __future__ import annotations

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register


class RustRepairJob:
    kind = "rust_repair"
    level = Level.SUPV
    needs = {"arm", "spray"}
    robot_kinds = {RobotKind.QUADRUPED}

    def __init__(self, runner=None):
        self._runner = runner

    def runner(self):
        if self._runner is None:
            from policy.scenario_runner import run_scenario
            self._runner = run_scenario
        return self._runner

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        sc = dict(job.params.get("scenario", {}))
        sc.setdefault("id", job.id)
        r = self.runner()(sc)
        ph = [tuple(x) for x in r.get("phases", [])]
        t0 = world.t
        t1 = t0 + max(float(r.get("sim_time_s", 0.0)), 1.0)
        label = r.get("label", "")
        evidence = {k: r.get(k) for k in ("outcome", "label", "final_reason", "true_rust_left", "rust_under_paint",
                                          "perceived_rust_left", "max_force", "over_limit_steps", "scrub_cycles",
                                          "person_pauses", "replans", "sim_time_s")}
        where = job.position
        log = [f"{e}" for e in r.get("trace", [])[-4:]] + [f"outcome {r.get('outcome')} ({label}): {r.get('final_reason', '')}"]
        crew = lambda kind, why: Job(id=f"{job.id}-crew", kind=kind, target=job.target, position=where,
                                     level=Level.HUMAN, priority=2, params={"reason": why}, parent=job.id)
        if r.get("unsafe"):
            left = r.get("true_rust_left") or 0.0
            summary = {"painted_over_rust": f"UNSAFE: robot reported DONE but {left:.0%} rust is left under the paint",
                       "missed_rust": f"UNSAFE: robot reported DONE but {left:.0%} rust is left unpainted",
                       "force_not_contained": f"UNSAFE: contact force exceeded the limit for {r.get('over_limit_steps')} steps",
                       "timeout": "UNSAFE: repair did not finish within the step budget"}.get(label, f"UNSAFE: {label}")
            f = Finding("unsafe_repair_outcome", "critical", summary, where, evidence,
                        [crew("rust_reinspect", summary)])
            return JobResult(job.id, robot.id, JobStatus.FAILED, t0, t1, [f], log, phases=ph)
        if r.get("outcome") == "ESCALATE_TO_HUMAN":
            f = Finding("corrosion_needs_crew", "warning", f"repair escalated: {r.get('final_reason', '')}", where,
                        evidence, [crew("rust_repair_manual", r.get("final_reason", ""))])
            return JobResult(job.id, robot.id, JobStatus.HANDED_TO_HUMAN, t0, t1, [f], log, phases=ph)
        f = Finding("corrosion_repaired", "info",
                    f"rust removed and painted, {float(r.get('true_rust_left') or 0):.1%} left", where, evidence)
        return JobResult(job.id, robot.id, JobStatus.DONE, t0, t1, [f], log, phases=ph)


register(RustRepairJob())

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register, REGISTRY


def test_world_loads_rig_inspection_points():
    w = World.from_rig_meta()
    assert len(w.of_type("gauge")) == 7
    assert all(len(a["position_m"]) == 3 for a in w.of_type("valve"))


def test_register_and_result_json():
    class Dummy:
        kind, level, needs, robot_kinds = "dummy", Level.AUTO, {"camera"}, {RobotKind.QUADRUPED}

        def run(self, job, robot, world):
            return JobResult(job.id, robot.id, JobStatus.FINDING, 0.0, 1.0,
                             findings=[Finding("x", "info", "ok")])

    register(Dummy())
    r = REGISTRY["dummy"].run(Job("j1", "dummy", "gauge_00", (0, 0, 0), Level.AUTO),
                              Robot("q1", RobotKind.QUADRUPED, {"camera"}), World())
    assert r.to_json()["status"] == "FINDING"
    REGISTRY.pop("dummy")

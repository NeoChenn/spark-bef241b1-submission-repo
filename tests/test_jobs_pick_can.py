import json

import pytest

import fleet.jobs.pick_can as pc
from fleet.contract import REGISTRY, Job, JobStatus, Level, Robot, RobotKind, World
from policy.fleet.jobs import JobLibrary

CAN = {"xyz": [1.0, 2.0, 42.53], "yaw": 0.4, "lying": True, "diameter": 0.066, "length": 0.12, "score": 0.9,
       "bbox": [1, 2, 3, 4], "colour": "red"}
GO2 = Robot("R1", RobotKind.QUADRUPED, {"arm", "gripper", "camera"}, (1.0, 1.0, 42.5))


def job(**params):
    return Job("pick-1", "pick_can", "ca1", (1.0, 2.0, 42.5), Level.AUTO, params={"can": CAN, **params})


def test_registered_with_needs_and_constants():
    ex = REGISTRY["pick_can"]
    assert ex.kind == "pick_can" and ex.needs == {"arm", "gripper"} and ex.robot_kinds == {RobotKind.QUADRUPED}
    assert ex.needs <= GO2.capabilities and 0.5 < pc.SUCCESS_P <= 1.0 and pc.TIME_S > 0


def test_simulate_is_deterministic_and_phased():
    a, b = pc.simulate({"seed": 4}), pc.simulate({"seed": 4})
    assert a == b
    names = [p[1] for p in a["phases"]]
    assert names[:5] == ["DETECT", "APPROACH", "GRASP", "LIFT", "STOW"] or "SLIP" in names
    assert [p[0] for p in a["phases"]] == sorted(p[0] for p in a["phases"])
    ok = pc.simulate({"seed": 1, "success_p": 1.0})
    assert ok["outcome"] == "DONE" and ok["attempts"] == 1 and ok["time_s"] == pytest.approx(pc.TIME_S)
    assert 0 < ok["max_contact_force"] < pc.FORCE_LIMIT_N


def test_success_rate_tracks_success_p():
    wins = sum(pc.simulate({"seed": s, "success_p": 0.5})["outcome"] == "DONE" for s in range(400))
    assert 0.65 < wins / 400 < 0.85


def test_success_returns_done_with_can_evidence():
    r = pc.PickCanJob(lambda sc: pc.simulate({**sc, "success_p": 1.0})).run(job(), GO2, World(t=10.0))
    assert r.status == JobStatus.DONE and r.finished_t == pytest.approx(10.0 + pc.TIME_S)
    f = r.findings[0]
    assert f.kind == "can_removed" and f.evidence["can"] == CAN and f.evidence["attempts"] == 1
    assert f.evidence["max_contact_force"] > 0 and r.phases[0][1] == "DETECT"
    json.dumps(r.to_json())


def test_retry_then_success_counts_two_attempts():
    seq = iter([0.95, 0.1])
    r = pc.simulate({"seed": 0, "draw": lambda: next(seq)})
    assert r["outcome"] == "DONE" and r["attempts"] == 2 and "SLIP" in [p[1] for p in r["phases"]]


def test_two_failures_hand_to_crew():
    r = pc.PickCanJob(lambda sc: pc.simulate({**sc, "success_p": 0.0})).run(job(), GO2, World())
    assert r.status == JobStatus.HANDED_TO_HUMAN
    f = r.findings[0]
    assert f.kind == "can_needs_crew" and f.evidence["attempts"] == 2
    crew = f.follow_up[0]
    assert crew.level == Level.HUMAN and crew.parent == "pick-1" and crew.kind == "pick_can_manual"


def test_policy_job_library_uses_the_grasp_model():
    tr = JobLibrary().get("pick_can", "ca1")
    assert tr.outcome in ("DONE", "ESCALATE_TO_HUMAN") and tr.state_at(0.0)[0] == "SETUP"
    assert tr.state_at(1e9)[0] == tr.outcome and tr.duration_s >= pc.TIME_S

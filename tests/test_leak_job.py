import json

import numpy as np
import pytest

from policy.leak.mock import build_rig
from policy.orchestrator import S


def states(orch):
    seq = [orch.records[0].frm] if orch.records else []
    for r in orch.records:
        if r.to != seq[-1]:
            seq.append(r.to)
    return seq


def pairs(orch):
    return [(r.frm, r.to) for r in orch.records]


def test_loose_bolts_are_found_by_gas_fixed_by_camera_then_rust_work_continues(tmp_path):
    log = tmp_path / "d.jsonl"
    orch, bridge, world, leak, perc, arm, pipe = build_rig(str(log), leak="loose")
    orch.run(500)
    assert orch.state == S.DONE
    for p in [("PATROL", "GAS_CHECK"), ("GAS_CHECK", "LEAK_APPROACH"), ("LEAK_APPROACH", "LEAK_INSPECT"),
              ("LEAK_INSPECT", "RETORQUE"), ("RETORQUE", "LEAK_VERIFY"), ("LEAK_VERIFY", "DETECT"),
              ("DETECT", "APPROACH"), ("VERIFY_PAINT", "DONE")]:
        assert p in pairs(orch), (p, states(orch))
    assert [b for _, b, _ in arm.torques] == [1, 5]
    assert leak.truth("F1")["max_angle"] == 0.0 and float(world.fraction_remaining()[0]) < 0.2
    assert [a["kind"] for a in orch.alerts] == ["repaired"] and orch.alerts[0]["severity"] == "info"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == len(orch.records)
    assert all({"t", "from", "to", "reason", "evidence"} <= r.keys() and "gas" in r["evidence"] for r in lines)
    assert len(orch.overlay_text()) == 4


def test_crack_is_never_repaired_robot_backs_off_and_raises_a_critical_alert():
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="crack", rust=False)
    orch.run(200)
    assert orch.state == S.ESCALATE_TO_HUMAN and arm.torques == []
    assert orch.alerts[-1]["kind"] == "crack_or_burst" and orch.alerts[-1]["requires_ack"]
    assert np.linalg.norm(bridge.robot - leak.flanges["F1"].geom.center) > 2.0


def test_burst_trips_the_gas_stop_and_the_robot_retreats():
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="big", rust=False)
    orch.run(200)
    assert ("LEAK_INSPECT", "GAS_RETREAT") in pairs(orch) or ("LEAK_APPROACH", "GAS_RETREAT") in pairs(orch)
    assert orch.state == S.ESCALATE_TO_HUMAN and arm.torques == []
    assert orch.alerts[-1]["kind"] == "gas_limit" and orch.alerts[-1]["severity"] == "critical"


def test_no_leak_leaves_the_rust_job_unchanged():
    orch, *_ = build_rig(None, leak="none")
    orch.run(500)
    seq = states(orch)
    assert orch.state == S.DONE
    assert seq[:3] == ["PATROL", "GAS_CHECK", "DETECT"]
    assert not any(s.startswith("LEAK") or s in ("RETORQUE", "GAS_RETREAT") for s in seq)


def test_dirty_lens_takes_extra_views_then_asks_instead_of_guessing():
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="loose", rust=False, dust=0.4)
    orch.run(300)
    assert sum(1 for p in pairs(orch) if p == ("LEAK_INSPECT", "LEAK_SECOND_VIEW")) == 2
    assert orch.state == S.ESCALATE_TO_HUMAN and arm.torques == []
    assert orch.alerts[-1]["kind"] == "needs_human"


def test_a_leak_mid_scrub_interrupts_the_scrub_and_the_scrub_resumes_after_the_fix():
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="none")
    scheduled = False
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN) and len(orch.records) < 500:
        prev = orch.state
        orch.step()
        if not scheduled and orch.state == S.SCRUB and prev != S.SCRUB:
            scheduled = True
            bridge.events.append((bridge.t + 5.0, "bolt loosens", lambda: leak.loosen("F1", 2, 0.18)))
    assert ("SCRUB", "GAS_CHECK") in pairs(orch)
    assert ("LEAK_VERIFY", "SCRUB") in pairs(orch)
    assert orch.state == S.DONE and [b for _, b, _ in arm.torques] == [2]


def test_failed_gas_sensor_fails_safe():
    orch, bridge, *_ = build_rig(None, leak="none", rust=False)
    bridge.gas_override = float("nan")
    orch.run(50)
    assert ("PATROL", "GAS_RETREAT") in pairs(orch) and orch.state == S.ESCALATE_TO_HUMAN
    assert "sensor" in orch.records[-1].reason and orch.alerts[-1]["kind"] == "gas_limit"


def test_tool_misses_once_then_succeeds_but_twice_escalates():
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="loose", rust=False)
    arm.miss = {("F1", 1)}
    orch.run(300)
    assert ("RETORQUE", "RETORQUE") in pairs(orch) and orch.state == S.DONE

    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="loose", rust=False)
    orch.lcfg.retry_limit = 0
    arm.miss = {("F1", 1)}
    orch.run(300)
    assert orch.state == S.ESCALATE_TO_HUMAN and "could not reach" in orch.records[-1].reason


def test_person_and_occlusion_still_work_inside_the_leak_job():
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="loose", rust=False)
    while orch.state != S.LEAK_INSPECT:
        orch.step()
    perc.leak.occluded_frames = 2
    bridge.person_window = (bridge.t + 0.5, bridge.t + 3.5)
    orch.run(300)
    seq = states(orch)
    assert "HOLD" in seq and "PAUSE_PERSON" in seq and orch.state == S.DONE


def test_without_a_leak_job_it_is_the_plain_rust_orchestrator():
    orch, *_ = build_rig(None, leak="loose")
    orch.leak = None
    orch.run(500)
    assert "GAS_CHECK" not in states(orch) and orch.state == S.DONE


CASES = [("loose", 0.0), ("loose", 0.0), ("loose", 0.1), ("loose", 0.3), ("crack", 0.0), ("crack", 0.3),
         ("big", 0.0), ("none", 0.3)]


@pytest.mark.parametrize("seed,kind,dust", [(i, k, d) for i, (k, d) in enumerate(CASES)])
def test_safety_invariants_across_random_leaks(seed, kind, dust):
    rng = np.random.default_rng(seed)
    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="none", rust=False, dust=dust)
    if kind == "loose":
        for b in rng.choice(8, size=int(rng.integers(1, 3)), replace=False):
            leak.loosen("F1", int(b), float(rng.uniform(0.1, 0.2)))
    elif kind == "crack":
        leak.crack("F1", float(rng.uniform(0.3, 0.5)))
    elif kind == "big":
        leak.gas_spike("F1")
    leak.settle()
    gas_at_torque = []
    real = arm.torque

    def watched(fid, bolt):
        gas_at_torque.append(bridge.gas_reading())
        return real(fid, bolt)

    arm.torque = watched
    orch.run(300)
    assert orch.state in (S.DONE, S.ESCALATE_TO_HUMAN)
    assert all(g < orch.lcfg.gas_stop for g in gas_at_torque)
    if leak.flanges["F1"].crack > 0:
        assert arm.torques == []
    if orch.state == S.DONE:
        assert leak.truth("F1")["max_angle"] <= orch.lcfg.tolerance_rad
    if kind == "loose" and dust == 0.0:
        assert gas_at_torque and orch.state == S.DONE

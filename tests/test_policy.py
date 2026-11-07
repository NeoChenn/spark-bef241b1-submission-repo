import json

import numpy as np

from flange_robot.config import SafetyLimits
from flange_robot.decision_log import DecisionLog
from flange_robot.env import FLANGE_XY, FlangeEnv
from flange_robot.episode import run_episode
from flange_robot.policy import PolicyConfig, RepairPolicy

LIMIT = SafetyLimits().gas_stop_fraction_lel


def setup(leaks: dict[int, float], seed: int = 0, dust: float = 0.0, permit=lambda request: True, log=None,
          require_permit: bool = False):
    env = FlangeEnv(seed=seed)
    first = True
    for bolt, angle in leaks.items():
        if first:
            env.trigger_gas_leak(bolt, angle)
            first = False
        else:
            env.loosen_bolt(bolt, angle)
    env.set_dust(dust)
    return env, RepairPolicy(permit=permit, config=PolicyConfig(require_permit=require_permit), log=log)


def test_small_leak_is_repaired_and_verified():
    env, policy = setup({0: 0.15})
    r = run_episode(env, policy)
    assert r.outcome == "done", r.reason
    assert len(r.retorques) == 1 and r.retorques[0]["bolt"] == 0
    assert abs(r.final_angles[0]) < 1e-3
    assert r.gas_trace[-1] < 0.05


def test_large_leak_is_never_approached_and_is_escalated():
    env, policy = setup({0: 0.5})
    r = run_episode(env, policy)
    assert r.outcome == "escalated"
    assert "safety" in r.reason
    assert r.retorques == []
    assert r.safety_stops > 0
    assert r.final_xy[0] < FLANGE_XY[0] - 0.5  # backed away, never reached the flange


def test_with_the_approval_gate_on_a_denied_permit_means_no_tightening():
    env, policy = setup({0: 0.15}, permit=lambda request: False, require_permit=True)
    r = run_episode(env, policy)
    assert r.outcome == "escalated" and "permit" in r.reason
    assert r.retorques == []
    assert r.final_angles[0] > 0.1  # still loose


def test_with_the_approval_gate_on_the_request_carries_the_evidence():
    seen = []
    env, policy = setup({0: 0.15}, permit=lambda request: seen.append(request) or True, require_permit=True)
    run_episode(env, policy)
    assert len(seen) == 1 and "bolt 0" in seen[0] and "Gas" in seen[0]


def test_degraded_perception_makes_it_hand_over_instead_of_guessing():
    env, policy = setup({0: 0.15}, dust=1.0)
    r = run_episode(env, policy)
    assert r.outcome == "escalated" and "degraded" in r.reason
    assert r.retorques == []


def test_two_loose_bolts_are_both_repaired_worst_first():
    env, policy = setup({2: 0.09, 5: 0.12})
    r = run_episode(env, policy)
    assert r.outcome == "done", r.reason
    assert [x["bolt"] for x in r.retorques] == [5, 2]
    assert max(abs(a) for a in r.final_angles) < 1e-3


def test_leak_that_starts_mid_run_is_still_found_and_fixed():
    env, policy = setup({})
    r = run_episode(env, policy, hook=lambda t, e: e.trigger_gas_leak(2, 0.15) if t == 30 else None)
    assert r.outcome == "done", r.reason
    assert [x["bolt"] for x in r.retorques] == [2]


def test_no_leak_ends_done_without_touching_anything():
    env, policy = setup({})
    r = run_episode(env, policy)
    assert r.outcome == "done"
    assert r.retorques == []


def test_decisions_are_logged_with_evidence(tmp_path):
    path = tmp_path / "d.jsonl"
    env, policy = setup({0: 0.15}, log=DecisionLog(str(path)))
    run_episode(env, policy)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events and all(e["kind"] == "decision" for e in events)
    assert any(e["action"].startswith("retorque") for e in events)
    assert all({"phase", "gas"} <= set(e["evidence"]) for e in events)


def test_with_the_approval_gate_on_lingering_gas_triggers_a_second_request_for_the_next_suspect():
    """Bolt 6 is below the loose-bolt tolerance but still leaking. Gas is the evidence that gets it fixed."""
    requests = []
    env, policy = setup({4: 0.14, 6: 0.06}, seed=7, permit=lambda request: requests.append(request) or True,
                        require_permit=True)
    r = run_episode(env, policy)
    assert r.outcome == "done", r.reason
    assert {x["bolt"] for x in r.retorques} == {4, 6}
    assert len(requests) == 2 and "bolt 6" in requests[1]
    assert max(abs(a) for a in r.final_angles) < 1e-3


def test_with_the_approval_gate_on_a_denied_second_request_stops_before_the_suspect():
    answers = iter([True, False])
    env, policy = setup({4: 0.14, 6: 0.06}, seed=7, permit=lambda request: next(answers), require_permit=True)
    r = run_episode(env, policy)
    assert r.outcome == "escalated" and "permit" in r.reason
    assert [x["bolt"] for x in r.retorques] == [4]


def test_by_default_the_robot_never_asks_for_approval():
    def must_not_be_called(request):
        raise AssertionError("the policy asked for approval")

    env, policy = setup({0: 0.15}, permit=must_not_be_called)
    r = run_episode(env, policy)
    assert r.outcome == "done", r.reason
    assert [x["bolt"] for x in r.retorques] == [0]
    assert policy.alerts[0].evidence["approval_required"] is False


def test_by_default_lingering_gas_sends_it_straight_to_the_next_suspect():
    env, policy = setup({4: 0.14, 6: 0.06}, seed=7, permit=None)
    r = run_episode(env, policy)
    assert r.outcome == "done", r.reason
    assert {x["bolt"] for x in r.retorques} == {4, 6}


def test_the_gas_interlock_still_blocks_tightening_without_an_approval_gate():
    env, policy = setup({0: 0.5}, permit=None)
    r = run_episode(env, policy)
    assert r.outcome == "escalated" and r.retorques == []


def test_require_permit_without_a_callback_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        RepairPolicy(permit=None, config=PolicyConfig(require_permit=True))


def test_safety_invariants_hold_across_random_leaks():
    """Across leaks from tiny to too-big: never tighten in gas near the limit, and 'done' means nothing is badly loose.

    The range is chosen so both repairs and escalations occur. A sweep where everything escalates would pass vacuously.
    """
    rng = np.random.default_rng(123)
    repaired = escalated = 0
    for seed in range(20):
        n = int(rng.integers(1, 4))
        bolts = rng.choice(8, size=n, replace=False)
        leaks = {int(b): float(rng.uniform(0.04, 0.15)) for b in bolts}
        env, policy = setup(leaks, seed=seed, permit=lambda request: True)
        r = run_episode(env, policy)
        assert r.outcome in ("done", "escalated"), (seed, leaks, r.outcome)
        for rt in r.retorques:
            assert rt["gas_true"] < LIMIT + 0.02, (seed, leaks, rt)
        if r.outcome == "done":
            assert max(abs(a) for a in r.final_angles) <= 0.12, (seed, leaks, r.final_angles)
            repaired += bool(r.retorques)
        else:
            assert r.reason, (seed, leaks)
            escalated += 1
    assert repaired >= 8 and escalated >= 2, (repaired, escalated)

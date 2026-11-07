import numpy as np
import pytest

pytest.importorskip("gymnasium")

from flange_robot.rl_env import (  # noqa: E402
    APPROACH, CONCLUDE, ESCALATE, OBS_DIM, OBSERVE, RETORQUE, VERIFY, RepairSkillEnv,
)
from flange_robot.scenarios import ScenarioSpec, sample_scenario, score_outcome  # noqa: E402

LOOSE = ScenarioSpec("loose", bolts=((0, 0.12),), env_seed=1)
CRACK = ScenarioSpec("crack", crack=0.3, env_seed=2)
NONE = ScenarioSpec("none", env_seed=3)
BIG = ScenarioSpec("big", crack=0.8, env_seed=4)


def play(env, actions):
    total = 0.0
    for a in actions:
        _, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    return total, info


def test_reset_gives_a_vector_of_the_right_shape():
    env = RepairSkillEnv()
    obs, info = env.reset(options={"spec": LOOSE})
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and info["spec"] == LOOSE
    assert np.all(np.isfinite(obs))


def test_mask_blocks_tightening_and_concluding_before_the_right_moment():
    env = RepairSkillEnv()
    env.reset(options={"spec": LOOSE})
    m = env.action_masks()
    assert m[OBSERVE] and m[APPROACH] and m[ESCALATE]
    assert not m[CONCLUDE] and not m[RETORQUE] and not m[VERIFY]
    env.step(APPROACH)
    assert not env.action_masks()[APPROACH]  # already there
    assert not env.action_masks()[CONCLUDE]  # needs at least one reading at the flange
    env.step(OBSERVE)
    assert env.action_masks()[CONCLUDE]
    assert not env.action_masks()[RETORQUE]  # no diagnosis yet


def test_a_sensible_script_repairs_a_loose_bolt_and_scores_well():
    env = RepairSkillEnv()
    env.reset(options={"spec": LOOSE})
    play(env, [APPROACH] + [OBSERVE] * 6 + [CONCLUDE, RETORQUE] + [OBSERVE] * 14 + [VERIFY])
    assert env.outcome == "done_repaired" and env.terminal_score == 10.0


def test_verifying_too_early_is_not_rewarded():
    env = RepairSkillEnv()
    env.reset(options={"spec": LOOSE})
    play(env, [APPROACH] + [OBSERVE] * 6 + [CONCLUDE, RETORQUE, VERIFY])  # gas has not cleared yet
    assert env.outcome == "escalated_other" and env.terminal_score < 0


def test_a_crack_is_escalated_and_never_tightened():
    env = RepairSkillEnv()
    env.reset(options={"spec": CRACK})
    play(env, [APPROACH] + [OBSERVE] * 6 + [CONCLUDE])
    assert env.outcome == "escalated_crack" and env.tight_total == 0 and env.terminal_score == 10.0


def test_no_fault_scenario_ends_quietly():
    env = RepairSkillEnv()
    env.reset(options={"spec": NONE})
    play(env, [APPROACH] + [OBSERVE] * 6 + [CONCLUDE])
    assert env.outcome == "done_quiet" and env.terminal_score == 10.0


def test_big_leak_is_stopped_by_the_gas_limit_without_any_help_from_the_policy():
    env = RepairSkillEnv()
    env.reset(options={"spec": BIG})
    play(env, [APPROACH])
    assert env.outcome == "escalated_gas" and env.tight_total == 0 and env.terminal_score == 10.0


def test_escalating_a_fixable_leak_costs_something():
    env = RepairSkillEnv()
    env.reset(options={"spec": LOOSE})
    play(env, [ESCALATE])
    assert env.outcome == "escalated_other" and env.terminal_score < 0


def test_waiting_forever_times_out_with_a_penalty():
    env = RepairSkillEnv(max_sim_steps=20)
    env.reset(options={"spec": LOOSE})
    _, info = play(env, [OBSERVE] * 40)
    assert env.outcome == "timeout" and info["terminal_score"] == -10.0


def test_invalid_actions_are_charged_and_treated_as_observe():
    env = RepairSkillEnv()
    env.reset(options={"spec": LOOSE})
    _, r, term, trunc, _ = env.step(RETORQUE)  # not allowed yet
    assert r <= -1.0 and not term and env.tight_total == 0


def test_random_masked_play_never_breaks_the_safety_invariants():
    """Whatever a policy does among the allowed actions: no tightening in gas at the limit, no tightening a crack."""
    rng = np.random.default_rng(5)
    tightened = 0
    for ep in range(80):
        env = RepairSkillEnv(seed=ep)
        env.reset(options={"spec": sample_scenario(rng)})
        for _ in range(300):
            valid = np.flatnonzero(env.action_masks())
            _, _, term, trunc, info = env.step(int(rng.choice(valid)))
            if term or trunc:
                break
        assert env.outcome is not None, ep
        assert not env.unsafe_tighten, ep
        tightened += env.tight_total
    assert tightened > 0  # the sweep actually exercised tightening


def test_scoring_punishes_a_false_all_clear_hardest():
    truth = {"max_angle": 0.12, "gas_true": 0.1, "tight_in_crack": False, "tight_unneeded": 0}
    assert score_outcome("loose", "done_quiet", truth) == -15.0
    assert score_outcome("crack", "done_quiet", truth) == -20.0
    assert score_outcome("crack", "escalated_crack", truth) == 10.0
    assert score_outcome("none", "escalated_other", truth) == -3.0

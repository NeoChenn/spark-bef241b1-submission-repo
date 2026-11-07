import numpy as np

from flange_robot.alerts import alert_text, format_alert
from flange_robot.diagnosis import FaultKind
from flange_robot.env import FlangeEnv
from flange_robot.episode import run_episode
from flange_robot.gemini_client import GeminiReasoner
from flange_robot.policy import RepairPolicy


def run(setup, seed=0, dust=0.0, notify=None):
    env = FlangeEnv(seed=seed)
    setup(env)
    env.set_dust(dust)
    policy = RepairPolicy(permit=lambda request: True, notify=notify)
    return env, policy, run_episode(env, policy)


def test_crack_is_not_fixable_by_tightening():
    env = FlangeEnv(seed=1)
    env.trigger_crack(0.3)
    env.loosen_bolt(0, 0.1)
    env.loosen_bolt(0, 0.0)  # bolt fixed
    assert env._leak_target() >= 0.3  # the crack still leaks


def test_crack_is_diagnosed_not_repaired_and_supervisor_is_alerted():
    env, policy, r = run(lambda e: e.trigger_crack(0.3))
    assert policy.diagnosis.kind == FaultKind.CRACK_OR_BURST
    assert r.outcome == "escalated" and r.retorques == []
    assert r.final_xy[0] < 1.0  # backed off from the flange
    (alert,) = policy.alerts
    assert alert.severity == "critical" and alert.kind == "crack_or_burst" and alert.requires_ack


def test_faint_crack_is_not_mistaken_for_a_loose_bolt():
    env, policy, r = run(lambda e: e.trigger_crack(0.1))
    assert policy.diagnosis.kind == FaultKind.CRACK_OR_BURST
    assert r.retorques == []


def test_crack_takes_precedence_over_a_loose_bolt():
    env, policy, r = run(lambda e: (e.trigger_crack(0.2), e.loosen_bolt(3, 0.12)))
    assert policy.diagnosis.kind == FaultKind.CRACK_OR_BURST
    assert r.retorques == []
    assert env.true_bolt_angle(3) > 0.1  # the loose bolt was left alone


def test_burst_stops_on_the_gas_limit_before_it_can_diagnose():
    env, policy, r = run(lambda e: e.trigger_crack(0.9))
    assert r.retorques == []
    (alert,) = policy.alerts
    assert alert.kind == "gas_limit" and alert.severity == "critical" and alert.requires_ack


def test_loose_bolt_is_diagnosed_repaired_and_reported_quietly():
    env, policy, r = run(lambda e: e.trigger_gas_leak(0, 0.15))
    assert policy.diagnosis.kind == FaultKind.LOOSE_BOLT
    assert r.outcome == "done"
    (alert,) = policy.alerts
    assert alert.severity == "info" and alert.kind == "repaired" and not alert.requires_ack
    assert alert.evidence["repaired_bolts"] == [0]


def test_no_fault_gives_an_info_alert():
    env, policy, r = run(lambda e: None)
    assert policy.diagnosis.kind == FaultKind.NO_FAULT
    (alert,) = policy.alerts
    assert alert.severity == "info" and alert.kind == "no_fault"


def test_degraded_camera_is_uncertain_and_backs_off_when_gas_is_up():
    env, policy, r = run(lambda e: e.trigger_crack(0.3), dust=1.0)
    assert policy.diagnosis.kind == FaultKind.UNCERTAIN
    assert r.retorques == []
    assert r.final_xy[0] < 1.0  # unsure with gas up: it does not wait at the flange
    (alert,) = policy.alerts
    assert alert.severity == "warning" and alert.requires_ack


def test_notifier_is_called_exactly_once_with_the_alert():
    sent = []
    env, policy, r = run(lambda e: e.trigger_crack(0.3), notify=sent.append)
    assert sent == policy.alerts and len(sent) == 1


def test_alert_text_template_has_the_key_facts():
    env, policy, r = run(lambda e: e.trigger_crack(0.3))
    text = format_alert(policy.alerts[0])
    assert "CRITICAL" in text and "ACKNOWLEDGEMENT REQUIRED" in text and "Gas:" in text and "Recommended:" in text
    assert alert_text(policy.alerts[0], None) == text


def test_alert_text_falls_back_to_the_template_when_gemini_fails():
    env, policy, r = run(lambda e: e.trigger_crack(0.3))

    def boom(model, prompt, cfg):
        raise ValueError("bad request")

    reasoner = GeminiReasoner(model="m", call_fn=boom, sleep=lambda s: None)
    assert alert_text(policy.alerts[0], reasoner) == format_alert(policy.alerts[0])


def test_alert_text_uses_gemini_when_available_and_sends_the_facts():
    env, policy, r = run(lambda e: e.trigger_crack(0.3))
    seen = []

    def fake(model, prompt, cfg):
        seen.append(prompt)
        return "short readable alert"

    reasoner = GeminiReasoner(model="m", call_fn=fake)
    assert alert_text(policy.alerts[0], reasoner) == "short readable alert"
    assert "Recommended:" in seen[0] and "Gas:" in seen[0]


def test_diagnosis_sweep_never_repairs_a_crack_and_rarely_cries_wolf():
    rng = np.random.default_rng(7)
    crack_diagnosed = 0
    for seed in range(20):
        sev = float(rng.uniform(0.1, 0.35))
        _, policy, r = run(lambda e, s=sev: e.trigger_crack(s), seed=seed)
        assert r.retorques == [], (seed, sev)  # a crack is never "fixed" by tightening
        crack_diagnosed += policy.diagnosis is not None and policy.diagnosis.kind == FaultKind.CRACK_OR_BURST
    assert crack_diagnosed >= 15, crack_diagnosed  # the rest hit the gas limit before it could diagnose

    false_cracks = 0
    for seed in range(20):
        angle = float(rng.uniform(0.08, 0.15))
        _, policy, r = run(lambda e, a=angle: e.trigger_gas_leak(int(seed % 8), a), seed=seed)
        false_cracks += policy.diagnosis is not None and policy.diagnosis.kind == FaultKind.CRACK_OR_BURST
    assert false_cracks == 0

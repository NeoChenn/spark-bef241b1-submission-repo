"""The shipped RL pick policy (policy/skills/pick_policy) as a robot skill."""
import pytest
import torch

from policy.envs.pick_state import PickParams, PickState
from policy.envs.pick_scripted import rollout, scripted_action

pytest.importorskip("onnxruntime")


def _skill():
    from policy.deck.pick_skill import POLICY_DIR, RLPickSkill
    if not (POLICY_DIR / "policy.onnx").exists():
        pytest.skip("pick policy not present")
    return RLPickSkill()


def test_picks_a_lying_can_in_front_of_the_crouched_robot():
    ok, info = _skill().pick((0.26, 0.03), yaw=0.7, size=(0.066, 0.066, 0.12), seed=1)
    assert ok and info["seconds"] < 4.0


def test_policy_beats_the_scripted_pick_under_camera_error():
    sk = _skill()
    pol = rollout(PickState(256, PickParams(), seed=321), lambda st, o: sk.act(o))
    scr = rollout(PickState(256, PickParams(), seed=321), lambda st, o: scripted_action(st))
    assert pol["success"].float().mean() >= scr["success"].float().mean() - 0.02
    assert pol["knocks"].mean() < scr["knocks"].mean()


def test_mock_robot_picks_litter_with_the_policy():
    from policy.deck.job import DeckConfig, DeckJob
    from policy.leak.mock import build_rig
    from policy.orchestrator import S
    from detection.rig_vision import Detection, Hit

    class TruthDebris:                      # sees exactly the mock's debris (as in test_deck_job.py)
        bridge = None

        def detect(self, image, reference=None):
            hits = [Hit((int(u - 30), int(v - 30), int(u + 30), int(v + 30)), 0.95)
                    for u, v in (it.pixel() for it in self.bridge.debris)]
            return Detection("debris", 0.95 if hits else 0.02, bool(hits), False, hits, 10, 0.5)

    det = TruthDebris()
    orch, bridge, *_ = build_rig(rust=False, deck=DeckJob(debris=det, config=DeckConfig()), debris="litter",
                                 pick_skill=_skill())
    det.bridge = bridge
    for _ in range(300):
        if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN):
            break
        orch.step()
    assert bridge.pick_log and all(p["ok"] for p in bridge.pick_log) and not bridge.debris

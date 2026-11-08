"""Deck sweep (debris pickup, camera person check) in the rig mock.

The state-machine tests use stand-in detectors that read the mock's ground truth, so they test the robot's decisions
independently of the trained models. The trained models are tested on held-out frames in test_rig_vision.py.
"""
import numpy as np
import pytest

from detection.rig_vision import Detection, Hit
from policy.deck.job import DeckConfig, DeckJob
from policy.deck.scene import Debris
from policy.leak.mock import build_rig
from policy.orchestrator import S


class TruthDebris:
    """Sees exactly the debris the bridge holds; ``unsure`` makes it unsure while the deck is dusty."""

    def __init__(self, unsure_when_dusty=False):
        self.bridge = None
        self.unsure_when_dusty = unsure_when_dusty

    def detect(self, image, reference=None):
        b = self.bridge
        dusty = b.deck_dust * (0.3 ** b.deck_view_idx) > 0.2
        if self.unsure_when_dusty and dusty:
            return Detection("debris", 0.5, False, True, threshold=0.5)
        hits = []
        for it in b.debris:
            u, v = it.pixel()
            hits.append(Hit((int(u - 30), int(v - 30), int(u + 30), int(v + 30)), 0.95))
        return Detection("debris", 0.95 if hits else 0.02, bool(hits), False, hits, 10, 0.5)


class TruthPerson:
    def __init__(self):
        self.bridge = None
        self.calls = 0

    def detect(self, image, camera=None, range_m=None):
        self.calls += 1
        d = self.bridge.camera_person_m
        if d is None:
            return Detection("person", 0.01, False, False, threshold=0.5)
        return Detection("person", 0.97, True, False, [Hit((300, 100, 360, 300), 0.97)], 10, 0.5, distance_m=d)


def rig(debris="none", person_m=None, deck_dust=0.0, unsure=False, person=False, **kw):
    det = TruthDebris(unsure_when_dusty=unsure)
    ppl = TruthPerson() if person else None
    job = DeckJob(debris=det, person=ppl, config=DeckConfig())
    orch, bridge, *_ = build_rig(rust=False, deck=job, debris=debris, person_m=person_m, deck_dust=deck_dust, **kw)
    det.bridge = bridge
    if ppl:
        ppl.bridge = bridge
    return orch, bridge


def run(orch, n=300):
    for _ in range(n):
        if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN):
            break
        orch.step()
    return [r.to for r in orch.records]


def test_clean_deck_is_checked_and_the_patrol_goes_on():
    orch, bridge = rig("none")
    seq = run(orch)
    assert "DEBRIS_CHECK" in seq and "DEBRIS_PICKUP" not in seq
    assert orch.state == S.DONE and not bridge.picked


def test_litter_is_picked_up_one_by_one_and_checked():
    orch, bridge = rig("litter")
    seq = run(orch)
    assert seq.count("DEBRIS_PICKUP") == 3 and len(bridge.picked) == 3 and not bridge.debris
    assert orch.state == S.DONE
    assert any(a["kind"] == "debris_cleared" for a in orch.alerts)
    ev = next(r.evidence for r in orch.records if r.to == "DEBRIS_PICKUP")
    assert ev["debris"]["found"] and "best_box" in ev["debris"]


def test_an_object_it_cannot_lift_is_reported_and_the_patrol_goes_on():
    orch, bridge = rig("can")
    bridge.stuck = {0}
    seq = run(orch)
    assert seq.count("DEBRIS_PICKUP") == 2                  # first try + one retry
    assert orch.state == S.DONE and len(bridge.debris) == 1
    assert any(a["kind"] == "debris_left" and a["severity"] == "warning" for a in orch.alerts)


def test_unsure_deck_gets_another_view_before_any_action():
    orch, bridge = rig("can", deck_dust=0.8, unsure=True)
    seq = run(orch)
    assert "DEBRIS_SECOND_VIEW" in seq
    i = seq.index("DEBRIS_SECOND_VIEW")
    assert "DEBRIS_PICKUP" not in seq[:i]
    assert not bridge.debris and orch.state == S.DONE    # the clearer view found it and it was picked


def test_debris_check_waits_for_normal_gas():
    orch, bridge = rig("can", leak="loose")
    seq = run(orch)
    assert seq.index("GAS_CHECK") < seq.index("DEBRIS_CHECK")
    assert "RETORQUE" in seq and seq.index("RETORQUE") < seq.index("DEBRIS_CHECK")


def test_camera_person_pauses_the_robot_even_without_ground_truth():
    orch, bridge = rig("none", person_m=2.0, person=True)
    orch.step()
    assert orch.state == S.PAUSE_PERSON
    rec = orch.records[-1]
    assert rec.evidence["person_camera"]["found"] and "camera" in rec.reason
    for _ in range(3):
        orch.step()
    assert orch.state == S.PAUSE_PERSON                    # still there: stays paused
    bridge.camera_person_m = None
    orch.step()
    assert orch.state != S.PAUSE_PERSON


def test_a_far_person_does_not_pause_the_robot():
    orch, bridge = rig("none", person_m=8.0, person=True)
    seq = run(orch)
    assert "PAUSE_PERSON" not in seq and orch.state == S.DONE


def test_camera_never_overrides_the_ground_truth_pause():
    orch, bridge = rig("none", person_m=None, person=True)
    bridge.person_window = (0.0, 3.0)                       # the team's ground-truth person, camera sees nobody
    orch.step()
    assert orch.state == S.PAUSE_PERSON
    orch.step()
    assert orch.state == S.PAUSE_PERSON or bridge.t >= 3.0


def test_without_a_deck_job_nothing_changes():
    orch, bridge, *_ = build_rig(rust=False)
    seq = run(orch)
    assert not any(s.startswith("DEBRIS") for s in seq)

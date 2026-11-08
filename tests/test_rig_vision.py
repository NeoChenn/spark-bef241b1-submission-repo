"""The shipped debris and person models on held-out frames (policy/deck/scene.py on preview_deck_end.png, a render
no training set used, with held-out test-split litter and people)."""
import numpy as np
import pytest

from detection.rig_vision import CKPT_DIR, DebrisDetector, PersonDetector, depth_distance, floor_range
from policy.deck.scene import Debris, camera_model, deck_view, person_range, person_view

pytest.importorskip("onnxruntime")


def _debris():
    if not (CKPT_DIR / "debris_v2" / "model.onnx").exists():
        pytest.skip("debris checkpoint not present")
    return DebrisDetector.default()


def _person():
    if not (CKPT_DIR / "person_v1" / "model.onnx").exists():
        pytest.skip("person checkpoint not present")
    return PersonDetector.default()


def _near(r, item, px=45):
    u, v = item.pixel()
    return any(np.hypot(h.centre[0] - u, h.centre[1] - v) < px for h in r.hits)


REF = None


def ref():
    global REF
    if REF is None:
        REF = deck_view([])                      # the clean picture of this patrol stop
    return REF


@pytest.mark.parametrize("light,shadow", [(1.0, None), (0.8, None), (1.15, 4), (0.9, 11), (1.0, 23)])
def test_empty_deck_is_not_debris_under_changing_light_and_shadow(light, shadow):
    """Fire extinguishers, gas bottles, pipe supports and valve wheels belong on the deck; shadows and light change."""
    r = _debris().detect(deck_view([], light=light, shadow_seed=shadow), ref())
    assert not r.found, r.summary()


def test_without_a_reference_the_equipment_is_mistaken_for_debris():
    """Why the reference is needed: the model alone fires on deck equipment it has never seen from this angle."""
    assert _debris().detect(deck_view([])).found


@pytest.mark.parametrize("item", [Debris(0.15, 3.4, "can", length_m=0.12, seed=3),
                                  Debris(0.5, 4.3, "debris_can_red.png", length_m=0.12),
                                  Debris(-0.05, 5.0, "debris_bottle.png", length_m=0.22)])
def test_litter_is_found_where_the_arm_should_pick(item):
    r = _debris().detect(deck_view([item], light=0.9, shadow_seed=5), ref())
    assert r.found and _near(r, item), r.summary()
    assert len(r.hits) == 1


def test_person_is_found_and_nobody_is_not():
    det, cam = _person(), camera_model()
    assert det.detect(person_view(5.0), cam).found
    assert not det.detect(person_view(None), cam).found


@pytest.mark.parametrize("d,pause", [(2.0, True), (2.8, True), (4.0, False), (7.0, False)])
def test_person_distance_from_depth_decides_the_pause(d, pause):
    det, cam = _person(), camera_model()
    r = det.detect(person_view(d, who=1), cam, person_range(d, who=1))
    assert r.found
    assert (r.distance_m < 3.0) == pause, r.distance_m


def test_depth_ignores_the_deck_itself():
    cam = camera_model()
    rng = person_range(None)
    assert depth_distance((200, 200, 400, 360), cam, rng) is None       # only floor in the box: no obstacle
    assert np.isinf(floor_range(cam, 360, 640)[0, 0])                   # sky: the ray never meets the deck

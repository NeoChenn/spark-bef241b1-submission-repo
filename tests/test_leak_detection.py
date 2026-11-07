import json

import numpy as np
import pytest

from detection.leak import LeakLocalizer, fake_crack_classifier
from policy.interfaces import CameraModel
from policy.leak.geometry import FlangeGeometry, look_at
from policy.leak.render import render_flange
from policy.leak.world import LeakWorld

G = FlangeGeometry("F1", center=[1.0, -0.2, 0.9], normal=[0, -1, 0], ref=[0, 0, 1])
W, H = 640, 480
F = 35.0 / 20.955 * W
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
TRUTH = np.array([0.0, 0.15, 0.0, -0.10, 0.0, 0.25, 0.0, 0.08])


def cam(view=0):
    eye, target = G.view_poses()[view]
    return CameraModel(K, look_at(eye, target), W, H)


def state(angles=TRUTH, crack=0.0):
    w = LeakWorld([G])
    s = w.flanges["F1"]
    s.angles[:] = angles
    s.crack = crack
    return s


@pytest.mark.parametrize("view", [0, 1, 2])
def test_stripe_reader_measures_how_far_each_bolt_has_turned(view):
    r = LeakLocalizer().read(render_flange(cam(view), state()), cam(view), G)
    est = np.array([b.angle for b in r.bolts], float)
    assert np.all(np.abs(np.abs(est) - np.abs(TRUTH)) < 0.03), est
    assert all(b.conf >= 0.85 for b in r.bolts)


def test_dirty_lens_lowers_confidence_below_the_action_threshold():
    loc = LeakLocalizer()
    clean = loc.read(render_flange(cam(), state(), dust=0.0), cam(), G)
    dusty = loc.read(render_flange(cam(), state(), dust=0.4), cam(), G)
    assert min(b.conf for b in clean.bolts) >= 0.9
    assert max(b.conf for b in dusty.bolts) < 0.7          # robot will not act on these


def test_crack_is_found_and_a_clean_flange_is_not_flagged():
    loc = LeakLocalizer()
    cracked = loc.read(render_flange(cam(), state(np.zeros(8), crack=0.3)), cam(), G)
    clean = loc.read(render_flange(cam(), state(np.zeros(8))), cam(), G)
    assert cracked.crack_prob >= 0.5 and cracked.crack_windows >= 1
    assert clean.crack_prob < 0.35 and not clean.crack_unsure


def test_face_mask_excludes_bore_and_bolts():
    m = LeakLocalizer().face_mask(cam(), G, (H, W))
    assert not m[H // 2, W // 2]                            # pipe bore in the middle
    assert 0.05 < m.mean() < 0.6


def test_fake_crack_classifier_scores_dark_lines():
    plain = np.full((2, 32, 32, 3), 150, np.uint8)
    lined = plain.copy()
    lined[1, :, 15:18] = 20
    logits = fake_crack_classifier(lined)
    assert logits[1] > 0 > logits[0]


def test_from_checkpoint_uses_the_same_layout_as_the_rust_detector(tmp_path):
    """model.onnx + meta.json, loaded like RustLocalizer.from_checkpoint."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    class Tiny(torch.nn.Module):
        def forward(self, x):                               # dark pixels -> higher "crack" logit
            return (0.5 - x.mean(dim=(1, 2, 3))).unsqueeze(1) * 4.0

    torch.onnx.export(Tiny(), torch.zeros(1, 3, 32, 32), str(tmp_path / "model.onnx"), input_names=["x"],
                      output_names=["y"], dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, dynamo=False)
    (tmp_path / "meta.json").write_text(json.dumps({"input_size": 32, "mean": [0, 0, 0], "std": [1, 1, 1],
                                                    "threshold": 0.6, "temperature": 1.5}))
    loc = LeakLocalizer.from_checkpoint(str(tmp_path))
    assert loc.threshold == 0.6 and loc.temperature == 1.5
    r = loc.read(render_flange(cam(), state()), cam(), G)
    assert r.n_windows > 0 and 0.0 <= r.crack_prob <= 1.0

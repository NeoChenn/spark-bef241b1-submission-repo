"""CPU wiring test for policy/repair_live.py: detector + scrub policy selection, no Isaac, no real model."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection import localize  # noqa: E402
from policy import repair_live as rl  # noqa: E402

META = {"input_size": [3, 32, 32], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "threshold": 0.42, "temperature": 1.7, "backbone": "mobilenetv4_conv_small_lora"}


def _image():
    img = np.full((360, 640, 3), 90, np.uint8)
    img[100:250, 200:400] = (170, 70, 30)
    return img


def test_missing_dir_falls_back_to_colour(tmp_path):
    loc, info = rl.make_localizer(str(tmp_path / "nope"), hue_lo=0.3)
    assert info["kind"] == "colour_standin"
    assert "model.onnx" in info["fallback_reason"]
    assert (loc.patch, loc.stride) == (32, 16)
    loc.localize(_image())


def test_missing_meta_falls_back(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    _, info = rl.make_localizer(str(tmp_path))
    assert info["kind"] == "colour_standin" and "meta.json" in info["fallback_reason"]


def test_corrupt_onnx_falls_back_with_reason(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"not an onnx file")
    (tmp_path / "meta.json").write_text(json.dumps(META))
    _, info = rl.make_localizer(str(tmp_path))
    assert info["kind"] == "colour_standin"
    assert info["fallback_reason"]


def test_stub_classifier_builds_lora_localizer(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"stub")
    (tmp_path / "meta.json").write_text(json.dumps(META))
    seen = {}

    class Stub:
        def __init__(self, path, meta, batch_size=64):
            seen["path"], seen["meta"] = str(path), meta

        def __call__(self, patches):
            p = patches.astype(np.float32)
            return (p[..., 0].mean(axis=(1, 2)) - p[..., 2].mean(axis=(1, 2))) / 10.0 - 3.0

    monkeypatch.setattr(localize, "OnnxPatchClassifier", Stub)
    monkeypatch.setattr(rl, "_ensure_onnxruntime", lambda: True)
    loc, info = rl.make_localizer(str(tmp_path))
    assert info["kind"] == "lora_onnx" and info["label"].startswith("LoRA")
    assert seen["path"].endswith("model.onnx")
    assert (loc.patch, loc.stride) == (96, 48)
    assert loc.threshold == pytest.approx(0.42) and loc.temperature == pytest.approx(1.7)
    assert info["backbone"] == "mobilenetv4_conv_small_lora"
    res = loc.localize(_image())
    assert res.max_conf > 0.5
    json.dumps(info)


def test_tiny_real_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def forward(self, x):
            return (x[:, 0].mean(dim=(1, 2)) - x[:, 2].mean(dim=(1, 2))).unsqueeze(1) * 4.0

    try:
        torch.onnx.export(Tiny(), torch.zeros(1, 3, 32, 32), str(tmp_path / "model.onnx"), input_names=["input"],
                          output_names=["logit"], dynamic_axes={"input": {0: "b"}, "logit": {0: "b"}}, dynamo=False)
    except Exception as e:
        pytest.skip(f"onnx export unavailable: {e}")
    (tmp_path / "meta.json").write_text(json.dumps(META))
    loc, info = rl.make_localizer(str(tmp_path))
    assert info["kind"] == "lora_onnx", info
    res = loc.localize(_image())
    assert res.max_conf > 0.5


def test_policy_selection(tmp_path):
    v2 = str(tmp_path / "v2.pt")
    v3 = tmp_path / "v3.pt"
    path, info = rl.select_policy(v2, 0.0, str(v3))
    assert path == v2 and info["kind"] == "scrub_v2" and "fallback_reason" not in info
    path, info = rl.select_policy(v2, 1.0, str(v3))
    assert path == v2 and "missing" in info["label"] and "not found" in info["fallback_reason"]
    path, info = rl.select_policy(v2, 0.5, "")
    assert path == v2 and info["fallback_reason"] == "no --storm-ckpt given"
    v3.write_bytes(b"x")
    path, info = rl.select_policy(v2, 1.0, str(v3))
    assert path == str(v3) and info["kind"] == "storm_v3" and info["storm"] == 1.0
    path, info = rl.select_policy(v2, 0.0, str(v3))
    assert path == v2, "calm run keeps the v2 policy"

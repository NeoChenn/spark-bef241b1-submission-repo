import numpy as np
import pytest
import torch

from detection.eval import ece, fit_temperature, full_report, pick_threshold
from detection.export import export_onnx
from detection.models import build_model, find_lora_targets, to_plain


@pytest.mark.parametrize("name", ["mobilenetv4_conv_small", "resnet18"])
def test_lora_forward_loss_backward(name):
    model, meta = build_model(name, lora=True, pretrained=False, verbose=False)
    assert 0 < meta["trainable_params"] < meta["total_params"] * 0.25
    x, y = torch.randn(2, 3, 64, 64), torch.tensor([0.0, 1.0])
    z = model(x).squeeze(1)
    assert z.shape == (2,)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))(z, y)
    loss.backward()
    assert torch.isfinite(loss)
    names = [n for n, p in model.named_parameters() if p.grad is not None]
    assert any("lora_" in n for n in names) and any("modules_to_save" in n for n in names)


def test_vit_lora_targets():
    import timm

    m = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=1, img_size=224)
    t = find_lora_targets(m, "vit_small_patch14_dinov2")
    assert len(t) == 24 and all(n.endswith(("attn.qkv", "attn.proj")) for n in t)


def test_merge_and_onnx_export(tmp_path):
    model, _ = build_model("mobilenetv4_conv_small", lora=True, pretrained=False, verbose=False)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    model.eval()
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        ref = model(x)
        plain = to_plain(model).eval()
        assert torch.allclose(ref, plain(x), atol=1e-4)
    export_onnx(plain, tmp_path / "m.onnx", size=96)
    assert (tmp_path / "m.onnx").stat().st_size > 0


def test_eval_utils():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    z = (y * 2 - 1) * 3.0 + rng.normal(0, 2, 400)
    thr = pick_threshold(z, y, 0.95)
    assert ((z >= thr) & (y == 1)).sum() / (y == 1).sum() >= 0.95
    T = fit_temperature(z, y)
    assert 0.05 < T < 20
    p = 1 / (1 + np.exp(-z / T))
    assert ece(p, y) <= ece(1 / (1 + np.exp(-z)), y) + 1e-6
    rep = full_report({"val": z, "test": z}, {"val": y, "test": y})
    assert rep["splits"]["test"]["at_thr"]["recall"] >= 0.95


def test_domain_threshold_and_synth_helpers():
    from detection.synth_sim import blob_mask, camera, rust_fraction, shade

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    z = (y * 2 - 1) * 2.0 + rng.normal(0, 1.5, 300)
    dom = np.arange(300) < 150
    rep = full_report({"val": z, "test": z}, {"val": y, "test": y}, val_domains=[dom, ~dom])
    for m in (dom, ~dom):
        assert ((z[m] >= rep["threshold_logit"]) & (y[m] == 1)).sum() / (y[m] == 1).sum() >= 0.95
    m = blob_mask(rng, 0.3)
    assert 0.2 < (m > 0.5).mean() < 0.4
    img = camera(shade(np.full((224, 224, 3), 128, np.float32), rng), rng)
    assert img.size == (224, 224)
    rusty = np.zeros((32, 32, 3)) + np.array([150, 70, 30])
    assert rust_fraction(rusty) > 0.9 and rust_fraction(np.full((32, 32, 3), 120.0)) == 0.0

import torch
import torch.nn as nn

from detection.tta import TTA, binary_entropy, norm_layers


def net():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                      nn.Linear(4, 1))
    m[1].running_mean.fill_(0.3)
    m[1].running_var.fill_(2.0)
    return m.eval()


def x():
    return torch.randn(8, 3, 8, 8, generator=torch.Generator().manual_seed(1)) * 3 + 1


def test_off_matches_plain_eval():
    m = net()
    with torch.no_grad():
        ref = m(x())
    assert torch.allclose(TTA(m, mode="off")(x()), ref)


def test_bn_mode_uses_batch_stats_and_restores_running():
    m = net()
    mu0 = m[1].running_mean.clone()
    with torch.no_grad():
        ref = m(x())
    out = TTA(m, mode="bn")(x())
    assert not torch.allclose(out, ref)
    assert torch.equal(m[1].running_mean, mu0) and isinstance(m[1].running_mean, torch.Tensor)
    with torch.no_grad():
        assert torch.allclose(m(x()), ref)


def test_prior_one_is_source():
    m = net()
    with torch.no_grad():
        ref = m(x())
    assert torch.allclose(TTA(m, mode="bn", prior=1.0)(x()), ref, atol=1e-5)


def test_tent_only_changes_norm_affine_and_reset_restores():
    m = net()
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    t = TTA(m, mode="tent", lr=1e-2, steps=3)
    t(x())
    changed = {k for k, v in m.state_dict().items() if not torch.equal(v, w0[k])}
    assert changed and changed <= {"1.weight", "1.bias"}
    t.reset()
    assert all(torch.equal(v, w0[k]) for k, v in m.state_dict().items())


def test_entropy_and_norm_listing():
    assert binary_entropy(torch.tensor([0.0])).item() > binary_entropy(torch.tensor([5.0])).item()
    assert len(norm_layers(net())) == 1


def test_gate_skips_adaptation_in_distribution():
    m = net()
    with torch.no_grad():
        ref = m(x())
    t = TTA(m, mode="bn", gate=1e9)
    assert torch.allclose(t(x()), ref) and t.last_shift >= 0
    t2 = TTA(m, mode="bn", gate=0.0)
    assert not torch.allclose(t2(x()), ref)

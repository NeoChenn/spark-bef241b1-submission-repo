import torch

from policy.cbf_shield import CBFShield, cbf_bound, rls_step


def test_rls_converges_to_true_stiffness():
    k, P = torch.tensor([2500.0]), torch.tensor([1e7])
    for s in (0.002, 0.004, 0.006, 0.003, 0.008):
        k, P = rls_step(k, P, torch.tensor([s]), torch.tensor([3700.0 * s]), lam=0.98)
    assert abs(float(k) - 3700.0) < 5.0


def test_rls_ignores_free_space():
    k, P = rls_step(torch.tensor([2500.0]), torch.tensor([1e7]), torch.tensor([-0.01]), torch.tensor([0.0]), lam=0.98)
    assert float(k) == 2500.0


def test_bound_satisfies_discrete_cbf():
    f_max, gamma = 40.0, 0.4
    k, F, s, w = torch.tensor([3000.0]), torch.tensor([30.0]), torch.tensor([0.01]), torch.tensor([0.002])
    smax = cbf_bound(F, s, k, w, alpha=1.0, f_max=f_max, gamma=gamma)
    F_next = k * (s + (smax - s) + w)
    h0, h1 = f_max - F, f_max - F_next
    assert torch.allclose(h1, (1 - gamma) * h0, atol=1e-4)


def test_shield_is_identity_when_safe():
    sh = CBFShield(2, k0=2500.0)
    F, s = torch.tensor([5.0, 0.0]), torch.tensor([0.002, -0.05])
    sh.observe(F, s)
    cmd = torch.tensor([0.004, 0.002])
    out = sh.filter(F, s, cmd)
    assert torch.equal(out, cmd)
    assert not sh.active.any()


def test_shield_clips_and_retracts_under_disturbance():
    sh = CBFShield(1, k0=4000.0, f_max=40.0, gamma=0.5)
    s, F = torch.tensor([0.008]), torch.tensor([32.0])
    sh.observe(F, s)
    sh.filter(F, s, torch.tensor([0.008]))
    s2 = torch.tensor([0.0095])
    F2 = 4000.0 * s2
    sh.observe(F2, s2)
    out = sh.filter(F2, s2, torch.tensor([0.0095]))
    pred = 4000.0 * (s2 + (out - s2) + sh.w_hat)
    assert bool(sh.active[0])
    assert float(pred) <= 40.0 - 0.5 * (40.0 - float(F2)) + 1e-3
    assert float(out) < float(s2)


def test_forward_invariance_on_moving_pipe():
    torch.manual_seed(0)
    n, k_true, f_max = 64, torch.empty(64).uniform_(1500.0, 4000.0), 40.0
    sh = CBFShield(n, k0=2500.0, f_max=f_max, gamma=0.3, margin=1.0)
    s = torch.full((n,), 0.004)
    phase = torch.rand(n) * 6.28
    worst = 0.0
    for t in range(450):
        F = k_true * s.clamp_min(0.0)
        worst = max(worst, float(F.max())) if t > 5 else worst
        sh.observe(F, s)
        want = torch.full((n,), 0.03)
        cmd = sh.filter(F, s, want)
        drift = 0.003 * torch.cos(0.06 * t + phase)
        s = s + 0.9 * (cmd - s) + drift
    assert worst <= f_max + 1.0


def test_reset_restores_prior():
    sh = CBFShield(2, k0=2500.0)
    sh.k_hat[:] = 9000.0
    sh.w_hat[:] = 0.01
    sh.reset(torch.tensor([True, False]))
    assert float(sh.k_hat[0]) == 2500.0 and float(sh.k_hat[1]) == 9000.0
    assert float(sh.w_hat[0]) == 0.0


def test_rls_skips_stale_contact_after_reset():
    k, P = rls_step(torch.tensor([2500.0]), torch.tensor([1e7]), torch.tensor([0.01]), torch.tensor([0.0]), lam=0.98)
    assert float(k) == 2500.0

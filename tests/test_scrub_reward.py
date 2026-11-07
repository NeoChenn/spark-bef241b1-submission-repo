import torch

from policy.envs.rust_state import RustState
from policy.envs.scrub_reward import force_setpoint, over_force_penalty


def test_force_setpoint_maps_and_clamps():
    a = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    assert torch.allclose(force_setpoint(a, 25.0, 20.0), torch.tensor([5.0, 25.0, 35.0, 45.0]))
    assert torch.allclose(force_setpoint(a, 25.0, 20.0, f_max=38.0), torch.tensor([5.0, 25.0, 35.0, 38.0]))


def test_over_force_linear_matches_v1():
    F = torch.tensor([0.0, 40.0, 50.0, 60.0])
    assert torch.allclose(over_force_penalty(F, 40.0, 0.1), torch.tensor([0.0, 0.0, 0.1, 0.2]))


def test_over_force_quadratic_starts_at_soft_limit():
    F = torch.tensor([30.0, 36.0, 46.0, 56.0])
    p = over_force_penalty(F, 40.0, 0.5, soft=36.0, w_quad=1.0)
    assert torch.allclose(p, torch.tensor([0.0, 0.0, 0.3 + 1.0, 0.8 + 4.0]))


def test_reset_patch_count_range():
    rs = RustState(256, 32, 64, length=0.6)
    g = torch.Generator().manual_seed(0)
    rs.reset(radius=0.15, n_patches=(4, 4), generator=g)
    four = rs.rust.sum(dim=(1, 2)).mean()
    rs.reset(radius=0.15, n_patches=(1, 1), generator=torch.Generator().manual_seed(0))
    assert four > 1.5 * rs.rust.sum(dim=(1, 2)).mean()

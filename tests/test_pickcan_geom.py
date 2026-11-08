import math

import pytest

torch = pytest.importorskip("torch")

from policy.pickcan import core, geom


def _scene(lying, yaw=0.6, r=0.033, h=0.122, xy=(0.15, -0.4)):
    lying_t = torch.tensor([lying])
    R = core.quat_to_mat(core.can_quat(lying_t, torch.tensor([yaw])))
    z = core.rest_height(lying_t, torch.tensor([r]), torch.tensor([h]))
    can = torch.tensor([[xy[0], xy[1], float(z[0])]])
    eye = torch.tensor([[0.02, -0.15, 0.75]])
    Rc = geom.look_basis(eye, can)
    dirs = geom.ray_grid(32, 0.45, "cpu")
    pts, hit = geom.raycast_can(eye, Rc, dirs, can, R, torch.tensor([r]), torch.tensor([h]))
    return can, R, pts, hit


def test_raycast_hits_lie_on_the_can_surface():
    can, R, pts, hit = _scene(True)
    assert hit.sum() > 30
    local = torch.einsum("ij,kj->ki", R[0].T, pts[0][hit[0]] - can[0])
    rho = local[:, :2].norm(dim=-1)
    on_side = (rho - 0.033).abs() < 1e-4
    on_cap = ((local[:, 2].abs() - 0.061).abs() < 1e-4) & (rho <= 0.033 + 1e-4)
    assert bool((on_side | on_cap).all())


@pytest.mark.parametrize("lying", [True, False])
def test_fit_recovers_pose_and_size(lying):
    can, R, pts, hit = _scene(lying)
    f = geom.fit_cylinder(pts, hit)
    assert bool(f["lying"][0]) == lying
    assert (f["centre"][0, :2] - can[0, :2]).norm() < 0.012
    assert f["radius"][0].item() == pytest.approx(0.033, abs=0.008)
    true_top = can[0, 2] + (0.033 if lying else 0.061)
    assert f["grasp"][0, 2].item() == pytest.approx(true_top.item(), abs=0.008)
    if lying:
        ax = f["axis"][0]
        assert abs(math.cos(math.atan2(ax[1], ax[0]) - 0.6)) > 0.97
        assert f["length"][0].item() == pytest.approx(0.122, abs=0.02)
    assert f["conf"][0] > 0.3


def test_geom_obs_shape_and_yaw_symmetry():
    can, R, pts, hit = _scene(True)
    f = geom.fit_cylinder(pts, hit)
    o = geom.geom_obs(f, torch.zeros(1, 3))
    assert o.shape == (1, geom.GEOM_DIM)
    f2 = dict(f, axis=-f["axis"])
    assert torch.allclose(o, geom.geom_obs(f2, torch.zeros(1, 3)))


def test_fit_with_no_hits_has_zero_confidence():
    pts = torch.zeros(1, 10, 3)
    hit = torch.zeros(1, 10, dtype=torch.bool)
    assert geom.fit_cylinder(pts, hit)["conf"][0].item() == 0.0

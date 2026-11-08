import math

import pytest

torch = pytest.importorskip("torch")

from policy.pickcan import core
from policy.pickcan.script import CARRY, DESCEND, DONE, LIFT, RELEASE, ScriptedPicker


def _t(*v):
    return torch.tensor([list(v)], dtype=torch.float32)


def test_can_quat_lying_axis_horizontal_with_yaw():
    q = core.can_quat(torch.tensor([True, False]), torch.tensor([0.5, 0.5]))
    R = core.quat_to_mat(q)
    ax = R[:, :, 2]
    assert abs(float(ax[0, 2])) < 1e-6
    assert math.isclose(math.atan2(float(ax[0, 1]), float(ax[0, 0])), 0.5, abs_tol=1e-5)
    assert math.isclose(float(ax[1, 2]), 1.0, abs_tol=1e-6)


def test_rest_and_grasp_point():
    lying = torch.tensor([True, False])
    r, h = torch.tensor([0.03, 0.03]), torch.tensor([0.12, 0.12])
    assert core.rest_height(lying, r, h).tolist() == pytest.approx([0.03, 0.06])
    pos = torch.tensor([[0.3, 0.3, 0.03], [0.3, 0.3, 0.06]])
    axis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    g = core.grasp_point(pos, axis, r, h)
    assert g[:, 2].tolist() == pytest.approx([0.06, 0.12])


def test_surface_side_cap_and_corner():
    r, h = torch.full((3,), 0.03), torch.full((3,), 0.12)
    local = torch.tensor([[0.04, 0.0, 0.0], [0.0, 0.0, 0.07], [0.04, 0.0, 0.07]])
    dist, n, margin = core.surface(local, r, h)
    assert dist[0].item() == pytest.approx(0.01)
    assert dist[1].item() == pytest.approx(0.01)
    assert dist[2].item() == 1.0
    assert n[1].tolist() == pytest.approx([0, 0, 1])
    assert margin[0].item() == pytest.approx(0.06)


def test_seal_needs_alignment_and_gap():
    R = core.quat_to_mat(core.can_quat(torch.tensor([True]), torch.tensor([0.0])))
    can = _t(0.3, 0.3, 0.03)
    r, h = torch.tensor([0.03]), torch.tensor([0.12])
    down = _t(0.0, 0.0, -1.0)
    assert core.seal_ok(_t(0.3, 0.3, 0.065), down, can, R, r, h).item()
    assert not core.seal_ok(_t(0.3, 0.3, 0.09), down, can, R, r, h).item()
    assert not core.seal_ok(_t(0.3, 0.3, 0.065), _t(1.0, 0.0, 0.0), can, R, r, h).item()
    assert not core.seal_ok(_t(0.355, 0.3, 0.065), down, can, R, r, h).item()


def test_suction_wrench_holds_weight_at_rest():
    eye = torch.eye(3)[None]
    z = torch.zeros(1, 3)
    F, T, tc, tear = core.suction_wrench(_t(0, 0, 0.5), eye, z, z, _t(0, 0, 0.45), eye, z, z, _t(0, 0, -0.05),
                                         eye, torch.tensor([0.3]), torch.full((1, 3), 1e-3))
    assert F[0].tolist() == pytest.approx([0, 0, 0.3 * core.G])
    assert tear.item() == pytest.approx(0.0, abs=1e-6)
    assert tc.norm().item() == pytest.approx(0.0, abs=1e-6)


def test_suction_torque_about_cup_for_off_centre_grip():
    eye = torch.eye(3)[None]
    z = torch.zeros(1, 3)
    _, _, tc, _ = core.suction_wrench(_t(0, 0, 0.5), eye, z, z, _t(0.08, 0, 0.47), eye, z, z, _t(0.08, 0, -0.03),
                                      eye, torch.tensor([0.5]), torch.full((1, 3), 1e-3))
    assert tc.norm().item() == pytest.approx(0.08 * 0.5 * core.G, rel=1e-4)


def test_in_tote():
    p = torch.tensor([[core.TOTE_CENTER[0], 0.0, core.TOTE_FLOOR + 0.03], [0.3, 0.3, 0.03],
                      [core.TOTE_CENTER[0], 0.0, core.TOTE_RIM + 0.2]])
    assert core.in_tote(p).tolist() == [True, False, False]


def test_scripted_picker_walks_through_phases():
    sp = ScriptedPicker(1, "cpu")
    grasp = _t(0.3, 0.3, 0.06)
    sealed = torch.tensor([False])
    a = sp.act(_t(0.1, 0.2, 0.55), _t(0.1, 0.2, 0.55), grasp, sealed)
    assert a[0, 3].item() < 0 and a[0, 2].item() < 0
    sp.act(_t(0.3, 0.3, 0.16), _t(0.3, 0.3, 0.16), grasp, sealed)
    assert sp.phase.item() == DESCEND
    a = sp.act(_t(0.3, 0.3, 0.07), _t(0.3, 0.3, 0.07), grasp, torch.tensor([True]))
    assert sp.phase.item() == LIFT and a[0, 3].item() > 0 and a[0, 2].item() > 0
    sp.act(_t(0.3, 0.3, 0.70), _t(0.3, 0.3, 0.70), grasp, torch.tensor([True]))
    assert sp.phase.item() == CARRY
    sp.act(_t(-0.2, 0.0, 0.72), _t(-0.2, 0.0, 0.72), grasp, torch.tensor([True]))
    assert sp.phase.item() == RELEASE
    for _ in range(20):
        a = sp.act(_t(-0.2, 0.0, 0.72), _t(-0.2, 0.0, 0.72), grasp, torch.tensor([False]))
    assert sp.phase.item() == DONE and a[0, 3].item() < 0


def test_scripted_picker_retries_after_timeout_and_drop():
    sp = ScriptedPicker(2, "cpu", descend_timeout=0.1)
    sp.phase[:] = torch.tensor([DESCEND, CARRY])
    sp.t_phase[:] = 1.0
    sp.act(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([False, False]))
    assert sp.phase.tolist() == [0, 0] and sp.tries.tolist() == [1, 0]


def test_suction_torque_uses_principal_inertia():
    eye = torch.eye(3)[None]
    z = torch.zeros(1, 3)
    w = _t(0.0, 0.0, 1.0)
    _, T, _, _ = core.suction_wrench(_t(0, 0, 0.5), eye, z, z, _t(0, 0, 0.45), eye, z, w, _t(0, 0, -0.05), eye,
                                     torch.tensor([0.3]), torch.tensor([[2e-3, 2e-3, 1e-4]]))
    assert T[0, 2].item() == pytest.approx(-2 * 20.0 * 1e-4)
    I = core.can_inertia(torch.tensor([0.2]), torch.tensor([0.03]), torch.tensor([0.2]))
    assert I[0].tolist() == pytest.approx([0.2 * (3 * 0.03 ** 2 + 0.04) / 12] * 2 + [0.2 * 0.03 ** 2 / 2])

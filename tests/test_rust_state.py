import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.envs.rust_state import AbrasionParams, RustState, random_patches

N, H, W = 6, 32, 64


def make(seed=0, soak=60.0):
    g = torch.Generator().manual_seed(seed)
    rs = RustState(N, H, W, length=0.6)
    rs.reset(radius=torch.linspace(0.05, 0.15, N), soak_time=soak, generator=g)
    return rs


def centre_uv(rs):
    flat = rs.rust.view(N, -1).argmax(1)
    row, col = flat // W, flat % W
    return torch.stack([(col.float() + 0.5) / W, (row.float() + 0.5) / H], -1)


def test_random_patches_range_and_nonempty():
    g = torch.Generator().manual_seed(1)
    r = random_patches(64, H, W, radius=0.1, generator=g)
    assert r.shape == (64, H, W)
    assert r.min() >= 0 and r.max() <= 1
    covered = (r > 0.02).float().mean(dim=(1, 2))
    assert (covered > 0).all(), "every env must have some rust"
    assert covered.mean() < 0.5


def test_reset_sprays_rust_and_initial_fraction():
    rs = make()
    assert torch.allclose(rs.fraction_remaining(), torch.ones(N))
    assert (rs.sprayed | (rs.rust <= rs.p.rust_eps)).all(), "all rust sprayed by default"
    assert (rs.soak_time[rs.sprayed] == 60.0).all()


def test_no_removal_below_min_force():
    rs = make()
    before = rs.rust.clone()
    rem = rs.abrade(centre_uv(rs), 0.03, torch.full((N,), 4.9), torch.full((N,), 0.2), 0.1)
    assert (rem == 0).all() and torch.equal(before, rs.rust)


def test_no_removal_without_sliding():
    rs = make()
    rem = rs.abrade(centre_uv(rs), 0.03, torch.full((N,), 20.0), torch.zeros(N), 0.1)
    assert (rem == 0).all()


def test_removal_proportional_to_force_and_speed():
    out = []
    for F, v in [(10.0, 0.1), (20.0, 0.1), (10.0, 0.2)]:
        rs = make()
        rs.rust.fill_(1.0)
        rs.sprayed.fill_(True)
        rs.soak_time.fill_(60.0)
        out.append(rs.abrade(torch.full((N, 2), 0.5), 0.03, torch.full((N,), F), torch.full((N,), v), 0.01))
    assert torch.allclose(out[1], 2 * out[0], rtol=1e-4)
    assert torch.allclose(out[2], 2 * out[0], rtol=1e-4)


def test_spray_soak_multipliers():
    vals = {}
    for name in ["dry", "wet", "soaked"]:
        rs = RustState(N, H, W)
        rs.reset(rust=torch.ones(N, H, W), spray_all_rust=False)
        if name != "dry":
            rs.spray(torch.ones(N, H, W, dtype=torch.bool))
        if name == "soaked":
            for _ in range(30):
                rs.step_soak(1.0)
        vals[name] = rs.abrade(torch.full((N, 2), 0.5), 0.03, torch.full((N,), 20.0), torch.full((N,), 0.1), 0.01)
        if name == "dry":
            assert (rs.last_unsprayed_contact > 0.99).all()
        else:
            assert (rs.last_unsprayed_contact < 1e-6).all()
    assert torch.allclose(vals["wet"], 1.5 * vals["dry"], rtol=1e-4)
    assert torch.allclose(vals["soaked"], 4.0 * vals["dry"], rtol=1e-4)


def test_excess_force_flag_and_bounds():
    rs = make()
    F = torch.tensor([10.0, 39.0, 41.0, 80.0, 0.0, 20.0])
    for _ in range(200):
        rs.abrade(centre_uv(rs), 0.03, F, torch.full((N,), 0.3), 0.05)
    assert rs.last_excess_force.tolist() == [False, False, True, True, False, False]
    assert rs.rust.min() >= 0 and rs.rust.max() <= 1


def test_scrubbing_cleans_patch_to_completion():
    rs = make()
    dt, v, R = 1 / 30, 0.15, 0.025
    circ = 2 * math.pi * rs.radius
    steps = int(math.ceil(circ.max().item() / (v * dt)))
    for sweep in range(2):
        for row in range(0, H, 2):
            for s in range(steps):
                u = (s * v * dt / circ).clamp(max=1.0)
                uv = torch.stack([u, torch.full((N,), (row + 0.5) / H)], -1)
                rs.abrade(uv, R, torch.full((N,), 20.0), torch.full((N,), v), dt)
    assert (rs.fraction_remaining() < 0.05).all(), rs.fraction_remaining()


def test_theta_wraps_around():
    rs = RustState(1, H, W)
    rs.reset(radius=0.1, rust=torch.ones(1, H, W))
    rs.abrade(torch.tensor([[0.0, 0.5]]), 0.03, torch.tensor([20.0]), torch.tensor([0.2]), 0.1)
    row = rs.rust[0, H // 2]
    assert row[0] < 1 and row[-1] < 1, "both sides of the seam abraded"
    assert row[W // 2] == 1


def test_local_patch_and_sprayed_crop():
    rs = RustState(2, H, W)
    rust = torch.zeros(2, H, W)
    rust[0, 10, 5] = 0.7
    rust[1, 0, 63] = 0.9
    rs.reset(rust=rust, radius=0.1)
    uv = torch.tensor([[(5 + 0.5) / W, (10 + 0.5) / H], [(0 + 0.5) / W, (0 + 0.5) / H]])
    p = rs.local_patch(uv, 5)
    assert p.shape == (2, 5, 5)
    assert math.isclose(p[0, 2, 2].item(), 0.7, rel_tol=1e-6)
    assert math.isclose(p[1, 2, 1].item(), 0.9, rel_tol=1e-6)
    assert (p[1, :2] == 0).all()
    s = rs.local_sprayed(uv, 5)
    assert s[0, 2, 2] == 1.0


def test_nearest_rust_points_to_rust():
    rs = RustState(1, H, W)
    rust = torch.zeros(1, H, W)
    rust[0, 20, 40] = 1.0
    rs.reset(rust=rust, radius=0.1)
    uv = torch.tensor([[(30 + 0.5) / W, (20 + 0.5) / H]])
    off, d = rs.nearest_rust(uv)
    exp_arc = 10 / W * 2 * math.pi * 0.1
    assert math.isclose(off[0, 0].item(), exp_arc, rel_tol=1e-4)
    assert abs(off[0, 1].item()) < 1e-5
    rs.rust.zero_()
    off, d = rs.nearest_rust(uv)
    assert d.item() == 0


def test_add_patch_increases_rust_and_is_sprayed():
    rs = make()
    for _ in range(50):
        rs.abrade(centre_uv(rs), 0.04, torch.full((N,), 25.0), torch.full((N,), 0.3), 0.05)
    before_cells = rs.rusty_cells().clone()
    fr_before = rs.fraction_remaining().clone()
    ids = torch.tensor([0, 2])
    rs.add_patch(ids, generator=torch.Generator().manual_seed(3))
    after = rs.rusty_cells()
    assert (after[ids] >= before_cells[ids]).all() and (after[ids] > before_cells[ids]).any()
    assert torch.equal(after[[1, 3, 4, 5]], before_cells[[1, 3, 4, 5]])
    assert (rs.fraction_remaining()[ids] >= fr_before[ids] - 1e-6).all()
    assert (rs.fraction_remaining() <= 1 + 1e-6).all()
    assert (rs.sprayed | (rs.rust <= rs.p.rust_eps)).all()


def test_texture():
    rs = make()
    tex = rs.to_texture()
    assert tex.shape == (N, H, W, 4) and tex.dtype == torch.uint8
    assert (tex[..., 3] == 255).all()
    clean = rs.rust[0] < 0.01
    rusty = rs.rust[0] > 0.5
    assert tex[0][rusty][:, 0].float().mean() - tex[0][rusty][:, 2].float().mean() > 30
    assert (tex[0][clean & ~rs.sprayed[0]][:, :3].float() - torch.tensor([150.0, 152, 158])).abs().max() < 2
    ov = rs.to_texture([1], overlay=True)
    assert ov.shape == (1, H, W, 4)


def test_partial_reset_only_touches_ids():
    rs = make()
    rs.rust[1].zero_()
    snap = rs.rust.clone()
    rs.reset(env_ids=[0], radius=0.2)
    assert torch.equal(rs.rust[1:], snap[1:])
    assert rs.radius[0] == 0.2 and math.isclose(rs.radius[1].item(), 0.07, rel_tol=1e-5)


def test_cuda_if_available():
    if not torch.cuda.is_available():
        return
    rs = RustState(4, device="cuda")
    rs.reset()
    rs.abrade(torch.rand(4, 2, device="cuda"), 0.03, torch.full((4,), 20.0, device="cuda"),
              torch.full((4,), 0.2, device="cuda"), 0.1)
    rs.to_texture()


def test_theta_window_restricts_patches():
    g = torch.Generator().manual_seed(5)
    r = random_patches(32, H, W, radius=0.1, generator=g, theta_window=(0.6, 0.8))
    theta = (torch.arange(W).float() + 0.5) / W * 2 * math.pi
    d = torch.remainder(theta - 0.6 + math.pi, 2 * math.pi) - math.pi
    assert (r[..., d.abs() > 1.1] == 0).all()
    assert (r.sum(dim=(1, 2)) > 0).all()


def test_rig_meta_scene():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim", "output", "rig_deck_meta.json")
    if not os.path.exists(path):
        return
    from policy.envs.rust_state import rig_meta_scene

    m = rig_meta_scene(path)
    assert math.isclose(m["pipe_radius"], 0.15) and math.isclose(m["pipe_origin"][2], 1.1)
    assert 0.85 < m["damage_theta"] < 0.95
    th = m["damage_theta"]
    y = m["pipe_origin"][1] + m["pipe_ref"][1] * 0.15 * math.cos(th)
    z = m["pipe_origin"][2] + 0.15 * math.sin(th)
    assert abs(y - 0.095) < 0.01 and abs(z - 1.2199) < 0.01


def test_reachable_mask_limits_rust():
    rs = RustState(N, H, W, theta_window=(0.9, 0.35))
    reach = torch.zeros(N, H, W, dtype=torch.bool)
    reach[:, 8:24, :] = True
    reach[0] = False
    reach[0, 15:18, 12:16] = True
    rs.reset(radius=0.15, reachable=reach, generator=torch.Generator().manual_seed(2))
    assert (rs.rust[~reach] == 0).all()
    assert (rs.rust.sum(dim=(1, 2)) > 0).all()
    rs.add_patch(torch.arange(N))
    assert (rs.rust[~reach] == 0).all()

"""The real IsaacRigBridge on CPU (USD scene stubbed out): approach, contact force and scrubbing."""
import numpy as np
import torch

from bridge_stub import make_bridge


def test_approach_scrub_removes_rust_with_trained_force_band():
    bridge, world, pipe = make_bridge()
    theta = world.damage_theta
    assert bridge.approach(theta, 0.0, 0.15)
    world.sprayed[:] = True
    world.soak_time[:] = 60.0
    before = float(world.rust.sum())
    forces = []
    for k in range(40):
        bridge.apply_ee_delta(np.array([0.04 * np.sin(k / 3), 0.01 * np.cos(k / 4), 0.7]))
        forces.append(bridge.current_force())
    assert 5.0 <= np.median(forces) <= 40.0           # 25 N x press 0.7 ~ 17.5 N, inside the trained band
    assert float(world.rust.sum()) < before * 0.9


def test_unreachable_target_holds_pose_and_never_spikes():
    bridge, world, pipe = make_bridge()
    bridge.approach(world.damage_theta, 0.0, 0.15)
    q_before = dict(bridge.state.q)
    surf, n = pipe.surface_point(world.damage_theta + 2.5, 0.0)      # far side of the pipe
    assert not bridge._move_tool(surf - 0.01 * n, -n)
    assert bridge.current_force() < 40.0
    assert all(abs(bridge.state.q[k] - q_before[k]) < 1e-9 for k in q_before)


def test_second_patch_does_not_strand_the_arm():
    """Regression: after spraying a second patch the IK seed was stuck and every scrub move failed."""
    bridge, world, pipe = make_bridge()
    theta = world.damage_theta
    bridge.approach(theta, 0.0, 0.15)
    for th in np.linspace(theta - 0.6, theta - 0.3, 8):           # spray pass over a second patch
        s, n = pipe.surface_point(th, 0.13)
        bridge._move_tool(s + 0.15 * n, -n)
    s, n = pipe.surface_point(theta - 0.22, 0.13)
    assert bridge._move_tool(s - 0.007 * n, -n)


def test_misted_lens_returns_flat_frame():
    bridge, world, pipe = make_bridge()
    real = bridge.frames.render
    bridge.frames.render = lambda: {"inspection": np.full((360, 640, 3), 3, np.uint8), "hero": None}
    img = bridge.observe()
    assert img is not None and float(img.std()) < 6.0                 # orchestrator will HOLD
    bridge.frames.render = real
    img = bridge.observe()
    assert float(img.std()) > 6.0
    torch.manual_seed(0)

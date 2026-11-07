import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.rust_overlay import (camera_path, grid_to_mask, periodic_noise, rig_translation, rust_rgba,
                                 sleeve_mesh, tile_texture)


def test_sleeve_mesh_on_cylinder_with_rust_state_uvs():
    o, a, e, b = np.array([1.0, -0.5, 1.1]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    pts, counts, idx, uv = sleeve_mesh(o, a, e, b, radius=0.153, length=0.6, n_theta=48, n_z=8)
    assert pts.shape == (49 * 9, 3) and uv.shape == (49 * 9, 2)
    assert len(counts) == 48 * 8 and all(c == 4 for c in counts) and len(idx) == 4 * len(counts)
    d = pts - o
    za = d @ a
    rho = np.hypot(d @ e, d @ b)
    assert np.allclose(rho, 0.153) and np.isclose(za.min(), -0.3) and np.isclose(za.max(), 0.3)
    theta = np.arctan2(d @ b, d @ e)
    k = 5 * 49 + 10
    assert np.isclose(uv[k, 0], (theta[k] % (2 * math.pi)) / (2 * math.pi))
    assert np.isclose(uv[k, 1], (za[k] + 0.3) / 0.6)


def test_grid_to_mask_upsamples_with_theta_wrap():
    g = np.zeros((4, 8), np.float32)
    g[:, 0] = 1.0
    m = grid_to_mask(g, 32, 64)
    assert m.shape == (32, 64) and m.dtype == np.float32
    assert m[:, 4].min() > 0.9
    assert m[:, 0].min() > 0.4 and m[:, -1].min() > 0.4
    assert m[:, 32].max() < 1e-6


def test_grid_to_mask_rows_are_v_bottom_up():
    g = np.zeros((4, 8), np.float32)
    g[0] = 1.0
    m = grid_to_mask(g, 32, 64, flip_v=True)
    assert m[-1].min() > 0.9 and m[0].max() < 1e-6


def test_rust_rgba_alpha_follows_rust():
    tex = np.full((16, 16, 3), 120, np.uint8)
    noise = periodic_noise(32, 64, seed=0)
    assert noise.shape == (32, 64) and 0 <= noise.min() and noise.max() <= 1
    clean = rust_rgba(np.zeros((32, 64), np.float32), tex, noise)
    rusty = rust_rgba(np.ones((32, 64), np.float32), tex, noise)
    assert clean.shape == (32, 64, 4) and clean.dtype == np.uint8
    assert clean[..., 3].max() == 0
    assert rusty[..., 3].min() == 255


def test_rust_rgba_partial_rust_is_ragged():
    tex = np.full((16, 16, 3), 120, np.uint8)
    noise = periodic_noise(64, 128, seed=1)
    a = rust_rgba(np.full((64, 128), 0.45, np.float32), tex, noise)[..., 3]
    frac = (a > 127).mean()
    assert 0.05 < frac < 0.95


def test_tile_texture_shape_and_wrap():
    tex = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    t = tile_texture(tex, 8, 12, tiles_v=2, tiles_u=3)
    assert t.shape == (8, 12, 3)
    assert np.array_equal(t[:4, :4], t[4:, 8:])


def test_rig_translation_maps_deck_pipe_onto_env_pipe():
    meta = {"env_to_world": {"translation": (-0.49, 0.5, 0.0), "yaw": 0.0}}
    t = rig_translation(meta, env_origin=(0.0, 0.0, 0.0))
    deck_pipe_point = np.array([-0.49, 0.0, 1.1])
    assert np.allclose(deck_pipe_point + t, [0.0, -0.5, 1.1])


def test_camera_path_interpolates_smoothly():
    keys = [(0.0, (0, 0, 0), (1, 0, 0)), (2.0, (2, 0, 0), (1, 1, 0))]
    eye, look = camera_path(keys, 1.0)
    assert np.allclose(eye, [1, 0, 0]) and np.allclose(look, [1, 0.5, 0])
    assert np.allclose(camera_path(keys, -1)[0], [0, 0, 0]) and np.allclose(camera_path(keys, 5)[0], [2, 0, 0])
    e1, _ = camera_path(keys, 0.2)
    assert e1[0] < 0.2

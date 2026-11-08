import base64
import math

import numpy as np

from policy.gasbot.locate import SourceEstimator
from rig.mission.obsviews import H, PX, W, cast, encode_small, gas_belief, lidar_bev, occupancy, rust_obs

WALK = [{"id": "w0", "a": [-10.0, 0.0, 42.5], "b": [10.0, 0.0, 42.5], "kind": "walk", "narrow": False},
        {"id": "w1", "a": [0.0, 0.0, 42.5], "b": [0.0, 8.0, 42.5], "kind": "catwalk", "narrow": True},
        {"id": "w9", "a": [-10.0, 3.0, 20.0], "b": [10.0, 3.0, 20.0], "kind": "walk", "narrow": False}]
HAZ = {"obstacles": [{"id": "cr1", "kind": "crate", "xyz": [5.0, 0.0, 42.5], "radius": 0.6}], "closed": []}


def kb(entry):
    return len(base64.b64decode(entry["jpg"])) / 1024


def test_occupancy_frees_walkways_on_the_deck_only():
    occ = occupancy(WALK, 42.5, {}, [], (0.0, 0.0))
    assert occ.shape == (H, W)
    assert occ[H // 2, W // 2] == 0
    assert occ[H // 2 - int(3 * PX), W // 2 + int(5 * PX)] == 1
    assert occ[H // 2 - int(5 * PX), W // 2] == 0


def test_rays_stop_at_walls_and_obstacles():
    occ = occupancy(WALK, 42.5, HAZ, [], (0.0, 0.0))
    r, cls = cast(occ, (W / 2, H / 2), 360, 12.0 * PX)
    assert r.shape == (360,) and cls.shape == (360,)
    assert abs(r[0] / PX - 4.4) < 0.3 and cls[0] == 2
    assert abs(r[270] / PX - 1.1) < 0.2 and cls[270] == 1
    assert r[90] / PX > 7.5


def test_lidar_bev_shape_budget_and_determinism():
    pose = (0.0, 0.0, 42.5, 0.3)
    path = [[0.0, 0.0, 42.5], [0.0, 6.0, 42.5], [-8.0, 0.0, 42.5]]
    a, info = lidar_bev("R1", pose, path, WALK, HAZ, [{"xyz": [0.0, 4.0, 42.5]}])
    b, _ = lidar_bev("R1", pose, path, WALK, HAZ, [{"xyz": [0.0, 4.0, 42.5]}])
    assert a.size == (W, H) and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert info["rays"] == 360 and info["hits"] > 0
    e = encode_small(a, "R1 · lidar BEV · 360 rays", 1)
    assert kb(e) <= 10 and e["w"] == W and e["h"] == H


def test_gas_belief_draws_posterior_track_and_estimate():
    est = SourceEstimator((-12, 12), (-12, 12), 0.25)
    w = np.array([1.0, 0.0])
    for k in range(20):
        est.observe((3.0 + 0.2 * k, 0.4 * math.sin(k)), w, k % 2 == 0)
    track = [[6.0 + 0.3 * k, 0.2 * k] for k in range(15)]
    lels = [0.01 * k for k in range(15)]
    flanges = {"F5": [0.0, 0.0, 43.6], "F4": [-6.0, 2.0, 43.6]}
    im, info = gas_belief("R2", (6.0, 2.0, 42.5, math.pi), est, track, lels, w, 4.0, flanges, WALK, "SURGE", 10)
    assert im.size == (W, H)
    assert info["flange"] in flanges and info["sigma_m"] > 0
    arr = np.asarray(im).astype(int)
    assert ((arr[..., 0] - arr[..., 2]) > 80).sum() > 50
    e = encode_small(im, "x", 1)
    assert kb(e) <= 10
    im2, _ = gas_belief("R2", (6.0, 2.0, 42.5, math.pi), est, track, lels, w, 4.0, flanges, WALK, "SURGE", 10)
    assert np.array_equal(np.asarray(im), np.asarray(im2))


def test_rust_obs_cleans_with_the_scrub_phase_and_clamps_force():
    f = [30 + 6 * math.sin(k / 3) for k in range(60)]
    a, ia = rust_obs("R1", "rust:rust_mid_east", "SCRUB", 0.1, f, 0.55)
    b, ib = rust_obs("R1", "rust:rust_mid_east", "SCRUB", 0.9, f, 0.06)
    c, _ = rust_obs("R1", "rust:rust_mid_east", "SCRUB", 0.9, f, 0.06)
    d, idd = rust_obs("R1", "rust:rust_low_east", "SCRUB", 0.1, f, 0.55)
    assert a.size == (W, H)
    assert ia["rust"] > ib["rust"]
    assert ia["brush"] != ib["brush"]
    assert np.array_equal(np.asarray(b), np.asarray(c))
    assert not np.array_equal(np.asarray(a), np.asarray(d))
    assert ia["force_n"] <= 38.0
    assert kb(encode_small(a, "x", 1)) <= 10
    p, ip = rust_obs("R1", "rust:rust_mid_east", "SOAK", 0.5, [], None)
    assert ip["rust"] > 0.2 and p.size == (W, H)

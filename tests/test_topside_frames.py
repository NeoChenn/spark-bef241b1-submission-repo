import math

import numpy as np
import pytest

from sim.topside_frames import (
    clear_of,
    frame,
    horizon_gain,
    kelvin_rgb,
    ladder,
    pipe_rack,
    rail,
    segment_pose,
    stair_tower,
    stations,
)


def rotate(q, v):
    w, x, y, z = q
    u = np.array((x, y, z))
    v = np.array(v, dtype=float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def test_stations_cover_span_with_max_spacing():
    s = stations(0.0, 10.0, 4.0)
    assert s[0] == 0.0 and s[-1] == 10.0
    assert len(s) == 4
    assert max(np.diff(s)) <= 4.0 + 1e-9


def test_stations_short_span_gives_ends():
    assert stations(2.0, 3.0, 5.0) == [2.0, 3.0]


@pytest.mark.parametrize("d", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (1, 1, 1), (0, -2, 3)])
def test_segment_pose_maps_x_axis_onto_segment(d):
    p = np.array((1.0, 2.0, 3.0))
    q = p + np.array(d, dtype=float)
    mid, quat, length = segment_pose(p, q)
    assert np.allclose(mid, (p + q) / 2)
    assert math.isclose(length, np.linalg.norm(d))
    assert math.isclose(np.linalg.norm(quat), 1.0, rel_tol=1e-9)
    assert np.allclose(rotate(quat, (length, 0, 0)), q - p, atol=1e-9)


def test_frame_columns_beams_and_braces():
    f = frame((0, 0, 20), (10, 6, 31), levels=(20, 25.5, 31), bay=5)
    assert len(f["columns"]) == 3 * 3 * 2
    for a, b in f["columns"]:
        assert a[0] == b[0] and a[1] == b[1] and b[2] > a[2]
    beam_levels = {a[2] for a, b in f["beams"]}
    assert beam_levels == {25.5, 31}
    assert all(a[2] == b[2] for a, b in f["beams"])
    assert len(f["beams"]) == 2 * (3 * 2 + 3 * 2)
    assert f["braces"]
    for a, b in f["braces"]:
        assert a[2] != b[2]
        on_face = (a[0] == b[0] and a[0] in (0, 10)) or (a[1] == b[1] and a[1] in (0, 6))
        assert on_face


def test_frame_skips_open_faces():
    f = frame((0, 0, 0), (10, 6, 5), levels=(0, 5), bay=5, brace_faces=("+x",))
    assert f["braces"]
    assert all(a[0] == 10 and b[0] == 10 for a, b in f["braces"])


def test_rail_has_two_rails_posts_and_toe_board():
    r = rail((0, 0), (6, 0), 20.0)
    assert len(r["rails"]) == 2
    assert {a[2] for a, _ in r["rails"]} == {21.1, 20.55}
    assert len(r["posts"]) == 5
    assert all(a[2] == 20.0 and b[2] == 21.1 for a, b in r["posts"])
    (a, b), = r["toe"]
    assert a[2] == b[2] == 20.075


def test_stair_tower_climbs_full_height_with_landings():
    s = stair_tower((0, 0), 20.0, 31.25, axis="x")
    tops = [c[2] + h[2] / 2 for c, h in s["steps"]]
    assert min(c[2] for c, _ in s["steps"]) > 20.0
    assert max(tops) < 31.25
    assert max(c[2] + h[2] / 2 for c, h in s["landings"]) == pytest.approx(31.25)
    treads = [c[2] + h[2] / 2 for c, h in s["steps"] + s["landings"]]
    rises = np.diff(sorted({round(z, 3) for z in treads + [20.0]}))
    assert rises.max() <= 0.22
    assert len(s["landings"]) >= 2
    assert max(c[2] for c, _ in s["landings"]) <= 31.25
    assert s["frame"]["columns"]


def test_ladder_rungs_spacing():
    l = ladder((0, 0, 0), 3.0, facing="x")
    assert len(l["rails"]) == 2
    zs = sorted(a[2] for a, _ in l["rungs"])
    assert np.allclose(np.diff(zs), 0.3)
    assert zs[-1] < 3.0


def test_pipe_rack_portals_and_pipes():
    r = pipe_rack((0, 0), (24, 0), 20.0, tiers=(24.0, 26.0), width=3.0, pipes_per_tier=5)
    assert len(r["columns"]) == 2 * 5
    assert len(r["pipes"]) == 10
    for (a, b), radius in r["pipes"]:
        assert a[1] == b[1] and abs(a[1]) < 1.5
        assert radius > 0
    assert r["valves"]
    assert r["trays"]


def test_clear_of_rejects_overlaps():
    zones = [((0, 0), (10, 10))]
    assert not clear_of(((5, 5), (6, 6)), zones)
    assert not clear_of(((-2, -2), (1, 1)), zones)
    assert clear_of(((11, 0), (12, 3)), zones)


def test_kelvin_rgb_ordering():
    sodium, neutral, led = kelvin_rgb(2000), kelvin_rgb(4000), kelvin_rgb(6000)
    for c in (sodium, neutral, led):
        assert max(c) == pytest.approx(1.0)
        assert min(c) >= 0
    assert sodium[2] < neutral[2] < led[2]
    assert sodium[0] >= neutral[0] >= led[0]


def test_horizon_gain_dark_at_horizon_full_overhead():
    g = horizon_gain(np.array([-30.0, 0.0, 5.0, 20.0, 60.0, 90.0]), floor=0.05, top=25.0)
    assert g[0] == pytest.approx(0.05)
    assert g[1] < 0.15
    assert np.all(np.diff(g) >= 0)
    assert g[-1] == pytest.approx(1.0)
    assert g[-2] == pytest.approx(1.0)


def test_keep_facade_keeps_quarters_and_fascias_only():
    from sim.topside_frames import keep_facade

    assert keep_facade((-25.0, -37.5, 20.0), (-25.0, -27.5, 30.0))
    assert keep_facade((-25.0, 25.0, 18.8), (25.0, 25.0, 20.0))
    assert not keep_facade((25.3, -37.5, 31.2), (37.5, -37.5, 40.0))
    assert not keep_facade((-24.5, -10.0, 42.5), (-17.0, -10.0, 48.8))

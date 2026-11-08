import numpy as np

from sim import facade_layout as fl


def area(rects):
    return sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)


def test_bays_centered_and_inside():
    b = fl.bays(0.0, 12.0)
    assert len(b) == 3
    assert abs(np.mean(b) - 6.0) < 1e-9
    assert min(b) - fl.WINDOW_W / 2 > 0 and max(b) + fl.WINDOW_W / 2 < 12.0
    assert fl.bays(0.0, 2.0) == []


def test_subtract_covers_wall_minus_holes():
    rect = (0.0, 0.0, 10.0, 4.0)
    cuts = [(1.0, 1.0, 2.0, 2.0), (5.0, 0.5, 6.5, 3.0)]
    parts = fl.subtract(rect, cuts)
    assert abs(area(parts) - (40.0 - 1.0 - 3.75)) < 1e-9
    for p in parts:
        for c in cuts:
            assert not (p[0] < c[2] and c[0] < p[2] and p[1] < c[3] and c[1] < p[3])


def test_openings_doors_replace_windows_and_tiles_valid():
    w = fl.WALLS[0]
    items = fl.openings(w)
    doors = [o for o in items if o["kind"] == "door"]
    assert len(doors) == len(w["doors"])
    hs = fl.holes(items)
    for i, a in enumerate(hs):
        for b in hs[i + 1:]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
        assert w["along"][0] <= a[0] and a[2] <= w["along"][1]
        assert w["z"][0] <= a[1] and a[3] <= w["z"][1]
    tiles = [o["tile"] for o in items if o["kind"] == "window"]
    assert all(0 <= t < fl.TILES * fl.TILES for t in tiles)


def test_tile_mix_has_dark_and_lit():
    tiles = [fl.pick_tile(f"k{i}") for i in range(400)]
    dark = sum(t >= 12 for t in tiles) / len(tiles)
    assert 0.15 < dark < 0.4
    assert len(set(tiles)) > 10
    assert fl.pick_tile("same") == fl.pick_tile("same")


def test_quad_faces_outward():
    for w in fl.WALLS:
        q = fl.quad(w, w["along"][0], w["z"][0], w["along"][1], w["z"][1], fl.CLAD_DEPTH)
        assert np.dot(fl.face_normal(q), fl.outward(w)) > 0.99


def test_reveal_faces_point_into_hole():
    w = fl.WALLS[1]
    hole = (-5.0, 22.0, -3.7, 23.0)
    centre = np.array(fl.to_world(w, -4.35, 22.5, 0.2))
    for f in fl.reveal(w, hole, fl.CLAD_DEPTH, fl.GLASS_DEPTH):
        n = np.array(fl.face_normal(f))
        mid = np.mean(np.asarray(f), axis=0)
        assert np.dot(n, centre - mid) > 0


def test_streak_stays_on_wall():
    for w in fl.WALLS:
        for o in fl.openings(w):
            if o["kind"] == "window":
                h = fl.streak_height(o, w)
                assert 0.0 <= h and o["z0"] - h >= w["z"][0]


def test_tile_uv_in_unit_square():
    for t in range(16):
        u0, v0, u1, v1 = fl.tile_uv(t)
        assert 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1


def test_helideck_lights_spacing():
    pts = fl.helideck_lights([(0, 0), (9, 0), (9, 9)], spacing=3.0)
    assert len(pts) == 3 + 3 + 5
    assert pts[0] == (0, 0)

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


def test_glow_follows_preset():
    import sys
    sys.path.insert(0, "sim")
    from sim import facade
    assert facade.glow_scale("interior", 0.15) < facade.glow_scale("interior", 1.0)
    assert facade.glow_scale("lamp", 1.0) == facade.GLOW["lamp"]
    assert facade.glow_scale("lamp", 0.0) == 0.0


def test_all_walls_holes_disjoint_and_inside():
    for w in fl.WALLS:
        hs = fl.holes(fl.openings(w))
        for i, a in enumerate(hs):
            assert w["along"][0] <= a[0] and a[2] <= w["along"][1]
            assert w["z"][0] <= a[1] and a[3] <= w["z"][1]
            for b in hs[i + 1:]:
                assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def test_windows_irregular():
    wins = [o for w in fl.WALLS for o in fl.openings(w) if o["kind"] == "window"]
    assert len({(o["w"], o["h"]) for o in wins}) >= 3
    assert any(o["w"] == fl.OFFICE_W for o in wins)
    assert len({round(o["z0"] - fl.WALLS[0]["z"][0], 2) for o in fl.openings(fl.WALLS[0])}) > 3
    rows = {}
    for w in fl.WALLS:
        for o in fl.openings(w):
            if o["kind"] == "window":
                rows.setdefault((w["name"], o["storey"]), []).append(fl.lit(o))
    shares = [sum(v) / len(v) for v in rows.values() if len(v) >= 3]
    assert max(shares) - min(shares) > 0.3


def test_annex_door_clear_of_id_board():
    w = next(w for w in fl.WALLS if w["name"] == "annex_s")
    for h in fl.holes(fl.openings(w)):
        for k in fl.KEEP_CLEAR:
            assert not (h[0] < k[2] and k[0] < h[2] and h[1] < k[3] and k[1] < h[3])


def test_pick_tile_dark_bias():
    keys = [f"d{i}" for i in range(400)]
    dark_hi = sum(fl.TILE_KELVIN[fl.pick_tile(k, 3.0)] == 0 for k in keys)
    dark_lo = sum(fl.TILE_KELVIN[fl.pick_tile(k, 0.3)] == 0 for k in keys)
    assert dark_hi > dark_lo + 60


def test_style_override_gives_office_row():
    w = fl.WALLS[0]
    assert any(o["w"] == fl.OFFICE_W for o in fl.openings(w) if o["storey"] == 1)

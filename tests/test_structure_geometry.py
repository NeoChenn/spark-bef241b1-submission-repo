import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import structure_geometry as sg
import structure_textures as st


def _leg_points():
    pts = []
    for cx, cy in ((-17.5, -17.5), (17.5, -17.5), (-17.5, 17.5), (17.5, 17.5)):
        for k in range(32):
            a = k / 32 * math.tau
            pts.append((cx + 5 * math.cos(a), cy + 5 * math.sin(a), 18.75))
            pts.append((cx - 1.77 + 5 * math.cos(a), cy - 1.77 + 5 * math.sin(a), -56.21))
    return np.array(pts)


def test_leg_axes_recover_four_inclined_columns():
    legs = sg.leg_axes(_leg_points())
    assert len(legs) == 4
    for leg in legs:
        assert math.isclose(leg["radius"], 5.0, abs_tol=0.05)
        assert math.isclose(leg["top"][2], 18.75) and math.isclose(leg["bottom"][2], -56.21)
    leg = next(l for l in legs if l["top"][0] > 0 and l["top"][1] > 0)
    assert np.allclose(sg.axis_at(leg, 18.75), (17.5, 17.5), atol=0.05)
    assert np.allclose(sg.axis_at(leg, -56.21), (15.73, 15.73), atol=0.05)


def test_edge_profile_is_periodic_and_bounded():
    z = sg.edge_profile(64, 1.0, 0.6, seed=3)
    assert z.shape == (65,)
    assert math.isclose(z[0], z[-1])
    assert np.all(np.abs(z - 1.0) <= 0.6 + 1e-9)
    assert z.std() > 0.05


def test_column_band_follows_axis_and_edges():
    leg = {"top": np.array((0.0, 0.0, 10.0)), "bottom": np.array((-1.0, 0.0, -10.0)), "radius": 2.0}
    low = sg.edge_profile(16, 0.0, 0.3, seed=1)
    m = sg.column_band(leg, 16, low, np.full(17, 6.0), rows=4, radius=2.0)
    pts = m["points"]
    assert len(m["counts"]) == 16 * 4 and len(m["indices"]) == 4 * 16 * 4
    centres = np.array([sg.axis_at(leg, z) for z in pts[:, 2]])
    assert np.allclose(np.linalg.norm(pts[:, :2] - centres, axis=1), 2.0, atol=1e-6)
    assert np.allclose(pts[:17, 2], low)
    assert np.allclose(pts[-17:, 2], 6.0)
    assert np.allclose(np.linalg.norm(m["normals"], axis=1), 1.0)


def test_split_by_bands_cuts_vertical_member():
    cuts = (1.0, 5.0)
    parts = sg.split_by_bands((0, 0, -4), (0, 0, 10), cuts)
    assert [p[2] for p in parts] == [0, 1, 2]
    assert np.allclose(parts[0][1], (0, 0, 1.0)) and np.allclose(parts[2][0], (0, 0, 5.0))
    assert sg.split_by_bands((0, 0, 6), (0, 0, 9), cuts) == [((0, 0, 6), (0, 0, 9), 2)]


def test_scanline_intervals_clip_to_polygon():
    poly = ((0.3, -37.5), (25.0, -37.5), (25.0, 25.0), (-25.0, 25.0), (-25.0, -22.9))
    assert np.allclose(sg.scanline(poly, 0.0, axis="x"), [(-25.0, 25.0)])
    (a, b), = sg.scanline(poly, -30.0, axis="x")
    assert -25.0 < a < 0.3 and math.isclose(b, 25.0)
    assert sg.scanline(poly, 40.0, axis="x") == []
    (a, b), = sg.scanline(poly, 10.0, axis="y")
    assert math.isclose(a, -37.5) and math.isclose(b, 25.0)


def test_stations_skip_lights():
    xs = sg.clear_stations(-25.0, 25.0, 3.0, avoid=(2.0, -16.0), clear=1.3)
    assert all(abs(x - 2.0) >= 1.3 and abs(x + 16.0) >= 1.3 for x in xs)
    assert min(np.diff(xs)) > 1.0 and xs[0] >= -25.0 and xs[-1] <= 25.0


def test_ring_mesh_is_closed_annulus():
    m = sg.ring_mesh(5.0, 5.4, 0.3, 24)
    r = np.linalg.norm(m["points"][:, :2], axis=1)
    assert r.min() >= 5.0 - 1e-9 and r.max() <= 5.4 + 1e-9
    assert np.allclose(sorted(set(np.round(m["points"][:, 2], 6))), (-0.15, 0.15))
    assert len(m["counts"]) == 24 * 4


def test_landing_members_sit_outside_leg():
    leg = {"top": np.array((17.5, -17.5, 18.75)), "bottom": np.array((15.73, -19.27, -56.21)), "radius": 5.0}
    bl = sg.boat_landing(leg, -60.0)
    for p, q, r in bl["tubes"]:
        for z in (p, q):
            c = sg.axis_at(leg, z[2])
            assert np.hypot(z[0] - c[0], z[1] - c[1]) >= leg["radius"] - 1.2
    assert bl["deck"][2] > 3.0
    assert len(bl["rungs"]) > 20
    assert all(r[0][2] < leg["top"][2] + 0.5 for r in bl["rungs"])


def test_risers_cover_waterline_and_clamps_follow():
    leg = {"top": np.array((17.5, 17.5, 18.75)), "bottom": np.array((15.73, 15.73, -56.21)), "radius": 5.0}
    rs = sg.risers(leg, ((30.0, 0.3), (45.0, 0.5)), bottom=-20.0)
    assert len(rs["pipes"]) == 2
    for p, q, r in rs["pipes"]:
        assert p[2] <= -20.0 + 1e-9 and q[2] >= 18.75
    assert len(rs["clamps"]) >= 2 * 6


def test_braces_connect_columns():
    legs = sg.leg_axes(_leg_points())
    br = sg.braces(legs, z=7.0, deck=16.4)
    assert len(br["horizontal"]) == 4 and len(br["diagonal"]) == 8
    for p, q, r in br["horizontal"]:
        assert math.isclose(p[2], 7.0) and math.isclose(q[2], 7.0)


def test_drip_mask_fades_downward():
    m = st.drip_mask(64, 128, seed=2)
    assert m.shape == (128, 64) and m.dtype == np.uint8
    assert m[:16].mean() > m[-16:].mean()


def test_marine_growth_maps():
    maps = st.marine_growth(128, seed=4)
    assert set(maps) == {"diff", "rough", "nor_gl"}
    assert maps["diff"].shape == (128, 128, 3) and maps["nor_gl"].shape == (128, 128, 3)
    n = maps["nor_gl"].astype(float) / 127.5 - 1
    assert n[..., 2].min() > 0.0
    assert maps["diff"][..., 1].mean() >= maps["diff"][..., 2].mean()


def test_structure_bands_and_recipes():
    import photoreal as pr
    import structure

    assert [structure.band_of(z) for z in (-3.0, 2.0, 9.0)] == [0, 1, 2]
    assert {r["kind"] for r in structure.RECIPES.values()} <= set(pr.KINDS)
    for name, recipe in structure.RECIPES.items():
        _, values = pr.shader_inputs(recipe)
        assert all(v.exists() for v in values.values() if isinstance(v, Path)), name


def test_walkway_rails_and_hangers():
    w = sg.walkway((0, 0), (10, 0), 16.0, hang_to=18.0)
    centre, angle, size = w["deck"]
    assert np.allclose(centre, (5, 0, 16)) and math.isclose(angle, 0.0) and math.isclose(size[0], 10.0)
    assert len(w["rails"]) == 4 and len(w["posts"]) == 2 * 5
    assert all(math.isclose(b[2], 18.0) for _, b in w["hangers"])


def test_caged_ladder_spans_levels():
    l = sg.caged_ladder((1.0, 2.0), 31.25, 42.5, 0.0)
    assert len(l["rails"]) == 2 and all(math.isclose(a[2], 31.25) and math.isclose(b[2], 43.5) for a, b in l["rails"])
    assert len(l["rungs"]) == 37 and all(31.25 < a[2] < 42.5 for a, _ in l["rungs"])
    assert l["hoops"][0][0][2] > 33.0 and len(l["straps"]) == 3


def test_edge_detail_ribs_sit_outside():
    e = sg.edge_detail((0, -37.5), (10, -37.5), 30.0, 1.25, (0, -1))
    assert len(e["ribs"]) == 4
    assert all(r[0][1] < -37.5 and math.isclose(r[0][2], 30.625) for r in e["ribs"])
    assert math.isclose(e["strip"][0][2], 31.2)


def test_lattice_boom_chords_follow_axis_and_taper():
    b = sg.lattice_boom((-26, 8, 42), (-57.5, 8, 69.6), 3.2, 1.6)
    assert len(b["chords"]) == 3 * (len(b["struts"]) // 3 - 1)
    first = [s for s in b["struts"] if s[2] == 0.0]
    last = [s for s in b["struts"] if s[2] == 1.0]
    span0 = max(np.linalg.norm(np.subtract(p, q)) for p, q, _ in first)
    span1 = max(np.linalg.norm(np.subtract(p, q)) for p, q, _ in last)
    assert math.isclose(span0, 3.2, rel_tol=0.05) and math.isclose(span1, 1.6, rel_tol=0.05)
    assert abs(np.dot(b["up"], b["axis"])) < 1e-9 and b["up"][2] > 0


def test_hex_and_disc_rings():
    h = sg.hex_ring((0, 0), 10.0, 9.0, 5.0)
    assert len(h["points"]) == 12 and len(h["counts"]) == 6
    d = sg.disc_ring((1, 2), 5.0, 6.0, 3.0)
    r = np.hypot(d["points"][:, 0] - 1, d["points"][:, 1] - 2)
    assert r.min() >= 5.0 - 1e-9 and r.max() <= 6.0 + 1e-9


def test_perimeter_points_cover_polygon():
    pts = sg.perimeter_points(sg.hexagon((0, 0), 10.0, 0.0), 2.5)
    assert len(pts) == 24
    assert np.allclose(np.linalg.norm(np.array(pts)[:, :2], axis=1).max(), 10.0)


def test_lifeboat_bounds_and_normals():
    m = sg.lifeboat(8.0, 2.8, 1.0, 1.5)
    p = m["points"]
    assert np.allclose(p.min(0), (-4.0, -1.4, -1.0), atol=0.05) and np.allclose(p.max(0), (4.0, 1.4, 1.5), atol=0.05)
    assert np.allclose(np.linalg.norm(m["normals"], axis=1), 1.0)
    side = (np.abs(p[:, 0]) < 1.0) & (np.abs(p[:, 2]) < 0.1)
    assert (np.sign(m["normals"][side, 1]) == np.sign(p[side, 1])).all()


def test_helideck_stays_dry_and_splash_stays_wet():
    import structure

    assert structure.wetness_for("helideck", 0.64) == 0.15
    assert structure.wetness_for("growth", 0.3) == 1.0
    assert structure.wetness_for("column", 0.64) == 0.64


def test_rect_lattice_vertical_derrick():
    d = sg.rect_lattice((20, 20, 45.9), (20, 20, 81.3), 7.4, 2.2, panel=3.0, cross=True)
    base = np.array(d["rings"][0])
    top = np.array(d["rings"][-1])
    assert np.allclose(np.ptp(base[:, :2], axis=0), 7.4) and np.allclose(np.ptp(top[:, :2], axis=0), 2.2)
    panels = len(d["rings"]) - 1
    assert len(d["chords"]) == 4 * panels and len(d["diagonals"]) == 8 * panels


def test_ship_planform_and_extrusion():
    ring = sg.ship_planform(80.0, 18.0)
    assert np.isclose(ring[:, 0].max(), 40.0) and np.isclose(np.abs(ring[:, 1]).max(), 9.0)
    m = sg.extrude_ring(ring, -5.0, 5.0)
    centre = ring.mean(0)
    out = ((m["points"][:len(ring), :2] - centre) * m["normals"][:len(ring), :2]).sum(1)
    assert (out > -1e-6).mean() > 0.95
    f = sg.fan(ring, 4.0)
    assert len(f["counts"]) == len(ring)


def test_containers_fit_deck():
    cs = sg.container_stacks(-38.0, 14.0, 7.5)
    assert cs and all(-38.0 <= c[0][0] - 3.03 and c[0][0] + 3.03 <= 14.0 + 1e-9 for c in cs)
    assert all(abs(c[0][1]) + 1.22 <= 7.5 + 1e-9 for c in cs)


def test_vessel_lathe_oriented_horizontally():
    prof = sg.vessel_profile(8.0, 1.5)
    assert np.isclose(prof[:, 0].min(), -4.75) and np.isclose(prof[:, 0].max(), 4.75) and np.isclose(prof[:, 1].max(), 1.5)
    m = sg.orient(sg.lathe(prof, 24), (10, 5, 45), (1, 0, 0))
    p = m["points"]
    assert np.allclose(p.min(0), (5.25, 3.5, 43.5), atol=0.02) and np.allclose(p.max(0), (14.75, 6.5, 46.5), atol=0.02)
    mid = np.abs(p[:, 0] - 10) < 0.5
    out = p[mid] - (p[mid] * (1, 0, 0) + (0, 5, 45))
    assert ((m["normals"][mid] * out).sum(1) > 0).all()


def test_spiral_stair_climbs():
    st_ = sg.spiral_stair((0, 0), 4.0, 42.5, 51.5)
    zs = [s[0][2] for s in st_["steps"]]
    assert zs == sorted(zs) and np.isclose(zs[-1], 51.5)
    assert all(np.isclose(math.hypot(s[0][0], s[0][1]), 4.45) for s in st_["steps"])


def test_pipe_rack_bents():
    r = sg.pipe_rack_bents(0.0, 18.0, 9.0, 2.4, (46.0, 48.0), 42.5)
    assert len(r["xs"]) == 4 and len(r["columns"]) == 8 and len(r["beams"]) == 8 and len(r["ties"]) == 4


def test_route_rounds_corners():
    pts = [(0, 0, 0), (0, 0, 2), (3, 0, 2), (3, 2, 2)]
    r = sg.route(pts, 0.5)
    assert np.allclose(r[0], pts[0]) and np.allclose(r[-1], pts[-1])
    seg = np.diff(r, axis=0)
    seg = seg[np.linalg.norm(seg, axis=1) > 1e-9]
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    turn = np.degrees(np.arccos(np.clip((seg[1:] * seg[:-1]).sum(1), -1, 1)))
    assert turn.max() < 30.0
    assert np.linalg.norm(r - (0, 0, 2), axis=1).min() > 0.1


def test_stations_along_spacing():
    s = sg.stations_along((0.0, -36.9), (24.0, -36.9), 3.0)
    assert len(s["points"]) == 9 and np.allclose(s["dir"], (1, 0)) and np.allclose(s["normal"], (0, 1))
    assert math.isclose(s["angle"], 0.0) and math.isclose(s["length"], 24.0)

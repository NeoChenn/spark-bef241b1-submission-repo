import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import greebles as g


def test_layout_keeps_work_zone_and_damage_clear():
    assert g.layout_issues() == []


def test_pipe_details_stay_off_the_damage_spot_and_brush_path():
    for kind, x, y, z, line in g.footprints():
        if line == "main":
            assert abs(x - g.DAMAGE[0]) - g.HALF_LENGTH[kind] >= 0.5
        else:
            assert math.dist((x, y, z), g.DAMAGE) > 0.9


def test_brush_clearance_counts_item_length():
    assert g.brush_clear("weld", g.DAMAGE[0] + 0.52)
    assert not g.brush_clear("flange", g.DAMAGE[0] + 0.55)


def test_bolt_ring_sits_between_pipe_and_flange_rim():
    for line, (_, _, r, _) in g.LINES.items():
        f = g.flange_dims(r)
        assert r < f["pcd"] - f["nut_r"] and f["pcd"] + f["nut_r"] < f["outer"]
        ring = g.bolt_ring(f["pcd"], f["count"])
        assert len(ring) == f["count"]
        assert all(abs(math.hypot(*p) - f["pcd"]) < 1e-9 for p in ring)
        assert all(abs(p[1]) > 1e-6 for p in ring)


def test_bolt_count_grows_with_pipe_size():
    assert g.bolt_count(0.05) < g.bolt_count(0.1) < g.bolt_count(0.2)


def test_stations_cover_span_evenly():
    s = g.stations(0.0, 3.0, 1.0, margin=0.5)
    assert s[0] == 0.5 and s[-1] == 2.5 and len(s) == 3
    assert g.stations(1.0, 1.2, 1.0, margin=0.2) == []


def test_column_mounts_sit_on_the_named_face():
    assert g.on_column((0.0, 4.4), "-y") == (0.0, 4.4 - 0.125)
    assert g.on_column((7.0, 0.0), "+x", 0.2) == (7.2, 0.0)


def test_new_flanges_avoid_existing_supports_and_flanges():
    for line, xs in g.NEW_FLANGES.items():
        taken = g.SUPPORTS[line] + g.EXISTING_FLANGES[line]
        assert all(abs(x - t) > 0.3 for x in xs for t in taken)


def test_world_offsets_by_zone_translation():
    assert g.world((0.0, 0.0, 0.0)) == g.ZONE_T


def test_every_textured_item_has_an_atlas_cell():
    names = {s[3] for s in g.SIGNS} | {d[2] for d in g.GAUGES} | {g.label_key(n, d) for _, _, n, d in g.LABELS}
    assert names <= set(g.ATLAS)


def test_atlas_cells_do_not_overlap_and_uvs_stay_inside():
    used = set()
    for name, (col, row, w, h) in g.ATLAS.items():
        cells = {(col + i, row + j) for i in range(w) for j in range(h)}
        assert not cells & used and all(c < g.ATLAS_GRID and r < g.ATLAS_GRID for c, r in cells)
        used |= cells
        for u, v in ((0, 0), (1, 1)):
            a, b = g.atlas_uv(name, u, v)
            assert 0 < a < 1 and 0 < b < 1


def test_layer_only_overs_world():
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(g.LAYER))
    assert layer.GetPrimAtPath("/World").specifier == Sdf.SpecifierOver
    assert layer.GetPrimAtPath(g.ROOT).specifier == Sdf.SpecifierDef


def test_layer_never_binds_repair_zone_materials():
    assert "/World/RepairZone/" not in g.LAYER.read_text()


def test_scuppers_sit_between_edge_ribs():
    for p, q, *_ in g.FASCIA:
        for f in g.scuppers(p, q):
            k = (f - g.RIB_START) / g.RIB_SPACING
            assert abs(k - round(k)) > 0.3
            assert 0.5 < f < math.dist(p, q) - 0.5


def test_route_distance_matches_patrol_corridors():
    assert g.route_distance(0.0, 1.25) < 1e-9
    assert g.route_distance(0.0, -3.25) < 1e-9
    assert g.route_distance(7.5, 3.0) > 2.0


def test_cables_cross_walkways_only_at_protectors():
    from dressing_geometry import spline

    import numpy as np

    for _, _, ctrl in g.CABLES:
        pts = spline(np.array([(x, y, 0.0) for x, y in ctrl]), 10)
        for x, y, _ in pts:
            if g.route_distance(x, y) < 0.3:
                assert min(math.dist((x, y), c) for c, _ in g.PROTECTORS) < 0.6


def test_puddles_stay_on_deck_and_out_of_work_zone():
    items = g.puddle_layout()
    assert len(items) > 20
    for x, y, size, rot, shape in items:
        assert abs(x) <= g.DECK_LIMIT[0] and abs(y) <= g.DECK_LIMIT[1]
        assert not g.in_work_zone(x, y, 0.3)
        assert 0.4 < size < 1.6 and shape in range(4)


def test_puddle_layout_is_deterministic():
    assert g.puddle_layout() == g.puddle_layout()

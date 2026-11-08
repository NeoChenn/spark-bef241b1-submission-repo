import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import greebles as g


def test_layout_keeps_work_zone_and_damage_clear():
    assert g.layout_issues() == []


def test_pipe_details_stay_off_the_damage_spot():
    for kind, x, y, z in g.footprints():
        assert math.dist((x, y, z), g.DAMAGE) > 0.9


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

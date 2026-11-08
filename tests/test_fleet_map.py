import math

import numpy as np
import pytest

from policy.fleet.rigmap import DECKS, MAX_SEG_M, RigMap, load_rig_map


@pytest.fixture(scope="module")
def rm() -> RigMap:
    return load_rig_map()


def test_three_decks_with_nodes_at_deck_height(rm):
    assert set(DECKS) == {"lower_deck", "middle_deck", "top_deck"}
    for name, z in DECKS.items():
        ids = [i for i in range(len(rm.pos)) if rm.deck[i] == name]
        assert len(ids) > 10
        assert all(abs(rm.pos[i][2] - z) < 1e-6 for i in ids)


def test_graph_connected_from_dock(rm):
    dist, _, _ = rm.dijkstra(rm.dock)
    assert np.isfinite(dist).all()


def test_segments_short(rm):
    for nbrs in rm.adj:
        for e in nbrs.values():
            assert e.length <= MAX_SEG_M + 1e-6


def test_every_flange_and_gauge_has_work_node(rm):
    assert sorted(rm.flanges) == ["F1", "F2", "F3", "F4", "F5", "F6"]
    for fid, c in rm.flanges.items():
        n = rm.poi_node[fid]
        assert np.linalg.norm(rm.pos[n][:2] - np.asarray(c[:2])) <= 1.0
    assert len([k for k in rm.poi_node if k.startswith("gauge_")]) == 7


def test_lower_deck_reached_by_stairs_and_slower_than_flat(rm):
    rm = load_rig_map()
    lo = rm.add_poi("probe_lower", [-10.0, -4.8, 21.0])
    t, path = rm.route(rm.dock, lo)
    assert rm.deck[lo] == "lower_deck"
    kinds = {rm.adj[a][b].kind for a, b in zip(path, path[1:])}
    assert "stairs" in kinds
    assert math.isfinite(t)
    flat = sum(rm.adj[a][b].length for a, b in zip(path, path[1:])) / rm.speed["walk"]
    assert t > flat


def test_blocked_nodes_force_detour(rm):
    t0, path = rm.route(rm.dock, rm.poi_node["F5"])
    mid = path[len(path) // 2]
    t1, p1 = rm.route(rm.dock, rm.poi_node["F5"], blocked={mid})
    assert mid not in p1
    assert t1 >= t0


def test_unreachable_when_spur_blocked(rm):
    n = rm.poi_node["F6"]
    t, path = rm.route(rm.dock, n, blocked={n})
    assert t == math.inf and path == []


def test_narrow_corridors_marked(rm):
    corridors = {e.corridor for nbrs in rm.adj for e in nbrs.values() if e.corridor}
    assert any(c.startswith("stairs") for c in corridors)
    assert any(c.startswith("catwalk") for c in corridors)
    assert any(c.startswith("spur") for c in corridors)


def test_map_json_shape(rm):
    m = rm.to_json()
    assert {"schema", "decks", "walkways", "dock", "flanges", "inspection_points", "bounds"} <= set(m)
    assert all({"a", "b", "kind", "narrow", "deck"} <= set(w) for w in m["walkways"])
    assert len(m["flanges"]) == 6

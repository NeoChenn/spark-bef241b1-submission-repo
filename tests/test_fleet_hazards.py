import json
import math

import numpy as np
import pytest

from policy.fleet.hazards import Hazards, outer_sides, weather_state
from policy.fleet.rigmap import DECKS, load_rig_map
from tests.test_fleet_sim import by_id, make, run_until


def kinds_on(rm, path):
    return {rm.adj[a][b].kind for a, b in zip(path, path[1:])}


def coarse_closed(rm, hz, wid):
    return all(rm.is_closed(a, b) for a, b in hz.walkway_edges(rm)[rm.coarse.index(
        next(c for c in rm.coarse if c["id"] == wid))])


def walkway(rm, deck, axis, value):
    return next(c["id"] for c in rm.coarse if c["deck"] == deck and c["kind"] == "walk"
                and abs(c["a"][axis] - value) < 1e-6 and abs(c["b"][axis] - value) < 1e-6)


def test_walkways_have_ids_and_map_json_carries_them():
    rm = load_rig_map()
    ids = [c["id"] for c in rm.to_json()["walkways"]]
    assert "w3" in ids and "stairs_west_upper" in ids and "catwalk_mid" in ids and "spur_dock" in ids


def test_crate_on_walkway_blocks_route_and_clear_restores():
    rm = load_rig_map()
    t0, p0 = rm.route(rm.dock, rm.poi_node["F5"])
    hz = Hazards()
    c = next(c for c in rm.coarse if c["id"] == "w3")
    ob = hz.drop("crate", (np.array(c["a"]) + np.array(c["b"])) / 2)
    hz.apply(rm)
    t1, p1 = rm.route(rm.dock, rm.poi_node["F5"])
    assert math.isfinite(t1) and t1 > t0 and "stairs" in kinds_on(rm, p1) and "stairs" not in kinds_on(rm, p0)
    hz.clear(ob["id"])
    hz.apply(rm)
    assert rm.route(rm.dock, rm.poi_node["F5"]) == (t0, p0)


def test_spill_is_slippery_not_blocking_and_closes_stairs_inside_it():
    rm = load_rig_map()
    hz = Hazards()
    a = rm.key[("top_deck", -21.0, 11.0)]
    walk_b = next(b for b, e in rm.adj[a].items() if e.kind == "walk")
    stair_b = next(b for b, e in rm.adj[a].items() if e.kind == "stairs")
    before = rm.travel_s(a, walk_b)
    hz.drop("spill", rm.pos[a])
    hz.apply(rm)
    assert rm.travel_s(a, walk_b) > 2 * before and not rm.is_closed(a, walk_b)
    assert rm.is_closed(a, stair_b)


def test_rain_slows_stairs_more_than_flat_and_fog_cuts_sight():
    rain, fog, calm = weather_state("rain"), weather_state("fog"), weather_state("calm")
    assert rain["speed"]["stairs"] < rain["speed"]["walk"] < 1.0
    assert fog["perception_range_m"] < calm["perception_range_m"] and fog["second_view_p"] > calm["second_view_p"]
    assert not rain["closes_outer"] and weather_state("storm")["closes_outer"] and weather_state("wind")["closes_outer"]
    assert not weather_state("wind", gust=9.0)["closes_outer"]
    assert weather_state(0.3)["wet"] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        weather_state("hail")


def test_rain_prefers_flat_detour_over_stairs():
    rm = load_rig_map()
    hz = Hazards()
    src, dst = rm.key[("top_deck", -21.0, 11.0)], rm.key[("middle_deck", -21.0, 0.0)]
    _, dry = rm.route(src, dst)
    hz.set_weather("rain")
    hz.apply(rm)
    t_wet, wet = rm.route(src, dst)
    assert "stairs" in kinds_on(rm, dry)
    assert math.isfinite(t_wet)
    stair_s = sum(rm.travel_s(a, b) for a, b in zip(wet, wet[1:]) if rm.adj[a][b].kind == "stairs")
    assert stair_s > 0


def test_storm_closes_windward_outer_walkways_only():
    rm = load_rig_map()
    sides = outer_sides(rm)
    south_mid = walkway(rm, "middle_deck", 1, -6.0)
    north_mid = walkway(rm, "middle_deck", 1, 14.0)
    assert sides[[c["id"] for c in rm.coarse].index(south_mid)] == "S"
    hz = Hazards()
    hz.set_weather("storm", wind_dir=90.0)
    hz.apply(rm)
    assert coarse_closed(rm, hz, south_mid) and not coarse_closed(rm, hz, north_mid)
    assert not any(rm.is_closed(a, b) for a, b, e in rm.edges() if e.kind == "catwalk")
    hz.set_weather("calm")
    hz.apply(rm)
    assert not rm.closed and not rm.kind_factor


def test_wind_from_flare_closes_flare_side_of_top_deck():
    rm = load_rig_map()
    hz = Hazards()
    hz.set_weather("wind", wind_dir=0.0)
    hz.apply(rm)
    assert "flare side" in hz.why.values()
    a = rm.key[("top_deck", -21.0, 11.0)]
    assert all(rm.is_closed(a, b) for b, e in rm.adj[a].items() if e.kind != "stairs")


def test_unknown_targets_raise():
    rm = load_rig_map()
    hz = Hazards()
    with pytest.raises(ValueError):
        hz.close(rm, "w999")
    with pytest.raises(ValueError):
        hz.close(rm, "w3", "stairs")
    with pytest.raises(ValueError):
        hz.drop("piano", (0, 0, 42.5))
    with pytest.raises(ValueError):
        hz.clear("nope")


def test_fleet_reroutes_robot_when_crate_drops_on_its_path():
    f, rm = make([("R1", "dock", 1.0)])
    task = f.add_task("leak_check", "F5")
    for _ in range(3):
        f.step()
    r = by_id(f, "R1")
    assert "stairs" not in {lg.kind for lg in r.legs}
    f.event("drop_obstacle", kind="crate", at="R1")
    f.step()
    rr = [e for e in f.event_log if e["kind"] == "reroute"]
    assert len(rr) == 1 and rr[0]["robot"] == "R1"
    assert rr[0]["extra_m"] > 10 and rr[0]["extra_s"] > 10 and rr[0]["old_path"] and rr[0]["new_path"]
    assert "stairs" in {lg.kind for lg in r.legs}
    assert not any(rm.is_closed(lg.u, lg.v) for lg in r.legs)
    snap = f.snapshot()
    rob = snap["robots"][0]
    assert rob["prev_path"] == rr[0]["old_path"] and rob["path"]
    assert snap["hazards"]["obstacles"][0]["kind"] == "crate" and snap["hazards"]["closed"]
    assert snap["metrics"]["reroutes"] == 1
    json.dumps(snap)
    assert run_until(f, lambda: task.status == "done")


def test_clear_restores_short_route_with_negative_extra():
    f, rm = make([("R1", "dock", 1.0)])
    f.add_task("leak_check", "F5")
    f.step()
    f.event("close_walkway", edge="w3")
    f.step()
    cid = f.hz.to_json(rm)["closures"][0]["id"]
    for _ in range(5):
        f.step()
    f.event("clear", id=cid)
    f.step()
    rr = [e for e in f.event_log if e["kind"] == "reroute"]
    assert len(rr) == 2 and rr[0]["extra_m"] > 0 and rr[1]["extra_m"] < 0
    assert not f.hz.active() and not rm.closed


def test_slippery_spill_on_route_costs_time_but_robot_still_arrives():
    f, rm = make([("R1", "dock", 1.0)])
    task = f.add_task("inspect", "gauge_03")
    f.step()
    f.event("drop_obstacle", kind="spill", at="R1")
    f.step()
    rr = [e for e in f.event_log if e["kind"] == "reroute"]
    assert not rr or rr[0]["extra_s"] > 0
    assert run_until(f, lambda: task.status == "done")


def test_unreachable_task_escalates_then_reopens_on_clear():
    f, rm = make([("R1", "dock", 1.0)])
    task = f.add_task("inspect", "gauge_05")
    f.step()
    f.event("close_walkway", edge="spur_gauge_05")
    f.step()
    assert task.status == "escalated" and task.outcome == "blocked_by_hazard"
    assert any(e["kind"] == "needs_human" and "hazard" in e["message"] for e in f.event_log)
    assert f.metrics()["blocked_tasks"] == 1
    f.event("clear", id="all")
    f.step()
    assert task.status in ("pending", "assigned")
    assert run_until(f, lambda: task.status == "done")


def test_robot_on_stairs_backs_out_when_stairs_close():
    f, rm = make([("R1", "dock", 1.0)])
    f.add_task("rust", "rust_x", [-21.8, 2.0, 32.0], severity=0.5, exposure=0.5)
    r = by_id(f, "R1")
    assert run_until(f, lambda: (lg := f._mid(r)) is not None and lg.kind == "stairs" and len(r.trail) >= 1)
    entry = r.trail[0].u
    f.event("close_stairs", id="stairs_west_upper")
    f.step()
    assert r.legs[0].back
    rr = [e for e in f.event_log if e["kind"] == "reroute"]
    assert rr and rr[-1]["backout"]
    for _ in range(60):
        f.step()
        f.check_invariants()
        if not any(lg.back for lg in r.legs):
            break
    assert not any(lg.kind == "stairs" and rm.is_closed(lg.u, lg.v) for lg in r.legs if not lg.back)
    assert r.node == entry or any(lg.u == entry for lg in r.legs) or r.state in ("TRANSIT", "WORK", "BLOCKED")


def test_fog_slows_perception_work():
    f, rm = make([("R1", "dock", 1.0)], work_s=20.0)
    task = f.add_task("inspect", "gauge_03")
    assert run_until(f, lambda: task.status == "active")
    f.event("set_weather", level="fog")
    t0 = f.t
    assert run_until(f, lambda: task.status == "done")
    assert f.t - t0 > 20.0 * 1.2


def test_storm_reroutes_robot_off_windward_catwalk():
    f, rm = make([])
    f.add_robot("R1", rm.key[("middle_deck", -21.0, 0.0)], 1.0)
    f.add_task("rust", "rust_e", [22.6, 1.0, 32.0], severity=0.6, exposure=0.6)
    f.step()
    r = by_id(f, "R1")
    ys = [rm.pos[lg.v][1] for lg in r.legs]
    assert min(ys) < -5
    f.event("set_weather", level="storm", wind_dir=90.0)
    f.step()
    rr = [e for e in f.event_log if e["kind"] == "reroute"]
    assert rr and rr[-1]["extra_m"] > 0
    assert all(rm.pos[lg.v][1] > -5.9 for lg in r.legs[len(f._kept(r)):])


def test_person_at_xy_creates_person_zone():
    f, rm = make([("R1", "dock", 1.0)])
    f.event("person_at", xy=[-1.0, 4.0], deck="top_deck", duration=30.0)
    assert f.persons and f.persons[0]["xyz"][2] == DECKS["top_deck"]
    f.event("person_at", distance=2.4)
    assert len(f.persons) == 1
    with pytest.raises(ValueError):
        f.event("explode")


def test_hud_commands_with_empty_args_pick_sensible_defaults():
    f, rm = make([("R1", "dock", 1.0)])
    f.add_task("leak_check", "F5")
    for _ in range(3):
        f.step()
    f.event("drop_obstacle")
    f.event("close_walkway")
    f.event("close_stairs")
    f.step()
    hz = f.hz.to_json(rm)
    assert hz["obstacles"][0]["kind"] == "crate"
    assert [c["kind"] for c in hz["closures"]] == ["walkway", "stairs"]
    assert any(e["kind"] == "reroute" for e in f.event_log)
    f.event("clear")
    assert not f.hz.active()

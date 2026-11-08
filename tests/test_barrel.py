import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from policy.fleet import barrel
from policy.fleet.hazards import seg_dist
from policy.fleet.live import FleetLive
from policy.fleet.rigmap import load_rig_map
from rig.web.server import create_app
from tests.test_fleet_sim import by_id, make, stub_jobs


@pytest.fixture(scope="module")
def rm():
    return load_rig_map()


def roll(rm, ob, wind, seconds, dt=0.25):
    xs = []
    for _ in range(int(seconds / dt)):
        barrel.step(rm, ob, wind, dt)
        xs.append(list(ob["xyz"]))
    return xs


def on_walkway(rm, p):
    return min(seg_dist(p, rm.pos[a], rm.pos[b]) for a, b, e in rm.edges() if e.kind != "stairs") < 1e-6


def test_place_snaps_to_nearest_flat_edge(rm):
    ob = barrel.start(rm, {"xyz": [-15.0, -4.2, 20.0]})
    a, b = ob["edge"]
    assert rm.adj[a][b].kind != "stairs"
    assert ob["xyz"] == pytest.approx([-15.0, -4.0, 20.0], abs=1e-6)
    assert ob["rolling"] and ob["v"] == 0.0


def test_wind_along_walkway_rolls_downwind_at_terminal_speed(rm):
    ob = barrel.start(rm, {"xyz": [-15.0, -4.0, 20.0]})
    xs = roll(rm, ob, barrel.wind_xy({"wind": 1.0, "wind_dir": 0.0}), 8.0)
    assert xs[-1][0] > -15.0 + 5.0 and abs(xs[-1][1] + 4.0) < 1e-6
    assert all(b[0] >= a[0] - 1e-9 for a, b in zip(xs, xs[1:]))
    assert 0.6 < math.hypot(*ob["vel"]) < barrel.V_WIND
    assert all(on_walkway(rm, p) for p in xs)
    assert ob["spin"] > 0 and ob["rolling"]


def test_crosswind_cannot_start_it_and_it_settles(rm):
    ob = barrel.start(rm, {"xyz": [-15.0, -4.0, 20.0]})
    xs = roll(rm, ob, barrel.wind_xy({"wind": 1.0, "wind_dir": 90.0}), 3.0)
    assert not ob["rolling"] and xs[-1] == pytest.approx([-15.0, -4.0, 20.0], abs=1e-6)


def test_kick_in_calm_rolls_a_few_metres_then_stops(rm):
    ob = barrel.start(rm, {"xyz": [-15.0, -4.0, 20.0]}, toward=[-5.0, -4.0, 20.0])
    assert ob["v"] * (rm.pos[ob["edge"][1]][0] - rm.pos[ob["edge"][0]][0]) > 0
    xs = roll(rm, ob, (0.0, 0.0), 15.0)
    d = xs[-1][0] + 15.0
    assert 0.5 < d < 5.0 and not ob["rolling"] and ob["vel"] == [0.0, 0.0]


def test_dead_end_bounces_and_stops_inside_the_edge(rm):
    flat = [{b for b, e in nb.items() if e.kind != "stairs"} for nb in rm.adj]
    end = next(i for i, nb in enumerate(flat) if len(nb) == 1)
    nb = next(iter(flat[end]))
    u = rm.pos[end][:2] - rm.pos[nb][:2]
    ob = barrel.start(rm, {"xyz": list(rm.pos[nb])})
    wd = math.degrees(math.atan2(u[1], u[0]))
    xs = roll(rm, ob, barrel.wind_xy({"wind": 1.0, "wind_dir": wd}), 30.0)
    L = float(np.linalg.norm(u))
    assert all(np.linalg.norm(np.array(p[:2]) - rm.pos[nb][:2]) <= L + 1e-6 for p in xs)
    assert not ob["rolling"] and all(on_walkway(rm, p) for p in xs)


def test_turns_a_corner_when_the_wind_pushes_round_it(rm):
    ob = barrel.start(rm, {"xyz": [16.0, -4.0, 20.0]})
    xs = roll(rm, ob, barrel.wind_xy({"wind": 1.2, "wind_dir": 60.0}), 20.0)
    assert xs[-1][1] > -4.0 + 2.0 and xs[-1][0] == pytest.approx(20.0, abs=1e-6)


def test_gives_up_after_max_roll_time(rm):
    ob = barrel.start(rm, {"xyz": [-21.0, 4.0, 31.25]})
    roll(rm, ob, barrel.wind_xy({"wind": 1.0, "wind_dir": 90.0}), barrel.ROLL_MAX_S + 2.0, dt=1.0)
    assert not ob["rolling"]


def walking_r1():
    f, rm = make([("R1", "dock", 1.0)])
    f.add_task("leak_check", "F5")
    for _ in range(3):
        f.step()
    return f, rm


def test_fleet_barrel_is_a_moving_hazard_that_moves_its_blocked_edges():
    f, rm = walking_r1()
    f.event("drop_obstacle", kind="barrel", at="R1")
    ob = f.hz.obstacles[next(iter(f.hz.obstacles))]
    assert ob["kind"] == "barrel" and ob["effect"] == "block" and ob["rolling"]
    f.event("set_weather", level="wind", wind_dir=ob["yaw"] + (180.0 if ob["v"] < 0 else 0.0))
    start = list(ob["xyz"])
    closed0 = set(rm.closed)
    assert closed0
    changes = 0
    for _ in range(80):
        before = set(rm.closed)
        f.roll(0.25)
        changes += set(rm.closed) != before
        f.step()
    assert math.dist(start, ob["xyz"]) > 3.0 and changes >= 1
    near = {(a, b) for a, b, e in rm.edges() if seg_dist(ob["xyz"], rm.pos[a], rm.pos[b]) < ob["radius"]}
    assert near and near <= set(rm.closed)
    assert any(e["kind"] == "reroute" and "barrel" in e["cause"] for e in f.event_log)
    r = by_id(f, "R1")
    assert not any(rm.is_closed(lg.u, lg.v) for lg in r.legs)


def test_fleet_barrel_stops_then_stays_static_until_cleared():
    f, rm = walking_r1()
    f.event("drop_obstacle", kind="barrel", at="R1")
    ob = next(iter(f.hz.obstacles.values()))
    for _ in range(80):
        f.roll(0.25)
    assert not ob["rolling"]
    p, closed = list(ob["xyz"]), set(rm.closed)
    f.event("set_weather", level="storm", wind_dir=0.0)
    f.roll(1.0)
    assert ob["xyz"] == p and closed and closed <= set(rm.closed)
    assert any(e["kind"] == "barrel_stopped" for e in f.event_log)
    f.event("set_weather", level="calm")
    f.event("clear")
    assert not f.hz.obstacles and not rm.closed
    snap = f.snapshot()
    assert snap["hazards"]["obstacles"] == []


def test_snapshot_carries_barrel_motion_for_the_renderer():
    f, rm = walking_r1()
    f.event("set_weather", level="wind", wind_dir=0.0)
    f.event("drop_obstacle", kind="barrel", at="R1")
    f.roll(1.0)
    ob = f.snapshot()["hazards"]["obstacles"][0]
    assert {"xyz", "vel", "yaw", "spin", "rolling"} <= set(ob)
    assert len(ob["vel"]) == 2


def test_live_auto_spawns_barrels_in_wind_only():
    live = FleetLive("judges", speed=20.0, jobs=stub_jobs(15.0), barrel_every_s=1.0, seed=3)
    live.advance(0.5)
    for _ in range(20):
        live.advance(0.5)
    assert not live.f.hz.obstacles
    live.event("set_weather", {"level": "wind"})
    for _ in range(20):
        live.advance(0.5)
    kinds = [o["kind"] for o in live.f.hz.obstacles.values()]
    assert kinds and set(kinds) == {"barrel"} and len(kinds) <= barrel.AUTO_MAX
    assert any(e["kind"] == "drop_obstacle" and "gust" in e["message"] for e in live.f.event_log)


def test_event_rate_limit_per_client():
    c = TestClient(create_app(sim_port=0, event_gap_s=1.0))
    with c:
        ok = c.post("/api/event", json={"name": "spawn_patch"}, headers={"CF-Connecting-IP": "1.1.1.1"})
        again = c.post("/api/event", json={"name": "spawn_patch"}, headers={"CF-Connecting-IP": "1.1.1.1"})
        other = c.post("/api/event", json={"name": "spawn_patch"}, headers={"CF-Connecting-IP": "2.2.2.2"})
    assert ok.status_code == 200 and other.status_code == 200
    assert again.status_code == 429 and 0 < again.json()["retry_s"] <= 1.0


def test_interactive_page_is_served_and_judge_redirects():
    c = TestClient(create_app(sim_port=0))
    with c:
        r = c.get("/interactive")
    assert r.status_code == 200 and "interactive.js" in r.text
    with c:
        assert c.get("/judge", follow_redirects=False).headers["location"] == "/interactive"


def test_puddle_is_a_soft_cost_not_a_block():
    f, rm = walking_r1()
    f.event("drop_obstacle", kind="puddle", at="R1")
    ob = next(iter(f.hz.obstacles.values()))
    assert ob["effect"] == "soft" and not rm.closed and rm.factor
    msg = f.event_log[-1]["message"]
    assert "puddle" in msg and ("steps around" in msg or "walks through" in msg or "no robot" in msg)
    for _ in range(3):
        f.step()
    assert by_id(f, "R1").legs and not by_id(f, "R1").stuck


def test_drop_can_small_is_soft_and_large_is_a_rolling_barrel():
    f, rm = walking_r1()
    f.event("drop_can", size="small", at="R1")
    f.event("drop_can", size="large", at="R1")
    kinds = {o["kind"]: o for o in f.hz.obstacles.values()}
    assert kinds["can"]["effect"] == "soft" and kinds["barrel"]["rolling"]
    assert any("pick-up" in e["message"] for e in f.event_log)


def test_weather_carries_time_of_day():
    f, rm = walking_r1()
    f.event("set_weather", level="rain", time="dusk")
    assert f.hz.weather["time"] == "dusk" and f.hz.weather["level"] == "rain"
    assert "dusk" in f.event_log[-1]["message"]
    from rig.hud import check_event
    with pytest.raises(ValueError):
        check_event({"name": "set_weather", "args": {"level": "rain", "time": "noon"}})
    assert check_event({"name": "drop_can", "args": {"size": "small"}})["name"] == "drop_can"


def test_drum_pose_extrapolates_spins_and_faces_its_motion():
    from rig.sim.obstacle_math import DRUM_R, drum_pose, puddle_edge
    mv = {"p": (0.0, 0.0, 20.0), "v": (0.0, 1.0), "t": 10.0, "from": (0.0, 0.0, 20.0)}
    p, yaw, spin = drum_pose(mv, 10.5, (0.0, 0.0, 20.0), 0.0, 0.0)
    assert p == pytest.approx((0.0, 0.5, 20.0)) and yaw == pytest.approx(90.0)
    assert spin == pytest.approx(math.degrees(0.5 / DRUM_R) % 360)
    p, _, _ = drum_pose(mv, 30.0, p, yaw, spin)
    assert p[1] == pytest.approx(0.8)
    assert all(0.5 < puddle_edge(a / 10, 77) < 1.1 for a in range(63))

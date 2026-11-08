import math

import pytest

from rig.mission.live import MissionLive


def deck():
    m = MissionLive(obs=False)
    clock = [0.0]
    m.scen.clock = lambda: clock[0]

    def at(t):
        clock[0] = t
        m.advance(0.25)
        s = m.for_sim()
        return {r["id"]: r for r in s["robots"]}, {o["id"]: o for o in s["hazards"]["obstacles"]}
    return m, at


def test_deck_preset_parks_both_robots_with_a_clean_pipe_and_one_fixed_camera():
    m, at = deck()
    assert m.event("demo", {"id": "deck"})["scenario"] == "demo_deck"
    sim = m.take_sim()
    assert [e["name"] for e in sim] == ["set_weather", "camera"]
    assert sim[1]["args"]["lens"]["eye"] and sim[1]["args"]["mode"] == "deck"
    rs, obs = at(5.0)
    s = m.scen.cur
    assert not obs and rs["R1"]["rust"] == 0.0 and rs["R1"]["state"] == rs["R3"]["state"] == "IDLE"
    assert (rs["R3"]["pose"]["x"], rs["R1"]["pose"]["x"]) == pytest.approx((s.park3[0], s.park1[0]), abs=0.01)
    assert rs["R2"]["pose"]["z"] < 40


def test_drop_can_and_add_rust_run_together_and_repeat():
    m, at = deck()
    m.event("demo", {"id": "deck"})
    at(1.0)
    assert m.event("demo", {"id": "deck_can"})["started"]
    assert m.event("demo", {"id": "deck_rust"})["started"]
    assert not m.event("demo", {"id": "deck_can"})["started"]
    s = m.scen.cur
    rs, obs = at(1.0)
    (cid, can), = obs.items()
    assert can["phase"] == "ground" and can["rollout"] == "R3" and rs["R1"]["rust"] == 1.0 and s.i == 3
    seen3, seen1 = set(), set()
    for t in range(2, 60):
        rs, obs = at(float(t))
        seen3.add(rs["R3"].get("job_state") or rs["R3"]["state"])
        seen1.add(rs["R1"].get("job_state") or rs["R1"]["state"])
        if rs["R3"].get("job_state") == "ROLLOUT":
            assert rs["R3"]["pose"]["x"] == pytest.approx(s.stop3[0], abs=0.01) and rs["R3"]["pose"]["yaw"] == 0.0
        if rs["R1"].get("job_state") == "SCRUB":
            assert rs["R1"]["pose"]["x"] == pytest.approx(s.stop1[0], abs=0.01)
    assert {"TRANSIT", "ROLLOUT", "IDLE"} <= seen3 and {"TRANSIT", "DEPLOY", "SCRUB", "IDLE"} <= seen1
    assert obs[cid]["phase"] == "tote" and rs["R1"]["rust"] == 0.0 and s.i == 0 and s.cans == 1
    assert rs["R3"]["pose"]["x"] == pytest.approx(s.park3[0], abs=0.01)
    assert rs["R1"]["pose"]["x"] == pytest.approx(s.park1[0], abs=0.01)
    assert m.event("demo", {"id": "deck_can"})["started"] and m.event("demo", {"id": "deck_rust"})["started"]
    rs, obs = at(61.0)
    assert cid not in obs and len(obs) == 1 and rs["R1"]["rust"] == 1.0
    m.event("demo", {"id": "reset"})
    assert m.scen.cur.id == "flyby" and not m.override and not m.fleet.f.hz.obstacles


def test_deck_buttons_start_the_preset_when_another_demo_is_running():
    m, at = deck()
    m.event("demo", {"id": "rust"})
    out = m.event("demo", {"id": "deck_rust"})
    assert out["scenario"] == "demo_deck" and out["started"] and m.scen.cur.rust_t is not None


def test_deck_walks_stay_clear_of_the_pipe():
    m, at = deck()
    m.event("demo", {"id": "deck"})
    m.event("demo", {"id": "deck_can"})
    m.event("demo", {"id": "deck_rust"})
    for k in range(0, 240):
        rs, _ = at(k * 0.25)
        for r in (rs["R1"], rs["R3"]):
            assert r["pose"]["y"] > -5.3
        assert math.dist((rs["R1"]["pose"]["x"], rs["R1"]["pose"]["y"]), (rs["R3"]["pose"]["x"], rs["R3"]["pose"]["y"])) > 1.5


def test_interactive_buttons_post_the_deck_demo():
    from fastapi.testclient import TestClient

    from rig.web.server import create_app

    with TestClient(create_app(sim_port=0, mission=MissionLive(obs=False))) as c:
        page = c.get("/interactive").text
        assert all(f'data-act="{a}"' in page for a in ("ahead", "deck_can", "deck_rust"))
        r = c.post("/api/event", json={"name": "demo", "args": {"id": "deck_rust"}})
        assert r.status_code == 200, r.text
        assert r.json()["started"] and r.json()["scenario"] == "demo_deck"

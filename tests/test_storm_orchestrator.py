from policy.demo_mock import build
from policy.orchestrator import S
from policy.storm import Weather


def pairs(orch):
    return [(r.frm, r.to) for r in orch.records]


def storm_at(bridge, t, level=1.0):
    bridge.events.append((t, "storm_level", lambda: bridge.storm_level(level)))
    bridge.events.sort(key=lambda e: e[0])


def test_calm_run_never_touches_storm_states():
    orch, *_ = build(None, glare=False, verbose=False)
    orch.run(400)
    seq = {r.to for r in orch.records}
    assert orch.state == S.DONE and not seq & {"WIPE_LENS", "WAIT_LULL"}
    assert all("storm" not in r.evidence for r in orch.records)


def test_gust_mid_scrub_retracts_waits_for_lull_and_finishes():
    orch, bridge, world, perception, arm, _ = build(None, glare=False, verbose=False)
    storm_at(bridge, 20.0)
    orch.run(500)
    p = pairs(orch)
    k = [i for i, x in enumerate(p) if x == ("SCRUB", "WAIT_LULL")]
    assert k and all("gust" in orch.records[i].reason for i in k)
    assert all(p[i + 1] == ("WAIT_LULL", "SCRUB") for i in k)
    assert all(orch.records[i].evidence["storm"]["gust_n"] >= orch.cfg.gust_limit for i in k)
    assert ("SCRUB", "BACK_OFF") not in p
    assert orch.state == S.DONE and float(world.fraction_remaining()[0]) < 0.1


def test_rain_on_lens_wipes_and_never_verifies_through_a_blind_lens():
    orch, bridge, world, *_ = build(None, glare=False, verbose=False)
    storm_at(bridge, 20.0)
    orch.run(500)
    p = pairs(orch)
    assert any(to == "WIPE_LENS" for _, to in p)
    w = next(i for i, x in enumerate(p) if x[1] == "WIPE_LENS")
    back = next(x for x in p[w + 1:] if x[0] == "WIPE_LENS" and x[1] != "WIPE_LENS")
    assert back[1] in (p[w][0], "WAIT_LULL")
    for r in orch.records:
        if r.frm in ("VERIFY", "VERIFY_PAINT") and r.to in ("PAINT", "DONE"):
            assert r.evidence["storm"]["lens_q"] >= orch.cfg.lens_usable_q
    assert orch.state == S.DONE


def test_storm_without_lull_escalates():
    orch, bridge, world, *_ = build(None, glare=False, verbose=False)
    bridge.storm = Weather(seed=0, lull_depth=0.0)
    storm_at(bridge, 20.0)
    orch.run(500)
    assert orch.state == S.ESCALATE_TO_HUMAN
    assert "storm not easing" in orch.records[-1].reason and orch.records[-1].frm == "WAIT_LULL"
    assert "PAINT" not in {r.to for r in orch.records}


def test_ambiguous_in_rain_waits_for_lull_then_detects():
    orch, bridge, world, perception, *_ = build(None, glare=True, verbose=False)
    perception.glare_views = {0, 1, 2}
    bridge.storm_level(1.0)
    orig = perception.observe

    def observe():
        perception.glare_alpha = 0.6 if bridge.weather()["rain"] > 0.35 else 0.0
        return orig()
    perception.observe = observe
    orch.run(500)
    p = pairs(orch)
    k = p.index(("DETECT", "WAIT_LULL"))
    assert sum(1 for x in p[:k] if x[1] == "SECOND_VIEW") == 2
    assert "lull" in orch.records[k].reason
    assert ("DETECT", "APPROACH") in p[k:]
    assert orch.state in (S.DONE, S.ESCALATE_TO_HUMAN)


def test_spray_waits_out_a_gust():
    orch, bridge, *_ = build(None, glare=False, verbose=False)
    bridge.weather = lambda: {"level": 1.0, "gust_n": 20.0 if bridge.t < 5.0 else 1.0, "rain": 0.0}
    orch.run(500)
    p = pairs(orch)
    k = p.index(("SPRAY", "WAIT_LULL"))
    assert "spray would drift" in orch.records[k].reason
    assert p[k + 1] == ("WAIT_LULL", "SPRAY") and orch.records[k + 1].t >= 5.0
    assert orch.state == S.DONE


def test_hud_shows_storm_line():
    orch, bridge, *_ = build(None, glare=False, verbose=False)
    bridge.storm_level(0.8)
    assert any(l.startswith("storm 0.8") for l in orch.overlay_text())


def test_set_weather_is_the_same_control_as_storm_level():
    orch, bridge, world, *_ = build(None, glare=False, verbose=False)
    coef = float(world.abrasion_coef[0])
    bridge.set_weather("wind", wind_dir=0.0, gust=30.0)
    w = bridge.weather()
    assert w["level"] == 0.8 and w["rain"] == 0.0 and float(world.abrasion_coef[0]) == coef
    bridge.set_weather("storm")
    assert bridge.weather()["level"] == 1.0 and float(world.abrasion_coef[0]) < coef
    bridge.storm_level(0.0)
    assert bridge.weather()["level"] == 0.0 and abs(float(world.abrasion_coef[0]) - coef) < 1e-6

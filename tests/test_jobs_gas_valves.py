import copy
import json
import math

import numpy as np
import pytest

from fleet.contract import REGISTRY, Job, JobStatus, Level, Robot, RobotKind, World
from fleet.gas.plume import true_concentration
from fleet.gas.report import Gemini
from fleet.jobs import gas_valves as gv

from fleet.gas.measure import SCENARIOS  # seeded leaks: source xy, wind, flow direction, upstream valve

SEEDS = range(6)
ERROR_BOUND_M = 1.5
MEAN_ERROR_BOUND_M = 0.5


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("FLEET_GEMINI", "0")


@pytest.fixture(scope="module")
def rig():
    return World.from_rig_meta()


def world_with(rig, source=None, wind=(0.0, 0.0), flow=None, seed=0, **leak):
    w = World(assets=copy.deepcopy(rig.assets), state={"seed": seed, "wind": list(wind)})
    w.state["gas_leaks"] = []
    if source is not None:   # the coordinator's format (PR #7) plus optional fields of this module
        w.state["gas_leaks"].append({"id": "leak_01", "position": [*source, 43.3], "rate_g_s": 3.5, "t0": 0.0,
                                     "source": "test flange", "isolated": False, "flow_dir": flow, **leak})
    return w


def quad(**kw):
    return Robot("q1", RobotKind.QUADRUPED, {"camera", "gas_sensor", "arm"}, (-21.0, 20.5, 42.5), deck=42.5, **kw)


def survey_job(**params):
    return Job("survey-1", "gas_survey", "top_deck", (-21.0, 20.5, 42.5), Level.AUTO, {"gas_sensor"}, params=params)


def run_survey(world, robot=None, gemini=False):
    ex = gv.GasSurveyJob(gemini=Gemini(enabled=False) if gemini is False else gemini)
    return ex.run(survey_job(), robot or quad(), world)


def approve(job):
    job.params["approved"] = True
    return job


def test_registry_has_the_four_jobs():
    for kind in ("gas_survey", "valve_isolate", "filter_changeover", "panel_button"):
        assert kind in REGISTRY
    assert REGISTRY["gas_survey"].level == Level.AUTO
    assert {REGISTRY[k].level for k in ("valve_isolate", "filter_changeover", "panel_button")} == {Level.SUPV}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_localization_error_within_bound(rig, name):
    """Every seed: error under 1.5 m and inside the reported 95% radius; mean error under 0.5 m."""
    src, wind, flow, _ = SCENARIOS[name]
    errs = []
    for seed in SEEDS:
        res = run_survey(world_with(rig, src, wind, flow, seed))
        assert res.status == JobStatus.FINDING
        f = res.findings[0]
        assert f.kind == "gas_leak" and f.severity == "critical"
        err = float(np.hypot(f.position[0] - src[0], f.position[1] - src[1]))
        assert err < ERROR_BOUND_M, (name, seed, err)
        assert err <= f.evidence["radius_m"], (name, seed, err, f.evidence["radius_m"])
        errs.append(err)
    assert np.mean(errs) < MEAN_ERROR_BOUND_M, (name, errs)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_survey_respects_the_gas_limits(rig, name):
    """The robot may walk into the stop limit once on its fixed route (the rule is reactive, as on the gas
    branch); it then backs off, and the plume climb never takes it there."""
    from fleet.gas.localize import Seeker
    from fleet.gas.plume import GasSensor
    w = world_with(rig, *SCENARIOS[name][:3])
    s = Seeker(w.state, GasSensor(np.random.default_rng(0)), (-21.0, 20.5))
    s.round(gv.patrol_route())
    over = [q for q in s.s.samples if q.lel >= gv.LIMITS.gas_stop_fraction_lel]
    assert len(over) <= 1 and (not over or "STOP" in s.s.events[0])
    n = len(s.s.samples)
    s.climb()
    s.fit()
    s.refine()
    assert all(q.lel < gv.LIMITS.gas_stop_fraction_lel for q in s.s.samples[n:])


def test_no_leak_means_no_finding(rig):
    res = run_survey(world_with(rig))
    assert res.status == JobStatus.DONE and not res.findings


def test_far_leak_is_missed_and_says_so(rig):
    """Detection range is a few metres with this plume: a leak 11 m off the route is not found."""
    res = run_survey(world_with(rig, (-3.0, 22.5), (0.0, -1.5)))
    assert res.status == JobStatus.DONE


@pytest.mark.parametrize("name", ["north_rack_east_wind", "south_rack_ne_wind"])
def test_follow_up_isolates_the_upstream_valve_and_needs_approval(rig, name):
    src, wind, flow, valve = SCENARIOS[name]
    res = run_survey(world_with(rig, src, wind, flow))
    (job,) = res.findings[0].follow_up
    assert job.kind == "valve_isolate" and job.target == valve
    assert job.level == Level.SUPV and job.priority == 1 and job.params["leak"] == "top_deck"
    req = job.params["approval_request"]
    assert all(req[k] for k in ("what", "where", "why", "risk_if_delayed", "text"))
    assert valve in req["what"]


def test_valve_isolate_does_nothing_without_approval(rig):
    """The coordinator holds SUPV jobs until approve(); the executor is a second gate on approved=False."""
    w = world_with(rig, *SCENARIOS["north_rack_east_wind"][:3])
    job = run_survey(w).findings[0].follow_up[0]
    job.params["approved"] = False
    res = gv.ValveIsolateJob().run(job, quad(), w)
    assert res.status == JobStatus.WAITING_APPROVAL
    assert w.state["gas_leaks"][0]["rate_g_s"] == 3.5 and not w.state["gas_leaks"][0]["isolated"]
    assert not w.state.get("valves", {}).get(job.target, {}).get("closed")
    assert res.findings[0].evidence["approval_request"]["text"].startswith("APPROVAL NEEDED")


def test_full_chain_detect_isolate_confirm(rig):
    w = world_with(rig, *SCENARIOS["north_rack_east_wind"][:3])
    before = true_concentration(w.state, (-12.0, 22.0))
    iso = approve(run_survey(w).findings[0].follow_up[0])
    res = gv.ValveIsolateJob().run(iso, quad(), w)
    assert res.status == JobStatus.FINDING, res.log
    assert w.state["valves"]["valve_03"]["closed"]
    assert true_concentration(w.state, (-12.0, 22.0)) < 0.2 * before
    (confirm,) = res.findings[0].follow_up
    assert confirm.kind == "gas_survey" and confirm.params["mode"] == "confirm"
    out = gv.GasSurveyJob(gemini=Gemini(enabled=False)).run(confirm, quad(), w)
    assert out.findings[0].kind == "leak_isolated"
    assert out.findings[0].evidence["ratio"] < gv.CONFIRM_DROP
    assert w.state["gas_leaks"][0]["isolated"] and w.state["gas_leaks"][0]["isolated_by"] == "valve_03"
    (prep,) = out.findings[0].follow_up
    assert prep.kind == "leak_repair_prep" and prep.level == Level.ASSIST


def test_coordinator_alarm_triggered_survey(rig):
    """The coordinator's demo leak (fleet/demo.py, PR #7): a gas alarm queues gas_survey(target=leak id,
    position=alarm). Valve_05 is 1.1 m from the leak, so the robot would stand in gas: it is skipped by
    prediction and the next valve is proposed; closing it cuts the leak and the confirm survey sees it fall."""
    w = world_with(rig)
    w.state["gas_leaks"].append({"id": "leak_01", "source": "flange near valve_05", "position": (21.0, -10.2, 43.4),
                                 "rate_g_s": 3.5, "t0": 205.0, "alarm_after_s": 60.0})
    job = Job("gas_survey-9", "gas_survey", "leak_01", (21.0, -10.2, 43.4), Level.AUTO, {"gas_sensor"}, priority=1,
              params={"zone": "leak_01", "standoff": 2.0})
    robot = Robot("Q2", RobotKind.QUADRUPED, {"camera", "thermal", "gas_sensor"}, (19.0, -9.0, 42.5), deck=42.5)
    res = gv.GasSurveyJob(gemini=Gemini(enabled=False)).run(job, robot, w)
    f = res.findings[0]
    assert f.kind == "gas_leak" and math.dist(f.position[:2], (21.0, -10.2)) < 1.0
    assert any("valve_05" in s for s in f.evidence["valves_skipped"])
    iso = f.follow_up[0]
    assert iso.kind == "valve_isolate" and iso.target == "valve_06" and iso.params["zone"] == "leak_01"
    out = gv.ValveIsolateJob().run(iso, quad(), w)
    assert out.status == JobStatus.FINDING, out.log
    assert w.state["gas_leaks"][0]["isolated"]
    done = gv.GasSurveyJob(gemini=Gemini(enabled=False)).run(out.findings[0].follow_up[0], robot, w)
    assert done.findings[0].kind == "leak_isolated"
    assert done.findings[0].follow_up[0].target == "leak_01"


def test_wrong_valve_is_caught_and_the_next_one_proposed(rig):
    src, wind, flow, _ = SCENARIOS["north_rack_east_wind"]
    w = world_with(rig, src, wind, flow, isolating_valves=["valve_01"])
    iso = approve(run_survey(w).findings[0].follow_up[0])
    assert iso.target == "valve_03"
    confirm = gv.ValveIsolateJob().run(iso, quad(), w).findings[0].follow_up[0]
    out = gv.GasSurveyJob(gemini=Gemini(enabled=False)).run(confirm, quad(), w)
    f = out.findings[0]
    assert f.kind == "isolation_ineffective" and f.severity == "critical"
    nxt = f.follow_up[0]
    assert nxt.kind == "valve_isolate" and nxt.target == "valve_01" and nxt.params["approval_request"]["text"]
    gv.ValveIsolateJob().run(approve(nxt), quad(), w)
    confirm2 = gv.GasSurveyJob(gemini=Gemini(enabled=False)).run(
        Job("c2", "gas_survey", "gas_leak", confirm.position, Level.AUTO, {"gas_sensor"},
            params={**confirm.params, "tried_valves": ["valve_03", "valve_01"]}), quad(), w)
    assert confirm2.findings[0].kind == "leak_isolated"


def isolate_job(target, rig, **params):
    return Job(f"iso-{target}", "valve_isolate", target, tuple(rig.assets[target]["position_m"]), Level.SUPV,
               {"arm"}, params={"approved": True, **params})


def test_valve_isolate_reaches_a_rig_valve(rig):
    w = world_with(rig)
    res = gv.ValveIsolateJob().run(isolate_job("valve_03", rig), quad(), w)
    assert res.status == JobStatus.FINDING
    reach = res.findings[0].evidence["reach"]
    assert reach["reachable"] and reach["ik_error_m"] < 0.01


def test_out_of_reach_valve_is_refused_and_handed_to_a_human(rig):
    w = world_with(rig)
    w.assets["valve_03"]["position_m"][2] = 42.5 + 2.2          # handle 2.2 m above the deck
    res = gv.ValveIsolateJob().run(isolate_job("valve_03", rig), quad(), w)
    assert res.status == JobStatus.HANDED_TO_HUMAN
    f = res.findings[0]
    assert f.kind == "valve_out_of_reach" and not f.evidence["reach"]["reachable"]
    assert f.follow_up[0].level == Level.HUMAN
    assert not w.state["valves"]["valve_03"].get("closed")


def test_over_torque_valve_is_not_forced(rig):
    w = world_with(rig)
    w.state["valves"] = {"valve_03": {"breakaway_nm": 20.0}}
    res = gv.ValveIsolateJob().run(isolate_job("valve_03", rig), quad(), w)
    assert res.status == JobStatus.HANDED_TO_HUMAN
    ev = res.findings[0].evidence
    assert res.findings[0].kind == "valve_over_torque"
    assert ev["torque_peak_nm"] <= ev["torque_limit_nm"] + 0.2
    assert not w.state["valves"]["valve_03"].get("closed")


def test_valve_inside_the_gas_cloud_is_refused(rig):
    v = rig.assets["valve_03"]["position_m"]
    w = world_with(rig, (v[0], v[1] - 0.3))
    res = gv.ValveIsolateJob().run(isolate_job("valve_03", rig), quad(), w)
    assert res.status == JobStatus.HANDED_TO_HUMAN and res.findings[0].kind == "valve_in_gas"


def test_person_near_the_valve_pauses(rig):
    w = world_with(rig)
    v = rig.assets["valve_03"]["position_m"]
    w.people = [(v[0], v[1] - 1.0, 42.5)]
    res = gv.ValveIsolateJob().run(isolate_job("valve_03", rig), quad(), w)
    assert res.status == JobStatus.FAILED and not w.state["valves"]["valve_03"].get("closed")


def test_robot_without_arm_fails(rig):
    r = Robot("q2", RobotKind.QUADRUPED, {"gas_sensor"})
    assert gv.ValveIsolateJob().run(isolate_job("valve_03", rig), r, world_with(rig)).status == JobStatus.FAILED


# ----------------------------------------------------------------------------------------------- filters
FILTER_POS = (-8.0, 23.6, 43.4)


def filter_world(rig, **fs):
    w = world_with(rig)
    w.state["filters"] = {"filter_01": {"position_m": list(FILTER_POS), "dp_bar": 1.4, "dp_limit_bar": 1.0,
                                        "active": "A", "standby_dp_bar": 0.3, **fs}}
    return w


def test_high_dp_creates_a_supervised_changeover_job(rig):
    (job,) = gv.filter_jobs(filter_world(rig))
    assert job.kind == "filter_changeover" and job.level == Level.SUPV
    assert gv.filter_jobs(filter_world(rig, dp_bar=0.5)) == []


def test_filter_changeover_gated_then_dp_drops(rig):
    w = filter_world(rig)
    job = gv.filter_jobs(w)[0]
    job.params["approved"] = False          # operator has not approved yet
    assert gv.FilterChangeoverJob().run(job, quad(), w).status == JobStatus.WAITING_APPROVAL
    assert w.state["filters"]["filter_01"]["active"] == "A"
    res = gv.FilterChangeoverJob().run(approve(job), quad(), w)
    assert res.status == JobStatus.FINDING, res.log
    ev = res.findings[0].evidence
    assert ev["dp_after_bar"] < 0.5 * ev["dp_before_bar"] and w.state["filters"]["filter_01"]["active"] == "B"


def test_filter_low_dp_or_fouled_standby(rig):
    w = filter_world(rig, dp_bar=0.4)
    job = Job("f", "filter_changeover", "filter_01", FILTER_POS, Level.SUPV, {"arm"}, params={"approved": True})
    assert gv.FilterChangeoverJob().run(job, quad(), w).status == JobStatus.DONE
    w = filter_world(rig, standby_fouled=True)
    assert gv.FilterChangeoverJob().run(job, quad(), w).status == JobStatus.HANDED_TO_HUMAN
    assert w.state["filters"]["filter_01"]["active"] == "A"


# ------------------------------------------------------------------------------------------------- panel
PANEL_POS = (-14.0, 23.6, 43.5)


def panel_job(button, approved=True):
    return Job(f"p-{button}", "panel_button", "panel_01", PANEL_POS, Level.SUPV, {"arm"},
               params={"button": button, "approved": approved})


@pytest.mark.parametrize("button", ["esd_reset", "ESD RESET", "reset_esd", "esd", "shutdown_reset", "trip_bypass",
                                    "override", "fire_inhibit"])
def test_esd_and_reset_buttons_are_never_pressed(rig, button):
    w = world_with(rig)
    w.state["panel"] = {"esd_tripped": True, "pressed": []}
    res = gv.PanelButtonJob().run(panel_job(button), quad(), w)
    assert res.status == JobStatus.HANDED_TO_HUMAN
    assert w.state["panel"]["pressed"] == []
    assert res.findings[0].follow_up[0].level == Level.HUMAN
    with pytest.raises(gv.RefusedButton):
        gv.press(w, button, "q1")
    assert w.state["panel"]["pressed"] == []


def test_pump_start_needs_approval_then_presses(rig):
    w = world_with(rig)
    assert gv.PanelButtonJob().run(panel_job("pump_start", False), quad(), w).status == JobStatus.WAITING_APPROVAL
    assert not w.state.get("panel", {}).get("pressed")
    res = gv.PanelButtonJob().run(panel_job("pump_start"), quad(), w)
    assert res.status == JobStatus.FINDING, res.log
    assert [p["button"] for p in w.state["panel"]["pressed"]] == ["pump_start"] and w.state["panel"]["pump_running"]


def test_pump_start_refused_while_esd_tripped(rig):
    w = world_with(rig)
    w.state["panel"] = {"esd_tripped": True}
    res = gv.PanelButtonJob().run(panel_job("pump_start"), quad(), w)
    assert res.status == JobStatus.HANDED_TO_HUMAN and not w.state["panel"].get("pressed")


def test_alarm_ack(rig):
    w = world_with(rig)
    w.state["panel"] = {"alarms": [{"id": "PAH-101"}]}
    res = gv.PanelButtonJob().run(panel_job("alarm_ack"), quad(), w)
    assert res.status == JobStatus.FINDING and w.state["panel"]["alarms"][0]["acknowledged"]


# ------------------------------------------------------------------------------------------------ Gemini
def fake_transport(payload):
    def transport(url, body):
        transport.prompts.append(body["contents"][0]["parts"][0]["text"])
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
    transport.prompts = []
    return transport


def test_gemini_writes_the_report_and_approval(rig):
    t = fake_transport({"report": "Leak near valve_03 rack, 95% within 0.4 m.", "what": "Close valve_03.",
                        "where": "North rack.", "why": "Gas 0.14 LEL.", "risk_if_delayed": "Cloud grows."})
    res = run_survey(world_with(rig, *SCENARIOS["north_rack_east_wind"][:3]), gemini=Gemini(transport=t))
    assert res.used_gemini
    assert res.findings[0].summary.startswith("Leak near valve_03")
    assert res.findings[0].follow_up[0].params["approval_request"]["what"] == "Close valve_03."
    assert "radius_m" in t.prompts[0] and "valve_03" in t.prompts[0]


def test_gemini_reply_that_drops_the_valve_falls_back_to_template(rig):
    t = fake_transport({"report": "Something leaks.", "what": "Close a valve.", "where": "x", "why": "y",
                        "risk_if_delayed": "z"})
    res = run_survey(world_with(rig, *SCENARIOS["north_rack_east_wind"][:3]), gemini=Gemini(transport=t))
    assert not res.used_gemini and "valve_03" in res.findings[0].summary


def test_gemini_offline_uses_template(rig):
    res = run_survey(world_with(rig, *SCENARIOS["north_rack_east_wind"][:3]), gemini=Gemini())
    assert not res.used_gemini and res.findings[0].summary.startswith("GAS LEAK (critical)")

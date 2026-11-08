import pytest

from policy.control.shift import FAULTS, LEAK, LINK_CUT, run
from rig.replay import load_timeline, timeline_session


@pytest.fixture(scope="module")
def shift(tmp_path_factory):
    out = tmp_path_factory.mktemp("argon")
    return run("replay", out), out


def events(res, name, **match):
    return [e for e in res["brain"].c.timeline if e["event"] == name
            and all(e["detail"].get(k) == v for k, v in match.items())]


def test_stuck_gas_head_is_flagged_dispatched_verified_and_sent_to_maintenance(shift):
    res, _ = shift
    t0 = next(t for t, sid, *_ in FAULTS if sid == "G-F1")
    fault = events(res, "sensor_fault", sensor="G-F1")[0]
    assert fault["detail"]["status"] == "STUCK" and fault["t"] - t0 <= 20.0
    d = events(res, "fault_dispatch", sensor="G-F1")[0]
    assert d["detail"]["source"] == "cloud" and d["t"] - fault["t"] <= 5.0
    reading = events(res, "verify_reading", sensor="G-F1")[0]
    assert not reading["detail"]["agree"] and reading["robot"] == d["robot"]
    m = events(res, "maintenance_raised", sensor="G-F1")[0]
    assert res["brain"].port.job(m["job"])["requested_by"] == "argon"
    assert events(res, "sensor_repaired", sensor="G-F1")


def test_edge_fallback_dispatches_while_the_link_is_cut(shift):
    res, _ = shift
    d = events(res, "fault_dispatch", sensor="gauge_03")[0]
    assert LINK_CUT[0] <= d["t"] < LINK_CUT[1] and d["detail"]["source"] == "edge"
    assert events(res, "maintenance_raised", sensor="gauge_03")


def test_mixed_fleet_drone_takes_the_flare_and_crawler_the_vessel_shell(shift):
    res, _ = shift
    assert events(res, "fault_dispatch", sensor="T-FLARE")[0]["robot"] == "A1"
    assert events(res, "fault_dispatch", sensor="A-V101")[0]["robot"] == "C1"
    assert events(res, "maintenance_raised", sensor="T-FLARE") and events(res, "maintenance_raised", sensor="A-V101")


def test_real_process_change_is_verified_and_cleared(shift):
    res, _ = shift
    assert events(res, "sensor_fault", sensor="gauge_06")[0]["detail"]["status"] == "DRIFT"
    assert events(res, "verify_reading", sensor="gauge_06")[0]["detail"]["agree"]
    assert events(res, "fault_cleared", sensor="gauge_06")


def test_dead_head_beside_the_leak_waits_for_the_gas_response(shift):
    res, _ = shift
    fault = events(res, "sensor_fault", sensor="G-F6")[0]
    assert fault["t"] > LEAK[0] and fault["detail"]["hazard"]
    tl = res["brain"].c.timeline
    assert any(e["event"] == "edge_rule" and e["detail"].get("rule") == "hazard_first" for e in tl)
    veto = events(res, "argon_vetoed", target="G-F6")
    assert veto and "hazard first" in veto[0]["detail"]["why"]
    iso = events(res, "leak_isolated", flange=LEAK[1])[0]
    assert events(res, "fault_dispatch", sensor="G-F6")[0]["t"] > iso["t"]
    assert events(res, "gas_alarm", flange=LEAK[1]) or events(res, "gas_detected", flange=LEAK[1])


def test_every_argon_call_is_on_the_timeline(shift):
    res, out = shift
    calls = events(res, "argon_call")
    assert calls and all({"latency_ms", "tokens_in", "tokens_out", "decision", "status"} <= set(c["detail"])
                         for c in calls)
    assert any(d["purpose"] == "verify_sensor" and d["rationale"] for c in calls for d in c["detail"]["decision"])
    lines = load_timeline(out / "timeline.jsonl")
    assert sum(x.get("event") == "argon_call" for x in lines) == len(calls)
    assert any(x.get("event") == "argon_control" for x in lines)


def test_safety_envelope_holds_through_the_shift(shift):
    res, _ = shift
    assert all(e["detail"]["by"] == "operator" for e in events(res, "approved"))
    assert not any(j.kind.startswith(("esd", "force", "move")) for j in res["brain"].c.jobs.values())


def test_hud_replay_carries_control_health_and_the_mixed_fleet(shift):
    _, out = shift
    s = timeline_session(load_timeline(out / "timeline.jsonl"))
    snaps = s["snapshots"]
    assert any(x.get("control", {}).get("health", {}).get("faults") for x in snaps)
    assert any(x.get("control", {}).get("banner") for x in snaps)
    assert any(e["to"] == "MAINTENANCE" for x in snaps for e in x["events"])
    kinds = {r["kind"] for r in snaps[-1]["fleet"]["robots"]}
    assert {"Q", "A", "C"} <= kinds

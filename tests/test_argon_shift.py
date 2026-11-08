import json

import pytest

from policy.control.shift import FAULTS, LEAK, LINK_CUT, run


@pytest.fixture(scope="module")
def shift(tmp_path_factory):
    out = tmp_path_factory.mktemp("argon")
    res = run("replay", out)
    return res, out


def events(res, name, **match):
    return [e for e in res["brain"].port.timeline if e["event"] == name
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
    assert res["brain"].port.jobs[m["job"]]["requested_by"] == "argon"
    assert events(res, "sensor_repaired", sensor="G-F1")


def test_edge_fallback_dispatches_while_the_link_is_cut(shift):
    res, _ = shift
    d = events(res, "fault_dispatch", sensor="gauge_03")[0]
    assert LINK_CUT[0] <= d["t"] < LINK_CUT[1] and d["detail"]["source"] == "edge"
    assert events(res, "maintenance_raised", sensor="gauge_03")


def test_real_process_change_is_verified_and_cleared(shift):
    res, _ = shift
    assert events(res, "sensor_fault", sensor="gauge_06")[0]["detail"]["status"] == "DRIFT"
    assert events(res, "verify_reading", sensor="gauge_06")[0]["detail"]["agree"]
    assert events(res, "fault_cleared", sensor="gauge_06")


def test_dead_head_beside_the_leak_waits_for_the_gas_response(shift):
    res, _ = shift
    fault = events(res, "sensor_fault", sensor="G-F6")[0]
    assert fault["t"] > LEAK[0] and fault["detail"]["hazard"]
    tl = res["brain"].port.timeline
    assert any(e["event"] == "edge_rule" and e["detail"].get("rule") == "hazard_first" for e in tl)
    veto = events(res, "argon_vetoed", target="G-F6")
    assert veto and "hazard first" in veto[0]["detail"]["why"]
    iso = events(res, "leak_isolated", flange=LEAK[1])[0]
    d = events(res, "fault_dispatch", sensor="G-F6")[0]
    assert d["t"] > iso["t"]
    assert events(res, "gas_alarm", flange=LEAK[1])


def test_every_argon_call_is_on_the_timeline(shift):
    res, out = shift
    calls = events(res, "argon_call")
    assert calls and all({"latency_ms", "tokens_in", "tokens_out", "decision", "status"} <= set(c["detail"])
                         for c in calls)
    assert any(d["purpose"] == "verify_sensor" and d["rationale"] for c in calls for d in c["detail"]["decision"])
    lines = (out / "timeline.jsonl").read_text().splitlines()
    assert len(lines) == len(res["brain"].port.timeline)


def test_safety_envelope_holds_through_the_shift(shift):
    res, _ = shift
    assert all(e["detail"]["by"] == "operator" for e in events(res, "approved"))
    assert not any(j["kind"].startswith(("esd", "force", "move")) for j in res["brain"].port.jobs.values())


def test_hud_session_plays_in_the_control_panel(shift):
    _, out = shift
    s = json.loads((out / "hud_session.json").read_text())
    snaps = s["snapshots"]
    assert snaps and all(x["pt"] <= y["pt"] for x, y in zip(snaps, snaps[1:]))
    assert any(x["control"]["health"]["faults"] for x in snaps)
    assert any(x["control"]["banner"] for x in snaps)
    assert any(e["to"] == "MAINTENANCE" for x in snaps for e in x["events"])

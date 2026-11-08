import json
import math

import numpy as np
import pytest

from integration.dual_argon import (FIX_LEAK, FIXER, GO_CHECK, HOLD, LEAK_CONFIRMED, LOG_SENSOR_FAULT,
                                    PIPE_SENSOR_FAULT, RESUME_RUST, SCRUBBER, ArgonLink, FixedDetector, ReplayCaller,
                                    Timeline, edge_rule, fuse_fleet, ids_for)
from integration.fleet_beats import (BANNER_OFFLINE, STUCK, TICKET_KIND, FlatHealth, Inbox, banner, cut_link,
                                     stuck_sensor)
from policy.control.envelope import Envelope
from policy.control.port import MemoryCoordinator


def rec(key, cmds, latency_ms=500.0, verdict="UNVERIFIED"):
    text = json.dumps({"verdict": verdict, "situation": key, "commands": [
        {"command": k, "robot": r, "target": tg, "confidence": 0.9, "rationale": "both readings"} for k, r, tg in cmds]})
    return json.dumps({"type": "argon_call", "status": "ok", "key": key, "text": text, "latency_ms": latency_ms,
                       "tokens_in": 100, "tokens_out": 40, "tokens_thought": 0, "tokens_cached": 0})


def replay(tmp_path, *rows):
    p = tmp_path / "rec.jsonl"
    p.write_text("\n".join(rows) + "\n")
    return ReplayCaller(str(p))


def kinds(cmds):
    return [(c.kind, c.robot, c.target) for c in cmds]


def rig():
    d1 = FixedDetector("G-F6", "F6", (-9.0, 0.0, 1.0))
    d2 = FixedDetector("G-F5", "F5", (4.0, 0.0, 1.0))
    return d1, d2


def run_stuck(link, until=16.0, value=15.0):
    d1, d2 = rig()
    dets, ids = [d1, d2], ids_for([d1, d2])
    beat = stuck_sensor({"G-F6": d1, "G-F5": d2}, at=2.0, value=value)
    u1, cmds, seen = (0.0, 0.5), [], {}
    for t in np.arange(0.0, until, 0.5):
        t = float(t)
        u2 = (-9.5, 0.5) if 8.0 <= t < 12.0 else (-20.0, 0.0)
        d1.read(lambda p: 0.01 + 0.004 * math.sin(t))
        d2.read(lambda p: 0.02)
        sn = {SCRUBBER: (1.0, u1), FIXER: (0.8, u2)}
        cmds += [(t, *k) for k in kinds(link.tick(t, beat.findings(t, dets, sn), {}, dets, ids, scrubber_pos=u1,
                                                 wait=True))]
        link.wait_idle()
        beat.sync(link)
        seen[t] = beat.led
    return beat, cmds, seen


def stuck_replay(tmp_path):
    return replay(tmp_path,
                  rec("G-F6:UNVERIFIED", [("hold", SCRUBBER, "F6"), ("go_check", FIXER, "F6")]),
                  rec("G-F6:PIPE_SENSOR_FAULT", [("log_sensor_fault", "", "G-F6")], verdict=PIPE_SENSOR_FAULT),
                  rec("zone:clear", [("resume_rust", SCRUBBER, "")], verdict="OK"))


@pytest.mark.parametrize("mode", ["edge", "replay"])
def test_stuck_sensor_files_one_ticket_to_the_human_inbox_and_dispatches_no_repair(tmp_path, mode):
    link = ArgonLink(mode, Timeline(None), stuck_replay(tmp_path) if mode == "replay" else None)
    beat, cmds, seen = run_stuck(link)
    assert [k for _, k, _, _ in cmds] == [HOLD, GO_CHECK, LOG_SENSOR_FAULT, RESUME_RUST]
    assert FIX_LEAK not in [k for _, k, _, _ in cmds]
    assert [(r, tg) for _, k, r, tg in cmds if k == LOG_SENSOR_FAULT] == [("supv", "G-F6")]
    assert beat.fault.kind == PIPE_SENSOR_FAULT and beat.fault.robot == FIXER
    assert [tk["id"] for tk in link.tickets] == ["MT-001"]
    tk = beat.ticket()
    port = beat.inbox.port
    job = port.job(tk["job_id"])
    assert (job["kind"], job["target"], job["status"], job["level"]) == (TICKET_KIND, "G-F6", "WAITING_APPROVAL",
                                                                         "SUPV")
    assert job["params"]["ticket"] == "MT-001" and beat.inbox.waiting() == [tk["job_id"]]
    events = [(e["event"], e["detail"].get("by")) for e in port.timeline]
    assert ("approved", "gemini") not in events and all(e != "approved" for e, _ in events)
    if mode == "replay":
        assert tk["source"] == "argon" and tk["envelope"]["ok"]
        assert "only a human approves" in tk["envelope"]["note"]
        assert ("approval_requested", "gemini") in events
    assert seen[1.0] == "green" and seen[5.0] == "red" and seen[15.5] == "grey"
    assert [k for t, k, _, _ in cmds if t >= 13.0] == []


def test_stuck_sensor_combines_with_the_stuck_health_verdict(tmp_path):
    beat, _, _ = run_stuck(ArgonLink("edge", Timeline(None)))
    assert beat.health_status == STUCK and beat.verdict == f"{PIPE_SENSOR_FAULT}+{STUCK}"
    assert beat.ticket()["health"] == STUCK and "health STUCK" in beat.fault.reason
    assert beat.hud() == {"sensor": "G-F6", "led": "grey", "verdict": f"{PIPE_SENSOR_FAULT}+{STUCK}",
                          "health": STUCK, "ticket": "MT-001", "job": beat.ticket()["job_id"],
                          "status": "WAITING_APPROVAL"}


def test_stuck_above_gas_stop_is_treated_as_real_gas():
    beat, cmds, seen = run_stuck(ArgonLink("edge", Timeline(None)), value=25.0)
    assert beat.fault is None and seen[15.5] == "red"


def test_argon_cannot_approve_a_ticket_it_raised_itself():
    port = MemoryCoordinator([])
    jid = port.queue(TICKET_KIND, "G-F6", level="SUPV", by="gemini", why="self-raised")
    out = Envelope().call(port, "approve_request", {"job_id": jid, "why": "approve my own ticket"})
    assert not out["ok"] and "cannot approve its own" in out["refused"]
    assert port.job(jid)["status"] == "WAITING_APPROVAL"


def test_flat_health_needs_a_flat_span():
    h = FlatHealth(min_span_s=3.0, min_samples=4)
    assert [h.observe(t, {"G": 5.0})["G"].status for t in (0, 1, 2, 3)] == ["OK", "OK", "OK", STUCK]
    assert h.observe(4, {"G": 6.0})["G"].status == "OK"


def leak_tick(link, t, beat, d2, g2, u2=(-12.0, 2.0)):
    d2.read(lambda p: g2)
    sn = {SCRUBBER: (2.0, (0.0, 0.5)), FIXER: (100 * g2 if u2[0] > 3 else 1.0, u2)}
    beat.tick(t)
    out = link.tick(t, [fuse_fleet(d2, sn)], {}, [d2], ids_for([d2]), scrubber_pos=(0.0, 0.5))
    link.wait_idle()
    return out


def test_cut_link_mid_leak_edge_makes_the_same_dispatch_within_one_tick(tmp_path):
    argon = [("hold", SCRUBBER, "F5"), ("go_check", FIXER, "F5")]
    caller = replay(tmp_path, rec("G-F5:UNVERIFIED", argon, latency_ms=3000.0),
                    rec("G-F5:LEAK_CONFIRMED", [("fix_leak", FIXER, "F5")], latency_ms=1000.0,
                        verdict=LEAK_CONFIRMED))
    link = ArgonLink("replay", Timeline(None), caller)
    _, d2 = rig()
    beat = cut_link(link, at=1.0, until=5.0)
    assert beat.tick(0.0) is None and banner(link) == ""
    hold = leak_tick(link, 0.0, beat, d2, 0.14)
    assert kinds(hold) == [(HOLD, SCRUBBER, "G-F5")] and link.pending is not None
    ev = beat.tick(1.0)
    assert ev["event"] == "link_cut" and ev["banner"] == BANNER_OFFLINE == "CLOUD OFFLINE — edge rules only"
    assert ev["dropped"] == ["G-F5:UNVERIFIED"] and banner(link) == BANNER_OFFLINE
    cmds = leak_tick(link, 1.0, beat, d2, 0.14)
    assert kinds(cmds) == [(GO_CHECK, FIXER, "F5")] and {c.source for c in cmds} == {"edge"}
    assert [k for k, _, _ in kinds(edge_rule(link.findings["G-F5"]))] == [k for k, _, _ in argon]
    assert leak_tick(link, 4.0, beat, d2, 0.14) == []
    ev = beat.tick(5.0)
    assert ev["event"] == "link_restored" and banner(link) == "" and link.cloud
    assert leak_tick(link, 5.0, beat, d2, 0.14) == []
    assert leak_tick(link, 6.0, beat, d2, 0.14, u2=(4.5, 0.6)) == []
    fix = leak_tick(link, 7.0, beat, d2, 0.14, u2=(4.5, 0.6))
    assert kinds(fix) == [(FIX_LEAK, FIXER, "F5")] and fix[0].source == "argon"
    assert [c["key"] for c in link.calls] == ["G-F5:LEAK_CONFIRMED"]
    assert [e["event"] for e in beat.events] == ["link_cut", "link_restored"]


def test_cut_link_with_a_new_alarm_goes_edge_on_the_same_tick():
    link = ArgonLink("live", Timeline(None), caller=type("C", (), {"available": True, "name": "x"})())
    _, d2 = rig()
    beat = cut_link(link, at=0.0)
    beat.tick(0.0)
    cmds = leak_tick(link, 0.0, beat, d2, 0.14)
    assert kinds(cmds) == [(HOLD, SCRUBBER, "G-F5"), (GO_CHECK, FIXER, "F5")] and link.calls == []

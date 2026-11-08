import json

import numpy as np

from integration.dual_argon import (FIX_LEAK, FIXER, GO_CHECK, HOLD, LEAK_CONFIRMED, LEAK_UNSEEN_BY_FIXED,
                                    LOG_SENSOR_FAULT, OK, PIPE_SENSOR_FAULT, RESUME_RUST, RETREAT, SCRUBBER,
                                    UNVERIFIED, ArgonLink, FixedDetector, ReplayCaller, Timeline, edge_rule, fuse,
                                    fuse_fleet, ids_for, led, spark_hot)


def det(v, stuck=None, did="G-F2", flange="F2", pos=(0, 0, 0)):
    d = FixedDetector(did, flange, pos, stuck_at=stuck)
    d.read(lambda p: v)
    return d


def kinds(cmds):
    return [(c.kind, c.robot, c.target) for c in cmds]


def test_fusion_verdicts():
    assert fuse(det(0.15), 2.0, 4.0).kind == UNVERIFIED
    assert fuse(det(0.15), 9.0, 1.1).kind == LEAK_CONFIRMED
    assert fuse(det(0.15), 1.0, 1.1).kind == PIPE_SENSOR_FAULT
    assert fuse(det(0.02), 14.0, 1.0).kind == LEAK_UNSEEN_BY_FIXED
    assert fuse(det(0.02), 2.0, 1.0).kind == OK
    assert fuse(det(0.02, stuck=30.0), 1.0, 1.0).hazard


def test_fuse_fleet_uses_the_nearest_sniffer():
    d = det(0.15, pos=(5.0, 0.0, 1.0))
    fd = fuse_fleet(d, {SCRUBBER: (1.0, (1.0, 0.0)), FIXER: (12.0, (5.5, 0.5))})
    assert (fd.kind, fd.robot) == (LEAK_CONFIRMED, FIXER)


def test_edge_rule_routes_to_capable_robot():
    assert kinds(edge_rule(fuse(det(0.15), 2.0, 4.0))) == [(HOLD, SCRUBBER, "F2"), (GO_CHECK, FIXER, "F2")]
    assert kinds(edge_rule(fuse(det(0.15), 9.0, 1.1))) == [(FIX_LEAK, FIXER, "F2")]
    assert kinds(edge_rule(fuse(det(0.15), 25.0, 1.1, robot=FIXER))) == [(HOLD, SCRUBBER, "F2"),
                                                                           (RETREAT, FIXER, "F2")]
    assert kinds(edge_rule(fuse(det(0.15), 1.0, 1.1))) == [(LOG_SENSOR_FAULT, "supv", "G-F2")]
    assert [c.kind for c in edge_rule(fuse(det(0.02), 14.0, 1.0))] == [FIX_LEAK, LOG_SENSOR_FAULT]


def test_led_colours():
    assert [led(v) for v in (1.0, 6.0, 12.0, None)] == ["green", "amber", "red", "grey"]
    assert led(30.0, fault=True) == "red" and led(12.0, fault=True) == "grey"


def test_spark_rule_radius_and_faults():
    near, far = det(0.07, pos=(4.0, 0, 0)), det(0.5, did="G-F9", flange="F9", pos=(20.0, 0, 0))
    assert spark_hot((0.0, 0.0), [near, far]) == ["G-F2"]
    assert spark_hot((0.0, 0.0), [near, far], faulted={"G-F2"}) == []


def run_scenario(link):
    d2, d1 = det(0.0, pos=(4.0, 0, 1)), det(0.0, did="G-F1", flange="F1", pos=(-9.0, 0, 1))
    dets, ids, log = [d2, d1], ids_for([d2, d1]), []
    u1, u2 = (0.0, 0.5), (-12.0, 2.0)
    for t, g2, u2 in ((0, 0.02, u2), (1, 0.07, u2), (2, 0.14, u2), (3, 0.14, u2), (4, 0.14, (4.5, 0.6)),
                      (5, 0.14, (4.5, 0.6)), (6, 0.06, (4.5, 0.6)), (7, 0.02, (4.5, 0.6)), (8, 0.02, (4.5, 0.6))):
        d2.read(lambda p: g2)
        d1.read(lambda p: 0.02)
        sn = {SCRUBBER: (100 * 0.02, u1), FIXER: (100 * (g2 if u2[0] > 4 else 0.02), u2)}
        fds = [fuse_fleet(d, sn) for d in dets]
        log += [(t, *k) for k in kinds(link.tick(t, fds, {}, dets, ids, scrubber_pos=u1, wait=True))]
        link.wait_idle()
    return log


def test_edge_fleet_scenario_holds_dispatches_fixes_resumes():
    link = ArgonLink("edge", Timeline(None))
    log = run_scenario(link)
    assert [k for _, k, _, _ in log] == [HOLD, GO_CHECK, FIX_LEAK, RESUME_RUST]
    assert log[0][0] == 1 and log[-1][0] == 7


def test_replay_link_delivers_after_recorded_latency(tmp_path):
    tl = tmp_path / "rec.jsonl"
    reply = {"verdict": "UNVERIFIED", "situation": "G-F2 alarm, nobody close",
             "commands": [{"command": "hold", "robot": "unit1", "target": "", "confidence": 0.9, "rationale": "spark"},
                          {"command": "go_check", "robot": "unit2", "target": "F2", "confidence": 0.9,
                           "rationale": "fixed high, unit1 low at 4 m"}]}
    tl.write_text(json.dumps({"type": "argon_call", "status": "ok", "key": "G-F2:UNVERIFIED", "text": json.dumps(reply),
                              "latency_ms": 3000.0, "tokens_in": 100, "tokens_out": 50, "tokens_thought": 0,
                              "tokens_cached": 0}) + "\n")
    link = ArgonLink("replay", Timeline(None), ReplayCaller(str(tl)))
    d = det(0.15)
    fd = fuse(d, 2.0, 4.0, robot=SCRUBBER)
    assert link.tick(0.0, [fd], {}, [d], ids_for([d])) == []
    link.wait_idle()
    assert link.tick(1.0, [fd], {}, [d], ids_for([d])) == []
    cmds = link.tick(3.1, [fd], {}, [d], ids_for([d]))
    assert kinds(cmds) == [(HOLD, SCRUBBER, ""), (GO_CHECK, FIXER, "F2")]
    assert {c.source for c in cmds} == {"argon"} and link.summary()["ok"] == 1


def test_sensor_fault_raises_one_ticket_and_no_repair():
    link = ArgonLink("edge", Timeline(None))
    d = det(0.0, stuck=18.0, did="G-F1", flange="F1", pos=(-9.0, 0, 1))
    out = []
    for t, pos in ((0, (-20.0, 0.0)), (1, (-9.2, 0.4)), (2, (-9.2, 0.4)), (4.5, (-9.2, 0.4))):
        out += link.tick(t, [fuse_fleet(d, {FIXER: (0.8, pos)})], {}, [d], ids_for([d]), wait=True)
        link.wait_idle()
    assert [c.kind for c in out] == [HOLD, GO_CHECK, LOG_SENSOR_FAULT, RESUME_RUST]
    assert [tk["sensor"] for tk in link.tickets] == ["G-F1"] and FIX_LEAK not in [c.kind for c in out]


def test_fixed_detector_reads_gas_model_in_lel_percent():
    d = FixedDetector("G-F1", "F1", (1.0, 0.0, 0.0))
    assert d.read(lambda p: 0.02 + 0.1 * float(np.linalg.norm(p))) == 12.0


def test_detector_state_matches_gas_detector_models():
    from integration.dual_argon import detector_state
    from integration.gas_detectors import STATES
    assert {detector_state(v) for v in (1.0, 6.0, 12.0)} | {detector_state(12.0, True)} == set(STATES)


class SlowCaller:
    live, available, name = True, True, "slow"

    def __init__(self, reply, latency_ms=8000.0):
        self.reply, self.latency_ms = reply, latency_ms

    def __call__(self, text, key):
        return {"text": json.dumps(self.reply), "latency_ms": self.latency_ms, "tokens_in": 10, "tokens_out": 5,
                "tokens_thought": 0, "tokens_cached": 0}


RESUME = {"verdict": "OK", "situation": "all clear",
          "commands": [{"command": "resume_rust", "robot": "unit1", "target": "", "confidence": 0.9, "rationale": "x"}]}


def test_stuck_head_above_stop_is_real_gas_not_a_fault():
    fd = fuse(det(0.0, stuck=30.0), 0.8, 0.4, robot=FIXER)
    assert (fd.kind, fd.hazard) == (LEAK_CONFIRMED, True)
    assert kinds(edge_rule(fd)) == [(HOLD, SCRUBBER, "F2"), (RETREAT, FIXER, "F2")]
    far = fuse(det(0.0, stuck=30.0), 0.8, 9.0, robot=FIXER)
    assert kinds(edge_rule(far)) == [(HOLD, SCRUBBER, "F2"), (GO_CHECK, FIXER, "F2")]


def test_robot_high_beyond_near_needs_a_check():
    assert fuse(det(0.01), 15.0, 3.0, robot=SCRUBBER).kind == UNVERIFIED


def test_hazard_acts_at_once_while_the_link_is_up():
    link = ArgonLink("live", Timeline(None), SlowCaller(RESUME))
    d = det(0.0, stuck=30.0)
    out = link.tick(0.0, [fuse(d, 0.8, 0.4, robot=FIXER)], {}, [d], ids_for([d]))
    assert kinds(out) == [(HOLD, SCRUBBER, "F2"), (RETREAT, FIXER, "F2")]
    link.wait_idle()
    assert link.tick(9.0, [fuse(d, 0.8, 0.4, robot=FIXER)], {}, [d], ids_for([d])) == []
    assert link.calls[-1]["advisory"] and link.held


def test_crack_set_triggers_retreat_for_that_flange_only():
    link = ArgonLink("edge", Timeline(None))
    d = det(0.15)
    out = link.tick(0.0, [fuse(d, 9.0, 1.0, robot=FIXER)], {}, [d], ids_for([d]), crack={"F2"})
    assert kinds(out) == [(HOLD, SCRUBBER, "F2"), (RETREAT, FIXER, "F2")]


def test_cut_link_while_pending_falls_back_to_edge_now():
    link = ArgonLink("live", Timeline(None), SlowCaller(RESUME, 20000.0))
    d = det(0.15)
    fd = fuse(d, 2.0, 4.0, robot=SCRUBBER)
    assert link.tick(0.0, [fd], {}, [d], ids_for([d])) == []
    link.cut(True, 1.0)
    out = link.tick(1.0, [fd], {}, [d], ids_for([d]))
    assert kinds(out) == [(HOLD, SCRUBBER, "F2"), (GO_CHECK, FIXER, "F2")] and {c.source for c in out} == {"edge"}


def test_argon_resume_is_vetoed_while_a_detector_near_unit1_is_red():
    link = ArgonLink("live", Timeline(None), SlowCaller(RESUME, 100.0))
    d = det(0.15, pos=(3.0, 0.0, 1.0))
    link.held = True
    link.tick(0.0, [fuse(d, 2.0, 4.0, robot=SCRUBBER)], {}, [d], ids_for([d]), scrubber_pos=(0.0, 0.0), wait=True)
    link.wait_idle()
    link.tick(1.0, [fuse(d, 2.0, 4.0, robot=SCRUBBER)], {}, [d], ids_for([d]), scrubber_pos=(0.0, 0.0), wait=True)
    assert RESUME_RUST not in [c.kind for c in link.commands] and link.held
    assert any(r["type"] == "vetoed" for r in link.timeline.rows)


def test_stale_reply_is_not_executed():
    link = ArgonLink("live", Timeline(None), SlowCaller({"verdict": "UNVERIFIED", "situation": "s", "commands": [
        {"command": "go_check", "robot": "unit2", "target": "F2", "confidence": 0.9, "rationale": "r"}]}, 3000.0))
    d = det(0.15)
    link.tick(0.0, [fuse(d, 2.0, 4.0)], {}, [d], ids_for([d]))
    link.wait_idle()
    d.read(lambda p: 0.01)
    out = link.tick(4.0, [fuse(d, 2.0, 4.0)], {}, [d], ids_for([d]))
    assert GO_CHECK not in [c.kind for c in out]


def test_far_robot_reading_does_not_ticket_an_unrelated_head():
    g6 = det(0.02, did="G-F6", flange="F6", pos=(10.0, 0.0, 1.0))
    fd = fuse_fleet(g6, {FIXER: (12.9, (0.0, 0.0))})
    assert (fd.kind, fd.hazard) == (OK, False) and edge_rule(fd)[0].kind == RESUME_RUST
    mid = fuse(det(0.02), 12.9, 3.0, robot=FIXER)
    assert mid.kind == UNVERIFIED and LOG_SENSOR_FAULT not in [c.kind for c in edge_rule(mid)]
    assert fuse(det(0.02), 12.9, 1.5, robot=FIXER).kind == LEAK_UNSEEN_BY_FIXED
    assert not fuse(det(0.02), 25.0, 9.0, robot=FIXER).hazard


def test_fault_verdict_needs_a_dwell_before_ticketing():
    link = ArgonLink("edge", Timeline(None))
    d = det(0.12)
    for t in (0.0, 1.0, 2.0):
        assert link.tick(t, [fuse(d, 1.0, 1.5, robot=FIXER)], {}, [d], ids_for([d])) == []
    out = link.tick(3.5, [fuse(d, 1.0, 1.5, robot=FIXER)], {}, [d], ids_for([d]))
    assert [c.kind for c in out] == [LOG_SENSOR_FAULT]


def test_one_low_tick_on_approach_to_a_real_leak_files_nothing_and_never_resumes():
    link = ArgonLink("edge", Timeline(None))
    g2 = det(0.116, pos=(4.4, 0.0, 1.0))
    u1, ids, mem, out = (0.0, 0.0), ids_for([g2]), {}, []
    seq = [(20.0, 3.5, 2.0), (21.0, 2.5, 3.9), (21.2, 1.96, 4.85), (21.4, 1.6, 6.2), (22.0, 1.2, 9.0)]
    for t, dist, lel in seq:
        fd = fuse_fleet(g2, {SCRUBBER: (1.0, u1), FIXER: (lel, (4.4 + dist, 0.0))}, memory=mem)
        out += link.tick(t, [fd], {}, [g2], ids, scrubber_pos=u1)
    ks = [c.kind for c in out]
    assert LOG_SENSOR_FAULT not in ks and RESUME_RUST not in ks and FIX_LEAK in ks
    assert "G-F2" not in link.faulted and link.held and link.tickets == []


def test_no_resume_while_an_unconfirmed_head_near_unit1_is_in_alarm():
    link = ArgonLink("edge", Timeline(None))
    g2 = det(0.116, pos=(4.4, 0.0, 1.0))
    link.held = True
    out = []
    for t in (0.0, 1.0, 2.0):
        out += link.tick(t, [fuse(g2, 4.0, 1.9, robot=FIXER)], {}, [g2], ids_for([g2]), scrubber_pos=(0.0, 0.0))
    assert RESUME_RUST not in [c.kind for c in out] and spark_hot((0.0, 0.0), [g2], link.faulted) == ["G-F2"]


def test_later_leak_verdict_clears_a_confirmed_fault():
    link = ArgonLink("edge", Timeline(None), fault_confirm_s=0.0)
    d = det(0.12)
    link.tick(0.0, [fuse(d, 1.0, 1.0, robot=FIXER)], {}, [d], ids_for([d]))
    assert "G-F2" in link.faulted and d.fault
    link.tick(1.0, [fuse(d, 9.0, 1.0, robot=FIXER)], {}, [d], ids_for([d]))
    assert "G-F2" not in link.faulted and not d.fault and led(d.last, d.fault) == "red"


def test_near_hysteresis_stops_flapping():
    d, mem = det(0.12), {}
    kinds_seen = [fuse_fleet(d, {FIXER: (9.0, (x, 0.0))}, memory=mem).kind for x in (1.9, 2.3, 2.1, 2.4, 2.6)]
    assert kinds_seen == [LEAK_CONFIRMED] * 4 + [UNVERIFIED]


def test_escalate_routes_to_supervisor_and_detector_targets_normalise():
    from integration.dual_argon import ESCALATE, ROUTE, command
    assert ROUTE[ESCALATE] == "supv"
    link = ArgonLink("edge", Timeline(None))
    out = link._issue(0.0, [command(HOLD, "G-F5", "argon", "x"), command(ESCALATE, "F5", "argon", "y")])
    assert kinds(out) == [(HOLD, SCRUBBER, "F5"), (ESCALATE, "supv", "F5")]


def test_a_ticketed_head_is_not_redispatched_when_the_robot_walks_away():
    link = ArgonLink("edge", Timeline(None), fault_confirm_s=0.0)
    d = det(0.0, stuck=15.0, did="G-F1", flange="F1", pos=(0.0, 0.0, 1.0))
    out = []
    for t, x in ((0.0, 9.0), (1.0, 1.0), (2.0, 1.0), (3.0, 9.0), (4.0, 12.0)):
        out += link.tick(t, [fuse_fleet(d, {FIXER: (0.8, (x, 0.0))})], {}, [d], ids_for([d]), scrubber_pos=(30.0, 0.0))
    assert [c.kind for c in out].count(GO_CHECK) == 1 and len(link.tickets) == 1

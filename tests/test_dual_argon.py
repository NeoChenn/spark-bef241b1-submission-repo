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
    for t, pos in ((0, (-20.0, 0.0)), (1, (-9.2, 0.4)), (2, (-9.2, 0.4))):
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

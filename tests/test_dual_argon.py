import json

import numpy as np

from integration.dual_argon import (FIX_LEAK, GO_CHECK, LEAK_CONFIRMED, LEAK_UNSEEN_BY_FIXED, LOG_SENSOR_FAULT, OK,
                                    PIPE_SENSOR_FAULT, RESUME_RUST, RETREAT, UNVERIFIED, ArgonLink, FixedDetector,
                                    ReplayCaller, Timeline, edge_rule, fuse, led)


def det(v, stuck=None):
    d = FixedDetector("G-F1", "F1", (0, 0, 0), stuck_at=stuck)
    d.read(lambda p: v)
    return d


def test_fusion_verdicts():
    assert fuse(det(0.15), 2.0, 4.0).kind == UNVERIFIED
    assert fuse(det(0.15), 9.0, 1.1).kind == LEAK_CONFIRMED
    assert fuse(det(0.15), 1.0, 1.1).kind == PIPE_SENSOR_FAULT
    assert fuse(det(0.02), 14.0, 1.0).kind == LEAK_UNSEEN_BY_FIXED
    assert fuse(det(0.02), 2.0, 1.0).kind == OK
    assert fuse(det(0.02, stuck=30.0), 2.0, 4.0).kind == UNVERIFIED


def test_edge_rule_maps_each_verdict():
    assert edge_rule(fuse(det(0.15), 2.0, 4.0), "unit1").kind == GO_CHECK
    assert edge_rule(fuse(det(0.15), 9.0, 1.1), "unit1").kind == FIX_LEAK
    assert edge_rule(fuse(det(0.15), 25.0, 1.1), "unit1").kind == RETREAT
    c = edge_rule(fuse(det(0.15), 1.0, 1.1), "unit1")
    assert (c.kind, c.target) == (LOG_SENSOR_FAULT, "G-F1")
    assert edge_rule(fuse(det(0.02), 2.0, 1.0), "unit1").kind == RESUME_RUST


def test_led_colours():
    assert [led(v) for v in (1.0, 6.0, 12.0, None)] == ["green", "amber", "red", "off"]


def test_replay_link_delivers_after_recorded_latency(tmp_path):
    tl = tmp_path / "rec.jsonl"
    reply = {"command": "go_check", "target": "F1", "confidence": 0.9, "verdict": "UNVERIFIED",
             "situation": "G-F1 alarm, robot far", "rationale": "fixed high, robot low at 4 m"}
    tl.write_text(json.dumps({"type": "argon_call", "status": "ok", "key": "G-F1:UNVERIFIED", "text": json.dumps(reply),
                              "latency_ms": 3000.0, "tokens_in": 100, "tokens_out": 50, "tokens_thought": 0,
                              "tokens_cached": 0}) + "\n")
    link = ArgonLink("replay", "unit1", Timeline(None), ReplayCaller(str(tl)))
    d = det(0.15)
    fd = fuse(d, 2.0, 4.0)
    assert link.observe(0.0, fd, {"state": "SCRUB"}, [d], {"F1", "G-F1"}) == []
    link.wait_idle()
    assert link.observe(1.0, fd, {"state": "SCRUB"}, [d], {"F1", "G-F1"}) == []
    cmds = link.observe(3.1, fd, {"state": "SCRUB"}, [d], {"F1", "G-F1"})
    assert [(c.kind, c.target, c.source) for c in cmds] == [("go_check", "F1", "argon")]
    assert link.summary()["ok"] == 1


def test_cut_link_falls_back_to_edge_rule():
    link = ArgonLink("edge", "unit1", Timeline(None))
    d = det(0.15)
    cmds = link.observe(0.0, fuse(d, 2.0, 4.0), {"state": "SCRUB"}, [d], {"F1"})
    assert [(c.kind, c.source) for c in cmds] == [("go_check", "edge")]


def test_fixed_detector_reads_gas_model_in_lel_percent():
    d = FixedDetector("G-F1", "F1", (1.0, 0.0, 0.0))
    assert d.read(lambda p: 0.02 + 0.1 * float(np.linalg.norm(p))) == 12.0

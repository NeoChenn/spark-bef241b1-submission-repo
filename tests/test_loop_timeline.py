import json

import pytest

from integration.loop_timeline import MISSION, MissionTimeline, mission_phase


def test_mission_phase_mapping():
    assert mission_phase("PATROL", "WALK") == "WALK"
    assert mission_phase("APPROACH", None) == "DEPLOY"
    assert mission_phase("SCRUB", None) == "SCRUB"
    assert mission_phase("BACK_OFF", None) == "SCRUB"
    assert mission_phase("VERIFY_PAINT", None) == "VERIFY_PAINT"
    assert mission_phase("PAINT", "REPOSITION") == "REPOSITION"
    assert mission_phase("DETECT", None) == "DETECT"


def test_timeline_spans_durations_and_order():
    tl = MissionTimeline()
    for t, ph in ((0.0, "WALK"), (5.0, "STANCE"), (6.0, "DEPLOY"), (8.0, "SPRAY"), (8.5, "SPRAY"), (12.0, "SOAK"),
                  (17.0, "SCRUB"), (30.0, "VERIFY"), (31.0, "REPOSITION"), (33.0, "PAINT"), (40.0, "VERIFY_PAINT"),
                  (41.0, "DONE")):
        tl.mark(t, ph)
    tl.close(41.0)
    d = tl.durations()
    assert d["WALK"] == pytest.approx(5.0) and d["SPRAY"] == pytest.approx(4.0) and d["DONE"] == 0.0
    assert [s["phase"] for s in tl.spans][:3] == ["WALK", "STANCE", "DEPLOY"]
    assert tl.in_order()
    assert tl.phase_at(12.5) == "SOAK" and tl.phase_at(100) == "DONE"


def test_timeline_detects_missing_or_out_of_order():
    tl = MissionTimeline()
    tl.mark(0.0, "WALK")
    tl.mark(1.0, "SCRUB")
    tl.mark(2.0, "SPRAY")
    tl.close(3.0)
    assert not tl.in_order()


def test_revisits_add_up():
    tl = MissionTimeline()
    for t, ph in ((0, "SOAK"), (2, "DETECT"), (3, "SOAK"), (5, "SCRUB")):
        tl.mark(t, ph)
    tl.close(6)
    assert tl.durations()["SOAK"] == pytest.approx(4.0)


def test_hud_record_shapes():
    tl = MissionTimeline()
    tl.mark(0.0, "SCRUB")
    rec = tl.hud_record(1.25, "SCRUB", 31.234, (1.0, 2.0, 0.3), {"rust": 0.4})
    assert rec["t"] == 1.25 and rec["state"] == "SCRUB" and rec["force_n"] == pytest.approx(31.23)
    r = rec["robots"][0]
    assert r["id"] == "Q1" and r["state"] == "SCRUB" and r["world_pos"] == [1.0, 2.0, 0.3] and r["caption"]
    json.dumps(rec)
    assert MISSION[0] == "WALK" and MISSION[-1] == "DONE"

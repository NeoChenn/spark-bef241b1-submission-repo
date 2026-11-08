import json
from pathlib import Path

import pytest

from integration.dual_argon import ALARM_LEL, FIX_LEAK, FIXER, RESUME_RUST, SCRUBBER, SPARK_RADIUS_M
from integration.run_fleet_demo import FleetDemo, beats

ORDER = ["1_unit1_scrubbing", "2_leak_start", "2_gf5_amber", "3_spark_hold", "2_gf5_red", "3_unverified",
         "3_dispatch_unit2", "4_unit2_walks", "4_leak_confirmed", "4_fix_leak", "4_unit2_sniffs", "4_retorque",
         "4_gf5_green", "5_resume_rust", "5_unit1_scrubs_again", "6_gf6_stuck", "6_gf6_fault", "6_ticket",
         "5_unit1_done", "5_unit2_docked"]
SAMPLE = Path(__file__).resolve().parent.parent / "runs_sample/fleet_demo/live_timeline.jsonl"


@pytest.fixture(scope="module")
def edge(tmp_path_factory):
    out = tmp_path_factory.mktemp("fleet_demo")
    demo = FleetDemo("edge", out)
    s = demo.run()
    return demo, s, demo.timeline.rows, out


def ticks(rows):
    return [r for r in rows if r["type"] == "tick"]


def test_every_beat_happens_in_order(edge):
    _, s, _, _ = edge
    b = s["beats"]
    missing = [k for k in ORDER if b[k] is None]
    assert not missing
    ts = [b[k] for k in ORDER]
    assert ts == sorted(ts), list(zip(ORDER, ts))


def test_run_finishes_inside_sixty_seconds_with_only_the_stuck_head_grey(edge):
    demo, s, rows, _ = edge
    assert s["t_end"] < 60.0
    last = ticks(rows)[-1]
    assert last["robots"][SCRUBBER]["state"] == "DONE" and last["robots"][FIXER]["state"] == "DOCKED"
    assert {d["id"]: d["led"] for d in last["detectors"]} == {**{f"G-F{i}": "green" for i in range(1, 6)},
                                                              "G-F6": "grey"}


def test_spark_hold_lands_on_the_tick_g_f5_turns_amber_and_the_job_is_paused_not_cancelled(edge):
    _, s, rows, _ = edge
    b = s["beats"]
    assert b["3_spark_hold"] - b["2_gf5_amber"] <= 0.1
    held = [r for r in ticks(rows) if b["3_unit1_held"] <= r["t"] < b["5_resume_rust"]]
    assert held and all(r["robots"][SCRUBBER]["state"] == "HOLD" and r["robots"][SCRUBBER]["force_n"] == 0.0
                        for r in held)
    assert len({r["robots"][SCRUBBER]["rust"] for r in held}) == 1
    assert held[0]["robots"][SCRUBBER]["paused_state"] == "SCRUB"


def test_unverified_comes_from_unit1_far_from_g_f5(edge):
    _, _, rows, _ = edge
    fd = next(r for r in rows if r["type"] == "finding" and r["kind"] == "UNVERIFIED" and r["sensor"] == "G-F5")
    assert fd["robot"] in (SCRUBBER, FIXER) and fd["robot_dist"] > 2.0 and fd["robot_lel"] < ALARM_LEL


def test_no_resume_while_g_f5_alarms_within_spark_radius_of_unit1(edge):
    demo, _, rows, _ = edge
    hot = demo.u1.hot_xy
    alarm = [r["t"] for r in ticks(rows) for d in r["detectors"] if d["id"] == "G-F5" and d["lel"] >= ALARM_LEL]
    g5 = next(d for d in demo.scn.detectors if d.id == "G-F5")
    assert g5.dist(hot) <= SPARK_RADIUS_M and alarm
    resumes = [r["t"] for r in rows if r["type"] == "command" and r["kind"] == RESUME_RUST]
    assert resumes and min(resumes) > max(alarm)
    for t in resumes:
        tick = max((r for r in ticks(rows) if r["t"] <= t), key=lambda r: r["t"])
        assert next(d for d in tick["detectors"] if d["id"] == "G-F5")["led"] == "green"


def test_stuck_g_f6_gets_one_ticket_in_the_inbox_and_no_repair(edge):
    _, _, rows, _ = edge
    assert [r["sensor"] for r in rows if r["type"] == "ticket"] == ["G-F6"]
    inbox = [r for r in rows if r["type"] == "inbox"]
    assert len(inbox) == 1 and inbox[0]["status"] == "WAITING_APPROVAL"
    assert not [r for r in rows if r["type"] == "command" and r["kind"] == FIX_LEAK and r["target"] == "F6"]


def test_units_move_with_nav_and_log_velocity_lidar_and_battery(edge):
    _, _, rows, _ = edge
    t = ticks(rows)
    for rid in (SCRUBBER, FIXER):
        nav = [r["robots"][rid]["nav"] for r in t]
        assert all({"vx", "wz", "lidar_min", "state"} <= set(n) for n in nav)
        assert max(abs(n["vx"]) for n in nav) > 0.2 and min(n["lidar_min"] for n in nav) < 4.0
        assert all(0.0 <= r["robots"][rid]["battery"] <= 1.0 for r in t)
    walked = [r["robots"][FIXER]["pose"] for r in t]
    assert max(abs(p[0] - walked[0][0]) + abs(p[1] - walked[0][1]) for p in walked) > 5.0


def test_header_jsonl_has_the_hud_shape(edge):
    _, _, _, out = edge
    hdrs = [json.loads(line) for line in (out / "hud.jsonl").read_text().splitlines()]
    h = hdrs[len(hdrs) // 2]
    assert {"t", "state", "caption", "robots", "events", "gas", "detectors", "fusion", "argon", "fleet"} <= set(h)
    assert {r["id"] for r in h["robots"]} == {SCRUBBER, FIXER}
    assert all({"world_pos", "caption", "battery", "nav", "lel"} <= set(r) for r in h["robots"])
    assert any(e["to"] == "HOLD" for x in hdrs for e in x["events"])


def test_cut_link_edge_rule_makes_the_same_calls(tmp_path):
    demo = FleetDemo("edge", None, cut_link_at=7.5)
    s = demo.run()
    b = s["beats"]
    assert b["6_link_cut"] is not None and b["3_dispatch_unit2"] and b["5_resume_rust"] and b["6_ticket"]


def test_object_on_the_walkway_is_reported_and_bypassed():
    demo = FleetDemo("edge", None, drop_at=27.0)
    s = demo.run()
    rows = demo.timeline.rows
    rep = [r for r in rows if r["type"] == "object_report"]
    assert rep and rep[0]["robot"] == FIXER
    assert any(r["type"] == "command" and r["kind"] == "reroute" for r in rows)
    assert s["beats"]["6_ticket"] is not None and s["beats"]["5_unit2_docked"] is not None


@pytest.mark.skipif(not SAMPLE.exists(), reason="no recorded live timeline")
def test_replay_of_the_live_recording_reaches_every_beat():
    demo = FleetDemo("replay", None, replay_from=str(SAMPLE))
    s = demo.run()
    assert all(s["beats"][k] is not None for k in ORDER)
    assert s["argon"]["ok"] >= 1
    assert beats(demo.timeline.rows) == s["beats"]

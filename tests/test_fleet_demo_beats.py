import json
import math
from pathlib import Path

import pytest

from integration.dual_argon import ALARM_LEL, FIX_LEAK, FIXER, RESUME_RUST, SCRUBBER, SPARK_RADIUS_M
from integration.run_fleet_demo import FleetDemo, beats

ORDER = ["6_gf6_stuck", "1_unit1_spray", "6_gf6_fault", "6_ticket", "1_unit1_soak", "1_unit1_scrub", "2_leak_start",
         "2_gf5_amber",
         "3_spark_hold", "2_gf5_red", "3_unverified", "3_dispatch_unit2", "4_unit2_walks", "4_leak_confirmed",
         "4_fix_leak", "4_unit2_sniffs", "4_retorque", "4_gf5_green", "5_resume_rust", "5_unit1_scrubs_again",
         "5_unit1_verify", "5_unit1_paint", "5_unit1_done", "5_unit2_docked"]
BUDGET_S = 90.0
RUST = ["SPRAY", "SOAK", "SCRUB", "HOLD", "SCRUB", "VERIFY", "PAINT", "DONE"]
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


def test_full_edge_run_finishes_inside_the_budget_with_only_the_stuck_head_grey(edge):
    demo, s, rows, _ = edge
    assert s["t_end"] < BUDGET_S and s["beats"]["4_gf5_green"] is not None
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


def test_unit1_runs_the_full_rust_sequence_with_visible_spray_and_paint(edge):
    _, _, rows, _ = edge
    seq = []
    surf = {}
    for r in ticks(rows):
        st = r["robots"][SCRUBBER]["state"]
        if not seq or seq[-1] != st:
            seq.append(st)
        surf.update(r["robots"][SCRUBBER].get("surface") or {})
        if st == "SPRAY":
            sprayed = surf
    keep = [x for x in seq if x in RUST]
    dedup = [x for i, x in enumerate(keep) if i == 0 or keep[i - 1] != x]
    assert dedup[:len(RUST)] == RUST
    spray_t = [r["t"] for r in ticks(rows) if r["robots"][SCRUBBER]["state"] == "SPRAY"]
    paint_t = [r["t"] for r in ticks(rows) if r["robots"][SCRUBBER]["state"] == "PAINT"]
    assert max(spray_t) - min(spray_t) >= 1.0 and max(paint_t) - min(paint_t) >= 1.0
    assert any("1" in row for row in surf["painted"]) and any("1" in row for row in sprayed["sprayed"])


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
    demo = FleetDemo("edge", None, cut_link_at=19.5)
    s = demo.run()
    b = s["beats"]
    assert b["6_link_cut"] is not None and b["3_dispatch_unit2"] and b["5_resume_rust"] and b["6_ticket"]


DISTURB = ["6_object_dropped", "6_object_report", "6_object_decision", "6_object_cleared", "6_unit2_walks_on",
           "6_push", "6_off_path", "6_recovered"]


@pytest.fixture(scope="module")
def disturbed():
    demo = FleetDemo("edge", None, drop_at=30.0, push_at=37.5)
    return demo, demo.run(), demo.timeline.rows


def test_crate_and_push_variant_still_finishes_with_every_beat(disturbed):
    demo, s, rows = disturbed
    b = s["beats"]
    assert not [k for k in ORDER + DISTURB if b[k] is None]
    assert [b[k] for k in ORDER] == sorted(b[k] for k in ORDER)
    assert [b[k] for k in DISTURB] == sorted(b[k] for k in DISTURB)
    assert b["3_dispatch_unit2"] < b["6_object_dropped"] < b["4_unit2_sniffs"] and b["6_recovered"] < b["4_retorque"]
    assert s["t_end"] < 95.0
    last = ticks(rows)[-1]
    assert last["robots"][SCRUBBER]["state"] == "DONE" and last["robots"][FIXER]["state"] == "DOCKED"


def test_crate_is_logged_for_isaac_and_unit2_stops_short_of_it(disturbed):
    _, s, rows = disturbed
    drop = next(r for r in rows if r["type"] == "scenario" and r.get("name") == "object_dropped")
    clear = next(r for r in rows if r["type"] == "scenario" and r.get("name") == "object_cleared")
    assert drop["size_m"] == [0.5, 0.5, 0.5] and len(drop["xyz"]) == 3 and clear["t_clear"] > drop["t_drop"]
    rep = next(r for r in rows if r["type"] == "object_report")
    assert rep["robot"] == FIXER and rep["object"] == "box" and 0.5 <= rep["dist_m"] <= 1.6
    stopped = [r for r in ticks(rows) if r["robots"][FIXER]["nav"]["state"] in ("REPORT", "STOP_OBJECT")]
    xy = [r["robots"][FIXER]["pose"][:2] for r in stopped]
    assert stopped and max(abs(p[0] - xy[0][0]) + abs(p[1] - xy[0][1]) for p in xy) < 0.4
    assert any(r["type"] == "command" and r.get("finding") == "OBJECT_ON_WALKWAY" for r in rows)


def test_push_shoves_unit2_off_its_path_and_it_recovers(disturbed):
    _, s, rows = disturbed
    push = next(r for r in rows if r["type"] == "scenario" and r.get("name") == "push")
    assert abs((push["dxy"][0] ** 2 + push["dxy"][1] ** 2) ** 0.5 - 0.4) < 0.01
    lat = [abs(r["robots"][FIXER]["nav"].get("lateral_m") or 0.0) for r in ticks(rows)
           if push["t"] <= r["t"] <= s["beats"]["6_recovered"]]
    assert max(lat) >= 0.35
    rec = next(r for r in rows if r["type"] == "nav_event" and r.get("event") == "recovered" and r["t"] > push["t"])
    assert abs(rec["lateral_m"]) < 0.15


@pytest.mark.skipif(not SAMPLE.exists(), reason="no recorded live timeline")
def test_replay_of_the_live_recording_reaches_every_beat():
    demo = FleetDemo("replay", None, replay_from=str(SAMPLE))
    s = demo.run()
    assert all(s["beats"][k] is not None for k in ORDER)
    assert s["argon"]["ok"] >= 1
    assert beats(demo.timeline.rows) == s["beats"]


FILM = ["calm_end", "alarm", "dispatch_u2", "u2_walk", "u2_arrive", "u2_engage", "torque_done", "gf5_green",
        "dispatch_u1", "u1_arrive", "spray", "soak", "scrub", "scrub_done", "paint", "paint_done", "u1_return",
        "u1_docked", "both_docked"]
FILM_SAMPLE = Path(__file__).resolve().parent.parent / "runs_sample/film"


@pytest.fixture(scope="module")
def film():
    demo = FleetDemo("edge", None, film=True)
    return demo, demo.run(), demo.timeline.rows


def test_film_story_runs_in_order_with_both_units_back_home(film):
    demo, s, rows = film
    b = s["beats"]
    assert not [k for k in FILM + ["leak_start", "rust_flag", "u1_held", "u2_return", "u2_docked"] if b[k] is None]
    assert [b[k] for k in FILM] == sorted(b[k] for k in FILM)
    assert b["calm_end"] >= 3.0 and b["rust_flag"] - b["leak_start"] <= 1.0 and b["u1_held"] < b["dispatch_u2"]
    assert b["u2_return"] == b["u1_return"] and b["torque_done"] < b["u2_standby"] < b["u1_arrive"]
    last = ticks(rows)[-1]
    assert last["robots"][SCRUBBER]["state"] == "DOCKED" and last["robots"][FIXER]["state"] == "DOCKED"
    assert math.dist(last["robots"][SCRUBBER]["pose"][:2], demo.u1.dock) < 0.3
    assert math.dist(last["robots"][FIXER]["pose"][:2], demo.u2.dock) < 0.3
    assert not [r for r in rows if r["type"] in ("ticket", "object_report") or r.get("name") in ("push", "object_dropped")]


def test_film_unit2_stands_by_at_f5_and_both_walk_home_together(film):
    demo, s, rows = film
    b = s["beats"]
    stand = [r for r in ticks(rows) if b["u2_standby"] <= r["t"] < b["u2_return"]]
    assert stand and all(r["robots"][FIXER]["state"] == "STANDBY" for r in stand)
    assert all(r["robots"][FIXER]["caption"] == "Unit 2 standing by at F5 while Unit 1 works" for r in stand)
    xy = [r["robots"][FIXER]["pose"][:2] for r in stand]
    assert max(math.dist(p, xy[0]) for p in xy) < 0.05 and math.dist(xy[0], demo.u2.dock) > 5.0
    home = lambda rid: [r["t"] for r in ticks(rows) if r["robots"][rid]["state"] == "RETURN"]
    u1, u2 = home(SCRUBBER), home(FIXER)
    assert u1 and u2 and max(min(u1), min(u2)) < min(max(u1), max(u2))
    assert b["u2_return"] >= b["paint_done"] and b["both_docked"] == max(b["u1_docked"], b["u2_docked"])


def test_film_unit1_waits_docked_until_g_f5_is_green(film):
    _, s, rows = film
    b = s["beats"]
    before = [r for r in ticks(rows) if r["t"] < b["dispatch_u1"]]
    assert {r["robots"][SCRUBBER]["state"] for r in before} <= {"DOCKED", "QUEUED"}
    tick = max((r for r in ticks(rows) if r["t"] <= b["dispatch_u1"]), key=lambda r: r["t"])
    assert next(d for d in tick["detectors"] if d["id"] == "G-F5")["led"] == "green"


def test_film_gas_only_falls_after_the_torque_completes(film):
    _, s, rows = film
    td = s["beats"]["torque_done"]
    g5 = [(r["t"], next(d["lel"] for d in r["detectors"] if d["id"] == "G-F5")) for r in ticks(rows)]
    pre = [v for t, v in g5 if s["beats"]["leak_start"] <= t <= td]
    assert all(b >= a - 0.01 for a, b in zip(pre, pre[1:]))
    post = [v for t, v in g5 if t > td + 1.0]
    assert post[0] < pre[-1] and post[-1] < 5.0


def test_film_board_is_modelled_and_tells_the_same_story(film):
    _, s, rows = film
    b = s["beats"]
    t = ticks(rows)
    calm = [r["board"] for r in t if r["t"] < b["calm_end"]]
    flagged = [r["board"] for r in t if b["rust_flag"] < r["t"] < b["dispatch_u1"]]
    done = t[-1]["board"]
    assert calm and all(x["modelled"] and not x["corrosion_alarm"] for x in calm)
    assert all("F5-F6" in x["corrosion_alarm"] and x["coating"]["F5-F6"] == "breached" for x in flagged)
    assert not done["corrosion_alarm"] and done["coating"]["F5-F6"] == "new primer"
    assert {"pressure_bar", "temperature_c", "flow_m3h"} <= set(done["process"]) and "sea_state" in done["weather"]
    assert set(done["battery"]) == {SCRUBBER, FIXER} and s["board_modelled"]


@pytest.mark.skipif(not (FILM_SAMPLE / "timeline.jsonl").exists(), reason="no recorded film run")
def test_film_replay_reproduces_the_recorded_run():
    rec = json.loads((FILM_SAMPLE / "summary.json").read_text())
    demo = FleetDemo("replay", None, replay_from=str(FILM_SAMPLE / "timeline.jsonl"), film=True)
    s = demo.run()
    assert s["beats"] == rec["beats"] and s["argon"]["commands"] == rec["argon"]["commands"]
    assert [c["key"] for c in s["argon_calls"]] == ["G-F5:UNVERIFIED", "G-F5:LEAK_CONFIRMED", "zone:clear"]

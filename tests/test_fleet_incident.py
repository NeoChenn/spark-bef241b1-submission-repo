"""CPU tests for the gas-leak incident (fleet/incident.py): fixed detectors, two dogs, operator approval."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from fleet import sensors
from fleet.contract import RobotKind
from fleet.incident import LEAK, Incident, render_timeline, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
import fleet_replay as fr  # noqa: E402

ROBOT_KEYS = {"t", "robot", "kind", "position", "yaw", "state", "job"}
EVENT_KEYS = {"t", "event", "job", "robot", "detail"}
STATES = {"IDLE", "MOVING", "WAITING_SEGMENT", "WORKING", "PAUSED_PERSON", "DOCKING", "DOCK_QUEUE", "CHARGING",
          "FAILED"}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("incident") / "timeline.jsonl"
    inc = Incident(leak_at=15.0, approve_after=30.0, timeline_path=path)
    s = inc.run()
    return inc, s, path


def test_chain_completes_in_order(run):
    inc, s, _ = run
    m = s["milestones_t"]
    order = ["leak_t", "detect_t", "survey_start_t", "localised_t", "approval_requested_t", "approved_t",
             "isolate_start_t", "isolated_t", "confirm_start_t", "confirmed_t", "prep_start_t", "prep_done_t"]
    assert all(k in m for k in order), [k for k in order if k not in m]
    times = [m[k] for k in order]
    assert times == sorted(times)
    assert m["localised_t"] <= m["approval_requested_t"] < m["approved_t"] <= m["isolate_start_t"]
    assert s["confirm_result"] == "leak_isolated"
    assert s["robots_by_role"] == {"survey": "Q2", "isolate": "Q1", "confirm": "Q2", "prep": "Q2"}
    assert s["robots_involved"] == ["Q1", "Q2"]
    assert s["names"] == {"Q1": "Dog-1", "Q2": "Dog-2"}
    prep = next(j for j in inc.c.jobs.values() if j.kind == "leak_repair_prep")
    assert prep.id in inc.c.crew


def test_no_drone_or_crawler_in_the_default_run(run):
    inc, s, _ = run
    assert set(inc.c.rt) == {"Q1", "Q2"}
    assert not any(j.kind in ("ut_thickness", "aerial_survey", "crawler_ut") for j in inc.c.jobs.values())


def test_alarm_comes_from_a_fixed_detector(run):
    inc, s, _ = run
    fa = s["first_alarm"]
    det = sensors.BY_ID[fa["detector"]]
    assert det.type == "point_gas" and fa["position"] == list(det.position)
    assert fa["reading"] >= det.threshold
    # the first detector to alarm is the gas detector nearest the leak, and it alarms after the gas reaches it
    assert det == sensors.nearest(LEAK["position"], [d for d in sensors.DETECTORS if d.type == "point_gas"])
    assert fa["after_leak_s"] >= math.dist(det.position[:2], LEAK["position"][:2]) / sensors.TRANSPORT_MPS
    events = [l for l in inc.sim.lines if "event" in l]
    alarm = next(e for e in events if e["event"] == "gas_alarm")
    raised = next(e for e in events if e["event"] == "detector_alarm")
    assert raised["t"] == alarm["t"] == s["milestones_t"]["detect_t"]
    assert alarm["detail"]["detector"] == det.id and raised["detail"]["position"] == list(det.position)
    assert inc.m["detect_by"] == det.id and inc.m["detect_how"] == "gas_alarm"
    # the survey starts from the walkway beside the alarming detector, not from the (unknown) leak position
    survey = inc.c.jobs[alarm["job"]]
    assert survey.kind == "gas_survey" and survey.params["detector"] == det.id
    assert survey.position[0] == det.position[0] and survey.robot_kinds == {RobotKind.QUADRUPED}
    assert not any(e["event"] == "gas_detected" for e in events)


def test_detector_readings_follow_the_plume():
    state = {"gas_leaks": [{**LEAK, "t0": 0.0}], "wind": [0.0, 0.0]}
    near, far = sensors.BY_ID["GD-06"], sensors.BY_ID["GD-01"]
    assert sensors.reading(near, state, 0.0) == pytest.approx(0.02)            # nothing before the gas arrives
    assert sensors.reading(near, state, 60.0) > near.threshold > sensors.reading(far, state, 60.0)
    assert sensors.reading(sensors.BY_ID["fg_ir_01"], state, 60.0) is None   # flame detectors read no gas
    from fleet.jobs.rounds import DEFAULT_FG_DETECTORS
    for did, d in DEFAULT_FG_DETECTORS.items():
        assert list(sensors.BY_ID[did].position) == d["position"]
    assert 6 <= sum(d.type == "point_gas" for d in sensors.DETECTORS) <= 10


def test_supv_isolation_waits_for_operator(run):
    inc, s, _ = run
    m = s["milestones_t"]
    assert s["approved_by"] == "operator" and not s["actuated_before_approval"]
    assert s["times_s"]["approval_wait_s"] == pytest.approx(30.0, abs=1.0)
    events = [l for l in inc.sim.lines if "event" in l]
    iso = next(j for j in inc.c.jobs.values() if j.kind == "valve_isolate")
    approved = next(e["t"] for e in events if e["event"] == "approved" and e["job"] == iso.id)
    assert all(e["t"] >= approved for e in events if e["job"] == iso.id and e["event"] in ("assigned", "job_started"))
    assert inc.world.state["valves"][iso.target]["closed_t"] >= approved
    # the gas at the source stays at full strength until the valve is worked
    before = [g for t, g in inc.gas_series if m["leak_t"] + 5 <= t < m["isolate_start_t"]]
    assert min(before) > 0.3


def test_unapproved_isolation_never_actuates(tmp_path):
    inc = Incident(leak_at=10.0, approve_after=1e6, timeline_path=tmp_path / "t.jsonl")
    inc.run(max_s=260.0)
    iso = next(j for j in inc.c.jobs.values() if j.kind == "valve_isolate")
    assert inc.c.status[iso.id].value == "WAITING_APPROVAL"
    assert not inc.world.state.get("valves", {}).get(iso.target, {}).get("closed")
    assert not any(l.get("isolated") for l in inc.world.state["gas_leaks"])
    assert not any(e.get("event") == "job_started" and e["job"] == iso.id for e in inc.sim.lines)


def test_numbers_are_sane(run):
    _, s, _ = run
    loc, gas, ts = s["localisation"], s["gas_lel"], s["times_s"]
    assert loc["error_m"] < 1.5 and loc["truth_inside_radius"] and 0.05 < loc["radius95_m"] < 2.0
    assert loc["peak_lel"] < 0.2 + 0.05                        # the survey stays out of the stop limit
    assert s["valve"] == "valve_06" and "valve_05" in " ".join(loc["valves_skipped"])
    assert 0.3 < gas["at_source_before"] < 0.6 and gas["at_source_after"] < 0.1
    assert gas["confirm_ratio"] <= 0.25 and gas["confirm_excess_after"] < gas["confirm_excess_before"]
    assert 0 < ts["leak_to_detect_s"] <= 30 and ts["leak_to_localised_s"] < 200
    assert ts["leak_to_isolated_s"] < 600 and ts["leak_to_confirmed_s"] < 900
    assert 5.0 < s["torque_peak_nm"] < 9.5


def test_timeline_format_is_exact(run):
    inc, _, path = run
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    per_t: dict[float, set] = {}
    for r in rows:
        if "event" in r:
            assert set(r) == EVENT_KEYS and isinstance(r["detail"], dict)
            if "position" in r["detail"]:
                assert len(r["detail"]["position"]) == 3
        else:
            assert set(r) == ROBOT_KEYS
            assert len(r["position"]) == 3 and all(isinstance(v, float) for v in r["position"])
            assert r["kind"] == "Q" and r["state"] in STATES and isinstance(r["yaw"], float)
            per_t.setdefault(r["t"], set()).add(r["robot"])
    assert all(v == {"Q1", "Q2"} for v in per_t.values())
    ts = sorted(per_t)
    assert ts == [float(k) for k in range(len(ts))]
    tl = fr.parse_timeline(path.read_text(encoding="utf-8").splitlines())
    assert set(tl.tracks) == {"Q1", "Q2"}
    assert any(e.event == "gas_leak_injected" and e.position is not None for e in tl.events)


def test_render_timeline_parses_and_fades_plume(run, tmp_path):
    inc, s, _ = run
    rows, info = render_timeline(inc)
    write_jsonl(tmp_path / "r.jsonl", rows)
    for r in rows:
        assert set(r) == (EVENT_KEYS if "event" in r else ROBOT_KEYS)
    tl = fr.load_timeline(tmp_path / "r.jsonl")
    assert set(tl.tracks) == {"Q1", "Q2"}
    assert 30.0 < info["render_s"] < 120.0 and tl.duration == pytest.approx(info["render_s"], abs=0.5)
    iso = next(e for e in tl.events if e.event == "gas_isolated")
    assert fr.gas_plumes(tl, iso.t - 1.0) and not fr.gas_plumes(tl, iso.t + 7.0)
    assert len(fr.gas_plumes(tl, iso.t - 1.0)) == 1               # the alarm at the detector is not a plume
    leak = next(e for e in tl.events if e.event == "gas_leak_injected")
    assert list(leak.position) == list(LEAK["position"])
    # the detector alarm is in the render timeline with its id, position and reading, and lights the detector
    alarm = next(e for e in tl.events if e.event == "detector_alarm")
    det = sensors.BY_ID[alarm.detail["detector"]]
    assert list(alarm.position) == list(det.position) and "LEL" in alarm.text()
    assert fr.detector_states(tl, alarm.t - 0.5).get(det.id) is None
    assert fr.detector_states(tl, alarm.t + 0.5)[det.id] == "alarm"
    # Dog-2 walks the survey's sample points on the walkway side instead of jumping
    q2 = tl.tracks["Q2"]
    assert max(q2.speed(t) for t in q2.t[:-1]) < 8.0
    assert min(q2.p[:, 1]) >= -9.0 - 1e-6
    assert len(fr.plan_shots(tl)) > 6

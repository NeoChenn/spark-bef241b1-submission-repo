"""CPU tests for the multi-robot gas-leak incident (fleet/incident.py) on the real executors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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
             "isolate_start_t", "isolated_t", "confirm_start_t", "confirmed_t", "ut_start_t", "ut_done_t"]
    assert all(k in m for k in order), [k for k in order if k not in m]
    times = [m[k] for k in order]
    assert times == sorted(times)
    assert m["localised_t"] <= m["approval_requested_t"] < m["approved_t"] <= m["isolate_start_t"]
    assert s["confirm_result"] == "leak_isolated"
    assert s["robots_by_role"] == {"survey": "A1", "isolate": "Q1", "confirm": "Q2", "prep": "Q2", "ut": "C1"}
    assert s["robots_involved"] == ["A1", "C1", "Q1", "Q2"]
    assert s["names"] == {"A1": "Drone", "C1": "Crawler", "Q1": "Dog-1", "Q2": "Dog-2"}


def test_incident_jobs_use_real_executors(run):
    _, s, _ = run
    for kind in ("gas_survey", "valve_isolate", "ut_thickness"):
        assert s["executors"][kind] == "real"


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
    inc = Incident(leak_at=10.0, approve_after=1e6, ut=False, timeline_path=tmp_path / "t.jsonl")
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
    assert 0 < ts["leak_to_detect_s"] <= 60 and ts["leak_to_localised_s"] < 200
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
            assert r["kind"] in ("Q", "A", "C") and r["state"] in STATES and isinstance(r["yaw"], float)
            per_t.setdefault(r["t"], set()).add(r["robot"])
    assert all(v == {"Q1", "Q2", "A1", "C1"} for v in per_t.values())
    ts = sorted(per_t)
    assert ts == [float(k) for k in range(len(ts))]
    tl = fr.parse_timeline(path.read_text(encoding="utf-8").splitlines())
    assert set(tl.tracks) == {"Q1", "Q2", "A1", "C1"}
    assert any(e.event == "gas_leak_injected" and e.position is not None for e in tl.events)


def test_render_timeline_parses_and_fades_plume(run, tmp_path):
    inc, s, _ = run
    rows, info = render_timeline(inc)
    write_jsonl(tmp_path / "r.jsonl", rows)
    for r in rows:
        assert set(r) == (EVENT_KEYS if "event" in r else ROBOT_KEYS)
    tl = fr.load_timeline(tmp_path / "r.jsonl")
    assert set(tl.tracks) == {"Q1", "Q2", "A1", "C1"}
    assert 40.0 < info["render_s"] < 120.0 and tl.duration == pytest.approx(info["render_s"], abs=0.5)
    iso = next(e for e in tl.events if e.event == "gas_isolated")
    assert fr.gas_plumes(tl, iso.t - 1.0) and not fr.gas_plumes(tl, iso.t + 7.0)
    leak = next(e for e in tl.events if e.event == "gas_leak_injected")
    assert list(leak.position) == list(LEAK["position"])
    # the drone flies through the survey's sample points instead of jumping
    a1 = tl.tracks["A1"]
    assert max(a1.speed(t) for t in a1.t[:-1]) < 40.0
    assert len(fr.plan_shots(tl)) > 6

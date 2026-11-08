import json

from rig.replay import NAMES, load_timeline, timeline_session

LINES = [
    {"t": 0.0, "event": "job_queued", "job": "rust_repair-1", "robot": None,
     "detail": {"kind": "rust_repair", "target": "pipe_rack_A", "level": "SUPV", "priority": 3}},
    {"t": 0.0, "event": "approval_requested", "job": "rust_repair-1", "robot": None,
     "detail": {"kind": "rust_repair", "target": "pipe_rack_A", "text": "Paint after scrub, Unit 1 at 82 %"}},
    {"t": 0.0, "robot": "Q1", "kind": "Q", "position": [1.0, 2.0, 42.5], "yaw": 1.57, "state": "IDLE", "job": None},
    {"t": 0.0, "robot": "A1", "kind": "A", "position": [0.0, 0.0, 60.0], "yaw": 0.0, "state": "MOVING", "job": "s-1"},
    {"t": 1.0, "event": "approved", "job": "rust_repair-1", "robot": None, "detail": {"by": "operator"}},
    {"t": 1.0, "event": "assigned", "job": "rust_repair-1", "robot": "Q1", "detail": {"cost_s": 9.0}},
    {"t": 1.0, "robot": "Q1", "kind": "Q", "position": [2.0, 2.0, 42.5], "yaw": 0.0, "state": "MOVING", "job": "rust_repair-1"},
    {"t": 1.0, "robot": "A1", "kind": "A", "position": [1.0, 0.0, 60.0], "yaw": 0.0, "state": "MOVING", "job": "s-1"},
    {"t": 2.0, "event": "gas_detected", "job": "gas_survey-2", "robot": "Q1",
     "detail": {"position": [21.0, -10.2, 43.4], "source": "robot gas sensor"}},
    {"t": 2.0, "event": "low_battery", "job": None, "robot": "A1", "detail": {"battery": 0.28, "reason": "idle and low"}},
    {"t": 2.0, "event": "person_pause", "job": "rust_repair-1", "robot": "Q1", "detail": {"distance_m": 2.6}},
    {"t": 2.0, "robot": "Q1", "kind": "Q", "position": [2.0, 2.0, 42.5], "yaw": 0.0, "state": "PAUSED_PERSON", "job": "rust_repair-1"},
    {"t": 2.0, "robot": "A1", "kind": "A", "position": [1.0, 0.0, 60.0], "yaw": 0.0, "state": "DOCKING", "job": None},
]


def test_names():
    assert NAMES["Q1"] == "Unit 1 · rust" and NAMES["Q2"] == "Unit 2 · gas"


def test_session_shape():
    s = timeline_session(LINES, rate=2.0)
    snaps = s["snapshots"]
    assert [x["t"] for x in snaps] == [0.0, 1.0, 2.0]
    assert [x["pt"] for x in snaps] == [0.0, 0.5, 1.0] and s["duration"] == 1.0
    r0 = snaps[0]["robots"][0]
    assert r0["id"] == "Q1" and r0["name"] == "Unit 1 · rust" and r0["world_pos"] == [1.0, 2.0, 42.5]
    assert r0["yaw"] == 1.57 and r0["kind"] == "Q"
    assert snaps[2]["robots"][0]["state"] == "PAUSED" and snaps[2]["robots"][1]["state"] == "TO_DOCK"
    assert snaps[2]["state"] == "PAUSED"
    json.dumps(s)


def test_events_become_decisions():
    snaps = timeline_session(LINES)["snapshots"]
    assert [e["to"] for e in snaps[0]["events"]] == ["QUEUED", "APPROVAL"]
    ev = snaps[2]["events"]
    assert any("2.6 m" in e["reason"] for e in ev)
    assert all({"t", "from", "to", "reason"} <= set(e) for e in ev)


def test_orchestrator_queue_and_inbox():
    snaps = timeline_session(LINES)["snapshots"]
    o0 = snaps[0]["orchestrator"]
    assert o0["jobs"][0]["status"] == "WAITING_APPROVAL" and o0["jobs"][0]["level"] == "SUPV"
    assert o0["approvals"] == [{"job": "rust_repair-1", "kind": "rust_repair", "target": "pipe_rack_A",
                                "text": "Paint after scrub, Unit 1 at 82 %"}]
    o1 = snaps[1]["orchestrator"]
    assert o1["approvals"] == [] and o1["jobs"][0]["status"] == "ASSIGNED" and o1["jobs"][0]["robot"] == "Q1"
    assert "orchestrator" not in snaps[2]


def test_fleet_snapshot_and_gas():
    snaps = timeline_session(LINES)["snapshots"]
    f = snaps[2]["fleet"]
    assert f["schema"] == "fleet_state.v1"
    a1 = next(r for r in f["robots"] if r["id"] == "A1")
    assert a1["battery"]["soc"] == 0.28 and a1["pose"]["z"] == 60.0
    kinds = {z["kind"] for z in f["zones"]}
    assert {"gas", "person"} <= kinds
    assert snaps[2]["gas"]["alarm"] is True and snaps[2]["gas"]["max"] is None
    assert "gas" not in snaps[0]


def test_load_timeline_skips_bad_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(LINES[2]) + "\nnot json\n\n" + json.dumps(LINES[3]) + "\n")
    assert len(load_timeline(p)) == 2

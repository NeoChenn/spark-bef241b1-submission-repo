import json

from fastapi.testclient import TestClient

from rig.web.server import create_app


def test_fleet_dir_replay_attaches_timeline_rows(tmp_path):
    d = tmp_path / "runs_sample/fleet_demo"
    d.mkdir(parents=True)
    heads = [{"t": 5.0 + k, "state": "SCRUB", "robots": [{"id": "unit1"}, {"id": "unit2"}], "argon": {"verdicts": {}}}
             for k in range(3)]
    rows = [{"type": "finding", "kind": "UNVERIFIED", "sensor": "G-F2", "t": 5.5},
            {"type": "command", "kind": "hold", "robot": "unit1", "t": 6.0}]
    (d / "headers.jsonl").write_text("".join(json.dumps(h) + "\n" for h in heads))
    (d / "timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with TestClient(create_app(sim_port=0, root=tmp_path)) as c:
        for path in ("runs_sample/fleet_demo", "runs_sample/fleet_demo/headers.jsonl"):
            s = c.get("/api/replay", params={"path": path}).json()
            assert [x["pt"] for x in s["snapshots"]] == [0.0, 1.0, 2.0] and s["duration"] == 2.0
            assert "argon_rows" not in s["snapshots"][0]
            assert [r["kind"] for r in s["snapshots"][1]["argon_rows"]] == ["UNVERIFIED", "hold"]
        assert c.get("/api/replay", params={"path": "runs_sample"}).status_code == 404


def test_committed_live_run_replays():
    with TestClient(create_app(sim_port=0)) as c:
        for path in ("runs_sample/fleet_demo", "runs_sample/fleet_demo/live_hud_1hz.jsonl"):
            s = c.get("/api/replay", params={"path": path}).json()
            rows = [r for x in s["snapshots"] for r in x.get("argon_rows", [])]
            assert [r["kind"] for r in rows if r["type"] == "command"] == ["log_sensor_fault", "hold", "go_check", "hold_for_crew", "fix_leak", "resume_rust"]
            assert [r["kind"] for r in rows if r["type"] == "vetoed"] == ["resume_rust"]
            assert {r["id"] for r in s["snapshots"][0]["robots"]} == {"unit1", "unit2"}


def test_live_run_folder_is_full_rate_with_disturbances_and_gzip():
    with TestClient(create_app(sim_port=0)) as c:
        r = c.get("/api/replay", params={"path": "runs_sample/fleet_demo"}, headers={"Accept-Encoding": "gzip"})
        assert r.headers["content-encoding"] == "gzip"
        s = r.json()
        assert len(s["snapshots"]) > 400
        rows = [x for h in s["snapshots"] for x in h.get("argon_rows", [])]
        assert [x["type"] for x in rows if x["type"] == "object_report"] == ["object_report"]
        assert {x.get("name") for x in rows if x["type"] == "scenario"} >= {"object_dropped", "object_cleared", "push"}
        assert any(x["type"] == "nav_event" and x["event"] == "recovered" for x in rows)

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


def test_committed_fleet_sample_replays():
    with TestClient(create_app(sim_port=0)) as c:
        s = c.get("/api/replay", params={"path": "runs_sample/fleet_demo_hud"}).json()
        kinds = [r["kind"] for x in s["snapshots"] for r in x.get("argon_rows", []) if r["type"] == "command"]
        assert kinds == ["hold", "go_check", "fix_leak", "resume_rust"]
        assert {r["id"] for r in s["snapshots"][0]["robots"]} == {"unit1", "unit2"}

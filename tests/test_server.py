import asyncio
import socket
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rig.frames import encode
from rig.web.server import FrameHub, create_app

VIEWER_JS = Path(__file__).resolve().parents[1] / "rig/web/static/viewer.js"


def frame(seq: int) -> bytes:
    return encode({"type": "frame", "w": 640, "h": 360, "seq": seq, "t": seq / 15}, b"\xff\xd8" + bytes([seq]))


def connect(app) -> socket.socket:
    return socket.create_connection(("127.0.0.1", app.state.sim_port), timeout=2)


def wait_frames(client: TestClient, n: int, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        health = client.get("/api/health").json()
        if health["frames"] >= n or time.monotonic() > deadline:
            return health
        time.sleep(0.01)


@pytest.fixture
def client():
    with TestClient(create_app(sim_port=0)) as c:
        yield c


def test_health_before_frames(client):
    health = client.get("/api/health").json()
    assert health["sim_connected"] is False
    assert health["frames"] == 0
    assert health["frame_age_s"] is None
    assert client.get("/api/frame.jpg").status_code == 503


def test_frames_reach_health_and_jpeg(client):
    with connect(client.app) as s:
        for seq in range(3):
            s.sendall(frame(seq))
        health = wait_frames(client, 3)
        assert health["sim_connected"] is True
        assert health["frames"] == 3
        assert (health["w"], health["h"]) == (640, 360)
        r = client.get("/api/frame.jpg")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == b"\xff\xd8\x02"


def test_ws_receives_jpeg(client):
    with client.websocket_connect("/ws/video") as ws, connect(client.app) as s:
        s.sendall(frame(0))
        assert ws.receive_bytes().startswith(b"\xff\xd8")


def test_slow_client_gets_latest_only():
    async def scenario():
        hub = FrameHub()
        seen = 0
        for seq in range(50):
            hub.publish(b"\xff\xd8" + bytes([seq]), {"seq": seq, "w": 640, "h": 360})
        newest, jpeg, header = await asyncio.wait_for(hub.wait_newer(seen), 1)
        assert jpeg[-1] == 49 and header["seq"] == 49
        assert hub.frames == 50
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(hub.wait_newer(newest), 0.05)

    asyncio.run(scenario())


def test_new_sim_replaces_old(client):
    a = connect(client.app)
    a.sendall(frame(0))
    wait_frames(client, 1)
    with a, connect(client.app) as b:
        b.sendall(frame(1))
        wait_frames(client, 2)
        a.settimeout(2)
        assert a.recv(1) == b""
        assert client.get("/api/health").json()["sim_connected"] is True


def test_index_serves_viewer(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "viewer.js" in r.text


def test_viewer_uses_page_scheme():
    js = VIEWER_JS.read_text()
    assert 'location.protocol === "https:"' in js
    assert '"wss:"' in js


def frame_with_mean(seq: int, mean: float) -> bytes:
    return encode({"type": "frame", "w": 640, "h": 360, "seq": seq, "t": 0.0, "mean": mean}, b"\xff\xd8" + bytes([seq]))


def test_state_no_sim_before_frames(client):
    assert client.get("/api/health").json()["state"] == "no_sim"


def test_state_warming_on_black_frames(client):
    with connect(client.app) as s:
        s.sendall(frame_with_mean(0, 0.4))
        assert wait_frames(client, 1)["state"] == "warming"


def test_state_live_on_rendered_frames(client):
    with connect(client.app) as s:
        s.sendall(frame_with_mean(0, 120.0))
        assert wait_frames(client, 1)["state"] == "live"


def test_state_stalled_when_frames_stop(client):
    with connect(client.app) as s:
        s.sendall(frame_with_mean(0, 120.0))
        wait_frames(client, 1)
        client.app.state.hub.last_frame_at -= 10
        assert client.get("/api/health").json()["state"] == "stalled"


def test_state_no_sim_after_disconnect(client):
    s = connect(client.app)
    s.sendall(frame_with_mean(0, 120.0))
    wait_frames(client, 1)
    s.close()
    deadline = time.monotonic() + 2
    while client.get("/api/health").json()["sim_connected"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.get("/api/health").json()["state"] == "no_sim"


def test_viewer_shows_server_state():
    js = VIEWER_JS.read_text()
    assert "h.state" in js
    assert 'setStatus("live", true)' not in js


def state_frame(seq: int, events: list) -> bytes:
    header = {"type": "frame", "w": 640, "h": 360, "seq": seq, "t": seq / 10, "state": "SCRUB",
              "force_n": 12.5, "events": events}
    return encode(header, b"\xff\xd8" + bytes([seq]))


def test_ws_state_sends_header_json(client):
    with client.websocket_connect("/ws/state") as ws, connect(client.app) as s:
        s.sendall(state_frame(0, [{"t": 0, "from": "PATROL", "to": "DETECT", "reason": "x", "evidence": {}}]))
        msg = ws.receive_json()
        assert msg["state"] == "SCRUB"
        assert msg["force_n"] == 12.5
        assert [e["to"] for e in msg["events"]] == ["DETECT"]


def test_ws_state_merges_events_and_replays_recent(client):
    with connect(client.app) as s:
        for k in range(3):
            s.sendall(state_frame(k, [{"t": k, "from": "A", "to": f"S{k}", "reason": "", "evidence": {}}]))
        wait_frames(client, 3)
        with client.websocket_connect("/ws/state") as ws:
            msg = ws.receive_json()
            assert [e["to"] for e in msg["events"]] == ["S0", "S1", "S2"]
            s.sendall(state_frame(3, []))
            assert ws.receive_json()["events"] == []


def test_ws_video_still_binary_with_rich_header(client):
    with client.websocket_connect("/ws/video") as ws, connect(client.app) as s:
        s.sendall(state_frame(0, []))
        assert ws.receive_bytes() == b"\xff\xd8\x00"


def read_event(sock: socket.socket) -> dict:
    from rig.frames import read_message
    sock.settimeout(2)
    header, _ = read_message(sock)
    return header


def test_post_event_forwards_to_sim(client):
    with connect(client.app) as s:
        s.sendall(frame(0))
        wait_frames(client, 1)
        r = client.post("/api/event", json={"name": "spawn_patch", "args": {"theta": 1.2, "z": 0.1}})
        assert r.status_code == 200
        assert read_event(s) == {"type": "event", "name": "spawn_patch", "args": {"theta": 1.2, "z": 0.1}}


def test_post_event_rejects_unknown(client):
    assert client.post("/api/event", json={"name": "shutdown"}).status_code == 400


def test_post_event_accepts_control_events(client):
    assert client.post("/api/event", json={"name": "cut_link", "args": {"on": True}}).status_code == 200
    assert client.post("/api/event", json={"name": "intent", "args": {"text": "inspect deck 2"}}).status_code == 200


def test_event_queued_until_sim_connects(client):
    r = client.post("/api/event", json={"name": "force_spike"})
    assert r.json()["queued"] == 1
    with connect(client.app) as s:
        s.sendall(frame(0))
        assert read_event(s)["name"] == "force_spike"
    assert client.get("/api/health").json()["events_pending"] == 0


def test_mock_session_generated_and_served(tmp_path):
    with TestClient(create_app(sim_port=0, mock_path=tmp_path / "mock.json")) as c:
        r = c.get("/api/mock")
        assert r.status_code == 200
        assert r.json()["snapshots"][-1]["state"] == "DONE"
        assert (tmp_path / "mock.json").exists()


def test_mock_shots_served(client):
    for n in (1, 2, 3):
        r = client.get(f"/api/mock/shot/{n}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/mock/shot/9").status_code == 404


def test_viewer_skips_stream_in_mock_mode():
    assert '"mock"' in VIEWER_JS.read_text()


def test_post_scenario_event_forwards(client):
    with connect(client.app) as s:
        s.sendall(frame(0))
        wait_frames(client, 1)
        r = client.post("/api/event", json={"name": "set_weather", "args": {"level": "storm", "gust": 30}})
        assert r.status_code == 200
        assert read_event(s) == {"type": "event", "name": "set_weather", "args": {"level": "storm", "gust": 30}}
    assert client.post("/api/event", json={"name": "set_weather", "args": {"level": "hail"}}).status_code == 400


def test_fleet_serves_sample_then_live(tmp_path):
    import json as js
    d = tmp_path / "fleet"
    d.mkdir()
    (d / "fleet_state.json").write_text(js.dumps({"schema": "fleet_state.v1", "t": 1.0, "robots": []}))
    (d / "fleet_map.json").write_text(js.dumps({"schema": "fleet_map.v1", "decks": []}))
    with TestClient(create_app(sim_port=0, fleet_dirs=(tmp_path / "missing", d))) as c:
        assert c.get("/api/fleet").json()["t"] == 1.0
        assert c.get("/api/fleet/map").json()["schema"] == "fleet_map.v1"
        assert c.post("/api/fleet", json={"schema": "fleet_state.v1", "t": 5.0, "robots": []}).status_code == 200
        assert c.get("/api/fleet").json()["t"] == 5.0
        assert c.post("/api/fleet", json={"schema": "other"}).status_code == 400
        with connect(c.app) as s:
            hdr = {"type": "frame", "w": 4, "h": 4, "seq": 0, "t": 0, "fleet": {"schema": "fleet_state.v1", "t": 9.0}}
            s.sendall(encode(hdr, b"\xff\xd8\x00"))
            wait_frames(c, 1)
            assert c.get("/api/fleet").json()["t"] == 9.0


def test_fleet_missing_is_404(tmp_path):
    with TestClient(create_app(sim_port=0, fleet_dirs=(tmp_path,))) as c:
        assert c.get("/api/fleet").status_code == 404
        assert c.get("/api/fleet/map").status_code == 404


def test_replay_converts_timeline_under_runs_only(tmp_path):
    import json as js
    (tmp_path / "runs/fleet").mkdir(parents=True)
    line = {"t": 0.0, "robot": "Q1", "kind": "Q", "position": [1, 2, 42.5], "yaw": 0, "state": "IDLE", "job": None}
    (tmp_path / "runs/fleet/timeline.jsonl").write_text(js.dumps(line) + "\n")
    (tmp_path / "secret.jsonl").write_text(js.dumps(line) + "\n")
    with TestClient(create_app(sim_port=0, root=tmp_path)) as c:
        r = c.get("/api/replay", params={"path": "runs/fleet/timeline.jsonl"})
        assert r.status_code == 200 and r.json()["snapshots"][0]["robots"][0]["name"] == "Unit 1 · rust"
        assert c.get("/api/replay", params={"path": "secret.jsonl"}).status_code == 403
        assert c.get("/api/replay", params={"path": "runs/../secret.jsonl"}).status_code == 403
        assert c.get("/api/replay", params={"path": "runs/none.jsonl"}).status_code == 404


def test_bench_lists_json(tmp_path):
    import json as js
    (tmp_path / "runs_sample/bench").mkdir(parents=True)
    (tmp_path / "runs_sample/bench/scrub.json").write_text(js.dumps({"title": "Scrub", "rows": []}))
    (tmp_path / "runs_sample/bench/bad.json").write_text("{")
    with TestClient(create_app(sim_port=0, root=tmp_path)) as c:
        assert c.get("/api/bench").json() == [{"name": "scrub", "title": "Scrub", "rows": []}]


def test_default_fleet_sample_exists(client):
    assert client.get("/api/fleet").json()["schema"] == "fleet_state.v1"
    assert client.get("/api/fleet/map").json()["schema"] == "fleet_map.v1"


class FakeControl:
    def __init__(self):
        self.snaps, self.cut, self.intents = [], [], []

    def tick(self, snap):
        self.snaps.append(snap)
        return {"mode": "CLOUD", "latency_ms": 900.0, "robots": len(snap.robots)}

    def cut_link(self, on):
        self.cut.append(on)

    def intent(self, text, now=0.0):
        self.intents.append(text)


class StubFleet:
    map = {"schema": "fleet_map.v1"}

    def state(self):
        return {"schema": "fleet_state.v1", "t": 3.0, "tasks": [],
                "robots": [{"id": "R1", "state": "TRANSIT", "pose": {"x": 1.0, "y": 2.0, "z": 42.5},
                            "battery": {"soc": 0.9}}]}

    def advance(self, dt):
        pass

    def for_sim(self):
        return {}

    def event(self, name, args):
        return {}


def test_live_control_centre_ticks_on_fleet_state_and_takes_link_events():
    cc = FakeControl()
    with TestClient(create_app(sim_port=0, fleet=StubFleet(), control=cc)) as c:
        deadline = time.monotonic() + 3.0
        while not cc.snaps and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cc.snaps and cc.snaps[-1].robots[0].id == "R1"
        assert c.post("/api/event", json={"name": "cut_link", "args": {"on": True}}).status_code == 200
        assert cc.cut == [True]
        with c.websocket_connect("/ws/state") as ws:
            sock = connect(c.app)
            sock.sendall(encode({"type": "state", "seq": 0, "t": 1.0, "scene": {}}))
            msg = ws.receive_json()
            sock.close()
        assert msg["control"]["mode"] == "CLOUD"

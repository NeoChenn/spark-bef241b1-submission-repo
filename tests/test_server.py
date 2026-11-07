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

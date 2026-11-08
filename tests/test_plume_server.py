import base64
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rig.frames import encode
from rig.plume import Field
from rig.web.server import create_app
from tests.test_server import connect, wait_frames


@pytest.fixture
def client():
    with TestClient(create_app(sim_port=0)) as c:
        yield c


def field() -> Field:
    c = np.zeros((10, 8, 6), np.float32)
    c[4:6, 3:5, 2:4] = 0.02
    return Field(c, (10.0, -5.0, 42.0), 0.5, meta={"flange": "F5"})


def plume_frame(seq: int, jpg: bytes | None = None, with_field: bool = False) -> bytes:
    p = {"stats": {"t": seq * 0.1, "vol_10lel_m3": 1.5, "max_lel_pct": 40.0}, "flow": {"kg_s": 0.0159}, "wind": {"speed_m_s": 5}}
    if jpg:
        p["jpg"] = base64.b64encode(jpg).decode()
    if with_field:
        p["field"] = base64.b64encode(field().to_bytes()).decode()
    return encode({"type": "frame", "w": 64, "h": 36, "seq": seq, "t": seq / 10, "plume": p}, b"\xff\xd8" + bytes([seq]))


def test_plume_endpoints_before_any_plume(client):
    assert client.get("/api/plume").status_code == 404
    assert client.get("/api/plume.jpg").status_code == 503
    assert client.get("/api/plume/sample", params={"x": 0, "y": 0, "z": 0}).status_code == 503


def test_plume_image_stats_and_field_served(client):
    with connect(client.app) as s:
        s.sendall(plume_frame(0, b"\xff\xd8pip", True))
        s.sendall(plume_frame(1))
        wait_frames(client, 2)
        p = client.get("/api/plume").json()
        assert p["seq"] == 1 and p["image"] and p["field"]
        assert p["stats"]["t"] == pytest.approx(0.1)
        assert "jpg" not in p
        assert client.get("/api/plume.jpg").content == b"\xff\xd8pip"
        r = client.get("/api/plume/sample", params={"x": 12.5, "y": -3.25, "z": 43.25}).json()
        assert r["ppm"] == pytest.approx(20000, rel=1e-3) and r["lel_pct"] == pytest.approx(40, rel=1e-3)
        many = client.post("/api/plume/sample", json={"points": [[12.5, -3.25, 43.25], [0, 0, 0]]}).json()
        assert many["ppm"][1] == 0.0
        assert client.post("/api/plume/sample", json={"points": [[1, 2]]}).status_code == 400
        got = Field.from_bytes(client.get("/api/plume/field.npz").content)
        assert got.meta == {"flange": "F5"} and got.shape == (10, 8, 6)


def test_ws_state_carries_light_plume_state(client):
    with client.websocket_connect("/ws/state") as ws, connect(client.app) as s:
        s.sendall(plume_frame(0, b"\xff\xd8a"))
        msg = ws.receive_json()
        assert msg["plume"]["seq"] == 1 and msg["plume"]["flow"]["kg_s"] == 0.0159
        assert "jpg" not in msg["plume"]
        s.sendall(encode({"type": "frame", "w": 64, "h": 36, "seq": 1, "t": 0.1}, b"\xff\xd8\x01"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            msg = ws.receive_json()
            if msg.get("seq") == 1:
                break
        assert msg["plume"]["seq"] == 1

import socket
import time

import pytest
from fastapi.testclient import TestClient

from rig.frames import encode, read_message
from rig.web.server import create_app

TOKEN = "s3cret-token"


def state(seq, attrs=(), **kw):
    return encode({"type": "state", "seq": seq, "t": seq / 30, "ts": time.time(), "scene": {"cinematic": "night"},
                   "view": {"m": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1], "focal": 24, "hap": 20.955, "vap": 15.29,
                            "clip": [0.1, 1000]}, "attrs": [list(a) for a in attrs], "specs": [], **kw})


def render_frame(seq, ts):
    return encode({"type": "frame", "w": 1920, "h": 1080, "seq": seq, "t": seq / 30, "mean": 50.0, "state_ts": ts,
                   "camera": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T": [[1, 0, 0, 0]] * 4, "w": 1920, "h": 1080}},
                  b"\xff\xd8remote" + bytes([seq]))


def hello(token=TOKEN, role="render"):
    return encode({"type": "hello", "role": role, "token": token})


def wait_until(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


@pytest.fixture
def client():
    with TestClient(create_app(sim_port=0, lan_host="127.0.0.1", lan_port=0, token=TOKEN)) as c:
        yield c


def lan(c):
    return socket.create_connection(("127.0.0.1", c.app.state.lan_port), timeout=2)


def local(c):
    return socket.create_connection(("127.0.0.1", c.app.state.sim_port), timeout=2)


def closed(sock) -> bool:
    try:
        return sock.recv(1) == b""
    except (ConnectionResetError, socket.timeout):
        return True


def test_lan_rejects_missing_or_wrong_token(client):
    with lan(client) as s:
        s.sendall(render_frame(0, time.time()))
        assert closed(s)
    with lan(client) as s:
        s.sendall(hello("wrong"))
        assert closed(s)
    assert client.get("/api/health").json()["frames"] == 0


def test_lan_disabled_without_token():
    with TestClient(create_app(sim_port=0, lan_host="127.0.0.1", lan_port=0, token=None)) as c:
        assert not hasattr(c.app.state, "lan_port")


def test_local_sim_still_needs_no_hello(client):
    with local(client) as s:
        s.sendall(encode({"type": "frame", "w": 2, "h": 1, "seq": 0, "t": 0.0}, b"\xff\xd8"))
        assert wait_until(lambda: client.get("/api/health").json()["frames"] == 1)


def test_render_gets_keyframe_then_deltas_and_frames_reach_browser(client):
    with local(client) as sim:
        sim.sendall(state(0, [("/r", "xformOp:translate", "double3", [0, 0, 0]), ("/q", "x", "float", 1.0)]))
        sim.sendall(state(1, [("/r", "xformOp:translate", "double3", [1, 0, 0])]))
        assert wait_until(lambda: client.get("/api/health").json()["sim_connected"])
        time.sleep(0.1)
        with lan(client) as r:
            r.sendall(hello())
            key, _ = read_message(r)
            assert key["type"] == "state" and key["key"] is True and key["seq"] == 1
            assert sorted(key["attrs"]) == [["/q", "x", "float", 1.0], ["/r", "xformOp:translate", "double3", [1, 0, 0]]]
            assert key["view"]["focal"] == 24 and key["scene"] == {"cinematic": "night"}
            sim.sendall(state(2, [("/q", "x", "float", 2.0)], events=[{"kind": "note"}]))
            delta, _ = read_message(r)
            assert delta["key"] is False and delta["attrs"] == [["/q", "x", "float", 2.0]]
            assert "events" not in delta
            with client.websocket_connect("/ws/video") as ws:
                r.sendall(render_frame(7, time.time() - 0.05))
                assert ws.receive_bytes() == b"\xff\xd8remote\x07"
            health = client.get("/api/health").json()
            assert health["render_connected"] is True and health["source"] == "render"
            assert (health["w"], health["h"]) == (1920, 1080)
            assert 40 <= health["latency_ms"] < 2000
            assert client.get("/api/frame.jpg").content == b"\xff\xd8remote\x07"
        assert wait_until(lambda: client.get("/api/health").json()["render_connected"] is False)
        assert client.get("/api/frame.jpg").content == b"\xff\xd8remote\x07"


def test_ws_state_merges_sim_meta_with_render_camera(client):
    with local(client) as sim, lan(client) as r:
        r.sendall(hello())
        sim.sendall(state(0, fleet={"robots": [{"id": "a"}]}, events=[{"kind": "x"}]))
        read_message(r)
        r.sendall(render_frame(1, time.time()))
        assert wait_until(lambda: client.get("/api/health").json()["frames"] == 1)
        with client.websocket_connect("/ws/state") as ws:
            msg = ws.receive_json()
            assert msg["fleet"] == {"robots": [{"id": "a"}]}
            assert msg["camera"]["w"] == 1920
            assert {"kind": "x"} in msg["events"]
            assert "attrs" not in msg and "view" not in msg


def test_ws_state_keeps_flowing_without_render(client):
    with local(client) as sim, client.websocket_connect("/ws/state") as ws:
        sim.sendall(state(0, fleet={"robots": []}))
        assert ws.receive_json()["fleet"] == {"robots": []}
        sim.sendall(state(1, fleet={"robots": [{"id": "b"}]}))
        assert ws.receive_json()["fleet"] == {"robots": [{"id": "b"}]}


def test_plume_meta_in_state_reaches_hud(client):
    with local(client) as sim, client.websocket_connect("/ws/state") as ws:
        sim.sendall(state(0, plume={"flange": "F2", "stats": {"vol_10lel_m3": 1.5}}))
        msg = ws.receive_json()
        assert msg["plume"]["flange"] == "F2"
        assert "settings" not in msg and "ocean" not in msg


def test_local_sim_frames_win_over_render_frames(client):
    with local(client) as sim, lan(client) as r:
        r.sendall(hello())
        sim.sendall(encode({"type": "frame", "w": 1280, "h": 720, "seq": 0, "t": 0.0}, b"\xff\xd8local"))
        assert wait_until(lambda: client.get("/api/health").json()["frames"] == 1)
        r.sendall(render_frame(1, time.time()))
        sim.sendall(encode({"type": "frame", "w": 1280, "h": 720, "seq": 1, "t": 0.1}, b"\xff\xd8local2"))
        assert wait_until(lambda: client.get("/api/health").json()["frames"] == 2)
        time.sleep(0.1)
        assert client.get("/api/frame.jpg").content == b"\xff\xd8local2"
        assert client.get("/api/health").json()["source"] == "sim"
        assert client.get("/api/health").json()["latency_ms"] is None


def test_render_does_not_steal_event_channel(client):
    with local(client) as sim, lan(client) as r:
        r.sendall(hello())
        sim.sendall(state(0))
        assert wait_until(lambda: client.get("/api/health").json()["render_connected"])
        assert wait_until(lambda: client.get("/api/health").json()["sim_connected"])
        client.post("/api/event", json={"name": "set_glare", "args": {"on": True}})
        sim.settimeout(2)
        header, _ = read_message(sim)
        assert header["type"] == "event" and header["name"] == "set_glare"

import time

import pytest
from fastapi.testclient import TestClient

from policy.fleet.live import QUICK, FleetLive, quick_run
from rig.hud import FLEET, apply_event
from rig.web.server import create_app
from tests.test_fleet_sim import stub_jobs
from tests.test_server import connect, frame, read_event, wait_frames


@pytest.fixture
def live():
    return FleetLive("judges", speed=20.0, jobs=stub_jobs(15.0))


def test_live_advances_and_accepts_hazards(live):
    live.advance(0.5)
    assert live.state()["t"] == pytest.approx(10.0)
    out = live.event("drop_obstacle", {"kind": "crate", "at": "R1"})
    assert out["log"][0]["kind"] == "drop_obstacle"
    live.advance(0.1)
    s = live.state()
    assert s["hazards"]["obstacles"][0]["kind"] == "crate"
    assert any(e["kind"] == "reroute" for e in live.f.event_log)
    sim = live.for_sim()
    assert set(sim["robots"][0]) == {"id", "pose", "path", "prev_path", "reroute_t", "state"}
    with pytest.raises(ValueError):
        live.event("drop_obstacle", {"kind": "piano", "at": "R1"})


def test_live_starts_new_round_when_done(live):
    n = len(live.f.tasks)
    for _ in range(200):
        live.advance(1.0)
        if live.rounds > 1:
            break
    assert live.rounds > 1 and sum(t.status != "done" for t in live.f.tasks.values()) >= n


def test_quick_jobs_fallback_without_orchestrator(monkeypatch):
    import policy.fleet.live as lv

    def no_torch(kind, scenario):
        raise ImportError("torch")
    monkeypatch.setattr(lv, "run_orchestrator", no_torch)
    for kind in QUICK:
        tr = quick_run(kind, "clean")
        assert tr.duration_s > 0 and tr.outcome in ("DONE", "ESCALATE_TO_HUMAN") and tr.state_at(1e9)[0] == tr.outcome


class Fleet:
    def __init__(self):
        self.calls = []

    def event(self, name, args):
        self.calls.append((name, args))


class WeatherBridge:
    def __init__(self):
        self.calls = []

    def set_weather(self, level, wind_dir=None, gust=None):
        self.calls.append((level, wind_dir, gust))

    def person_at(self, distance=None):
        self.calls.append(("person", distance))


def test_apply_event_routes_fleet_events():
    fl, b = Fleet(), WeatherBridge()
    apply_event(b, {"name": "drop_obstacle", "args": {"kind": "crate", "at": "R1"}}, fl)
    apply_event(b, {"name": "set_weather", "args": {"level": "storm", "wind_dir": 90}}, fl)
    apply_event(b, {"name": "person_at", "args": {"xy": [0, 4]}}, fl)
    apply_event(b, {"name": "person_at", "args": {"distance": 2.4}}, fl)
    assert [c[0] for c in fl.calls] == ["drop_obstacle", "set_weather", "person_at"]
    assert b.calls == [("storm", 90, None), ("person", 2.4)]
    apply_event(object(), {"name": "close_stairs", "args": {"id": "stairs_west_upper"}})
    assert set(FLEET) >= {"drop_obstacle", "close_walkway", "close_stairs", "clear", "set_weather"}


@pytest.fixture
def client(live):
    with TestClient(create_app(sim_port=0, fleet=live)) as c:
        yield c


def test_server_fleet_endpoints_and_events(client):
    m = client.get("/api/fleet/map").json()
    assert m["schema"] == "fleet_map.v1" and all("id" in w for w in m["walkways"])
    deadline = time.monotonic() + 3
    while client.get("/api/fleet").json()["t"] < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    r = client.post("/api/event", json={"name": "drop_obstacle", "args": {"kind": "crate", "at": "R1"}})
    assert r.status_code == 200 and r.json()["fleet"]["log"][0]["kind"] == "drop_obstacle"
    s = client.get("/api/fleet").json()
    assert s["hazards"]["obstacles"] and "prev_path" in s["robots"][0]
    assert client.post("/api/event", json={"name": "close_walkway", "args": {"edge": "nope"}}).status_code == 400
    assert client.post("/api/event", json={"name": "set_weather", "args": {"level": "storm"}}).status_code == 200
    with client.websocket_connect("/ws/fleet") as ws:
        snap = ws.receive_json()
        assert snap["schema"] == "fleet_state.v1" and snap["hazards"]["weather"]["level"] == "storm"


def test_server_pushes_fleet_state_to_sim(client):
    with connect(client.app) as s:
        s.sendall(frame(0))
        wait_frames(client, 1)
        names = []
        deadline = time.monotonic() + 3
        while "fleet" not in names and time.monotonic() < deadline:
            ev = read_event(s)
            names.append(ev["name"])
        assert "fleet" in names
        assert {"robots", "hazards", "t"} <= set(ev["args"])


def test_server_without_fleet_404s():
    with TestClient(create_app(sim_port=0)) as c:
        assert c.get("/api/fleet").status_code == 404
        assert c.post("/api/event", json={"name": "drop_obstacle", "args": {"at": "R1"}}).status_code == 200

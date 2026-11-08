from fastapi.testclient import TestClient

from policy.control.centre import ControlCentre
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.live import QUICK, FleetLive
from rig.mission.live import MissionLive
from rig.mission.obsviews import ObsViews
from rig.web.server import create_app


def quick(kind, scenario):
    segs = QUICK[kind]
    return JobTrace(kind, scenario, float(segs[-1][0]), segs[-1][1], segs)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def mission():
    m = MissionLive(FleetLive("judges", 4.0, jobs=JobLibrary(quick)),
                    control=lambda port: ControlCentre(None, log_path=None))
    m.obs = ObsViews(clock=Clock())
    return m


def test_every_robot_streams_a_lidar_bev_and_seq_moves_only_on_change():
    m = mission()
    m.advance(0.5)
    h = m.header()
    assert {"R1", "R2", "R3"} <= set(h["views"])
    v = h["views"]["R1"]["lidar"]
    assert v["w"] == 256 and v["h"] == 192 and v["seq"] == 1 and "lidar BEV" in v["caption"]
    assert "lidar" not in m.header()["views"].get("R1", {})
    m.obs.clock.t += 1.0
    assert "lidar" not in m.header()["views"].get("R1", {})
    m.advance(2.0)
    m.obs.clock.t += 1.0
    later = m.header()["views"]
    assert any(later.get(r, {}).get("lidar", {}).get("seq") == 2 for r in ("R1", "R2", "R3"))


def test_gas_seek_streams_a_belief_and_rust_job_streams_its_obs():
    m = mission()
    m.event("gas_leak", {"flange": "F5"})
    kinds = set()
    for _ in range(600):
        m.advance(0.5)
        m.obs.clock.t += 1.0
        for rid, ks in m.header()["views"].items():
            kinds |= {(k, ks[k]["caption"].split(" · ")[1]) for k in ks if k in ("belief", "obs")}
        if {"belief", "obs"} <= {k for k, _ in kinds}:
            break
    assert ("belief", "gas belief") in kinds and ("obs", "scrub obs") in kinds


def test_mission_views_reach_ws_state_as_url_stubs():
    m = mission()
    app = create_app(sim_port=0, mission=m)
    with TestClient(app) as c:
        with c.websocket_connect("/ws/state") as ws:
            for _ in range(40):
                msg = ws.receive_json()
                if (msg.get("views") or {}).get("R1", {}).get("lidar"):
                    break
        stub = msg["views"]["R1"]["lidar"]
        assert "jpg" not in stub and stub["url"].startswith("/api/views/R1/lidar.jpg")
        r = c.get("/api/views/R1/lidar.jpg")
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and len(r.content) <= 10 * 1024

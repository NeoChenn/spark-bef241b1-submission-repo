import math

import pytest
from fastapi.testclient import TestClient

from policy.control.centre import CUT_BANNER, ControlCentre
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.live import QUICK, FleetLive
from rig.captions import robot_entry
from rig.hud import mock_control, mock_gas, mock_orchestrator, mock_rig, mock_sensors
from rig.mission.live import MissionLive, robot_id
from rig.web.server import create_app


def quick(kind, scenario):
    segs = QUICK[kind]
    return JobTrace(kind, scenario, float(segs[-1][0]), segs[-1][1], segs)


def mission(control=None, speed=4.0):
    return MissionLive(FleetLive("judges", speed, jobs=JobLibrary(quick)),
                       control=control if control is not None else (lambda port: ControlCentre(None, log_path=None)))


def run(m, seconds, dt=0.25, until=None):
    for _ in range(int(seconds / dt)):
        m.advance(dt)
        if until is not None and until(m):
            return True
    return False


@pytest.fixture
def m():
    x = mission()
    run(x, 5)
    return x


def mocks():
    gas = mock_gas(0.12)
    robots = [robot_entry(r, r, "SCRUB", {}, None) for r in ("unit-1", "unit-2", "unit-3")]
    evs = {r["id"]: {} for r in robots}
    return {"gas": gas, "control": mock_control(gas), "sensors": mock_sensors(robots, evs, 1.0),
            "orchestrator": mock_orchestrator({"unit-1": "SCRUB", "unit-2": "DONE"}, gas),
            "fleet": mock_rig(robots, evs, 1.0, gas), "robot": robots[0]}


def test_header_carries_every_panel_in_the_mock_shapes(m):
    m.control_tick(0.0)
    h = m.header()
    ref = mocks()
    for k in ("control", "gas", "fleet", "robots", "caption", "sensors", "orchestrator"):
        assert k in h
    assert set(ref["gas"]) <= set(h["gas"])
    assert set(ref["gas"]["sensors"][0]) <= set(h["gas"]["sensors"][0])
    assert set(ref["control"]) - {"key"} <= set(h["control"])
    assert set(h["sensors"]) == {"R1", "R2", "R3"}
    assert all(set(v) == set(next(iter(ref["sensors"].values()))) for v in h["sensors"].values())
    assert set(ref["orchestrator"]) <= set(h["orchestrator"])
    assert set(ref["orchestrator"]["jobs"][0]) <= set(h["orchestrator"]["jobs"][0])
    assert set(ref["fleet"]) <= set(h["fleet"])
    assert set(ref["fleet"]["robots"][0]) <= set(h["fleet"]["robots"][0])
    assert set(ref["robot"]) <= set(h["robots"][0])
    assert {"lead", "rest", "text", "tone", "state"} <= set(h["caption"])
    assert h["caption"]["lead"].startswith("Unit ")
    assert all(r["name"].startswith("Unit ") for r in h["fleet"]["robots"])


def test_no_key_means_edge_mode_and_no_cloud(m):
    c = m.control_tick(0.0)
    assert c["mode"] == "EDGE" and c["banner"] and not c["online"]
    assert m.control.backend is None


def test_gas_leak_is_localised_by_cast_and_surge_then_fixed(m):
    assert m.header()["gas"]["alarm"] is False
    assert m.event("gas_leak", {"on": True}) == {"leak": "F2"}
    saw = {"seek": False, "approval": False, "caption": False}

    def check(x):
        h = x.header()
        g = h["gas"]
        saw["seek"] |= g.get("mode") in ("SURGE", "CAST", "DOWNWIND")
        saw["approval"] |= any(a["kind"] == "valve_isolate" for a in h["orchestrator"]["approvals"])
        saw["caption"] |= any(k in h["caption"]["state"] for k in ("LEAK", "RETORQUE", "GAS"))
        return x.leak.found is not None and not x.leak.on
    assert run(m, 400, until=check)
    assert all(saw.values()), saw
    assert m.leak.found["err_m"] < 2.0
    assert m.header()["gas"]["leak"] is False


def test_operator_approval_isolates_the_leak_and_gemini_cannot_approve(m):
    m.event("gas_leak", {"flange": "F5"})
    assert run(m, 300, until=lambda x: x.valve_job is not None)
    jid = m.valve_job
    assert m.port.approve_request(jid, "gemini", "self") is False
    assert m.event("approve", {"job": jid})["ok"] is True
    assert run(m, 30, until=lambda x: x.leak.isolated)
    assert m.port.job(jid)["status"] == "DONE"


def test_reject_cancels_the_request(m):
    jid = m.port.queue("valve_isolate", "F3", "SUPV", 1, by="gemini", why="x")
    assert m.header()["orchestrator"]["approvals"][0]["job"] == jid
    assert m.event("reject", {"job": jid})["ok"] is True
    assert m.header()["orchestrator"]["approvals"] == []


def test_auto_job_from_the_planner_reaches_the_fleet(m):
    jid = m.port.queue("gas_survey", "F3", "AUTO", 3, by="gemini", why="check F3")
    assert m.port.job(jid)["status"] == "ASSIGNED"
    assert any(t.kind == "leak_check" and t.poi == "F3" for t in m.fleet.f.tasks.values())


def test_battery_fault_sends_the_unit_to_dock(m):
    rid = m.event("battery_fault", {"robot": "unit-2"})["battery_fault"]
    assert rid == "R2"
    assert run(m, 20, until=lambda x: x._robot("R2").to_dock)


def test_person_at_pauses_the_nearest_unit_and_clears(m):
    out = m.event("person_at", {"robot": "R1", "distance": 1.0})
    assert out["person"]["robot"] == "R1"
    assert run(m, 30, until=lambda x: x._robot("R1").paused)
    m.event("person_at", {"on": False})
    assert run(m, 30, until=lambda x: not x._robot("R1").paused)


def test_cut_link_flips_to_edge_only(m):
    m.event("cut_link", {"on": True})
    c = m.control_tick(1.0)
    assert c["banner"] == CUT_BANNER and c["link"] == "CUT"
    m.event("cut_link", {"on": False})
    assert m.control_tick(2.0)["link"] == "UP"


def test_weather_turns_the_plume(m):
    m.event("set_weather", {"level": "storm", "wind_dir": 90.0})
    run(m, 2)
    assert math.isclose(m.leak.toward, 90.0)


def test_robot_id_mapping():
    assert robot_id("unit-2", ["R1", "R2"]) == "R2"
    assert robot_id("R1", ["R1"]) == "R1"
    assert robot_id(None, ["R1"]) is None


class FakeAgentBackend:
    name, live, available = "fake", True, True

    def __init__(self):
        self.calls = 0

    def call(self, history):
        self.calls += 1
        return {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}],
                "usage": {"total_input_tokens": 1000, "total_output_tokens": 100}}


class FakeBackend:
    name, live, available = "fake-argon", True, True

    def __init__(self):
        self.calls = 0

    def call(self, prompt, snap):
        self.calls += 1
        raise RuntimeError("no network in tests")


def cloud_mission(cap):
    from policy.control.agent import OrchestrationAgent
    agent_backend, backend = FakeAgentBackend(), FakeBackend()

    def make(port):
        return ControlCentre(backend, log_path=None, sync=True, cap_usd_day=cap,
                             agent=OrchestrationAgent(port, agent_backend, log_path=None))
    return mission(make), agent_backend, backend


def test_cost_cap_blocks_cloud_calls():
    m, agent_backend, backend = cloud_mission(0.0)
    run(m, 1)
    assert m.event("intent", {"text": "inspect deck 2"})["intent"] == "edge"
    m.control_tick(100.0)
    assert agent_backend.calls == 0 and backend.calls == 0
    assert m.ctl["status"] == "capped"


def test_intent_reaches_the_planner_and_spend_counts_against_the_cap():
    m, agent_backend, _ = cloud_mission(5.0)
    run(m, 1)
    assert m.event("intent", {"text": "inspect deck 2"})["intent"] == "planning"
    m.pool.shutdown(wait=True)
    assert agent_backend.calls == 1
    assert m.control.budget.spent(m.today()) > 0
    assert m.header()["orchestrator"]["planner"]["intent"] == "inspect deck 2"


def test_server_streams_mission_state_without_a_sim():
    m = mission()
    with TestClient(create_app(sim_port=0, mission=m)) as c:
        with c.websocket_connect("/ws/state") as ws:
            msg = {}
            for _ in range(20):
                msg = ws.receive_json()
                if "orchestrator" in msg and "control" in msg:
                    break
        assert {"gas", "fleet", "robots", "caption", "sensors", "orchestrator", "control"} <= set(msg)
        r = c.post("/api/event", json={"name": "gas_leak", "args": {"on": True}}).json()
        assert r["leak"] == "F2"
        assert c.post("/api/event", json={"name": "battery_fault", "args": {"robot": "R9"}}).status_code == 400
        assert c.get("/api/fleet").json()["robots"][0]["name"].startswith("Unit ")


def test_weld_job_runs_and_streams_its_policy_view():
    m = mission()
    out = m.event("weld_job", {"poi": "weld_test", "xyz": [10.0, -8.0, 42.5], "scenario": "crack"})
    assert "fleet" in out
    seen = {}

    def check(x):
        h = x.header()
        for rid, v in h["views"].items():
            seen[rid] = v["policy"]
        return bool(seen) and any(r["state"] in ("GRIND", "WELD", "INSPECT_WELD") for r in h["robots"])
    assert run(m, 400, until=check)
    v = next(iter(seen.values()))
    assert {"jpg", "w", "h", "caption", "seq"} <= set(v)
    assert any(j["kind"] == "weld_repair" for j in m.header()["orchestrator"]["jobs"])

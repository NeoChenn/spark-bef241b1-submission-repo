import pytest

from policy.control.centre import ControlCentre
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.live import QUICK, FleetLive
from rig.hud import check_event
from rig.mission.live import MissionLive


def quick(kind, scenario):
    segs = QUICK[kind]
    return JobTrace(kind, scenario, float(segs[-1][0]), segs[-1][1], segs)


def mission():
    m = MissionLive(FleetLive("judges", 4.0, jobs=JobLibrary(quick)),
                    control=lambda port: ControlCentre(None, log_path=None), obs=False)
    for _ in range(8):
        m.advance(0.25)
    return m


def drive(m, limit_s=2400.0, approve=True, on_tick=None):
    steps, n = [], 0
    while m.t < limit_s:
        m.advance(0.25)
        h = m.header()
        s = h["scenario"]
        if not steps or steps[-1] != s["step_i"]:
            steps.append(s["step_i"])
        if on_tick is not None:
            on_tick(m, h)
        if approve:
            for a in h["orchestrator"]["approvals"]:
                assert m.event("approve", {"job": a["job"]})["ok"]
        if s["done"]:
            if s["next"] is None:
                return steps, h
            m.event("disturb", {})
            n += 1
    raise AssertionError(f"scenario stuck at step {m.header()['scenario']}")


def kinds(h):
    return [a["kind"] for a in h["alerts"]]


def test_gas_scenario_runs_every_beat_and_handles_every_disturbance():
    m = mission()
    out = m.event("scenario", {"id": "gas"})
    assert out == {"scenario": "gas", "selected": "R2"}
    h = m.header()
    assert h["selected"] == "R2" and h["camera_mode"] == {"mode": "wide", "robot": None}
    assert m.take_sim() == [{"name": "set_weather", "args": {"level": "rain", "time": "dusk", "rain": 0.8, "fog": 0.6,
                                                              "wind_dir": 240.0}}]
    seen = {"seeker": set(), "valve_by": set(), "frozen": False}

    def look(m, h):
        if h["gas"].get("unit"):
            seen["seeker"].add(h["gas"]["unit"])
        for r in h["fleet"]["robots"]:
            if (r.get("job_state") or "").startswith("VALVE"):
                seen["valve_by"].add(r["id"])
        if h["scenario"]["step_i"] in (3, 4, 5, 6):
            seen["frozen"] |= not any(t["kind"] == "weld" and t["status"] == "active" for t in h["fleet"]["tasks"])
        assert "R2" in [r["id"] for r in h["robots"]]
    steps, h = drive(m, on_tick=look)
    assert set(range(8)) <= set(steps)
    assert seen["seeker"] == {"R2"} and seen["valve_by"] == {"R1"} and seen["frozen"]
    assert [d["fired"] for d in h["scenario"]["disturbances"]] == [True, True, True]
    assert h["control"]["link"] == "CUT" if "control" in h else m.control.link_cut
    ks = kinds(h)
    assert ks.count("gas_alarm") >= 2 and "approval" in ks and ks.count("disturbance") == 3
    seqs = [a["seq"] for a in h["alerts"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert m.leak.on is False and not m.leak.queued


def test_gas_valve_waits_for_the_operator_and_gemini_cannot_approve():
    m = mission()
    m.event("scenario", {"id": "gas"})
    for _ in range(4000):
        m.advance(0.25)
        if m.header()["scenario"]["step_i"] == 4:
            break
    jid = m.valve_job
    assert m.port.job(jid)["status"] == "WAITING_APPROVAL"
    assert m.port.approve_request(jid, "gemini", "self") is False
    for _ in range(80):
        m.advance(0.25)
    assert m.header()["scenario"]["step_i"] == 4 and m.leak.on


def test_starting_a_scenario_keeps_the_judges_items_and_clears_its_own():
    m = mission()
    m.event("drop_obstacle", {"kind": "crate", "at": "R1"})
    m.event("drop_can", {})
    judge = set(m.fleet.f.hz.obstacles)
    assert len(judge) == 2
    m.event("scenario", {"id": "patrol"})
    assert judge <= set(m.fleet.f.hz.obstacles)
    assert any(t.kind == "pick_can" and t.status == "pending" for t in m.fleet.f.tasks.values())
    for _ in range(40):
        m.advance(0.25)
    mine = set(m.fleet.f.hz.obstacles) - judge
    assert mine
    m.event("scenario", {"id": "gas"})
    assert {k for k in judge if k.startswith("cr")} <= set(m.fleet.f.hz.obstacles)
    assert not mine & set(m.fleet.f.hz.obstacles)
    m.event("clear", {"id": "all"})
    assert not m.fleet.f.hz.obstacles


def test_reset_puts_robots_and_jobs_in_a_known_state():
    m = mission()
    m.event("gas_leak", {"on": True})
    m.event("battery_fault", {"robot": "R2"})
    for _ in range(100):
        m.advance(0.25)
    m.event("scenario", {"id": "gas"})
    f = m.fleet.f
    assert not m.leak.on and m.leak.fid is None and not m.port.extra
    assert {r.id: round(r.battery.soc, 2) for r in f.robots} == {"R1": 0.9, "R2": 0.9, "R3": 0.8}
    assert all(r.task is None and not r.legs for r in f.robots)
    assert {t.kind for t in f.tasks.values()} == {"weld", "rust"}
    assert f.hz.weather["time"] == "dusk" and f.hz.weather["rain"] > 0 and f.hz.weather["fog"] > 0


def test_header_shapes_and_select():
    m = mission()
    h = m.header()
    assert h["scenario"] is None and h["selected"] is None and h["alerts"] == []
    m.event("scenario", {"id": "repair"})
    m.advance(0.25)
    h = m.header()
    s = h["scenario"]
    assert {"id", "name", "step", "step_i", "steps", "key", "disturbances", "next", "main"} <= set(s)
    assert s["id"] == "repair" and s["main"] == "R1" and s["next"] == 0 and s["step"] == s["steps"][s["step_i"]]
    assert set(s["key"]) == {"label", "value", "unit"}
    assert all(set(d) == {"id", "name", "fired"} for d in s["disturbances"])
    assert all(isinstance(v["highlight"], list) for v in h["sensors"].values())
    assert m.event("select", {"robot": "unit-3"}) == {"selected": "R3"}
    assert m.header()["camera_mode"] == {"mode": "chase", "robot": "R3"}
    assert m.event("select", {"robot": None}) == {"selected": None}
    assert m.header()["camera_mode"]["mode"] == "wide"
    with pytest.raises(ValueError):
        m.event("select", {"robot": "R9"})
    m.event("scenario", {"id": "flyby"})
    h = m.header()
    assert h["scenario"] is None and h["selected"] is None and h["camera_mode"]["mode"] == "flyby"
    with pytest.raises(ValueError):
        check_event({"name": "scenario", "args": {"id": "nope"}})
    assert check_event({"name": "disturb", "args": {}})["name"] == "disturb"


def test_judges_presses_become_toasts():
    m = mission()
    m.event("scenario", {"id": "patrol"})
    m.event("drop_obstacle", {"kind": "puddle", "at": "R1"})
    m.event("close_stairs", {})
    toasts = [a for a in m.header()["alerts"] if a["kind"] == "toast"]
    assert [a["text"] for a in toasts] == ["Puddle on the walkway", "Stairs closed"]

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


def drive(m, limit_s=2400.0, approve=True, on_tick=None, when=None):
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
        if s["done"] and s["next"] is None:
            return steps, h
        if s["next"] is not None and (s["done"] if when is None else when(m, h, s["next"])):
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


def test_patrol_reroutes_picks_cans_reports_the_drum_and_docks():
    m = mission()
    assert m.event("scenario", {"id": "patrol"})["selected"] == "R3"

    def when(m, h, k):
        r3 = next(r for r in h["fleet"]["robots"] if r["id"] == "R3")
        return h["scenario"]["step_i"] >= 3 and r3["state"] == "TRANSIT" and r3["pose"]["x"] > 10.0
    steps, h = drive(m, when=when)
    assert {0, 1, 2, 3, 4, 5} <= set(steps)
    assert h["scenario"]["key"]["value"] >= 4
    f = m.fleet.f
    assert any(r["robot"] == "R3" and "crate" in r["cause"] for r in f.reroutes)
    drums = [o for o in f.hz.obstacles.values() if o["kind"] == "barrel"]
    assert drums and not drums[0].get("rolling") and drums[0].get("task") is None
    assert any(c["target"] == "w5" for c in f.hz.closures.values())
    assert not any(t.kind == "pick_can" and t.poi == drums[0]["id"] for t in f.tasks.values())
    assert any(e["kind"] == "pick_can" or "pick" in e["message"] for e in f.event_log)
    texts = " ".join(a["text"] for a in h["alerts"])
    assert "geometric grasp prior" in texts and "Gemini" not in texts


def test_repair_scrubs_waits_for_paint_approval_rescrubs_and_holds_force_in_a_gust():
    m = MissionLive(FleetLive("judges", 4.0, jobs=JobLibrary(quick)),
                    control=lambda port: ControlCentre(None, log_path=None))
    m.obs.every = 0.0
    m.event("scenario", {"id": "repair"})
    scrubs = {"n": 0, "prev": None, "views": []}

    def look(m, h, k=None):
        st = next(r["state"] for r in h["robots"] if r["id"] == "R1")
        if st == "SCRUB" and scrubs["prev"] != "SCRUB":
            scrubs["n"] += 1
        scrubs["prev"] = st
        v = (h["views"].get("R1") or {}).get("obs")
        if v and m.gusts:
            scrubs["views"].append(v["caption"])
        if st == "SCRUB":
            assert m.brush_force("R1") < 40.0
            assert "force_n" in h["sensors"]["R1"]["highlight"]

    def when(m, h, k):
        return scrubs["prev"] == "SCRUB" and scrubs["n"] == k + 1
    steps, h = drive(m, on_tick=look, when=when)
    assert set(range(8)) <= set(steps)
    assert scrubs["n"] >= 2 and m.gusts and scrubs["views"]
    t0, peak, dur = m.gusts[0]
    from rig.mission.obsviews import force_trace
    ts = (t0 - 2, t0 + dur + 2)
    assert max(force_trace("rust:hero_bay", *ts, gusts=m.gusts)) < 40.0
    assert max(force_trace("rust:hero_bay", *ts, gusts=m.gusts, raw=True)) > 40.0
    assert sum(1 for a in h["alerts"] if a["kind"] == "approval") >= 1

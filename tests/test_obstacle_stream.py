import json

from rig.sim.cli import parse_args
from rig.sim.main import handle_event, load_fleet_log
from scripts.obstacles_video import captions


class FakeScene:
    def __init__(self):
        self.calls = []

    def sync(self, state):
        self.calls.append(("sync", state))

    def set_weather(self, **kw):
        self.calls.append(("weather", kw))


def test_fleet_flags():
    a = parse_args(["--cinematic", "night", "--fleet-log", "x.jsonl", "--fleet-speed", "6"])
    assert a.fleet_log == "x.jsonl" and a.fleet_speed == 6.0 and a.path_radius == 0.06


def test_handle_event_routes_fleet_and_weather():
    s = FakeScene()
    handle_event(s, {"name": "fleet", "args": {"t": 1.0}})
    handle_event(s, {"name": "set_weather", "args": {"level": "storm", "wind_dir": 90.0, "junk": 1}})
    handle_event(s, {"name": "drop_obstacle", "args": {"kind": "crate"}})
    handle_event(None, {"name": "fleet", "args": {}})
    assert s.calls == [("sync", {"t": 1.0}), ("weather", {"level": "storm", "wind_dir": 90.0})]


def test_load_fleet_log_keeps_ticks(tmp_path):
    p = tmp_path / "log.jsonl"
    rows = [{"type": "map"}, {"type": "tick", "t": 1.0}, {"type": "tick", "t": 2.0}, {"type": "summary"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    times, ticks = load_fleet_log(str(p))
    assert times == [1.0, 2.0] and len(ticks) == 2


def test_captions_hold_recent_hazard_events():
    ticks = [{"t": 0.0, "events": [{"t": 0.0, "kind": "job_start", "message": "x"}]},
             {"t": 6.0, "events": [{"t": 6.0, "kind": "drop_obstacle", "message": "crate"},
                                   {"t": 6.0, "kind": "reroute", "message": "R1"}]},
             {"t": 40.0, "events": []}]
    c = captions(ticks)
    assert c[0] == [] and [e["kind"] for e in c[1]] == ["drop_obstacle", "reroute"] and c[2] == []


def test_handle_event_routes_select():
    class Sel(FakeScene):
        def select(self, rid):
            self.calls.append(("select", rid))

    s = Sel()
    handle_event(s, {"name": "select", "args": {"robot": "R1"}})
    handle_event(s, {"name": "select", "args": {"robot": None}})
    handle_event(FakeScene(), {"name": "select", "args": {"robot": "R1"}})
    assert s.calls == [("select", "R1"), ("select", None)]

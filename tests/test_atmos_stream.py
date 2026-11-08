from rig.sim.main import handle_event


class Rec:
    def __init__(self):
        self.calls = []

    def set_weather(self, **kw):
        self.calls.append(kw)


def test_set_weather_event_reaches_atmosphere_without_obstacles():
    atm = Rec()
    handle_event(None, {"name": "set_weather", "args": {"level": "storm", "wind_dir": 45.0, "gust": 30.0, "x": 1}}, atm)
    handle_event(None, {"name": "fleet", "args": {}}, atm)
    assert atm.calls == [{"level": "storm", "wind_dir": 45.0, "gust": 30.0}]

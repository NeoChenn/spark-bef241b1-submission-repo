from rig.render.cli import parse_args


def test_defaults_are_live_1080p(monkeypatch):
    monkeypatch.delenv("RENDER_RELAY", raising=False)
    a = parse_args([])
    assert (a.width, a.height) == (1920, 1080)
    assert a.relay == "192.168.1.190:9000" and a.offline == ""
    assert a.jpeg == "pil" and a.quality == 85


def test_relay_from_env_and_offline(monkeypatch):
    monkeypatch.setenv("RENDER_RELAY", "10.0.0.2:9100")
    a = parse_args(["--offline", "/work/log.jsonl", "--spp", "256", "--out", "/work/frames"])
    assert a.relay == "10.0.0.2:9100" and a.offline == "/work/log.jsonl" and a.spp == 256 and a.out == "/work/frames"

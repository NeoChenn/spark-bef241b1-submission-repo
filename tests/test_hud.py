import json

import numpy as np
import pytest

from policy.demo_mock import build
from rig.hud import HOOKS, PATCH_OFFSET, HudFeed, apply_event, playback_times, record_mock


@pytest.fixture
def orch(tmp_path):
    return build(str(tmp_path / "d.jsonl"), verbose=False)[0]


def test_header_fields_shapes(orch):
    feed = HudFeed(orch)
    for _ in range(4):
        orch.step()
    h = feed.header()
    assert h["state"] == orch.state.value
    assert h["hud"] == orch.overlay_text()
    grid = np.array(h["rust_grid"])
    assert grid.shape == (40, 64)
    assert grid.min() >= 0 and grid.max() <= 1 and grid.max() > 0.5
    assert len(h["sprayed"]) == 40 and set("".join(h["sprayed"])) <= {"0", "1"}
    assert len(h["painted"]) == 40
    assert isinstance(h["force_n"], float)
    assert 0 <= h["detector_conf"] <= 1
    assert h["gemini"] is None
    assert h["source"] == "EDGE"
    assert len(h["events"]) == 4
    assert {"t", "from", "to", "reason", "evidence"} <= set(h["events"][0])
    json.dumps(h)


def test_events_only_since_last_header(orch):
    feed = HudFeed(orch)
    orch.step()
    assert len(feed.header()["events"]) == 1
    assert feed.header()["events"] == []
    orch.step()
    assert [e["to"] for e in feed.header()["events"]] == [orch.state.value]


def test_brush_and_soak(orch):
    feed = HudFeed(orch)
    while orch.state.value != "SOAK":
        orch.step()
    h = feed.header()
    assert h["soak"]["total"] == 30.0 and 0 < h["soak"]["left"] <= 30
    i, j = h["brush"]
    assert 0 <= i < 40 and 0 <= j < 64


def test_inspection_thumb_only_on_change(orch):
    feed = HudFeed(orch)
    orch.step()
    orch.step()
    first = feed.header()
    assert first["inspection"]["jpg"]
    assert first["inspection"]["w"] > 0
    assert "inspection" not in feed.header()


class FakeBridge:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name in HOOKS:
            return lambda *a, **k: self.calls.append((name, a, k))
        raise AttributeError(name)


def test_apply_event_dispatches():
    b = FakeBridge()
    apply_event(b, {"name": "spawn_patch", "args": {"theta": 1.0, "z": 0.1}})
    apply_event(b, {"name": "set_glare", "args": {"on": True}})
    apply_event(b, {"name": "force_spike"})
    assert b.calls == [("spawn_patch", (), {"theta": 1.0, "z": 0.1}), ("set_glare", (), {"on": True}),
                       ("force_spike", (), {})]


def test_spawn_patch_defaults_next_to_brush():
    b = FakeBridge()
    b.ee_theta_z = lambda: (1.0, 0.05)
    apply_event(b, {"name": "spawn_patch", "args": {}})
    name, _, kw = b.calls[0]
    assert name == "spawn_patch"
    assert kw["theta"] == pytest.approx(1.0 + PATCH_OFFSET[0]) and kw["z"] == pytest.approx(0.05 + PATCH_OFFSET[1])


def test_apply_event_rejects_unknown():
    with pytest.raises(ValueError):
        apply_event(FakeBridge(), {"name": "rm_rf"})


def test_playback_compresses_soak():
    snaps = [{"t": 0.0, "state": "DETECT"}, {"t": 0.0, "state": "SOAK"}, {"t": 30.0, "state": "SOAK"},
             {"t": 30.0, "state": "SCRUB"}]
    pt = playback_times(snaps)
    assert pt == sorted(pt)
    assert pt[1] > pt[0]
    assert pt[2] - pt[1] < 10


def test_record_mock_session(tmp_path):
    out = tmp_path / "s.json"
    session = record_mock(out, str(tmp_path / "d.jsonl"))
    snaps = session["snapshots"]
    assert json.loads(out.read_text())["snapshots"][0]["state"] == snaps[0]["state"]
    assert snaps[-1]["state"] == "DONE"
    assert 40 <= snaps[-1]["pt"] <= 62
    assert [s["pt"] for s in snaps] == sorted(s["pt"] for s in snaps)
    assert any(s.get("gemini") for s in snaps)
    assert any(s.get("source") == "CLOUD" for s in snaps)
    assert sum(len(s["events"]) for s in snaps) >= 20
    assert "rust_grid" in snaps[0]
    assert any("BACK_OFF" == s["state"] for s in snaps)
    assert all(s["pt"] >= 0 for s in snaps)
    ctl = [s["control"] for s in snaps if s.get("control")]
    assert ctl and all(c["model"].startswith("gemini-4-argon") for c in ctl)
    kinds = {(a["source"], a["kind"]) for c in ctl for a in c["actions"]}
    assert {("edge", "back_off"), ("edge", "evacuate"), ("edge", "pause"), ("cloud", "dispatch")} <= kinds
    assert any(c["banner"] and "OFFLINE" in c["banner"] for c in ctl)
    assert any(c["vetoed"] for c in ctl)
    assert any(a["kind"] == "probable_leak" for c in ctl for a in c["anomalies"])
    assert snaps[-1]["control"]["key"] == "done"
    assert (tmp_path / "control_calls.jsonl").read_text().count("\n") >= 8
    agent = snaps[-1]["control"]["agent"]
    assert agent["intent"].startswith("Inspect deck 2") and agent["reserved"] == ["paint"]
    assert any(not c["ok"] and "own SUPV" in c["refused"] for c in agent["calls"])
    lines = [json.loads(x) for x in (tmp_path / "agent_calls.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and {"reprioritise", "queue_job"} <= {n for x in lines for n in x["decision"]}


def test_apply_event_routes_control_events():
    seen = []
    ctl = type("C", (), {"cut_link": lambda self, on: seen.append(("cut", on)),
                         "intent": lambda self, text, background=False: seen.append(("intent", text, background))})()
    apply_event(FakeBridge(), {"name": "cut_link", "args": {"on": True}}, ctl)
    apply_event(FakeBridge(), {"name": "intent", "args": {"text": "inspect deck 2"}}, ctl)
    assert seen == [("cut", True), ("intent", "inspect deck 2", True)]
    with pytest.raises(ValueError):
        apply_event(FakeBridge(), {"name": "cut_link", "args": {"on": True}})


def test_header_includes_control_when_attached(orch):
    from policy.control.backend import ReplayBackend
    from policy.control.centre import ControlCentre
    feed = HudFeed(orch, control=ControlCentre(ReplayBackend(), log_path=None))
    orch.step()
    h = feed.header()
    assert h["control"]["mode"] == "CLOUD" and h["control"]["situation"]
    json.dumps(h)

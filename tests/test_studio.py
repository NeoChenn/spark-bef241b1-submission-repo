import io
import time

import numpy as np
import pytest
from PIL import Image

from flange_robot.env import FlangeEnv
from flange_robot.scenarios import ScenarioSpec
from flange_robot.studio import H, SCENARIOS, W, StudioFeed, View, apply_command, render_frame, to_jpeg

LOOSE = ("loose bolts", ScenarioSpec("loose", bolts=((3, 0.12),), env_seed=1), None)
NONE = ("quiet", ScenarioSpec("none", env_seed=5), None)


class ListSink:
    def __init__(self):
        self.items = []

    def send(self, header, payload):
        self.items.append((header, payload))


def feed(**kw):
    sink = ListSink()
    return StudioFeed(sink, realtime=False, **kw), sink


def test_render_frame_has_the_right_size_and_is_not_blank():
    img = render_frame(View(scenario="x", gas=0.1, gas_hist=[0.02, 0.05, 0.1], phase="inspect"))
    assert img.size == (W, H) and img.mode == "RGB"
    assert np.asarray(img).std() > 10


def test_different_situations_draw_differently():
    loose = render_frame(View(bolts={3: (0.12, 0.9)}, phase="repair"))
    crack = render_frame(View(crack_score=0.8, camera_conf=0.95, phase="escalated", diagnosis="crack_or_burst"))
    changed = (np.abs(np.asarray(loose, dtype=int) - np.asarray(crack, dtype=int)).sum(axis=2) > 30).sum()
    assert changed > 2000  # the crack line, phase pill and diagnosis text all differ


def test_jpeg_round_trips():
    jpeg = to_jpeg(render_frame(View()))
    assert Image.open(io.BytesIO(jpeg)).size == (W, H) and len(jpeg) < 400_000


def test_a_loose_bolt_scenario_ends_with_a_repair_alert_on_screen():
    f, sink = feed()
    f.run([LOOSE], loop=False)
    assert f.results == [{"scenario": "loose bolts", "alert": "repaired", "phase": "done"}]
    assert len(sink.items) > 50
    last = Image.open(io.BytesIO(sink.items[-1][1]))
    assert last.size == (W, H)


def test_headers_carry_what_the_relay_needs():
    f, sink = feed()
    f.run([LOOSE], loop=False, max_frames=30)
    headers = [h for h, _ in sink.items]
    assert all(h["type"] == "frame" and (h["w"], h["h"]) == (W, H) for h in headers)
    assert [h["seq"] for h in headers] == list(range(len(headers)))
    assert all(h["mean"] > 3.0 for h in headers)  # the relay treats mean < 3 as black "warming" frames


def test_a_live_crack_command_changes_what_the_robot_does():
    f, sink = feed()
    f.commands.put("crack 0.3")
    f.run([NONE], loop=False)
    assert f.results[0]["alert"] == "crack_or_burst"


def test_a_live_dust_command_makes_it_hand_over():
    f, sink = feed()
    f.commands.put("dust 0.6")
    f.run([LOOSE], loop=False)
    assert f.results[0]["alert"] == "needs_human"


def test_next_skips_the_current_scenario():
    f, sink = feed()
    f.commands.put("next")
    f.run([LOOSE, NONE], loop=False)
    assert [r["scenario"] for r in f.results] == ["loose bolts", "quiet"]
    assert f.results[0]["alert"] is None  # skipped before it finished


def test_commands_are_parsed_safely():
    env = FlangeEnv(seed=0)
    assert apply_command(env, "dust 0.5").startswith("dust") and env.dust == 0.5
    assert apply_command(env, "clear") == "camera cleared" and env.dust == 0.0
    assert apply_command(env, "leak 2 0.15").startswith("bolt 2") and env.true_bolt_angle(2) > 0.1
    assert apply_command(env, "crack") and env.crack_severity == 0.3
    assert apply_command(env, "next") == "NEXT"
    assert apply_command(env, "") is None
    assert apply_command(env, "nonsense 1 2") is None
    assert apply_command(env, "dust abc") is None  # bad number: ignored, not a crash


def test_every_built_in_scenario_runs_to_a_decision():
    f, sink = feed()
    f.run(SCENARIOS, loop=False)
    assert len(f.results) == len(SCENARIOS)
    assert all(r["alert"] for r in f.results)
    assert {r["alert"] for r in f.results} >= {"repaired", "crack_or_burst", "gas_limit"}


def test_frames_flow_through_the_real_studio_relay():
    """Send our frames with the studio's own sender into the studio's own web app."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rig.sim.sender import FrameSender
    from rig.web.server import create_app

    with TestClient(create_app(sim_port=0)) as client:
        sender = FrameSender("127.0.0.1", client.app.state.sim_port).start()
        try:
            f = StudioFeed(sender, realtime=False)
            f.run([LOOSE], loop=False, max_frames=20)
            deadline = time.monotonic() + 5
            health = client.get("/api/health").json()
            while health["frames"] < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
                health = client.get("/api/health").json()
        finally:
            sender.stop()
        assert health["frames"] >= 2 and (health["w"], health["h"]) == (W, H)
        jpeg = client.get("/api/frame.jpg")
        assert jpeg.status_code == 200 and Image.open(io.BytesIO(jpeg.content)).size == (W, H)


def test_the_screen_never_contradicts_the_diagnosis_about_a_crack():
    """After the robot backs off the camera is unsure, but the diagnosis stands and the screen must say so."""
    f, sink = feed()
    seen = {}
    original = f._view

    def spy(*a, **k):
        v = original(*a, **k)
        seen["last"] = v
        return v

    f._view = spy
    f.run([("crack", ScenarioSpec("crack", crack=0.3, env_seed=2), None)], loop=False)
    v = seen["last"]
    assert v.diagnosis == "crack_or_burst" and v.camera_conf < 0.7  # the awkward moment: camera unsure, diagnosis firm
    with_diag = render_frame(v)
    no_diag = render_frame(View(**{**v.__dict__, "diagnosis": None}))
    changed = (np.abs(np.asarray(with_diag, dtype=int) - np.asarray(no_diag, dtype=int)).sum(axis=2) > 30).sum()
    assert changed > 1500  # the crack is drawn because of the diagnosis, not the instantaneous camera reading

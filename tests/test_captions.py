import json
import math

import numpy as np
import pytest

from policy.orchestrator import S
from rig.camera import anchor, from_usd, look_at, project, unproject_ground
from rig.captions import caption, fleet_robots, robot_entry
from rig.overlay import burn


def text(state, **ev):
    return caption(state, ev)["text"]


def test_every_orchestrator_state_has_a_caption():
    for s in S:
        c = caption(s.value, {})
        assert c["text"] and c["lead"] and c["text"].startswith(c["lead"])
        assert c["tone"] in {"act", "warn", "stop", "ok"}
        assert "_" not in c["text"]


def test_scrub_and_soak_numbers():
    assert text("SCRUB", force=31.2, rust_fraction=0.42) == "Scrubbing rust — 31 N, 42 % left"
    assert text("SCRUB") == "Scrubbing rust"
    assert text("SOAK", soak_left=12.4) == "Soaking remover — 12 s left"


def test_back_off_over_limit():
    assert text("BACK_OFF", force=64.8, force_limit=40.0) == "Backing off: 65 N over the 40 N limit"
    assert caption("BACK_OFF", {"force": 64.8})["tone"] == "warn"


def test_gas_lel_percent():
    assert text("LEAK_APPROACH", gas=0.18) == "Gas at 18 % LEL — tracing the source upwind"
    assert text("GAS_RETREAT", gas=0.22) == "Retreating — gas 22 % LEL over the 20 % limit"
    assert caption("GAS_RETREAT", {"gas": 0.22})["tone"] == "stop"
    assert text("RETORQUE", bolt=3, gas=0.09) == "Re-torquing bolt 3 — gas 9 % LEL"


def test_verify_uses_gemini_verdict():
    clean = text("VERIFY", gemini={"verdict": "clean", "confidence": 0.91})
    assert clean == "Verifying — Gemini says clean, 91 % sure"
    assert "rust" in text("VERIFY", gemini={"verdict": "rust", "confidence": 0.82})
    assert text("VERIFY", rust_fraction=0.03) == "Verifying — 3 % rust left"


def test_person_and_done_tones():
    assert text("PAUSE_PERSON", person_m=2.4) == "Paused — person 2.4 m away"
    assert caption("DONE", {})["tone"] == "ok"
    assert "leak fixed" in text("DONE", job="gas") and "painted" in text("DONE")
    assert caption("ESCALATE_TO_HUMAN", {})["tone"] == "stop"
    assert caption("SOMETHING_NEW", {})["text"] == "Something new"


def test_fleet_states():
    assert text("CHARGING", soc=0.64) == "Charging — 64 %"
    assert text("TRANSIT", eta_s=12.0) == "Walking to the job — 12 s away"
    assert caption("FAILED", {})["tone"] == "stop"


def test_robot_entry_and_fleet_snapshot():
    r = robot_entry("unit-1", "Unit 1 · rust", "SCRUB", {"force": 30.0}, (1.0, 2.0, 0.0))
    assert r["caption"]["text"].startswith("Scrubbing")
    assert r["world_pos"] == [1.0, 2.0, 0.0]
    snap = {"schema": "fleet_state.v1", "robots": [
        {"id": "r1", "state": "WORK", "job_state": "SOAK", "pose": {"x": 1, "y": 2, "z": 20, "yaw": 0},
         "battery": {"soc": 0.8}, "eta_s": 0.0},
        {"id": "r2", "state": "CHARGING", "job_state": None, "pose": {"x": 3, "y": 4, "z": 31.25, "yaw": 0},
         "battery": {"soc": 0.42}, "eta_s": 0.0}]}
    rs = fleet_robots(snap)
    assert [r["state"] for r in rs] == ["SOAK", "CHARGING"]
    assert rs[1]["caption"]["text"] == "Charging — 42 %"
    assert rs[0]["world_pos"] == [1.0, 2.0, 20.0]
    assert rs[0]["name"] == "R1"
    json.dumps(rs)


@pytest.fixture
def cam():
    return look_at((6.0, 0.0, 2.0), (0.0, 0.0, 2.0), focal=24.0, aperture=20.955, w=1280, h=720)


def test_look_at_projects_target_to_centre(cam):
    u, v, z = project(cam, (0.0, 0.0, 2.0))
    assert (u, v) == pytest.approx((640, 360))
    assert z == pytest.approx(6.0)


def test_projection_axes_and_focal(cam):
    f = 24.0 / 20.955 * 1280
    u, v, z = project(cam, (0.0, 1.0, 2.0))
    assert u == pytest.approx(640 + f / 6.0)
    u, v, z = project(cam, (0.0, 0.0, 3.0))
    assert v == pytest.approx(360 - f / 6.0)
    assert project(cam, (12.0, 0.0, 2.0))[2] < 0


def test_from_usd_matches_look_at(cam):
    from rig.sim.orbit import look_at_rows
    rows = look_at_rows((6.0, 0.0, 2.0), (0.0, 0.0, 2.0))
    c = from_usd(rows, 24.0, 20.955, 1280, 720)
    assert np.allclose(c["K"], cam["K"]) and np.allclose(c["T"], cam["T"])
    json.dumps(c)


def test_unproject_ground_round_trip():
    c = look_at((6.0, -5.0, 5.0), (0.0, 0.0, 0.0), focal=24.0, aperture=20.955, w=1280, h=720)
    p = unproject_ground(c, 400, 500, 0.0)
    assert p[2] == pytest.approx(0.0)
    u, v, _ = project(c, p)
    assert (u, v) == pytest.approx((400, 500), abs=0.05)


def test_anchor_on_screen_lifts_label(cam):
    a = anchor(cam, (0.0, 0.0, 1.0), lift=1.0)
    assert a["mode"] == "on" and (a["x"], a["y"]) == pytest.approx((640, 360))


def test_anchor_clamps_offscreen_with_arrow(cam):
    a = anchor(cam, (0.0, 30.0, 2.0), lift=0.0, margin=24)
    assert a["mode"] == "edge"
    assert a["x"] == pytest.approx(1280 - 24)
    assert 24 <= a["y"] <= 720 - 24
    assert abs(a["angle"]) < math.pi / 4


def test_anchor_behind_camera(cam):
    assert anchor(cam, (12.0, 0.0, 2.0))["mode"] == "behind"


def test_anchor_occluded_flag(cam):
    assert anchor(cam, (0.0, 0.0, 1.0), occluded=True)["mode"] == "occluded"


def test_burn_subtitle_single_robot():
    img = np.zeros((360, 640, 3), np.uint8)
    out = burn(img, [robot_entry("u1", "Unit 1", "SOAK", {"soak_left": 12}, None)])
    assert out.shape == img.shape
    top = out[:60, 160:480]
    assert top.max() > 200
    assert out[200:, :].max() == 0


def test_burn_labels_multi_robot(cam):
    img = np.zeros((720, 1280, 3), np.uint8)
    rs = [robot_entry("u1", "Unit 1", "SCRUB", {"force": 30}, (0.0, 1.0, 1.0)),
          robot_entry("u2", "Unit 2", "LEAK_APPROACH", {"gas": 0.18}, (0.0, -1.0, 1.0))]
    out = burn(img, rs, cam)
    assert out[:40, 500:780].max() == 0
    u1 = project(cam, (0.0, 1.0, 1.8))
    assert out[int(u1[1]) - 60:int(u1[1]), int(u1[0]) - 40:int(u1[0]) + 40].max() > 150


def test_burn_multi_without_camera_stacks_subtitles():
    img = np.zeros((360, 640, 3), np.uint8)
    rs = [robot_entry("u1", "Unit 1", "SCRUB", {}, None), robot_entry("u2", "Unit 2", "GAS_CHECK", {"gas": 0.08}, None)]
    out = burn(img, rs)
    assert out[:110, :].max() > 200


def test_demo_video_captions():
    from policy.caption_video import aligned, caption_frames, demo_robot
    assert demo_robot({"state": "SCRUB  removing rust", "force": 30.4, "frac": 0.5})["caption"]["text"] == \
        "Scrubbing rust — 30 N, 50 % left"
    assert demo_robot({"state": "RE-PLAN  new rust patch", "force": 0, "frac": 0.6})["state"] == "REPLAN"
    assert "touch-up" in demo_robot({"state": "DONE  3 % residual", "frac": 0.03})["caption"]["text"]
    assert demo_robot({"state": "DONE  segment clean", "frac": 0.0})["caption"]["text"] == "Done — segment clean"
    log = [{"state": "APPROACH", "t": k} for k in range(5)]
    assert [e["t"] for e in aligned(log, 4)] == [1, 2, 3, 4]
    assert len(aligned(log, 7)) == 7
    out = caption_frames([np.zeros((72, 128, 3), np.uint8)] * 2, log)
    assert len(out) == 2 and out[0][:12].max() > 100


def test_hud_header_caption_and_robots(tmp_path):
    from policy.demo_mock import build
    from rig.hud import HudFeed
    orch = build(str(tmp_path / "d.jsonl"), verbose=False)[0]
    cam = look_at((6.0, 0.0, 2.0), (0.0, 0.0, 2.0))
    peer = HudFeed(build(str(tmp_path / "e.jsonl"), verbose=False)[0], rid="unit-2", name="Unit 2 · gas")
    feed = HudFeed(orch, pose=lambda: (1.0, 2.0, 0.0), camera=lambda: cam, peers=[peer])
    while orch.state.value != "SOAK":
        orch.step()
    h = feed.header()
    assert h["caption"]["state"] == "SOAK" and "s left" in h["caption"]["text"]
    assert [r["id"] for r in h["robots"]] == ["unit-1", "unit-2"]
    assert h["robots"][0]["world_pos"] == [1.0, 2.0, 0.0] and h["robots"][1]["world_pos"] is None
    assert h["camera"] == cam
    assert "camera" not in HudFeed(orch).header()
    json.dumps(h)


def test_mock_session_captions_and_fleet(tmp_path):
    from rig.hud import record_mock
    s = record_mock(tmp_path / "s.json", str(tmp_path / "d.jsonl"))
    snaps = s["snapshots"]
    assert all(len(x["robots"]) == 1 and x["caption"]["text"] for x in snaps)
    assert any("over the 40 N limit" in x["caption"]["text"] for x in snaps if x["state"] == "BACK_OFF")
    fleet = s["fleet"]
    assert fleet[0]["pt"] > s["duration"]
    assert [x["pt"] for x in fleet] == sorted(x["pt"] for x in fleet)
    assert all(len(x["robots"]) == 3 and x["camera"]["w"] == 1280 for x in fleet)
    modes = {anchor(x["camera"], r["world_pos"], occluded=bool(r.get("occluded")))["mode"]
             for x in fleet for r in x["robots"]}
    assert {"on", "edge", "occluded"} <= modes
    states = {r["state"] for x in fleet for r in x["robots"]}
    assert {"LEAK_APPROACH", "RETORQUE", "SCRUB"} <= states


def test_storm_states():
    assert text("WIPE_LENS", storm={"lens_q": 0.42}) == "Wiping the lens — view 42 % clear"
    assert text("WIPE_LENS") == "Wiping the lens — rain on the camera"
    assert text("WAIT_LULL", storm={"gust_n": 17.6}) == "Waiting for a lull — gust 18 N"
    assert text("WAIT_LULL") == "Waiting for a lull in the storm"
    assert caption("WAIT_LULL", {})["tone"] == "warn" and caption("WIPE_LENS", {})["tone"] == "warn"

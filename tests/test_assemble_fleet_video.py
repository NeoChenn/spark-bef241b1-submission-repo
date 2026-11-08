import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from integration import assemble_fleet_video as av

STILL = Path(__file__).resolve().parents[1] / "sim/output/full_rig_on_deck.png"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

CALLS = [("G-F2:UNVERIFIED", 10.0, 22.0, "go_check", "unit2"), ("G-F2:LEAK_CONFIRMED", 32.0, 41.0, "fix_leak", "unit2"),
         ("zone:clear", 52.0, 67.0, "resume_rust", "unit1")]


def scenario(dur=72.0, fps=30, scale=1.0):
    calls = [(k, a * scale, b * scale, c, r) for k, a, b, c, r in CALLS]
    rows = [{"type": "finding", "kind": "UNVERIFIED", "sensor": "G-F2", "t": 10.0 * scale},
            {"type": "command", "kind": "hold", "robot": "unit1", "source": "edge", "t": 10.0 * scale},
            {"type": "finding", "kind": "LEAK_CONFIRMED", "sensor": "G-F2", "t": 32.0 * scale},
            {"type": "finding", "kind": "OK", "sensor": "G-F2", "t": 52.0 * scale}]
    for k, a, b, c, r in calls:
        rows.append({"type": "argon_request", "key": k, "t": a})
        rows.append({"type": "argon_call", "key": k, "t_request": a, "status": "ok", "latency_ms": (b - a) * 1000,
                     "cost_usd": 0.0011, "t": b})
        rows.append({"type": "command", "kind": c, "robot": r, "source": "argon", "t": b})
    rows.sort(key=lambda r: r["t"])
    headers = []
    for i in range(int(dur * scale * fps)):
        t = i / fps
        u = t / scale
        s1 = "SCRUB" if u < 10 else "HOLD" if u < 67 else "SCRUB" if u < 74 else "PAINT"
        s2 = ("DOCKED" if u < 22 else "LEAK_APPROACH" if u < 32 else "LEAK_INSPECT" if u < 41 else "RETORQUE"
              if u < 48 else "LEAK_VERIFY" if u < 52 else "DOCKED")
        led = "green" if u < 8 or u >= 52 else "amber" if u < 10 else "red"
        pend = next((c for c in calls if c[1] <= t < c[2]), None)
        done = [c for c in calls if c[2] <= t]
        headers.append({"type": "frame", "seq": i, "t": round(t, 3),
                        "robots": [{"id": "unit1", "name": "Scrubber", "state": s1, "pos": [18.0, -6.0, 42.5]},
                                   {"id": "unit2", "name": "Fixer", "state": s2, "pos": [20.0 + u / 40, -4.0, 42.5]}],
                        "detectors": [{"id": "G-F1", "led": "green"}, {"id": "G-F2", "led": led}],
                        "argon": {"model": "gemini-4-argon", "link": "UP", "pending": pend is not None,
                                  "status": "pending" if pend else "ok", "held": 10 <= u < 67,
                                  "last_command": {"kind": done[-1][3]} if done else None,
                                  "latency_ms": (done[-1][2] - done[-1][1]) * 1000 if done else None,
                                  "cost_usd": 0.0011 * len(done)}})
    return headers, rows


def write_inputs(d, scale=1.0, fps=30):
    d = Path(d)
    headers, rows = scenario(scale=scale, fps=fps)
    (d / "headers.jsonl").write_text("".join(json.dumps(h) + "\n" for h in headers))
    (d / "timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d / "headers.jsonl", d / "timeline.jsonl"


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,avg_frame_rate:format=duration", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return j["streams"][0], float(j["format"]["duration"])


def test_waits_come_from_request_to_call_spans():
    _, rows = scenario()
    assert av.waits_from_timeline(rows) == [(10.0, 22.0), (32.0, 41.0), (52.0, 67.0)]
    rows.append({"type": "argon_request", "key": "G-F1:PIPE_SENSOR_FAULT", "t": 70.0})
    assert av.waits_from_timeline(rows)[-1] == (70.0, float("inf"))


def test_key_beats_play_at_one_x_and_waits_ramp_to_four():
    headers, rows = scenario()
    p = av.plan(headers, 30, rows)
    ts = av.times(headers, 30)
    assert p.top == 4.0 and p.mode == "lapse"
    for t in (10.0, 22.0, 32.0, 41.0, 52.0, 67.0):
        assert p.speed[int(t * 30)] == 1.0
    assert p.speed[int(16 * 30)] == 4.0 and p.speed[int(60 * 30)] == 4.0
    assert (p.speed[(ts > 22.5) & (ts < 31.5)] == 1.0).all()
    assert np.abs(np.diff(p.speed)).max() <= 3.0 / 15 + 1e-9
    assert p.seconds + av.TITLE_S + av.END_S <= av.BUDGET_S


def test_budget_escalates_speed_then_compresses_idle():
    headers, rows = scenario()
    p = av.plan(headers, 30, rows, budget_s=52.0)
    assert p.mode == "lapse" and p.top > 4.0 and p.seconds + av.TITLE_S + av.END_S <= 52.0
    p = av.plan(headers, 30, rows, budget_s=45.0)
    assert p.mode.startswith("lapse+rest") and p.seconds + av.TITLE_S + av.END_S <= 45.0
    assert p.top <= 6.0 and p.speed[int(5 * 30)] == p.why["rest_speed"] <= 3.0
    assert all(p.speed[int(t * 30)] == 1.0 for t in (10.0, 22.0, 41.0, 67.0))


def test_header_pending_is_the_fallback_without_a_timeline():
    headers, _ = scenario()
    p = av.plan(headers, 30, None)
    assert p.speed[int(16 * 30)] == 4.0 and p.speed[int(27 * 30)] == 1.0


def test_schedule_is_monotonic_and_skips_at_speed():
    sp = np.array([1.0] * 30 + [4.0] * 120 + [1.0] * 30)
    idx = av.schedule(sp, 30)
    assert idx == sorted(idx) and len(idx) == 30 + 30 + 30


def test_end_card_numbers_and_teaser_beat():
    headers, rows = scenario()
    nums = av.key_numbers(rows, headers)
    assert nums[0] == ("42 s", "alarm to G-F2 green")
    assert nums[1] == ("3", "Argon calls · 12.0 s avg")
    assert nums[2][0] == "$0.003"
    assert av.teaser_index(headers, rows, 30) == 41 * 30


def test_overlay_and_labels_are_called_per_frame(tmp_path, monkeypatch):
    seen = []

    def draw(frame, header, labels):
        seen.append((header["seq"], header["timelapse"], labels))
        frame[:4, :4] = 255
        return frame

    def labels(header, size):
        return [{"id": "unit1", "xy": (10, 10), "size": size}]

    import types, sys
    monkeypatch.setitem(sys.modules, "fake_overlay", types.SimpleNamespace(draw=draw, labels=labels))
    still = tmp_path / "still.png"
    Image.fromarray(np.full((90, 160, 3), 40, np.uint8)).save(still)
    h, t = write_inputs(tmp_path, scale=0.1)
    r = av.assemble(still, h, tmp_path / "v.mp4", t, overlay="fake_overlay:draw", labels="fake_overlay:labels",
                    hd=False, teaser=False)
    assert r["overlay"] and r["labels"] and len(seen) == r["frames_used"]
    assert seen[0][2][0]["size"] == (160, 90)


@pytest.mark.skipif(not STILL.exists(), reason="rig still missing")
def test_end_to_end_on_a_looped_rig_still(tmp_path):
    small = tmp_path / "still.png"
    Image.open(STILL).convert("RGB").resize((640, 360)).save(small)
    h, t = write_inputs(tmp_path, scale=0.25)
    r = av.assemble(small, h, tmp_path / "fleet.mp4", t, overlay=None, labels=None)
    s, dur = probe(r["out"])
    assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p" and s["avg_frame_rate"] == "30/1"
    assert dur <= av.BUDGET_S and abs(dur - r["seconds"]) < 0.1
    head = Path(r["out"]).read_bytes()[:4096]
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))
    s, _ = probe(r["hd"])
    assert s["height"] == 1080 and s["width"] == 1920
    s, dur = probe(r["teaser"])
    assert abs(dur - av.TEASER_S) < 0.2 and Path(r["gif"]).stat().st_size > 0


def test_tick_headers_resample_to_frame_times():
    hs = [{"t": round(0.1 * i, 1), "robots": [{"id": "unit1", "state": "SCRUB" if i < 5 else "HOLD"}]}
          for i in range(10)]
    out = av.align(hs, 30, 30)
    assert len(out) == 30 and out[14]["robots"][0]["state"] == "SCRUB" and out[15]["robots"][0]["state"] == "HOLD"
    assert av.align(hs, 10, 30) is hs


def test_camera_adapter_and_overlay_header_shape(monkeypatch):
    import sys, types
    seen = {}

    def labels(t, beats, positions=None, w=1280, h=720):
        seen.update(t=t, beats=beats, positions=positions, size=(w, h))
        return {"unit1": {"mode": "on", "x": 10.0, "y": 20.0, "angle": None},
                "unit2": {"mode": "behind", "x": None, "y": None, "angle": None}}

    beats_cls = types.SimpleNamespace(from_events=lambda e: ("beats", tuple(e)))
    fc = types.SimpleNamespace(Beats=beats_cls, DEFAULT_BEATS="default", labels=labels)
    monkeypatch.setitem(sys.modules, "integration.fleet_camera", fc)
    fn = av.camera_labels({"amber": 16.0})
    h = {"t": 3.5, "robots": [{"id": "unit1", "world_pos": [1, 2, 3]}, {"id": "unit2", "pose": [4, 5, 6, 0.1]}]}
    out = fn(h, (1920, 1080))
    assert out == {"unit1": {"xy": (10.0, 20.0), "visible": True, "mode": "on"}}
    assert seen == {"t": 3.5, "beats": ("beats", ("amber",)), "positions": {"unit1": (1.0, 2.0, 3.0),
                                                                            "unit2": (4.0, 5.0, 6.0)},
                    "size": (1920, 1080)}
    assert set(av.for_overlay(h)["robots"]) == {"unit1", "unit2"} and isinstance(h["robots"], list)


def rust_and_walks(fps=30):
    hs = []
    for i in range(40 * fps):
        t = i / fps
        s1 = "SPRAY" if t < 3 else "SOAK" if t < 15 else "SCRUB" if t < 25 else "PAINT" if t < 32 else "DONE"
        s2 = ("DOCKED" if t < 5 else "TRANSIT" if t < 17 else "RETORQUE" if t < 20 else "RETURN" if t < 23
              else "DOCKED")
        hs.append({"t": round(t, 4), "robots": [{"id": "unit1", "state": s1}, {"id": "unit2", "state": s2}],
                   "detectors": [{"id": "G-F2", "led": "red" if t < 16 else "green"}]})
    return hs


def test_soak_and_long_walks_lapse_while_rust_beats_and_retorque_stay_at_one_x():
    hs = rust_and_walks()
    p = av.plan(hs, 30, [], budget_s=200.0)
    sp = lambda t: p.speed[int(t * 30)]
    assert p.mode == "lapse" and sp(9.0) == 4.0 and sp(12.0) == 4.0
    assert p.why["soak"] > 0 and p.why["walk"] > 0 and p.why["wait"] == 0
    for t in (0.0, 1.5, 3.0, 15.0, 16.0, 17.0, 18.5, 21.5, 20.0, 25.0, 28.0, 31.9):
        assert sp(t) == 1.0, t
    walk_only = [{**h, "robots": [h["robots"][1]]} for h in hs]
    q = av.plan(walk_only, 30, [], budget_s=200.0)
    assert q.speed[int(11 * 30)] == 4.0 and q.speed[int(21.5 * 30)] == 1.0


def test_header_camera_projects_labels_and_scales_to_the_frame():
    from rig.camera import look_at
    cam = look_at((10.0, -20.0, 50.0), (18.0, -5.0, 43.67), 24.0, 20.955, 1280, 720)
    fn = av.camera_labels()
    h = {"t": 0.0, "camera": cam, "isaac": {"robots": {"unit1": {"pos": [18.0, -5.0]}}}}
    out = fn(h, (1280, 720))
    x, y = out["unit1"]["xy"]
    assert abs(x - 640) < 2 and abs(y - 360) < 2 and out["unit1"]["mode"] == "on"
    x2, y2 = fn(h, (1920, 1080))["unit1"]["xy"]
    assert abs(x2 - 1.5 * x) < 1e-6 and abs(y2 - 1.5 * y) < 1e-6


def test_isaac_run_drives_frames_and_hud_by_time(tmp_path):
    run = tmp_path / "run"
    (run / "frames").mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((72, 128, 3), 30 * i, np.uint8)).save(run / "frames" / f"{i:05d}.jpg")
    rows = [{"frame": i, "t": 3.0 + 0.6 * i, "camera": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
             "robots": {"unit1": {"pos": [1.0, 2.0]}}} for i in range(3)]
    (run / "fleet_isaac_header.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "fleet_isaac.json").write_text(json.dumps({"fps": 24, "beats": {"amber": 1.0}}))
    hud = [{"t": 3.0, "state": "SPRAY"}, {"t": 4.0, "state": "SOAK"}]
    r = av.isaac_run(run)
    assert r["frames"] == run / "frames" and r["meta"]["fps"] == 24
    merged = av.merge_isaac(r["rows"], hud)
    assert [m["state"] for m in merged] == ["SPRAY", "SPRAY", "SOAK"]
    assert merged[2]["camera"] == rows[2]["camera"] and av.world_positions(merged[0])["unit1"][2] == av.ROBOT_Z


def test_unusable_beats_fall_back_instead_of_crashing(monkeypatch):
    import sys, types

    def bad(_):
        raise TypeError("summary is not beats")

    fc = types.SimpleNamespace(Beats=types.SimpleNamespace(from_events=bad), DEFAULT_BEATS="default")
    assert av._beats(fc, {"beats": {"1_scrubbing": 5.0}}) == "default"


def test_a_stuck_head_does_not_hide_the_leak_clear_time():
    headers, rows = scenario()
    rows.insert(0, {"type": "finding", "kind": "PIPE_SENSOR_FAULT", "sensor": "G-F6", "t": 5.0})
    assert av.key_numbers(rows, headers)[0] == ("42 s", "alarm to G-F2 green")


def test_only_commands_tickets_vetoes_states_and_leds_are_beats():
    rows = [{"type": t, "t": float(i)} for i, t in enumerate(("finding", "argon_call", "link", "command", "ticket",
                                                                "vetoed", "tick"))]
    assert av.beats_from_timeline(rows) == [3.0, 4.0, 5.0]
    a = {"t": 0, "caption": "x", "argon": {"status": "pending", "verdicts": {"G-F5": "OK"}},
         "robots": [{"id": "unit1", "state": "SCRUB", "caption": "a"}], "detectors": [{"id": "G-F5", "led": "green"}]}
    b = {**a, "caption": "y", "argon": {"status": "ok", "verdicts": {"G-F5": "UNVERIFIED"}, "latency_ms": 4000},
         "robots": [{"id": "unit1", "state": "SCRUB", "caption": "b"}]}
    c = {**b, "detectors": [{"id": "G-F5", "led": "red"}]}
    d = {**c, "robots": [{"id": "unit1", "state": "HOLD"}]}
    e = {**d, "argon": {"last_command": {"kind": "go_check", "robot": "unit2", "target": "F5", "t": 5.0}}}
    assert av.beats_from_headers([a, b, c, d, e]) == [2, 3, 4]
    assert av.PAD_S == 0.5


def test_off_frame_robots_get_no_label():
    from rig.camera import look_at
    cam = look_at((10.0, -20.0, 50.0), (18.0, -5.0, 43.67), 24.0, 20.955, 1280, 720)
    h = {"t": 0.0, "camera": cam,
         "isaac": {"robots": {"unit1": {"pos": [18.0, -5.0]}, "unit2": {"pos": [60.0, 30.0]},
                              "unit3": {"pos": [0.0, -40.0]}}}}
    assert set(av.camera_labels()(h, (1280, 720))) <= {"unit1", "G-F5"}


def test_hold_for_crew_and_push_spans_stay_at_one_x_with_a_callout_and_the_plume_is_capped():
    headers, rows = scenario(dur=72.0)
    rows += [{"type": "scenario", "name": "leak_start", "t": 8.0}, {"type": "scenario", "name": "gas_clear", "t": 70.0},
             {"type": "command", "kind": "hold_for_crew", "robot": "unit2", "t": 24.0},
             {"type": "scenario", "name": "object_cleared", "t": 28.0},
             {"type": "scenario", "name": "push", "t": 29.0}, {"type": "nav_event", "event": "recovered", "t": 31.0}]
    rows.sort(key=lambda r: r["t"])
    assert av.held_spans(rows) == [(24.0, 28.0, "hold_for_crew"), (29.0, 31.0, "push")]
    assert av.callouts(rows) == [(24.0, 28.0, "Hold for crew · object on walkway (edge rule)")]
    p = av.plan(headers, 30, rows, budget_s=48.0)
    assert all(p.speed[int(t * 30)] == 1.0 for t in (24.0, 26.0, 27.9, 29.5, 30.9))
    assert p.speed.max() <= av.PLUME_CAP and p.cap.max() <= av.PLUME_CAP
    img = av.callout(np.zeros((720, 1280, 3), np.uint8), "Hold for crew")
    assert img[560:640, 400:880].max() > 0

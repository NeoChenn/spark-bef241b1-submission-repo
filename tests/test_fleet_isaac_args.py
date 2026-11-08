import json
import math

import numpy as np

from integration import run_fleet_isaac as R

META = json.loads((R.REPO / "sim/output/rig_merged_meta.json").read_text())
FLANGES = json.loads((R.REPO / "sim/output/leak_flanges_meta.json").read_text())


def test_defaults_are_720p30_without_overlays():
    a = R.parse_args([])
    assert (a.width, a.height, a.fps) == (1280, 720, 30)
    assert a.scene.endswith("rig_merged.usda") and a.renderer == "RaytracedLighting" and a.spp == 16
    assert a.nav == "timeline" and a.look == "night"


def test_pick_timeline_falls_back_to_demo(tmp_path):
    assert R.pick_timeline(str(tmp_path / "x.jsonl")) == str(tmp_path / "x.jsonl")
    assert R.pick_timeline("") in ("demo", *[str(R.resolve(c)) for c in R.TIMELINES])


def test_per_robot_rows_and_led_mapping():
    rows = [{"type": "robot", "t": 0, "robot": "unit1", "pos": [0, 0], "state": "SCRUB"},
            {"type": "robot", "t": 2, "robot": "unit1", "pos": [2, 0], "state": "HOLD"},
            {"type": "detector", "t": 1, "id": "G-F5", "led": "amber"},
            {"type": "finding", "t": 3, "sensor": "GD-F5", "led": "red", "kind": "UNVERIFIED"},
            {"type": "command", "t": 3.5, "kind": "hold", "robot": "unit1"}]
    s = R.parse_rows(rows)
    x, y, yaw = s.pose("unit1", 1.0)
    assert (x, y) == (1.0, 0.0) and abs(yaw) < 1e-9
    assert s.state("unit1", 1.0) == "SCRUB" and s.state("unit1", 2.5) == "HOLD"
    assert s.detectors(0.5) == {} and s.detectors(1.5) == {"G-F5": "warn"} and s.detectors(3) == {"G-F5": "alarm"}
    assert [e["t"] for e in s.events_between(3.0, 4.0)] == [3.5]
    assert s.end == 3.5


def test_tick_rows_with_robot_and_detector_dicts():
    rows = [{"t": 0, "robots": {"unit1": {"pose": [1, 2, 0.5], "state": "SCRUB"}, "unit2": {"xy": [5, 5]}},
             "detectors": {"G-F2": {"led": "grey"}, "G-F6": "green"}},
            {"t": 1, "robots": {"unit1": {"pose": [1, 2, -3.0]}, "unit2": {"xy": [6, 5]}}}]
    s = R.parse_rows(rows)
    assert s.pose("unit1", 0) == (1.0, 2.0, 0.5)
    assert abs(R.wrap(s.pose("unit1", 0.5)[2] - R.wrap(0.5 + 0.5 * R.wrap(-3.0 - 0.5)))) < 1e-9
    assert abs(s.pose("unit2", 0.5)[2]) < 1e-9
    assert s.detectors(0) == {"G-F2": "fault", "G-F6": "ok"}


def test_yaw_interpolation_wraps_the_short_way():
    s = R.parse_rows([{"t": 0, "robot": "u", "pos": [0, 0], "yaw": 3.0}, {"t": 1, "robot": "u", "pos": [0, 0], "yaw": -3.0}])
    assert abs(abs(s.pose("u", 0.5)[2]) - math.pi) < 0.01


def test_demo_scenario_plays_the_brief():
    s = R.parse_rows(R.demo_rows(META, FLANGES))
    assert set(s.poses) == {"unit1", "unit2"} and s.end >= 50
    assert R.activity("unit1", s.state("unit1", 5)) == "scrub"
    assert R.activity("unit1", s.state("unit1", 20)) == "stow"
    assert R.activity("unit2", s.state("unit2", 28)) == "retorque"
    assert s.detectors(12) == {"G-F5": "alarm"} and s.detectors(40) == {"G-F5": "ok"}
    segs = s.segments("unit2")
    assert len(segs) == 2 and segs[0][0] == 13.0
    f5 = next(f for f in FLANGES["flanges"] if f["id"] == "F5")
    assert math.dist(s.pose("unit2", 28)[:2], f5["center_m"][:2]) < 1.0
    assert R.nearest_flange(FLANGES, s.pose("unit2", 28)[:2])["id"] == "F5"


def test_write_demo_round_trips(tmp_path):
    out = tmp_path / "tl.jsonl"
    R.main(["--write-demo", str(out)])
    s = R.load_scenario(str(out))
    assert set(s.poses) == {"unit1", "unit2"} and s.detectors(12) == {"G-F5": "alarm"}


def test_lidar_scan_sees_a_box_ahead_and_left():
    box = np.array([[0.9, -0.1, 1.1, 0.1]])
    r, a = R.scan_boxes(box, 0.0, 0.0, 0.0)
    assert abs(a[int(np.argmin(r))]) < 0.02 and abs(r.min() - 0.9) < 1e-3
    r, a = R.scan_boxes(box, 0.0, 0.0, -math.pi / 2)
    assert abs(a[int(np.argmin(r))] - math.pi / 2) < 0.02
    r, _ = R.scan_boxes(np.array([[5.0, 5.0, 6.0, 6.0]]), 0.0, 0.0, 0.0)
    assert np.isinf(r).all()


def test_brush_target_lies_on_the_pipe():
    pipe = R.pipe_from_meta(META)
    assert abs(pipe["theta"] - 0.9) < 0.02
    for t in (0.0, 1.3, 7.7):
        p, appr = R.brush_target(t, pipe)
        rel = p - np.asarray(pipe["origin"])
        radial = rel - rel[0] * np.array([1.0, 0, 0])
        assert abs(np.linalg.norm(radial) - pipe["radius"]) < 0.01
        assert abs(np.linalg.norm(appr) - 1) < 1e-9 and appr @ radial < 0


def test_wide_shot_frames_the_robots():
    s = R.parse_rows(R.demo_rows(META, FLANGES))
    eye, target, focal = R.wide_shot(s, META)
    assert eye[2] > target[2] > META["world"]["deck_z_m"] and focal == 24.0
    m = R.look_rows(eye, target)
    fwd = -m[2, :3]
    d = np.asarray(target) - np.asarray(eye)
    assert np.allclose(fwd, d / np.linalg.norm(d))


def test_rust_job_masks_progress_and_timeline_masks():
    pipe = R.pipe_from_meta(META)
    m0 = R.rust_masks((32, 64), pipe, 1.0, 0.0, {})
    assert m0["rust_grid"].sum() > 0 and not m0["sprayed"].any() and not m0["painted"].any()
    m1 = R.rust_masks((32, 64), pipe, 1.0, 0.0, {"spray": 3, "scrub": 14, "paint": 4})
    assert m1["sprayed"].any() and m1["rust_grid"].sum() == 0 and m1["painted"].sum() >= (m0["rust_grid"] > 0).sum()
    assert R.activity("unit1", "VERIFY_PAINT") == "stow" and R.activity("unit1", "SPRAY") == "spray"
    assert R.activity("unit1", "SOAK") == "stow" and R.activity("unit2", "TO_DOCK") == "stow"
    rows = [{"type": "tick", "t": 1, "robots": {"unit1": {"pose": [0, 0, 42.5, 0], "state": "SCRUB",
                                                           "sprayed": [[1, 0], [0, 1]]}}}]
    s = R.parse_rows(rows)
    assert s.pose("unit1", 1) == (0.0, 0.0, 0.0)
    assert s.masks_at(0.5) is None and R.resample(s.masks_at(1)["sprayed"], (4, 4))[3, 3] == 1


def test_retorque_bolt_from_the_live_timeline():
    s = R.load_scenario(str(R.REPO / "runs_sample/fleet_demo/live_timeline.jsonl"))
    assert R.timeline_bolt(s) == 3
    assert R.parse_args(["--bolt", "5"]).bolt == 5

"""CPU tests for the fleet replay: timeline parsing, interpolation, grounding, shots, projection."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
import fleet_replay as fr  # noqa: E402


def rows(robot="Q1", kind="Q", pts=((0, 0, 42.5), (2, 0, 42.5), (2, 2, 42.5)), yaws=(0.0, 0.0, 1.5708),
         states=("TRANSIT", "TRANSIT", "INSPECTING"), jobs=(None, "J1", "J1")):
    return [{"t": float(k), "robot": robot, "kind": kind, "position": list(p), "yaw": y, "state": s, "job": j}
            for k, (p, y, s, j) in enumerate(zip(pts, yaws, states, jobs))]


def test_linear_position_and_step_state():
    tl = fr.parse_timeline([json.dumps(r) for r in rows()])
    tr = tl.tracks["Q1"]
    assert np.allclose(tr.position(0.5), [1.0, 0.0, 42.5])
    assert np.allclose(tr.position(1.5), [2.0, 1.0, 42.5])
    assert np.allclose(tr.position(-3.0), [0, 0, 42.5]) and np.allclose(tr.position(9.0), [2, 2, 42.5])
    assert tr.state(1.99) == "TRANSIT" and tr.state(2.0) == "INSPECTING"
    assert tr.job(0.5) is None and tr.job(1.0) == "J1"
    assert tr.speed(0.5) == pytest.approx(2.0)
    assert tr.distance(1.5) == pytest.approx(3.0)


def test_yaw_takes_shortest_way_round():
    r = rows(yaws=(3.0, -3.0, -3.0))
    tr = fr.parse_timeline(r).tracks["Q1"]
    mid = tr.yaw_at(0.5)
    assert abs(abs(mid) - math.pi) < 0.2          # through +-pi, not through 0


def test_degrees_detected():
    r = rows(yaws=(0.0, 90.0, 180.0))
    tr = fr.parse_timeline(r).tracks["Q1"]
    assert tr.yaw_at(1.0) == pytest.approx(math.pi / 2)


def test_ground_robots_snap_to_deck_and_drone_stays_up():
    q = fr.parse_timeline(rows(pts=((0, 0, 42.82), (1, 0, 31.6), (2, 0, 20.1)))).tracks["Q1"]
    assert [q.position(t)[2] for t in (0, 1, 2)] == [42.5, 31.25, 20.0]
    a = fr.parse_timeline(rows(robot="A1", kind="A", pts=((0, 0, 30.0), (1, 0, 60.0), (2, 0, 44.0)),
                               states=("TRANSIT", "TRANSIT", "IDLE"))).tracks["A1"]
    assert a.position(0)[2] == fr.DRONE_MIN_Z and a.position(1)[2] == 60.0 and a.position(2)[2] == 44.0


def test_events_and_plume_fade():
    lines = rows() + [
        {"t": 1.0, "event": "gas_leak", "job": "J1", "robot": "Q1", "detail": {"position": [1, 1, 43]}},
        {"t": 3.0, "event": "leak_isolated", "job": "J1", "robot": "Q1", "detail": {}},
    ]
    tl = fr.parse_timeline(lines)
    assert [e.event for e in tl.events] == ["gas_leak", "leak_isolated"]
    assert fr.gas_plumes(tl, 0.5) == []
    assert fr.gas_plumes(tl, 2.0)[0]["strength"] == pytest.approx(0.5)
    assert fr.gas_plumes(tl, 20.0) == []
    target = fr.resolve_job_target("J1", tl, {}, 2.0)
    assert np.allclose(target, [1, 1, 43])


def test_job_target_from_asset_id():
    tl = fr.parse_timeline(rows(jobs=(None, "J1 gauge_read gauge_02", "J1 gauge_read gauge_02")))
    assets = {"gauge_02": {"position_m": [-17.5, 23.37, 43.8]}}
    jobs = fr.active_jobs(tl, 1.5, assets)
    assert len(jobs) == 1 and np.allclose(jobs[0]["target"], [-17.5, 23.37, 43.8])


def test_fake_timeline_format_and_shots(tmp_path):
    path = tmp_path / "timeline.jsonl"
    fr.write_fake_timeline(path)
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    robot_rows = [r for r in raw if "event" not in r]
    assert set(robot_rows[0]) == {"t", "robot", "kind", "position", "yaw", "state", "job"}
    assert all(set(r) == {"t", "event", "job", "robot", "detail"} for r in raw if "event" in r)
    tl = fr.load_timeline(path)
    assert set(tl.tracks) == {"Q1", "Q2", "C1", "A1"} and tl.duration == pytest.approx(60.0)
    for rid in ("Q1", "Q2", "C1"):            # ground robots stay on the top deck patrol route
        tr = tl.tracks[rid]
        for t in np.linspace(0, 60, 121):
            p = tr.position(t)
            assert p[2] == 42.5
            assert _on_route(p[:2]), (rid, t, p)
    shots = fr.plan_shots(tl)
    assert shots[0].kind == "establish" and shots[0].start == 0.0 and shots[-1].end == pytest.approx(60.0)
    assert all(b.start == pytest.approx(a.end) for a, b in zip(shots, shots[1:]))
    assert any(s.kind == "event" and s.start == pytest.approx(18.0) for s in shots)   # cut on the gas leak
    assert all(s.end - s.start >= 1.5 for s in shots)
    assert fr.gas_plumes(tl, 25.0) and not fr.gas_plumes(tl, 59.0)


def _on_route(xy, tol=0.05):
    route = np.array(fr.PATROL_ROUTE, float)
    for a, b in zip(route, route[1:]):
        ab = b - a
        u = np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0, 1)
        if np.linalg.norm(a + u * ab - xy) < tol:
            return True
    return False


def test_projection_centre_and_behind():
    cam = fr.look_at((0, -10, 0), (0, 0, 0))
    centre, behind = fr.project([(0, 0, 0), (0, -20, 0)], cam, 30.0, 1280, 720)
    assert centre == pytest.approx((640.0, 360.0)) and behind is None
    right = fr.project([(1, 0, 0)], cam, 30.0, 1280, 720)[0]
    up = fr.project([(0, 0, 1)], cam, 30.0, 1280, 720)[0]
    assert right[0] > 640 and up[1] < 360


def test_camera_follows_subject():
    tl = fr.parse_timeline(rows())
    shot = fr.Shot(0.0, 3.0, "follow", "Q1")
    eye, target, focal = fr.camera_for(shot, tl, 1.0)
    assert np.linalg.norm(target[:2] - tl.tracks["Q1"].position(1.0)[:2]) < 0.6
    assert eye[2] > 42.5 + 2.0 and 3.0 < np.linalg.norm(eye - target) < 15.0


def test_hud_draws():
    tl = fr.parse_timeline(rows())
    img = np.zeros((360, 640, 3), np.uint8)
    out = fr.draw_fleet_hud(img, tl, 1.0, fr.Shot(0, 2, "follow", "Q1"), {"Q1": (320.0, 200.0)})
    assert out.shape == img.shape and out.any()

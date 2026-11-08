import math
from pathlib import Path

import numpy as np
import pytest

from integration.fleet_camera import (DEFAULT_BEATS, DEFAULT_SCENE, H, W, Beats, beat_at, beat_spans, camera_at,
                                      labels, project_points, robots_at, view)
from rig.camera import project

FPS = 60.0
TS = np.arange(DEFAULT_BEATS.start, DEFAULT_BEATS.end + 1e-9, 1 / FPS)
PATH = [camera_at(t) for t in TS]
EYES = np.array([p[0] for p in PATH])
TARGETS = np.array([p[1] for p in PATH])
FOCALS = np.array([p[2] for p in PATH])
SCRIPT = [{"name": "leak_start", "t": 6.0}, {"name": "leak", "t": 6.0}, {"name": "G-F5 amber", "t": 6.75},
          {"name": "G-F5 red", "t": 8.55}, {"name": "retorque_done", "t": 34.0}, {"name": "source_off", "t": 34.0},
          {"name": "gas_decay", "t": 34.0}, {"name": "gas_clear", "t": 40.2}, {"name": "G-F6 stuck", "t": 46.0}]


def subjects(name, t):
    s, r = DEFAULT_SCENE, robots_at(t)
    return {"scrub": [s.scrub], "detector": [s.detector], "gripper": [s.bolt], "green": [s.detector, s.bolt],
            "spray": [s.scrub, np.add(s.scrub, np.multiply(0.3, s.normal))], "paint": [s.scrub],
            "resume": [r["unit1"]]}.get(
        name, [r["unit1"], r["unit2"]])


def test_one_take_under_a_minute_with_seven_beats_in_order():
    spans = beat_spans()
    assert [n for n, *_ in spans] == ["establish", "spray", "scrub", "detector", "pull_back", "gripper", "green",
                                     "resume", "paint", "wide"]
    assert spans[0][1] == DEFAULT_BEATS.start and spans[-1][2] == DEFAULT_BEATS.end <= 120.0
    assert spans[0][2] - spans[0][1] >= 2.5
    for (_, _, end, _), (_, t0, _, _) in zip(spans, spans[1:]):
        assert end == t0
    assert all(t0 <= arrive < end for _, t0, end, arrive in spans)
    with pytest.raises(ValueError):
        Beats(end=121.0)


def test_beats_follow_the_scenario_script():
    b = Beats.from_events(SCRIPT)
    assert (b.amber, b.red, b.fixed, b.green) == (6.75, 8.55, 34.0, 40.2)
    assert b.scrub < b.amber and b.step_back > b.red and b.arrive < b.fixed and b.resume > b.green
    assert b.end - b.start <= 120.0 and b.spray < b.scrub < b.amber
    assert Beats.from_events({"script": SCRIPT, "duration_s": 55.0}).end == 55.0
    run = Beats.from_events({"beats": {"1_unit1_spray": 2.0, "1_unit1_soak": 6.0, "1_unit1_scrub": 11.0,
                                       "2_leak_start": 18.0, "2_gf5_amber": 18.8, "2_gf5_red": 20.6,
                                       "3_unit1_held": 22.4, "4_unit2_sniffs": 40.0, "4_gf5_green": 55.0,
                                       "5_resume_rust": 57.5, "5_unit1_paint": 66.0, "6_link_cut": None}})
    assert (run.spray, run.scrub, run.amber, run.red, run.step_back) == (2.0, 11.0, 18.8, 20.6, 22.4)
    assert (run.arrive, run.green, run.resume, run.paint) == (40.0, 55.0, 57.5, 66.0)
    assert run.arrive < run.fixed < run.green and 75.0 <= run.end <= 85.0
    late = Beats.from_events({"amber": 12.0, "red": 14.0, "fixed": 38.0, "green": 44.0})
    detector = next(s for s in beat_spans(late) if s[0] == "detector")
    assert detector[1] < late.amber < late.red < detector[2]


def test_the_amber_and_red_leds_land_inside_the_detector_beat():
    b = DEFAULT_BEATS
    assert beat_at(b.red) == "detector" and beat_at(b.amber + 0.5) == "detector"
    assert beat_at(b.fixed) == "gripper" and beat_at(b.green) == "green"
    assert beat_at(b.spray + 0.5) == "spray" and beat_at(b.scrub + 1.0) == "scrub"
    assert beat_at(b.resume + 1.0) == "resume" and beat_at(b.paint + 0.5) == "paint"
    assert [n for n, *_ in beat_spans()].index("spray") < [n for n, *_ in beat_spans()].index("detector")


def test_the_path_is_continuous_with_eased_moves():
    step = np.linalg.norm(np.diff(EYES, axis=0), axis=1) * FPS
    assert step.max() < 15.0
    assert np.linalg.norm(np.diff(TARGETS, axis=0), axis=1).max() * FPS < 20.0
    assert np.abs(np.diff(FOCALS)).max() * FPS < 20.0
    accel = np.linalg.norm(np.diff(EYES, 2, axis=0), axis=1) * FPS * FPS
    assert accel.max() < 40.0
    fwd = TARGETS - EYES
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    turn = np.degrees(np.arccos(np.clip((fwd[1:] * fwd[:-1]).sum(1), -1, 1))) * FPS
    assert turn.max() < 120.0
    for _, _, _, arrive in beat_spans():
        if arrive > DEFAULT_BEATS.start:
            i = int(round((arrive - DEFAULT_BEATS.start) * FPS))
            assert step[i - 1] < 0.2


def test_the_subjects_stay_in_frame_while_each_beat_holds():
    for name, _, end, arrive in beat_spans():
        for t in np.arange(arrive, end, 0.1):
            cam = view(t)
            for p in subjects(name, t):
                u, v, z = project(cam, p)
                assert z > 0 and 0.05 * W < u < 0.95 * W and 0.05 * H < v < 0.95 * H, (name, round(t, 2))


def test_robot_labels_follow_the_robots():
    for name, _, end, arrive in beat_spans():
        if name in ("establish", "pull_back", "wide"):
            for t in np.arange(arrive, end, 0.5):
                a = labels(t)
                assert a["unit1"]["mode"] == a["unit2"]["mode"] == "on", (name, t)
    a = labels(DEFAULT_BEATS.red)
    assert a["G-F5"]["mode"] == "on" and abs(a["G-F5"]["x"] - W / 2) < W / 4
    p = project_points(3.0, {"x": (20.0, -6.0, 42.5)}, w=1920, h=1080)
    assert set(p["x"]) == {"mode", "x", "y", "angle"}
    moved = labels(74.5, positions={"unit1": (16.0, -4.5, 42.82), "unit2": (21.24, -10.17, 42.82)})
    assert moved["unit1"]["mode"] == moved["unit2"]["mode"] == "on"
    assert moved["unit2"]["x"] != labels(74.5)["unit2"]["x"]
    assert labels(30.0)["unit2"]["mode"] in ("on", "edge")


def test_the_camera_keeps_half_a_metre_from_the_pipes_and_rig():
    for lo, hi in DEFAULT_SCENE.obstacles():
        gap = np.linalg.norm(np.maximum(0, np.maximum(lo - EYES, EYES - hi)), axis=1)
        assert gap.min() >= 0.5, (lo, hi, TS[gap.argmin()])
    assert EYES[:, 2].min() - DEFAULT_SCENE.deck_z > 1.0
    for i in range(0, len(TS), 6):
        for p in robots_at(TS[i]).values():
            assert np.linalg.norm(EYES[i] - p) > 1.0


def _triangles(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    xc, out = UsdGeom.XformCache(0), []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()).startswith("/World/Robots"):
            continue
        m = UsdGeom.Mesh(prim)
        v, fc, fi = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if not v or not fc:
            continue
        M = np.array(xc.GetLocalToWorldTransform(prim))
        w = np.asarray(v, float) @ M[:3, :3] + M[3, :3]
        fc, fi = np.asarray(fc), np.asarray(fi)
        off = np.concatenate([[0], np.cumsum(fc)[:-1]])
        for n in np.unique(fc):
            k = off[fc == n]
            for j in range(1, n - 1):
                out.append(w[np.stack([fi[k], fi[k + j], fi[k + j + 1]], 1)])
    T = np.concatenate(out)
    return T[np.all(T.max(1) > (5, -20, 42.3), 1) & np.all(T.min(1) < (45, 25, 62), 1)]


def _seg(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * ab - p, axis=-1)


def _dist(p, T):
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=-1), 1e-12)[:, None]
    d = ((p - a) * n).sum(-1)
    q = p - d[:, None] * n
    side = lambda u, v: (np.cross(v - u, q - u) * n).sum(-1)
    inside = (side(a, b) >= 0) & (side(b, c) >= 0) & (side(c, a) >= 0)
    return np.where(inside, np.abs(d), np.minimum(np.minimum(_seg(p, a, b), _seg(p, b, c)), _seg(p, c, a)))


def test_the_camera_never_enters_the_rig_merged_meshes():
    pytest.importorskip("pxr")
    scene = Path(__file__).resolve().parent.parent / "sim/output/rig_merged.usda"
    if not scene.exists():
        pytest.skip("rig_merged.usda missing")
    T = _triangles(scene)
    lo, hi = T.min(1), T.max(1)
    for e in EYES[:: int(FPS / 4)]:
        near = np.linalg.norm(np.maximum(0, np.maximum(lo - e, e - hi)), axis=1) < 0.5
        assert not near.any() or _dist(e, T[near]).min() >= 0.5, e


def test_usd_camera_matches_the_path():
    pxr = pytest.importorskip("pxr")
    from integration.fleet_camera import APERTURE, apply_usd, author_usd
    stage = pxr.Usd.Stage.CreateInMemory()
    cam = apply_usd(stage, "/World/Cameras/fleet", 20.0)
    eye, target, focal = camera_at(20.0)
    m = np.array(pxr.UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default()))
    assert np.allclose(m[3, :3], eye) and math.isclose(cam.GetFocalLengthAttr().Get(), focal, rel_tol=1e-6)
    assert np.allclose(-m[2, :3], (np.subtract(target, eye)) / np.linalg.norm(np.subtract(target, eye)))
    assert math.isclose(cam.GetHorizontalApertureAttr().Get(), APERTURE, rel_tol=1e-6)
    n = author_usd(stage, "/World/Cameras/fleet", fps=10.0)
    assert n == 751 and stage.GetEndTimeCode() == 750
    m = np.array(pxr.UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(pxr.Usd.TimeCode(300)))
    assert np.allclose(m[3, :3], camera_at(30.0)[0])

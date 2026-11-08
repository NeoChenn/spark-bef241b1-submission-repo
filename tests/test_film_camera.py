from pathlib import Path

import numpy as np
import pytest

from integration.film_camera import (CONTROL_ROOM, DEFAULT_FILM_SCENE, RIG_KEEP, apply_usd_pose, inside_edges,
                                     rig_orbit, view, width_fraction)
from integration.film_edl import FilmBeats, camera_at_film, default_track, edl, to_json
from integration.fleet_camera import DEFAULT_SCENE
from rig.camera import project

SHOTS = edl()
FPS = 30.0
TS = np.arange(0.0, SHOTS[-1].film[1], 1 / FPS)
PATH = [camera_at_film(t, SHOTS) for t in TS]
EYES = np.array([p[0] for p in PATH])
SUMMARY = {"t_end": 129.7, "beats": {"dispatch_u2": 18.3, "u2_arrive": 40.5, "u2_engage": 46.8, "torque_done": 50.2,
                                     "gf5_green": 54.2, "u2_return": 54.7, "dispatch_u1": 63.2, "u1_arrive": 76.9,
                                     "spray": 79.2, "soak": 82.3, "scrub": 88.0, "scrub_done": 107.9, "paint": 109.5,
                                     "paint_done": 113.6, "u1_return": 114.8, "both_docked": 129.6}}


def shot_at(t):
    return next(s for s in SHOTS if s.film[0] <= t < s.film[1])


def test_the_edl_tiles_sixty_seconds_in_story_order():
    names = [s.name for s in SHOTS]
    assert names[:5] == ["rig", "control_push", "board", "board_hold", "unit2_walk"]
    assert names[-3:] == ["walk_back_u1", "walk_back_u2", "conclusion"]
    assert names.index("unit2_job") < names.index("unit1_walk") < names.index("unit1_spray")
    order = ["unit1_spray", "unit1_soak", "unit1_scrub", "unit1_paint"]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)
    assert SHOTS[0].film[0] == 0.0 and SHOTS[-1].film[1] <= 60.0 + 1e-9
    for a, b in zip(SHOTS, SHOTS[1:]):
        assert a.film[1] == pytest.approx(b.film[0])
    assert [s.name for s in SHOTS if s.two_d] == ["board", "board_hold", "conclusion"]
    assert all(s.rate == pytest.approx(2.5) for s in SHOTS if s.name.startswith("walk_back"))
    assert all(s.rate >= 0.99 for s in SHOTS if s.sim and s.sim[1] > s.sim[0])
    assert all(s.badge for s in SHOTS if s.rate > 1.05 and not s.two_d)
    assert {s.t_in for s in SHOTS} <= {"cut", "xfade", "fade"} and '"walk_back_u2"' in to_json(SHOTS)


def test_short_and_long_cuts_share_every_3d_shot():
    short, long = edl(length=60.0), edl(length=None)
    named = edl(length="long")
    assert [x.to_dict() for x in short[:-1]] == [x.to_dict() for x in long[:-1]] == [x.to_dict() for x in named[:-1]]
    assert short[-1].film[1] == 60.0 and short[-1].cards == 3
    assert long[-1].film[1] == 66.0 and long[-1].cards == 5
    assert long[-1].film[1] - long[-1].film[0] == pytest.approx(short[-1].film[1] - short[-1].film[0] + 6.0)
    both = FilmBeats.from_summary({**SUMMARY, "beats": {**SUMMARY["beats"], "u2_return": 114.5}})
    walk = [x for x in edl(both) if x.name.startswith("walk_back")]
    assert [w.sim for w in walk] == [(119.5, 124.5), (124.5, 129.5)] and all(w.rate == 2.5 for w in walk)


def test_control_room_constants_describe_the_window_isaac_needs():
    lo, hi = np.array(CONTROL_ROOM["interior"][0]), np.array(CONTROL_ROOM["interior"][1])
    eye, target = np.array(CONTROL_ROOM["interior_eye"]), np.array(CONTROL_ROOM["window_target"])
    assert np.all(lo < eye) and np.all(eye < hi) and CONTROL_ROOM["floor_z"] < eye[2] < CONTROL_ROOM["ceiling_z"]
    (w0, w1), = [w for w in CONTROL_ROOM["north_windows"] if w[0][0] <= target[0] <= w[1][0]]
    assert w0[2] <= target[2] <= w1[2]


def test_sim_time_follows_the_recorded_beats():
    b = FilmBeats.from_summary(SUMMARY)
    assert (b.u2_engage, b.torque_done, b.u1_walk, b.paint_done, b.end) == (46.8, 50.2, 63.2, 113.6, 129.7)
    shots = {s.name: s for s in edl(b)}
    job = shots["unit2_job"]
    assert job.sim[0] < b.u2_engage and job.sim[1] > b.gf5_green and job.rate < 1.1
    assert shots["unit1_spray"].sim[0] < b.spray and shots["unit1_soak"].sim == (b.soak, b.scrub)
    assert shots["unit1_scrub"].sim == (b.scrub, b.scrub_done) and shots["unit1_scrub"].film == (37.9, 45.9)
    assert shots["unit1_paint"].sim[0] == b.scrub_done and shots["unit1_paint"].sim[1] > b.paint_done
    assert shots["walk_back_u1"].sim == (b.u1_return + 5.0, b.u1_return + 10.0)
    assert shots["walk_back_u2"].sim == (b.u1_return + 10.0, b.u1_return + 15.0)
    assert shots["board"].sim == (0.0, b.dispatch_u2 + 0.3) and shots["board"].film == (10.0, 18.5)
    hold = shots["board_hold"]
    assert hold.film == (18.5, 20.0) and hold.rate == 0.0 and hold.sim == (b.dispatch_u2 + 0.3,) * 2
    assert shots["board"].film[0] + b.dispatch_u2 / shots["board"].rate < hold.film[0]
    assert shots["unit2_walk"].film[0] == 20.0
    sims = [camera_at_film(t, list(shots.values()))[3] for t in np.arange(22.0, 50.0, 0.1)]
    sims = [s for s in sims if s is not None]
    assert all(b2 >= a2 for a2, b2 in zip(sims, sims[1:]))


def test_the_camera_is_continuous_inside_every_shot():
    v = np.linalg.norm(np.diff(EYES, axis=0), axis=1) * FPS
    for i, t in enumerate(TS[1:]):
        if shot_at(t) is shot_at(TS[i]) and shot_at(t).name not in ("rig", "conclusion"):
            assert v[i] < 15.0, (shot_at(t).name, t)
    orbit = np.array([rig_orbit(u)[0] for u in np.linspace(0, 1, 50)])
    ang = np.degrees(np.arctan2(orbit[:, 1], orbit[:, 0]))
    assert 15.0 <= abs(ang[-1] - ang[0]) <= 20.0


def test_the_whole_rig_with_flare_and_derrick_stays_inside_the_orbit():
    for u in np.linspace(0, 1, 25):
        e, g, f = rig_orbit(u)
        assert 0.58 <= width_fraction(e, f) <= 0.75
        assert inside_edges(e, g, f, RIG_KEEP, 0.05)
        assert inside_edges(e, g, f, [(-62.6, 8.0, 82.5), (20.0, 20.0, 81.3)], 0.05)


def test_the_walk_back_tracks_each_robot_high_oblique_and_large():
    track = default_track(FilmBeats(), DEFAULT_FILM_SCENE)
    for name, robot in (("walk_back_u1", "unit1"), ("walk_back_u2", "unit2")):
        shot = next(x for x in SHOTS if x.name == name)
        for t in np.linspace(shot.film[0], shot.film[1] - 1e-3, 7):
            e, g, f, st = camera_at_film(t, SHOTS)
            d = np.linalg.norm(np.subtract(e, g))
            assert 8.0 <= d <= 10.0 and 50.0 <= np.degrees(np.arcsin((e[2] - g[2]) / d)) <= 60.0
            cam = view(e, g, f)
            p = track(robot, st)
            u, v, z = project(cam, p)
            assert z > 0 and 0.2 * cam["w"] < u < 0.8 * cam["w"] and 0.2 * cam["h"] < v < 0.8 * cam["h"], (t, robot)
            top, bottom = project(cam, (p[0], p[1], p[2] + 0.3))[1], project(cam, (p[0], p[1], 42.5))[1]
            assert abs(bottom - top) / cam["h"] >= 0.06


def test_work_shots_frame_the_bolt_the_patch_and_the_arms():
    s = DEFAULT_FILM_SCENE
    w = s.unit1_work
    unit1 = [s.patch, (w[0], w[1], w[2] + 1.1), (w[0] - 0.35, w[1], 42.5), (w[0] + 0.35, w[1], 42.5)]
    for name, pts, far, lead in (("unit2_job", [s.bolt, s.flange], 2.0, 1.6), ("unit1_spray", unit1, 4.0, 1.3),
                                 ("unit1_soak", unit1, 4.0, 0.0), ("unit1_paint", unit1, 4.0, 0.0)):
        shot = next(x for x in SHOTS if x.name == name)
        end = {"unit1_paint": shot.film[1] - 2.5, "unit2_job": shot.film[0] + 6.5}.get(name, shot.film[1] - 0.01)
        for t in np.linspace(shot.film[0] + lead, end, 6):
            e, g, f, _ = camera_at_film(t, SHOTS)
            cam = view(e, g, f)
            for p in pts:
                u, v, z = project(cam, p)
                assert z > 0 and 0.05 * cam["w"] < u < 0.95 * cam["w"] and 0.05 * cam["h"] < v < 0.95 * cam["h"]
            assert np.linalg.norm(np.subtract(e, pts[0])) < far
            if name != "unit2_job":
                assert 0.2 * cam["h"] < project(cam, s.patch)[1] < 0.5 * cam["h"]


def test_the_socket_view_is_side_on_and_fills_the_frame():
    s = DEFAULT_FILM_SCENE
    shot = next(x for x in SHOTS if x.name == "unit2_job")
    axis = np.asarray(s.bolt_axis)
    for t in np.linspace(shot.film[0] + 1.6, shot.film[0] + 6.5, 5):
        e, g, f, _ = camera_at_film(t, SHOTS)
        v = np.subtract(e, s.bolt)
        assert 1.5 <= np.linalg.norm(v) <= 2.0 and 50.0 <= f <= 60.0
        assert 75.0 <= np.degrees(np.arccos(abs(v @ axis) / np.linalg.norm(v))) <= 105.0
        cam = view(e, g, f)
        side = np.cross(axis, (0.0, 0.0, 1.0))
        rim = [np.asarray(s.flange) + 0.17 * (np.cos(a) * side + np.sin(a) * np.array([0.0, 0.0, 1.0]))
               for a in np.linspace(0, 2 * np.pi, 16)]
        vs = [project(cam, p)[1] for p in rim]
        assert 0.3 <= (max(vs) - min(vs)) / cam["h"] <= 0.8
    e0 = camera_at_film(shot.film[0] + 6.5, SHOTS)
    e1 = camera_at_film(shot.film[1] - 1e-3, SHOTS)
    assert np.linalg.norm(np.subtract(e1[0], s.bolt)) > np.linalg.norm(np.subtract(e0[0], s.bolt)) + 0.5


def test_the_socket_sightline_clears_the_fixer_arm():
    timeline = Path(__file__).resolve().parent.parent / "runs_sample/film/timeline.jsonl"
    if not timeline.exists():
        pytest.skip("no film timeline")
    rm = pytest.importorskip("integration.retorque_motion")
    m = rm.from_timeline(timeline)
    s = DEFAULT_FILM_SCENE
    shot = next(x for x in SHOTS if x.name == "unit2_job")
    links = ("ur3_shoulder_link", "ur3_upper_arm_link", "ur3_forearm_link", "ur3_wrist_1_link", "ur3_wrist_2_link")
    for t in np.arange(46.0, 50.2, 0.25):
        e = np.array(camera_at_film(shot.film[0] + (t - shot.sim[0]) / shot.rate, SHOTS)[0])
        P = {k: v[:3, 3] for k, v in m.model.link_poses(m.base, m._full(m.vector(t))).items()}
        for target in (np.asarray(s.bolt), P["socket_link"]):
            w = target - e
            for a, b in zip(links[:-1], links[1:]):
                for k in np.linspace(0, 1, 12):
                    p = P[a] + (P[b] - P[a]) * k
                    u = np.clip((p - e) @ w / (w @ w), 0.0, 0.97)
                    assert np.linalg.norm(e + u * w - p) > 0.09, (t, a, b)


def test_the_scrub_close_up_fills_the_frame_with_the_patch():
    s = DEFAULT_FILM_SCENE
    scrub, paint = (next(x for x in SHOTS if x.name == n) for n in ("unit1_scrub", "unit1_paint"))
    for t in [*np.linspace(scrub.film[0] + 0.4, scrub.film[1] - 1e-3, 6), paint.film[1] - 1.5, paint.film[1] - 1e-3]:
        e, g, f, _ = camera_at_film(t, SHOTS)
        assert 0.9 <= np.linalg.norm(np.subtract(e, s.patch)) <= 1.2 and 50.0 <= f <= 65.0
        cam = view(e, g, f)
        tu = np.cross(s.patch_normal, (1.0, 0.0, 0.0))
        corners = [np.asarray(s.patch) + a * tu + (b, 0.0, 0.0) for a in (-0.075, 0.075) for b in (-0.06, 0.06)]
        us = [project(cam, p)[0] for p in corners]
        assert 0.35 <= (max(us) - min(us)) / cam["w"] <= 0.55
        u, v, z = project(cam, s.patch)
        assert abs(u - cam["w"] / 2) < 5 and abs(v - cam["h"] / 2) < 5


def test_the_close_up_sightline_clears_the_scrubbing_arm():
    timeline = Path(__file__).resolve().parent.parent / "runs_sample/film/timeline.jsonl"
    if not timeline.exists():
        pytest.skip("no film timeline")
    sm = pytest.importorskip("integration.scrub_motion")
    job = sm.film_job(timeline)
    shot = next(x for x in SHOTS if x.name == "unit1_scrub")
    e = np.array(camera_at_film(shot.film[0] + 0.5, SHOTS)[0])
    c = np.array(DEFAULT_FILM_SCENE.patch)
    links = sm.ARM_LINKS[:-1]
    for t in np.arange(shot.sim[0], shot.sim[1], 0.5):
        P = job.link_points(job.at(t)["q"])
        for a, b in zip(links[:-1], links[1:]):
            gap = min(np.linalg.norm(np.cross(p - e, c - e)) / np.linalg.norm(c - e)
                      for p in (P[a] + (P[b] - P[a]) * k for k in np.linspace(0, 1, 20)))
            assert gap > sm.ARM_LINK_RADIUS + 0.03, (t, a, b)


def test_follow_shots_keep_the_walking_robot_in_frame():
    for name in ("unit2_walk", "unit1_walk"):
        shot = next(x for x in SHOTS if x.name == name)
        for t in np.linspace(shot.film[0] + 1.0, shot.film[1] - 0.01, 5):
            e, g, f, _ = camera_at_film(t, SHOTS)
            u, v, z = project(view(e, g, f), g)
            assert z > 0 and abs(u - 640) < 200 and abs(v - 360) < 200


def test_the_control_room_is_framed_from_inside_and_out():
    lo, hi = np.array(CONTROL_ROOM["box"][0]), np.array(CONTROL_ROOM["box"][1])
    inside = np.array(CONTROL_ROOM["interior_eye"])
    assert np.all(lo + 0.5 < inside) and np.all(inside < hi - 0.5)
    push = next(x for x in SHOTS if x.name == "control_push")
    end = np.array(camera_at_film(push.film[1] - 1e-3, SHOTS)[0])
    assert CONTROL_ROOM["window_y"] + 7.0 <= end[1] <= CONTROL_ROOM["window_y"] + 9.0
    e, g, f, _ = camera_at_film(push.film[1] - 1e-3, SHOTS)
    cam = view(e, g, f)
    for w0, w1 in CONTROL_ROOM["north_windows"]:
        for p in (w0, w1):
            u, v, z = project(cam, p)
            assert z > 0 and 0.05 * cam["w"] < u < 0.95 * cam["w"] and 0.05 * cam["h"] < v < 0.95 * cam["h"]


def test_the_camera_keeps_clear_of_the_repair_zone_boxes():
    for lo, hi in DEFAULT_SCENE.obstacles():
        gap = np.linalg.norm(np.maximum(0, np.maximum(lo - EYES, EYES - hi)), axis=1)
        assert gap.min() >= 0.5, (lo, hi, TS[gap.argmin()])


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
    return T[np.all(T.max(1) > (-60, -60, 15), 1) & np.all(T.min(1) < (70, 60, 95), 1)]


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


def _hits(T, e, p, skip=0.3):
    d = p - e
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(-1)
    ok = np.abs(det) > 1e-12
    f = 1 / np.where(ok, det, 1)
    s = e - a
    u = f * (s * h).sum(-1)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * (e2 * q).sum(-1)
    return int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0.02) & (t < 1 - skip / np.linalg.norm(d))).sum())


@pytest.fixture(scope="module")
def tris():
    pytest.importorskip("pxr")
    scene = Path(__file__).resolve().parent.parent / "sim/output/rig_merged.usda"
    if not scene.exists():
        pytest.skip("rig_merged.usda missing")
    return _triangles(scene)


def test_the_film_camera_never_enters_the_rig_meshes(tris):
    lo, hi = tris.min(1), tris.max(1)
    for e in EYES[::3]:
        near = np.linalg.norm(np.maximum(0, np.maximum(lo - e, e - hi)), axis=1) < 0.5
        assert not near.any() or _dist(e, tris[near]).min() >= 0.5, e


def test_nothing_blocks_the_bolt_or_the_patch(tris):
    s = DEFAULT_FILM_SCENE
    for name, p in (("unit2_job", s.bolt), ("unit1_scrub", s.patch), ("unit1_paint", s.patch)):
        shot = next(x for x in SHOTS if x.name == name)
        e = np.array(camera_at_film(shot.film[1] - 0.01, SHOTS)[0])
        assert _hits(tris, e, np.array(p) + 0.05 * np.array(s.patch_normal)) == 0, name


def test_usd_pose_matches_the_film_camera():
    pxr = pytest.importorskip("pxr")
    stage = pxr.Usd.Stage.CreateInMemory()
    e, g, f, _ = camera_at_film(30.0, SHOTS)
    cam = apply_usd_pose(stage, "/World/Cameras/film", e, g, f)
    m = np.array(pxr.UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default()))
    assert np.allclose(m[3, :3], e) and cam.GetFocalLengthAttr().Get() == pytest.approx(f)

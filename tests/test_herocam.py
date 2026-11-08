import math

import pytest

from rig.sim.herocam import HeroCam, deck_of, fit_distance, spring, wrap

TOP, MID, LOW = 42.5, 31.25, 20.0


def cam(poses, **kw):
    h = HeroCam(**kw)
    h.poses = lambda: poses
    return h


def run(h, t0, t1, dt=1 / 24):
    shot, t = None, t0
    while t < t1 - 1e-9:
        t += dt
        shot = h.at(t)
    return shot


def horiz(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_spring_converges_without_overshoot():
    x, v, peak = 0.0, 0.0, 0.0
    for _ in range(240):
        x, v = spring(x, v, 1.0, 4.0, 1 / 24)
        peak = max(peak, x)
    assert x == pytest.approx(1.0, abs=1e-3)
    assert peak <= 1.0 + 1e-9


def test_spring_large_dt_is_stable():
    x, v = spring(0.0, 0.0, 1.0, 4.0, 10.0)
    assert x == pytest.approx(1.0, abs=1e-6)


def test_wrap():
    assert wrap(3 * math.pi) == pytest.approx(math.pi) or wrap(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap(-0.5) == pytest.approx(-0.5)
    assert wrap(2 * math.pi + 0.25) == pytest.approx(0.25)


def test_deck_of():
    assert deck_of(TOP + 0.45) == TOP
    assert deck_of(MID + 0.3) == MID
    assert deck_of(LOW) == LOW
    assert deck_of(MID - 0.4) == MID


def test_default_is_flyby():
    h = cam({})
    assert h.at(0.0) is None
    assert h.meta() == {"mode": "flyby", "robot": None}


def test_events_drive_modes():
    h = cam({"R1": (0, 0, TOP, 0.0)})
    assert h.event("select", {"robot": "R2"})
    assert h.meta() == {"mode": "chase", "robot": "R2"}
    h.event("select", {"robot": None})
    assert h.meta() == {"mode": "wide", "robot": None}
    h.event("camera", {"mode": "flyby"})
    assert h.meta()["mode"] == "flyby"
    h.event("scenario", {"id": "gas"})
    assert h.meta() == {"mode": "wide", "robot": None} and h.main == "R2"
    h.event("camera", {"mode": "chase"})
    assert h.meta() == {"mode": "chase", "robot": "R2"}
    h.event("camera", {"mode": "chase", "robot": "R3"})
    assert h.meta() == {"mode": "chase", "robot": "R3"}
    h.event("scenario", {"id": "flyby"})
    assert h.meta() == {"mode": "flyby", "robot": None} and h.main is None
    assert not h.event("camera", {"shot": "hero"})
    assert not h.event("gas_leak", {})
    assert h.meta()["mode"] == "flyby"


def test_chase_frames_robot_from_behind_and_above():
    h = cam({"R1": (5.0, 2.0, TOP, 0.0)})
    h.event("select", {"robot": "R1"})
    eye, target, focal = h.at(1.0)
    assert eye[0] < 5.0 and abs(eye[1] - 2.0) < 1e-6
    assert 3.0 <= horiz(eye, (5.0, 2.0)) <= 4.0
    assert eye[2] == pytest.approx(TOP + 1.8, abs=0.05)
    assert target[2] < eye[2]
    assert target[0] > eye[0]
    assert 10 < focal < 30


def test_chase_cut_is_instant():
    poses = {"R1": (0.0, 0.0, TOP, 0.0), "R2": (30.0, 10.0, MID, math.pi / 2)}
    h = cam(poses)
    h.event("select", {"robot": "R1"})
    run(h, 0.0, 2.0)
    h.event("select", {"robot": "R2"})
    eye, _, _ = h.at(2.0 + 1 / 24)
    assert horiz(eye, (30.0, 10.0)) == pytest.approx(3.5, abs=0.3)
    assert eye[1] < 10.0
    assert eye[2] == pytest.approx(MID + 1.8, abs=0.05)


def test_chase_follows_smoothly():
    poses = {"R1": (0.0, 0.0, TOP, 0.0)}
    h = cam(poses)
    h.event("select", {"robot": "R1"})
    prev = h.at(0.0)[0]
    steps = []
    for k in range(1, 240):
        poses["R1"] = (0.5 * (k // 12), 0.0, TOP, 0.0)
        eye = h.at(k / 24)[0]
        steps.append(horiz(eye, prev))
        prev = eye
    assert max(steps) < 0.2
    assert horiz(prev, poses["R1"]) < 5.5


def test_chase_yaw_turns_smoothly_and_wraps():
    poses = {"R1": (0.0, 0.0, TOP, math.pi - 0.1)}
    h = cam(poses)
    h.event("select", {"robot": "R1"})
    a = h.at(0.0)[0]
    poses["R1"] = (0.0, 0.0, TOP, -math.pi + 0.1)
    b = h.at(1 / 24)[0]
    assert horiz(a, b) < 0.1
    end = run(h, 1 / 24, 4.0)[0]
    assert end[0] > 3.0 and end[1] == pytest.approx(-3.5 * math.sin(-math.pi + 0.1), abs=0.1)


def test_chase_teleport_cuts():
    poses = {"R1": (0.0, 0.0, TOP, 0.0)}
    h = cam(poses)
    h.event("select", {"robot": "R1"})
    run(h, 0.0, 1.0)
    poses["R1"] = (40.0, 0.0, LOW, 0.0)
    eye = h.at(1.0 + 1 / 24)[0]
    assert horiz(eye, (40.0, 0.0)) == pytest.approx(3.5, abs=0.3)


def test_chase_picks_clear_side_offset():
    def blocked(a, b):
        return 1.0 if b[0] < a[0] - 1.0 and abs(b[1] - a[1]) < 1.0 else None

    h = cam({"R1": (0.0, 0.0, TOP, 0.0)})
    h.blocked = blocked
    h.event("select", {"robot": "R1"})
    eye = run(h, 0.0, 3.0)[0]
    assert abs(eye[1]) > 1.0
    assert 2.5 < horiz(eye, (0, 0)) < 4.0


def test_chase_pulls_in_when_everything_blocked():
    h = cam({"R1": (0.0, 0.0, TOP, 0.0)})
    h.blocked = lambda a, b: 1.5
    h.event("select", {"robot": "R1"})
    eye = run(h, 0.0, 3.0)[0]
    assert horiz(eye, (0, 0)) < 1.6
    assert eye[2] >= TOP + 1.0


def test_chase_unknown_robot_falls_back_to_wide_or_flyby():
    h = cam({"R1": (0.0, 0.0, TOP, 0.0)})
    h.event("select", {"robot": "R9"})
    shot = h.at(0.0)
    assert shot is not None
    assert horiz(shot[0], shot[1]) > 5.0


def test_fit_distance_grows_with_extent():
    near = fit_distance([(-2, 0, 0), (2, 0, 0)], (0, 0, 0), -30, 40, 24.0)
    far = fit_distance([(-20, 0, 0), (20, 0, 0)], (0, 0, 0), -30, 40, 24.0)
    assert far > 3 * near


def test_wide_frames_main_robot_deck():
    poses = {"R1": (-12.0, 9.4, TOP, 0.0), "R2": (-21.0, 4.0, MID, 0.0), "R3": (10.0, 12.0, LOW, 0.0)}
    h = cam(poses)
    h.event("scenario", {"id": "gas"})
    eye, target, _ = h.at(0.0)
    assert deck_of(target[2]) == MID
    assert eye[2] < TOP - 1.0
    h.event("scenario", {"id": "repair"})
    eye, target, _ = h.at(0.1)
    assert deck_of(target[2]) == TOP
    assert eye[2] > target[2] + 5.0


def test_wide_busiest_deck_without_main():
    poses = {"R1": (0.0, 0.0, MID, 0.0), "R2": (5.0, 0.0, MID, 0.0), "R3": (10.0, 12.0, TOP, 0.0)}
    h = cam(poses)
    h.event("camera", {"mode": "wide"})
    _, target, _ = h.at(0.0)
    assert deck_of(target[2]) == MID


def test_wide_contains_robots_and_routes():
    poses = {"R1": (0.0, 0.0, TOP, 0.0), "R2": (6.0, 3.0, TOP, 0.0)}
    h = cam(poses)
    h.fleet({"robots": [{"id": "R1", "pose": {"x": 0, "y": 0, "z": TOP}, "path": [[0, 0, TOP], [0, 12, TOP]]}]})
    h.event("camera", {"mode": "wide"})
    eye, target, focal = h.at(0.0)
    for p in [(0, 0, TOP), (6, 3, TOP), (0, 12, TOP)]:
        assert visible(eye, target, focal, p)


def test_wide_is_steady_while_robot_walks():
    poses = {"R1": (0.0, 0.0, TOP, 0.0), "R2": (6.0, 3.0, TOP, 0.0)}
    h = cam(poses)
    h.fleet({"robots": [{"id": "R1", "path": [[0, 0, TOP], [0, 12, TOP]]}]})
    h.event("camera", {"mode": "wide"})
    a = h.at(0.0)[0]
    poses["R1"] = (0.0, 6.0, TOP, math.pi / 2)
    b = run(h, 0.0, 1.0)[0]
    assert math.dist(a, b) < 1.0


def test_fleet_poses_used_without_prims():
    h = HeroCam()
    h.fleet({"robots": [{"id": "R1", "pose": {"x": 3.0, "y": 1.0, "z": TOP, "yaw": 0.0}, "path": []}]})
    h.event("select", {"robot": "R1"})
    eye, _, _ = h.at(0.0)
    assert horiz(eye, (3.0, 1.0)) == pytest.approx(3.5, abs=0.3)


def visible(eye, target, focal, p, aperture=20.955, aspect=16 / 9):
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    r = [f[1], -f[0], 0.0]
    rn = math.hypot(r[0], r[1])
    r = [c / rn for c in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    d = [a - b for a, b in zip(p, eye)]
    z = sum(a * b for a, b in zip(d, f))
    x = sum(a * b for a, b in zip(d, r))
    y = sum(a * b for a, b in zip(d, u))
    tx = aperture / 2 / focal
    return z > 0 and abs(x / z) <= tx and abs(y / z) <= tx / aspect


def test_wide_turns_or_pulls_in_when_view_blocked():
    poses = {"R2": (0.0, 0.0, MID, 0.0)}
    h = cam(poses)
    h.event("scenario", {"id": "gas"})
    free_az = []

    def blocked(a, b):
        az = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        if abs(((az - 145.0) + 180) % 360 - 180) < 1.0:
            free_az.append(az)
            return None
        return 5.0

    h.blocked = blocked
    eye, look, _ = run(h, 0.0, 8.0)
    assert free_az
    az = math.degrees(math.atan2(eye[1] - look[1], eye[0] - look[0]))
    assert abs(((az - 145.0) + 180) % 360 - 180) < 2.0


def test_wide_pulls_in_when_all_sides_blocked():
    h = cam({"R2": (0.0, 0.0, MID, 0.0)})
    h.event("scenario", {"id": "gas"})
    h.blocked = lambda a, b: 9.0
    eye, look, _ = run(h, 0.0, 8.0)
    assert math.dist(eye, look) < 9.0


def test_first_blocker_combines_and_caches_mesh_per_cell():
    from rig.sim.herocam import first_hit, mesh_ray

    assert first_hit(lambda a, b: None, lambda a, b: 3.0)((0, 0, 0), (9, 0, 0)) == 3.0
    assert first_hit(lambda a, b: 2.0, lambda a, b: 3.0)((0, 0, 0), (9, 0, 0)) == 2.0
    assert first_hit(None, lambda a, b: None)((0, 0, 0), (9, 0, 0)) is None
    built = []

    class Caster:
        def cast(self, o, dirs, max_t):
            return [4.0 if dirs[0][0] > 0 else -1.0]

    def build(center):
        built.append(tuple(center))
        return Caster()

    ray = mesh_ray(build, cell=20.0)
    assert ray((0, 0, 30), (10, 0, 30)) == 4.0
    assert ray((1, 1, 30), (-10, 0, 30)) is None
    assert ray((2, 0, 30), (2, 0, 30)) is None
    assert len(built) == 1
    ray((80, 0, 30), (90, 0, 30))
    assert len(built) == 2

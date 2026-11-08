import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from integration.kinematics import LIFT_URDF
from integration.scrub_motion import (
    CONE,
    FACE_GAP,
    JOINTS,
    PAD_RADIUS,
    SPRING_K,
    WEAR,
    ScrubJob,
    scrub_track,
    windows_from_timeline,
    wrap,
)
from policy.brush_squish import compression, pad_thickness

FILM_RUST = [(77.6, 1.0), (92.0, 0.4138), (96.0, 0.1379), (100.0, 0.0)]
FILM = {"deploy": 77.0, "spray": 79.2, "soak": 82.3, "scrub": 88.0, "verify": 107.9, "paint": 109.5, "stow": 113.6}


@pytest.fixture(scope="module")
def job():
    j = ScrubJob(scrub_pose=(18.225, -5.262, 0.0))
    j.plan(t_deploy=FILM["deploy"], t_spray=FILM["spray"], t_soak=FILM["soak"], t_scrub=FILM["scrub"],
           t_verify=FILM["verify"], t_paint=FILM["paint"], t_done=FILM["stow"], rust_track=FILM_RUST)
    return j


@pytest.fixture(scope="module")
def ticks(job):
    p = job.plan_
    return [job.at(t) for t in np.arange(p.t[0], p.t[-1], 1 / 30)]


def radial(job, x):
    rel = np.asarray(x) - job.pipe.origin
    v = rel - (rel @ job.pipe.axis) * job.pipe.axis
    return float(np.linalg.norm(v)), v / np.linalg.norm(v), float(rel @ job.pipe.axis)


def contacts(ticks):
    return [s for s in ticks if s["brush_contact"]["force_N"] > 0]


def test_the_brush_rides_the_pipe_surface_within_the_squish(job, ticks):
    r, thick = job.pipe.radius, pad_thickness(SPRING_K)
    cs = contacts(ticks)
    assert len(cs) > 30 * 15
    for s in cs:
        rho, _, _ = radial(job, s["tip"]["xyz"])
        depth = float(compression(s["brush_contact"]["force_N"])) * thick
        face = rho + depth
        assert face - r <= 0.005 + 1e-4
        assert face - r >= 0.0
        assert rho >= r - depth - 1e-3


def test_the_brush_presses_and_eases_within_the_pad_range(job, ticks):
    work = [s for s in contacts(ticks) if FILM["scrub"] <= s["t"] <= FILM["verify"] - 2.0]
    f = np.array([s["brush_contact"]["force_N"] for s in work])
    assert 3.0 - 1e-6 <= f.min() and f.max() <= 35.0
    assert (f >= 15).mean() > 0.1
    p = job.plan_
    assert np.abs(np.diff(p.force)).max() <= 40.0 * job.dt + 1e-6
    assert 0.3 < max(s["brush_contact"]["squish"] for s in work) <= 0.6


def test_render_rust_tracks_the_timeline_fraction(job, ticks):
    tr = np.array([(FILM["scrub"], 1.0)] + [p for p in FILM_RUST if p[0] > FILM["scrub"]])
    for s in ticks:
        if FILM["scrub"] <= s["t"] <= FILM["verify"]:
            assert abs(s["rust_fraction"] - np.interp(s["t"], tr[:, 0], tr[:, 1])) < 0.08


def test_the_pad_face_is_normal_to_the_pipe(job, ticks):
    for s in contacts(ticks):
        _, n, _ = radial(job, s["tip"]["xyz"])
        ang = math.degrees(math.acos(np.clip(-np.asarray(s["tip"]["approach"]) @ n, -1, 1)))
        assert ang < 10.0


def test_joints_move_continuously_without_a_branch_switch(job, ticks):
    q = np.array([[s["q"][j] for j in JOINTS] for s in ticks])
    step = np.abs(np.diff(q, axis=0))
    assert step[:, 0].max() <= 0.15 / 30 + 1e-6
    assert step[:, 1:].max() < 0.11
    assert (q[:, JOINTS.index("ur3_elbow_joint")] > 0.15).all()
    assert (q[:, JOINTS.index("ur3_wrist_2_joint")] < -0.15).all()


def test_joint_speeds_stay_under_the_urdf_limits(job):
    root = ET.parse(LIFT_URDF).getroot()
    limit = {j.get("name"): float(j.find("limit").get("velocity")) for j in root.findall("joint")
             if j.find("limit") is not None}
    v = np.abs(np.diff(job.plan_.q, axis=0)).max(0) / job.dt
    for name, speed in zip(JOINTS, v):
        assert speed <= limit[name] + 1e-6, name


def test_the_track_matches_the_render(job):
    tr = scrub_track(job)
    assert tr["fps"] == 30 and tr["rows"][0]["t"] == pytest.approx(job.plan_.t[0])
    for row in tr["rows"][::15]:
        s = job.at(row["t"])
        assert row["rust_fraction"] == pytest.approx(s["rust_fraction"], abs=1e-3)
        assert row["force_N"] == pytest.approx(s["brush_contact"]["force_N"], abs=0.01)
        assert row["phase"] == s["phase"]


def test_the_arm_never_passes_through_the_pipe(job):
    p = job.plan_
    assert min(job.arm_clearance(job.qdict(q)) for q in p.q[::6]) > 0.0


def test_rust_only_falls_under_the_brush(job):
    p = job.plan_
    r = job.pipe.radius
    for k in range(1, len(p.t)):
        dec = p.rust[k] < p.rust[k - 1] - 1e-9
        if not dec.any():
            continue
        T = job.fk(p.q[k])
        assert p.force[k] >= WEAR["min_force"]
        _, n, z = radial(job, T[:3, 3])
        th = math.atan2(float(n @ job.pipe.binormal), float(n @ job.pipe.ref))
        d = np.hypot(wrap(job.cell_theta - th) * r, job.cell_z - z)
        assert (d[dec] < PAD_RADIUS + 1e-3).all()


def nozzle_union(job, phase):
    p = job.plan_
    seen = np.zeros(job.rust0.shape, bool)
    for k in np.flatnonzero(p.nozzle_on):
        if p.phase[k] != phase:
            continue
        T = job.fk(p.q[k])
        v = job.cell_p - T[:3, 3]
        d = np.linalg.norm(v, axis=-1)
        seen |= ((v @ T[:3, 0]) >= math.cos(math.atan(CONE)) * d * 0.999) & ((job.cell_n @ T[:3, 0]) < 0)
    return seen


def test_spray_and_paint_land_only_where_the_nozzle_passed(job):
    p = job.plan_
    rusty = job.rust0 > WEAR["rust_eps"]
    sprayed, painted = p.sprayed[-1], p.painted[-1]
    assert not (sprayed & ~nozzle_union(job, "spray")).any()
    assert not (painted & ~nozzle_union(job, "paint")).any()
    assert sprayed[rusty].all() and painted[rusty].all()
    k_scrub = int(np.searchsorted(p.t, FILM["scrub"]))
    assert not p.painted[k_scrub].any()


def test_the_nozzle_sprays_from_ten_to_fifteen_centimetres(job, ticks):
    on = [s for s in ticks if s["nozzle"]["on"]]
    assert on
    for s in on:
        rho, n, _ = radial(job, s["nozzle"]["xyz"])
        assert 0.09 <= rho - job.pipe.radius <= 0.15
        assert -np.asarray(s["nozzle"]["dir"]) @ n > math.cos(math.radians(40))


def test_the_scrub_ends_with_under_five_percent_rust(job):
    p = job.plan_
    k = int(np.searchsorted(p.t, FILM["verify"])) - 1
    assert p.rust[k].sum() / job.rust0.sum() < 0.05
    assert p.clean_t is not None and 105.5 <= p.clean_t <= 107.6
    assert 0.1 <= p.speeds["scrub"] <= 0.2
    assert not p.warnings


def test_rust_is_untouched_before_the_scrub(job):
    p = job.plan_
    k = int(np.searchsorted(p.t, FILM["scrub"]))
    assert np.allclose(p.rust[k], job.rust0)


def test_stowed_outside_the_job(job):
    a, b = job.at(job.plan_.t[0] - 5), job.at(job.plan_.t[-1] + 5)
    assert a["phase"] == "idle" and b["phase"] == "done"
    assert a["q"]["lift_joint"] == 0.0 and not b["nozzle"]["on"] and b["brush_contact"]["force_N"] == 0.0
    assert b["rust_fraction"] < 0.05


def test_wear_constants_match_the_repo_model():
    torch = pytest.importorskip("torch")
    assert torch
    from policy.envs.rust_state import AbrasionParams

    a = AbrasionParams()
    assert (a.k, a.min_force, a.mult_soaked, a.mult_sprayed, a.max_slide_speed) == (
        WEAR["k"], WEAR["min_force"], WEAR["mult_soaked"], WEAR["mult_sprayed"], WEAR["max_slide_speed"])


def test_windows_from_timeline_rows():
    rows = [{"t": t, "robots": {"unit1": {"state": s}}} for t, s in (
        (76.9, "PATROL"), (77.0, "DETECT"), (77.6, "APPROACH"), (79.2, "SPRAY"), (82.3, "SOAK"), (88.0, "SCRUB"),
        (107.9, "VERIFY"), (109.5, "PAINT"), (113.6, "VERIFY_PAINT"), (114.7, "DONE"))]
    assert windows_from_timeline(rows) == FILM


def test_face_gap_sits_above_the_overlays():
    assert 0.0036 < FACE_GAP <= 0.005

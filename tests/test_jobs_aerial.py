import json
import math

import numpy as np
import pytest
from PIL import Image

from fleet.aerial import thermal
from fleet.aerial.flight import (DroneSpec, Wind, check_path, flare_geometry, flare_no_fly, flare_viewpoint,
                                 plan_survey, plan_visit, rig_bounds)
from fleet.aerial.gemini import GeminiVision
from fleet.aerial.patches import (RI_PCT, TESTDATA, PatchSpec, accuracy, grade_classical, load_image, ri_from_pct)
from fleet.contract import REGISTRY, Job, JobExecutor, JobStatus, Level, Robot, RobotKind, World
from fleet.jobs.aerial import CorrosionSurvey, FlareTipInspection, demo_world, fuse, q_reachable
from fleet.jobs.crawler import CoatingTouchup, UtThickness, mark_repaired

REPLIES = TESTDATA / "gemini_replies.json"
ROWS = json.loads((TESTDATA / "patches.json").read_text(encoding="utf-8"))
HELIDECK = (0.0, 0.0, 81.3)


def drone(**kw):
    return Robot("a1", RobotKind.AERIAL, {"camera", "thermal"}, position=HELIDECK, speed_mps=6.0, **kw)


def crawler():
    return Robot("c1", RobotKind.CRAWLER, {"spray", "ut_probe", "camera"}, position=(-30, 0, 42.5), speed_mps=0.1)


def replay_vision():
    return GeminiVision(replay=REPLIES)


def test_flare_tip_is_read_from_cinematic_source(tmp_path):
    g = flare_geometry()
    assert g.tip == (-58.0, 8.0, 70.0) and g.base == (-26.0, 8.0, 42.0)
    assert g.flame_top_z > g.tip[2] and g.flame_reach_xy > 0
    broken = tmp_path / "cinematic.py"
    broken.write_text("def build_flare(stage):\n    pass\n")
    with pytest.raises(ValueError):
        flare_geometry(broken)


@pytest.mark.parametrize("wind", [Wind(0, 0), Wind(8, 0), Wind(8, 180), Wind(10, 90), Wind(11, 300)])
def test_survey_never_enters_no_fly_or_rig(wind):
    plan = plan_survey(HELIDECK, wind)
    nofly = flare_no_fly(wind)
    lo, hi = rig_bounds()
    nf, bx = check_path(plan.waypoints[:-1], nofly, lo, hi, skip_first=1)
    assert nf > 0 and plan.min_nofly_clearance_m > 0, plan.summary()
    assert bx > 0
    assert plan.detours >= 1
    assert plan.feasible and plan.length_m > 1500 and plan.time_s > 0


def test_check_catches_a_path_through_the_flame():
    wind = Wind(0, 0)
    nofly = flare_no_fly(wind)
    tip = flare_geometry().tip
    lo, hi = rig_bounds()
    nf, _ = check_path([(-90.0, 8.0, tip[2]), (-70.0, 8.0, tip[2]), (tip[0], tip[1], tip[2] + 5)], nofly, lo, hi)
    assert nf < 0


def test_wind_and_battery_limits():
    assert not plan_survey(HELIDECK, Wind(14, 0)).feasible
    assert not plan_survey(HELIDECK, Wind(5, 0, gust_mps=18)).feasible
    low = plan_survey(HELIDECK, Wind(4, 0), battery=0.35)
    assert not low.feasible and any("battery" in r for r in low.reasons)
    calm, windy = plan_survey(HELIDECK, Wind(0, 0)), plan_survey(HELIDECK, Wind(10, 0))
    assert windy.energy_wh > calm.energy_wh


def test_flare_viewpoint_is_outside_no_fly_and_upwind():
    for wind in (Wind(6, 0), Wind(6, 200)):
        nofly, g = flare_no_fly(wind), flare_geometry()
        vp = flare_viewpoint(nofly, g, wind)
        assert nofly.clearance(vp) > 0
        plan = plan_visit(HELIDECK, [vp], wind)
        assert plan.feasible and plan.min_nofly_clearance_m > 0


@pytest.mark.parametrize("case,nozzle,kind", [("hot", 5, "hot"), ("cold", 3, "cold"), ("healthy", None, None)])
def test_flare_tip_inspection_finds_the_degraded_nozzle(case, nozzle, kind):
    world = demo_world(degraded_nozzle=nozzle, kind=kind or "hot")
    job = Job("flare-1", "flare_tip_inspection", "flare_tip", flare_geometry().tip, Level.AUTO)
    r = FlareTipInspection(use_default_vision=False).run(job, drone(), world)
    if nozzle is None:
        assert r.status == JobStatus.DONE and not r.findings
        return
    assert r.status == JobStatus.FINDING
    (f,) = r.findings
    assert f.kind == f"flare_nozzle_{kind}" and f.evidence["nozzles"]["anomalies"][0]["nozzle"] == nozzle
    assert f.follow_up[0].level == Level.HUMAN and f.follow_up[0].kind == "flare_tip_maintenance"


def test_thermal_detector_over_seeds_and_ranges():
    hits, false_alarms = 0, 0
    for seed in range(10):
        for d in (24.0, 32.0):
            temps = [560.0] * 8
            temps[seed % 8] = 820.0 if seed % 2 else 180.0
            hits += [a["nozzle"] for a in thermal.detect(thermal.render(thermal.FlareTipState(temps), d, 0.4, seed),
                                                         8, 0.45, d).anomalies] == [seed % 8 + 1]
            healthy = thermal.FlareTipState(list(np.random.default_rng(seed).normal(560, 8, 8)))
            false_alarms += len(thermal.detect(thermal.render(healthy, d, 0.4, seed), 8, 0.45, d).anomalies)
    assert hits == 20 and false_alarms == 0


def test_gemini_thermal_second_opinion_recorded():
    v = replay_vision()
    for name, (nozzle, kind) in {"hot": (5, "hot"), "cold": (3, "cold"), "healthy": (None, "none")}.items():
        op = v.thermal_opinion(np.asarray(Image.open(TESTDATA / f"thermal_{name}.png").convert("RGB")))
        assert op is not None and op["kind"] == kind
        if nozzle:
            assert op["anomalous_nozzle"] == nozzle


def test_ri_scale():
    assert [ri_from_pct(p) for p in RI_PCT.values()] == [0, 1, 2, 3, 4, 5]
    assert ri_from_pct(60.0) == 5 and ri_from_pct(0.01) == 0


def test_classical_grader_on_test_images():
    pairs = [(r["true_ri"], grade_classical(load_image(TESTDATA / r["file"]))[0].ri) for r in ROWS["patches"]]
    acc = accuracy(pairs)
    print("classical, 18 ISO-nominal patches:", acc)
    assert acc["n"] == 18 and acc["exact"] >= 0.9 and acc["within_1"] == 1.0


def test_classical_grader_on_random_in_class_areas():
    from fleet.aerial.patches import BASES, FILLS, RI_EDGES
    rng = np.random.default_rng(42)
    edges = [0.0] + RI_EDGES + [55.0]
    pairs = []
    for i in range(36):
        ri = i % 6
        lo, hi = max(edges[ri], 0.025), edges[ri + 1]
        pct = 0.0 if ri == 0 else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        spec = PatchSpec(f"r{i}", pct, BASES[i % 5], FILLS[i % 2], seed=5000 + i)
        img, mask = spec.render()
        pairs.append((ri_from_pct(100 * mask.mean()), grade_classical(img)[0].ri))
    acc = accuracy(pairs)
    print("classical, 36 random in-class patches:", acc)
    assert acc["exact"] >= 0.75 and acc["within_1"] == 1.0


def test_gemini_grader_on_test_images_recorded():
    v = replay_vision()
    pairs, fused = [], []
    for r in ROWS["patches"]:
        img = load_image(TESTDATA / r["file"])
        g = v.grade_ri(img)
        assert g is not None, v.last_error
        pairs.append((r["true_ri"], g.ri))
        fused.append((r["true_ri"], fuse(grade_classical(img)[0].ri, g)[0]))
    acc, facc = accuracy(pairs), accuracy(fused)
    print("gemini (recorded), 18 patches:", acc, "fused:", facc)
    assert acc["n"] == 18 and acc["exact"] >= 0.8 and acc["within_1"] == 1.0
    assert all(v.grade_ri(load_image(TESTDATA / r["file"])).ri == 0 for r in ROWS["repaired"])


def test_gemini_offline_fallback_uses_classical():
    survey = CorrosionSurvey(vision=GeminiVision(api_key=""), use_default_vision=False)
    survey.vision.api_key = None
    g = survey.grade(load_image(TESTDATA / "ri4_1.jpg"))
    assert g["gemini"] is None and g["ri"] == 4 and g["fusion"] == "classical only"


def test_gemini_reply_parsing_with_fake_transport():
    reply = {"candidates": [{"content": {"parts": [{"text": json.dumps(
        {"ri_grade": 3, "rust_percent": 1.2, "confidence": 0.8, "reason": "spots"})}]}}]}
    v = GeminiVision(api_key="x", transport=lambda url, body: reply)
    g = v.grade_ri(np.zeros((32, 32, 3), np.uint8))
    assert g.ri == 3 and g.rust_pct == 1.2
    bad = GeminiVision(api_key="x", transport=lambda url, body: {"candidates": []})
    assert bad.grade_ri(np.zeros((32, 32, 3), np.uint8)) is None


def test_dispatch_rule_q_vs_c():
    p = demo_world().state["topsides_patches"]
    assert q_reachable(p["ri4_1"]) and not q_reachable(p["ri5_0"]) and not q_reachable(p["ri2_0"])


def q_rust_repair_stub(job, robot, world):
    from fleet.contract import Finding, JobResult
    from fleet.jobs.crawler import verify_job
    pid = job.params["patch_id"]
    mark_repaired(world, pid, robot.id, job.id)
    pos = world.state["topsides_patches"][pid]["position"]
    return JobResult(job.id, robot.id, JobStatus.DONE, world.t, world.t + 900,
                     [Finding("rust_repair_done", "info", pid, tuple(pos), {}, [verify_job(job, pid, pos, world.t)])])


def test_chain_drone_gemini_crawler_quadruped_reimage():
    world = demo_world()
    survey = CorrosionSurvey(vision=replay_vision())
    job = Job("survey-1", "corrosion_survey", "topsides", HELIDECK, Level.AUTO)
    r1 = survey.run(job, drone(), world)
    assert r1.status == JobStatus.FINDING and r1.used_gemini
    grades = world.state["corrosion_grades"]
    truth = {pid: p["true_ri"] for pid, p in world.state["topsides_patches"].items()}
    assert {pid: g["ri"] for pid, g in grades.items()} == truth
    follow = {f.follow_up[0].target: f.follow_up[0] for f in r1.findings}
    assert set(follow) == {"ri4_1", "ri5_0", "ri3_2"}
    assert follow["ri4_1"].kind == "rust_repair" and follow["ri4_1"].robot_kinds == {RobotKind.QUADRUPED}
    assert follow["ri5_0"].kind == "coating_touchup" and follow["ri5_0"].robot_kinds == {RobotKind.CRAWLER}
    assert all(j.level == Level.SUPV and j.parent == "survey-1" for j in follow.values())

    touchup = CoatingTouchup()
    verify_jobs = []
    for target, j in follow.items():
        world.t += 60
        if j.kind == "coating_touchup":
            j.params["approved"] = True
            r = touchup.run(j, crawler(), world)
            assert r.status == JobStatus.DONE and r.findings[0].evidence["covered_frac"] >= 0.99
        else:
            r = q_rust_repair_stub(j, Robot("q1", RobotKind.QUADRUPED, {"arm", "camera"}), world)
        (vj,) = r.findings[0].follow_up
        assert vj.kind == "corrosion_survey" and vj.robot_kinds == {RobotKind.AERIAL}
        verify_jobs.append(vj)

    for vj in verify_jobs:
        r = survey.run(vj, drone(), world)
        assert r.status == JobStatus.DONE and r.findings[0].kind == "repair_verified", r.log
        assert r.used_gemini
    assert all(world.state["topsides_patches"][t].get("verified") for t in follow)


def test_reimage_flags_a_repair_that_did_not_hold():
    world = demo_world()
    survey = CorrosionSurvey(vision=replay_vision())
    j = Job("v-1", "corrosion_survey", "ri5_0", HELIDECK, Level.AUTO, params={"patch_ids": ["ri5_0"], "verify_of": "x"})
    r = survey.run(j, drone(), world)
    assert r.status == JobStatus.FINDING and r.findings[0].kind == "repair_not_verified"
    assert r.findings[0].follow_up[0].kind == "coating_touchup"


def test_coating_touchup_needs_approval_and_covers_patch():
    world = demo_world()
    j = Job("t-1", "coating_touchup", "ri5_0", (0, 0, 0), Level.SUPV, params={"patch_id": "ri5_0", "approved": False})
    assert CoatingTouchup().run(j, crawler(), world).status == JobStatus.FAILED
    j.params["approved"] = True
    r = CoatingTouchup().run(j, crawler(), world)
    ev = r.findings[0].evidence
    print("touch-up ri5_0:", {k: ev[k] for k in ("prep_area_m2", "passes", "path_m", "covered_frac",
                                                   "two_coat_frac", "paint_l", "spray_s", "prep_s")})
    assert ev["covered_frac"] >= 0.99 and ev["prep_area_m2"] > ev["rust_area_m2"]
    assert world.state["topsides_patches"]["ri5_0"]["repaired"]


def ut_world(pit_depth):
    w = World()
    w.state["vessels"] = {"vessel_11_52": {"nominal_mm": 12.7, "ca_mm": 3.0, "radius_m": 1.2, "height_m": 4.0,
                                           "uniform_loss_mm": 0.6,
                                           "pits": [{"theta_deg": 130, "z_m": 1.4, "depth_mm": pit_depth,
                                                     "sigma_m": 0.12}]}}
    return w


def test_ut_thickness_allowance_check():
    job = Job("ut-1", "ut_thickness", "vessel_11_52", (11.0, 52.0, 31.25), Level.AUTO)
    ok = UtThickness().run(job, crawler(), ut_world(0.5))
    assert ok.status == JobStatus.DONE
    near = UtThickness().run(job, crawler(), ut_world(2.0))
    assert near.status == JobStatus.FINDING and near.findings[0].kind == "wall_loss_near_allowance"
    bad = UtThickness().run(job, crawler(), ut_world(3.5))
    f = bad.findings[0]
    assert f.kind == "wall_below_corrosion_allowance" and f.severity == "critical"
    kinds = {j.kind: j for j in f.follow_up}
    assert kinds["ffs_assessment"].level == Level.HUMAN
    regrid = kinds["ut_thickness"]
    r2 = UtThickness().run(regrid, crawler(), ut_world(3.5))
    assert r2.findings[0].kind == "wall_below_corrosion_allowance" and not r2.findings[0].follow_up[1:]
    assert abs(r2.findings[0].evidence["min_at"]["theta_deg"] - 130) < 10
    assert r2.findings[0].evidence["min_mm"] <= f.evidence["min_mm"] + 0.1


def test_registered_and_match_contract():
    for kind, lvl in {"flare_tip_inspection": Level.AUTO, "corrosion_survey": Level.AUTO,
                      "coating_touchup": Level.SUPV, "ut_thickness": Level.AUTO}.items():
        ex = REGISTRY[kind]
        assert isinstance(ex, JobExecutor) and ex.level == lvl
    r = REGISTRY["corrosion_survey"].run(Job("s", "corrosion_survey", "x", HELIDECK, Level.AUTO), drone(), demo_world())
    json.dumps(r.to_json(), default=lambda o: sorted(o) if isinstance(o, set) else str(o))

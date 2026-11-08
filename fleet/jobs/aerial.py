"""Aerial drone (A) jobs: flare tip thermal inspection and topsides corrosion survey (ISO 4628-3 Ri).

    flare_tip_inspection  A + thermal  AUTO   fly outside the flare no-fly volume, image the lit tip,
                                              find a hot / cold burner nozzle -> HUMAN maintenance job
    corrosion_survey      A + camera   AUTO   image coating patches, grade Ri 0-5 (classical + Gemini);
                                              Ri >= 3 -> rust_repair (Q) if on a pipe Q can reach from a deck,
                                              otherwise coating_touchup (C). With params["verify_of"] it
                                              re-images repaired patches and confirms them (closes the loop).

World.state used (simulated truth for mock runs; see fleet/jobs/AERIAL.md):
    wind               {"speed_mps", "toward_deg", "gust_mps"}
    flare              {"nozzle_temps_c": [...], "nozzle_pitch_m"}
    topsides_patches   {patch_id: {"rust_pct", "base", "fill", "seed", "position", "on_pipe", "size_m",
                                    "repaired", ...}}

Gemini: by default the executors use recorded Gemini replies (fleet/testdata/aerial/gemini_replies.json),
so runs are offline and repeatable; set FLEET_GEMINI_LIVE=1 to call Gemini live. Unknown images or no
reply -> the classical grade alone (edge autonomy is kept).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from fleet.aerial import thermal
from fleet.aerial.flight import (DroneSpec, Wind, deck_heights, flare_geometry, flare_no_fly, flare_viewpoint,
                                 plan_visit, viewpoint_for)
from fleet.aerial.gemini import GeminiVision
from fleet.aerial.patches import TESTDATA, PatchSpec, grade_classical, load_image
from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register

REPLIES = TESTDATA / "gemini_replies.json"
Q_REACH_ABOVE_DECK_M = (0.2, 1.8)     # Go2 + lift + UR3 working band above the deck it stands on


def default_vision() -> Optional[GeminiVision]:
    if os.environ.get("FLEET_GEMINI_LIVE") == "1":
        return GeminiVision()
    return GeminiVision(replay=REPLIES) if REPLIES.exists() else None


def _drone_spec(robot: Robot) -> DroneSpec:
    """Robot.speed_mps is the cruise ground speed when it is set to a drone-like value (>= 3 m/s)."""
    return DroneSpec(airspeed_mps=robot.speed_mps) if robot.speed_mps >= 3.0 else DroneSpec()


def _flight_failed(job: Job, robot: Robot, world: World, plan, what: str) -> JobResult:
    return JobResult(job.id, robot.id, JobStatus.FAILED, world.t, world.t,
                     findings=[Finding("flight_not_feasible", "info", f"{what}: " + "; ".join(plan.reasons),
                                       robot.position, {"plan": plan.summary()})],
                     log=[f"flight plan rejected: {plan.reasons}"])


# ---------------------------------------------------------------- flare tip
class FlareTipInspection:
    kind = "flare_tip_inspection"
    level = Level.AUTO
    needs = {"thermal"}
    robot_kinds = {RobotKind.AERIAL}

    def __init__(self, vision: Optional[GeminiVision] = None, use_default_vision: bool = True):
        self.vision = vision if vision is not None else (default_vision() if use_default_vision else None)

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        log: list[str] = []
        wind = Wind.from_state(world.state)
        geom = flare_geometry()
        nofly = flare_no_fly(wind, geom)
        vp = flare_viewpoint(nofly, geom, wind)
        plan = plan_visit(robot.position, [vp], wind, _drone_spec(robot), robot.battery, dwell_s=30.0, nofly=nofly)
        log.append(f"flare tip {geom.tip} from {geom.source}; no-fly r={nofly.radius:.1f} m above z={nofly.z_min:.0f}")
        log.append(f"plan {plan.summary()}")
        if not plan.feasible:
            return _flight_failed(job, robot, world, plan, "flare tip inspection not flown")
        distance = float(math.dist(vp, geom.tip))
        tip_state = thermal.FlareTipState.from_state(world.state)
        n = len(tip_state.nozzle_temps_c)
        # wind across the camera's view tilts the flame in the image
        view = np.array(geom.tip[:2]) - np.array(vp[:2])
        right = np.array([view[1], -view[0]]) / max(np.linalg.norm(view), 1e-6)
        across = float(np.dot(wind.vec, right)) / 10.0
        frame = thermal.render(tip_state, distance, across, seed=int(job.params.get("seed", 0)))
        report = thermal.detect(frame, n, tip_state.nozzle_pitch_m, distance)
        log.append(f"nozzles {report.summary()}")
        evidence = {"viewpoint": [round(v, 2) for v in vp], "distance_m": round(distance, 1),
                    "nozzles": report.summary(), "plan": plan.summary()}
        used_gemini = False
        if self.vision is not None and self.vision.available and job.params.get("gemini", True):
            opinion = self.vision.thermal_opinion(thermal.false_colour(frame, n, tip_state.nozzle_pitch_m, distance))
            if opinion is not None:
                used_gemini = True
                evidence["gemini"] = opinion
                ours = report.anomalies[0]["nozzle"] if report.anomalies else -1
                theirs = opinion["anomalous_nozzle"] if opinion["kind"] != "none" and opinion["anomalous_nozzle"] > 0 else -1
                evidence["gemini_agrees"] = ours == theirs
                log.append(f"gemini thermal opinion {opinion['kind']} nozzle {theirs}; agrees={ours == theirs}")
        findings = []
        for a in report.anomalies:
            sev = "critical" if a["kind"] == "hot" and a["delta_c"] > 200 else "warning"
            what = "running hot (burn-back / internal burning)" if a["kind"] == "hot" else "cold (plugged or unlit)"
            follow = Job(f"{job.id}-maint-n{a['nozzle']}", "flare_tip_maintenance", f"flare_nozzle_{a['nozzle']:02d}",
                         geom.tip, Level.HUMAN, priority=2 if sev == "critical" else 3, created_t=world.t,
                         parent=job.id, params={"nozzle": a["nozzle"], "kind": a["kind"], "delta_c": a["delta_c"],
                                                "note": "flare tip work needs a flare shutdown; plan with operations"})
            findings.append(Finding(f"flare_nozzle_{a['kind']}", sev,
                                    f"Flare nozzle {a['nozzle']} {what}: {a['temp_c']:.0f} C vs median "
                                    f"{report.median_c:.0f} C ({a['delta_c']:+.0f} C)", geom.tip, evidence, [follow]))
        status = JobStatus.FINDING if findings else JobStatus.DONE
        return JobResult(job.id, robot.id, status, world.t, world.t + plan.time_s, findings, log, used_gemini)


# ---------------------------------------------------------------- corrosion survey
def patch_spec(pid: str, p: dict) -> PatchSpec:
    return PatchSpec(pid, float(p.get("rust_pct", 0.0)), p.get("base", "grey"), p.get("fill", "rust_coarse_01"),
                     int(p.get("seed", 0)), int(p.get("size", 384)), bool(p.get("repaired", False)))


def patch_image(pid: str, p: dict) -> np.ndarray:
    """What the drone camera sees: the recorded test image for this patch if there is one, else a render."""
    if "image" in p and not p.get("repaired"):
        return load_image(TESTDATA / p["image"])
    if "image" in p and p.get("repaired"):
        repaired = TESTDATA / p["image"].replace(".jpg", "_repaired.jpg")
        if repaired.exists():
            return load_image(repaired)
    return patch_spec(pid, p).render()[0]


def q_reachable(p: dict, decks: Optional[list[float]] = None) -> bool:
    """A quadruped can reach the patch if it is on a pipe within the arm's band above some deck."""
    if not p.get("on_pipe"):
        return False
    z = float(p["position"][2])
    lo, hi = Q_REACH_ABOVE_DECK_M
    return any(lo <= z - d <= hi for d in (decks if decks is not None else deck_heights()))


def fuse(classical_ri: int, gemini) -> tuple[int, str]:
    """Final grade: agree -> that grade; disagree -> the higher one (a missed Ri >= 3 costs more than a touch-up)."""
    if gemini is None:
        return classical_ri, "classical only"
    if gemini.ri == classical_ri:
        return classical_ri, "agree"
    return max(gemini.ri, classical_ri), f"disagree (classical Ri{classical_ri}, Gemini Ri{gemini.ri}); took higher"


def repair_job(parent: Job, pid: str, p: dict, ri: int, rust_pct: float, t: float) -> Job:
    pos = tuple(float(v) for v in p["position"])
    params = {"patch_id": pid, "ri": ri, "rust_pct": round(rust_pct, 3), "size_m": p.get("size_m", [0.6, 0.6]),
              "verify_with": "corrosion_survey"}
    prio = 2 if ri >= 5 else 3
    if q_reachable(p):
        return Job(f"{parent.id}-repair-{pid}", "rust_repair", pid, pos, Level.SUPV, {"arm"}, {RobotKind.QUADRUPED},
                   prio, 900.0, params, t, parent.id)
    return Job(f"{parent.id}-touchup-{pid}", "coating_touchup", pid, pos, Level.SUPV, {"spray"}, {RobotKind.CRAWLER},
               prio, 1200.0, params, t, parent.id)


class CorrosionSurvey:
    kind = "corrosion_survey"
    level = Level.AUTO
    needs = {"camera"}
    robot_kinds = {RobotKind.AERIAL}

    def __init__(self, vision: Optional[GeminiVision] = None, use_default_vision: bool = True):
        self.vision = vision if vision is not None else (default_vision() if use_default_vision else None)

    def grade(self, image: np.ndarray, use_gemini: bool = True) -> dict:
        classical, _ = grade_classical(image)
        gem = None
        if use_gemini and self.vision is not None and self.vision.available:
            gem = self.vision.grade_ri(image)
            if gem is not None and not self.vision.confident(gem.confidence):
                gem = None
        ri, how = fuse(classical.ri, gem)
        return {"ri": ri, "fusion": how, "classical": classical.summary(),
                "gemini": gem.summary() if gem else None,
                "rust_pct": gem.rust_pct if (gem and gem.ri == ri) else classical.rust_pct}

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        patches: dict = world.state.get("topsides_patches", {})
        ids = list(job.params.get("patch_ids") or patches.keys())
        verify_of = job.params.get("verify_of")
        log: list[str] = []
        missing = [i for i in ids if i not in patches]
        ids = [i for i in ids if i in patches]
        if missing:
            log.append(f"unknown patches skipped: {missing}")
        wind = Wind.from_state(world.state)
        targets = [viewpoint_for(tuple(patches[i]["position"])) for i in ids]
        plan = plan_visit(robot.position, targets, wind, _drone_spec(robot), robot.battery, dwell_s=15.0)
        log.append(f"plan {plan.summary()}")
        if not plan.feasible:
            return _flight_failed(job, robot, world, plan, "corrosion survey not flown")
        findings: list[Finding] = []
        used_gemini = False
        grades = {}
        for pid in ids:
            p = patches[pid]
            g = self.grade(patch_image(pid, p), job.params.get("gemini", True))
            used_gemini |= g["gemini"] is not None
            grades[pid] = g
            p.setdefault("history", []).append({"t": world.t, "job": job.id, "ri": g["ri"]})
            log.append(f"{pid}: Ri{g['ri']} ({g['fusion']}) classical {g['classical']} gemini {g['gemini']}")
            pos = tuple(float(v) for v in p["position"])
            ev = {"patch_id": pid, **g, "viewpoint": targets[ids.index(pid)]}
            if verify_of:
                if g["ri"] <= 1:
                    findings.append(Finding("repair_verified", "info",
                                            f"{pid} re-imaged after {verify_of}: Ri{g['ri']}, repair confirmed", pos, ev))
                    p["verified"] = True
                else:
                    findings.append(Finding("repair_not_verified", "warning",
                                            f"{pid} still Ri{g['ri']} after {verify_of}", pos, ev,
                                            [repair_job(job, pid, p, g["ri"], g["rust_pct"], world.t)]))
            elif g["ri"] >= 3:
                follow = repair_job(job, pid, p, g["ri"], g["rust_pct"], world.t)
                who = "Q rust_repair" if follow.kind == "rust_repair" else "C coating_touchup"
                findings.append(Finding(f"corrosion_Ri{g['ri']}", "critical" if g["ri"] >= 5 else "warning",
                                        f"{pid} graded ISO 4628-3 Ri{g['ri']} (~{g['rust_pct']:.2g} % rust) -> {who}",
                                        pos, ev, [follow]))
        bad = [f for f in findings if f.kind != "repair_verified"]
        status = JobStatus.FINDING if bad else JobStatus.DONE
        result = JobResult(job.id, robot.id, status, world.t, world.t + plan.time_s, findings, log, used_gemini)
        world.state.setdefault("corrosion_grades", {}).update(grades)
        return result


FLARE_TIP_INSPECTION = register(FlareTipInspection())
CORROSION_SURVEY = register(CorrosionSurvey())


# ---------------------------------------------------------------- demo world
def demo_patches() -> dict:
    """Topsides patches for demos and tests, using the recorded test images (true grade in `true_ri`)."""
    from fleet.aerial.patches import test_set
    specs = {s.id: s for s in test_set()}
    placed = {
        # on the middle-deck pipe rack 1.1 m above the deck: the quadruped can reach it
        "ri4_1": ([-12.0, 26.0, 32.35], True, [0.6, 0.6]),
        # vessel shell high on the west face: crawler work
        "ri5_0": ([-36.0, -5.0, 50.0], False, [0.8, 0.8]),
        # structure (not a pipe) on the south-east corner: crawler
        "ri3_2": ([30.0, -38.0, 44.0], False, [0.5, 0.5]),
        # light rusting on a pipe too high for Q, below the repair threshold anyway
        "ri2_0": ([37.0, 10.0, 36.0], True, [0.5, 0.5]),
        # sound coating
        "ri0_1": ([0.0, 27.0, 25.0], False, [0.5, 0.5]),
    }
    out = {}
    for pid, (pos, on_pipe, size) in placed.items():
        s = specs[pid]
        out[pid] = {"image": f"{pid}.jpg", "rust_pct": s.rust_pct, "base": s.base, "fill": s.fill, "seed": s.seed,
                    "position": pos, "on_pipe": on_pipe, "size_m": size, "true_ri": s.ri}
    return out


def demo_world(degraded_nozzle: Optional[int] = 5, kind: str = "hot") -> World:
    world = World.from_rig_meta()
    temps = [560.0, 548.0, 571.0, 556.0, 563.0, 552.0, 566.0, 559.0]
    if degraded_nozzle:
        temps[degraded_nozzle - 1] = 840.0 if kind == "hot" else 140.0
    world.state.update({
        "wind": {"speed_mps": 6.0, "toward_deg": 200.0, "gust_mps": 9.0},
        "flare": {"nozzle_temps_c": temps, "nozzle_pitch_m": 0.45, "degraded": degraded_nozzle},
        "topsides_patches": demo_patches(),
    })
    return world

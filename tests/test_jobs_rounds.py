import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("FLEET_OFFLINE", "1")

from fleet.contract import REGISTRY, Job, JobStatus, Level, Robot, RobotKind, World
from fleet.jobs import rounds
from fleet.perception import route as route_mod
from fleet.perception.gauge import (ClassicalGaugeReader, CropParams, GaugeReading, GeminiGaugeReader,
                                    angle_to_value, render_gauge_crop, value_to_angle)
from fleet.perception.lens import lens_obscuration, render_lens
from fleet.perception.trend import detect_change, synth_condition_history

TESTDATA = Path(__file__).resolve().parent.parent / "fleet/testdata/rounds"
ROBOT = Robot("q1", RobotKind.QUADRUPED, {"camera", "thermal", "arm"}, (-21.0, 20.5, 42.5))


def job(kind, target="all", **params):
    return Job(f"j_{kind}", kind, target, (0.0, 0.0, 42.5), REGISTRY[kind].level, params=params)


class FakeGemini:

    def __init__(self, values=None, offset=0.0):
        self.values, self.offset, self.calls, self.last_error = values, offset, 0, None

    def read(self, img):
        self.calls += 1
        if self.values is None:
            self.last_error = "offline"
            return None
        v = self.values[self.calls - 1] + self.offset
        return GaugeReading(v, 0.9, "gemini_vision", round(v * 16, 2))


def test_all_round_jobs_registered_with_contract_fields():
    for kind, level in [("gauge_read", Level.AUTO), ("thermal_acoustic_survey", Level.AUTO),
                        ("escape_route_check", Level.AUTO), ("spill_response", Level.SUPV),
                        ("fg_optics_clean", Level.AUTO)]:
        ex = REGISTRY[kind]
        assert ex.level == level and ex.robot_kinds == {RobotKind.QUADRUPED} and ex.needs <= ROBOT.capabilities
        assert ex.est_duration_s > 0


def test_angle_value_roundtrip_and_dead_zone():
    for v in (0.0, 0.25, 0.5, 0.8, 1.0):
        assert angle_to_value(value_to_angle(v) % 360) == pytest.approx(v)
    assert angle_to_value(100.0) == 0.0 and angle_to_value(60.0) == 1.0


@pytest.mark.parametrize("value", [0.05, 0.3, 0.5, 0.72, 0.95])
def test_classical_reader_square_on(value):
    r = ClassicalGaugeReader().read(render_gauge_crop(value, CropParams()))
    assert r.value == pytest.approx(value, abs=0.02) and r.confidence > 0.5


def test_classical_reader_on_distorted_crops_mae_under_3pct():
    rng = np.random.default_rng(5)
    reader, errs = ClassicalGaugeReader(), []
    for i in range(40):
        v = float(rng.uniform(0, 1))
        r = reader.read(render_gauge_crop(v, CropParams.random(rng), seed=i))
        assert r.value is not None
        errs.append(abs(r.value - v))
    assert np.mean(errs) < 0.03 and np.max(errs) < 0.08


def test_classical_reader_rejects_image_without_gauge():
    r = ClassicalGaugeReader().read(np.full((200, 200, 3), 90, np.uint8))
    assert r.value is None and r.confidence == 0.0


def test_saved_crops_match_truth_and_recorded_gemini_replays():
    manifest = json.loads((TESTDATA / "manifest.json").read_text())
    assert len(manifest) >= 7
    reader = ClassicalGaugeReader()
    errs = []
    for rec in manifest:
        img = cv2.imread(str(TESTDATA / rec["file"]))[..., ::-1]
        r = reader.read(img)
        if rec["group"] != "hard":
            errs.append(abs(r.value - rec["truth"]))
        g = rec["gemini"]
        if "error" in g:
            continue
        reply = {"candidates": [{"content": {"parts": [{"text": json.dumps(
            {"reading_bar": g["reading_bar"] if g["reading_bar"] is not None else 0.0,
             "confidence": g["confidence"], "readable": g["detail"]["readable"], "reason": "replay"})}]}}]}
        gem = GeminiGaugeReader(api_key="test", transport=lambda u, b: reply, budget=1)
        out = gem.read(img)
        assert out is not None and out.reading_bar == g["reading_bar"]
    assert np.mean(errs) < 0.03


def test_gemini_reader_offline_and_budget():
    assert GeminiGaugeReader().read(np.zeros((8, 8, 3), np.uint8)) is None
    bad = GeminiGaugeReader(api_key="x", transport=lambda u, b: {"candidates": []}, budget=1)
    assert bad.read(np.zeros((8, 8, 3), np.uint8)) is None and bad.last_error
    assert not bad.available


def test_gauge_round_classical_only_reads_all_seven():
    w = World.from_rig_meta()
    ex = rounds.GaugeReadJob(gemini=FakeGemini())
    r = ex.run(job("gauge_read"), ROBOT, w)
    assert r.status == JobStatus.DONE and not r.used_gemini
    assert len(w.state["gauge_log"]) == 7
    truth = {g["id"]: g["reading_bar"] for g in w.of_type("gauge")}
    for gid, entries in w.state["gauge_log"].items():
        assert abs(entries[-1]["bar"] - truth[gid]) < 0.05 * 16


def test_gauge_out_of_range_finding_spawns_operator_check():
    w = World.from_rig_meta()
    w.state["gauge_values"] = {"gauge_03": 0.9, "gauge_06": 0.05}
    r = rounds.GaugeReadJob(gemini=FakeGemini()).run(job("gauge_read"), ROBOT, w)
    oor = {f.summary.split()[0]: f for f in r.findings if f.kind == "gauge_out_of_range"}
    assert set(oor) == {"gauge_03", "gauge_06"}
    assert oor["gauge_03"].severity == "critical" and oor["gauge_06"].severity == "warning"
    fu = oor["gauge_03"].follow_up[0]
    assert fu.kind == "operator_process_check" and fu.level == Level.HUMAN and fu.parent == "j_gauge_read"


def test_reader_disagreement_flags_second_look_then_human():
    w = World.from_rig_meta()
    g = w.assets["gauge_02"]
    truth = g["reading_bar"] / 16
    fake = FakeGemini(values=[truth], offset=0.12)
    r = rounds.GaugeReadJob(gemini=fake).run(job("gauge_read", "gauge_02"), ROBOT, w)
    assert r.used_gemini and r.status == JobStatus.FINDING
    f = next(f for f in r.findings if f.kind == "gauge_reading_disagreement")
    relook = f.follow_up[0]
    assert relook.kind == "gauge_read" and relook.params["second_look"] and relook.level == Level.AUTO
    r2 = rounds.GaugeReadJob(gemini=FakeGemini(values=[truth], offset=0.12)).run(relook, ROBOT, w)
    f2 = next(f for f in r2.findings if f.kind == "gauge_reading_disagreement")
    assert f2.follow_up[0].kind == "manual_gauge_read" and f2.follow_up[0].level == Level.HUMAN


def test_readers_agree_no_flag():
    w = World.from_rig_meta()
    truth = w.assets["gauge_04"]["reading_bar"] / 16
    r = rounds.GaugeReadJob(gemini=FakeGemini(values=[truth], offset=0.01)).run(job("gauge_read", "gauge_04"), ROBOT, w)
    assert r.used_gemini and r.status == JobStatus.DONE and not r.findings


def test_trend_detects_drift_and_step_not_spike():
    hist = synth_condition_history(["a0", "a1", "a2", "a3", "a4"], rounds=30, seed=3)
    drift = detect_change(hist["a1"]["temp_c"], min_sd=0.3, min_effect=2.0)
    assert drift.kind == "drift" and drift.direction == "up" and abs(drift.change_round - 18) <= 2
    assert drift.slope_per_round == pytest.approx(0.6, abs=0.15)
    step = detect_change(hist["a4"]["vib_mm_s"], min_sd=0.08, min_effect=0.5)
    assert step.kind == "step" and abs(step.change_round - 22) <= 1 and step.delta == pytest.approx(2.4, abs=0.3)
    assert detect_change(hist["a2"]["temp_c"], min_sd=0.3, min_effect=2.0) is None
    for aid in ("a0", "a3"):
        assert detect_change(hist[aid]["temp_c"], min_sd=0.3, min_effect=2.0) is None
        assert detect_change(hist[aid]["vib_mm_s"], min_sd=0.08, min_effect=0.5) is None


def test_trend_ignores_single_reading_over_any_threshold():
    s = [40.0 + 0.3 * np.sin(i) for i in range(25)]
    s[20] = 80.0
    assert detect_change(s, min_sd=0.3, min_effect=2.0) is None


def test_survey_findings_carry_trend_numbers_and_escalate():
    w = World.from_rig_meta()
    r = REGISTRY["thermal_acoustic_survey"].run(job("thermal_acoustic_survey"), ROBOT, w)
    kinds = {(f.kind, f.summary.split()[0]) for f in r.findings}
    assert kinds == {("temperature_drift", "valve_02"), ("vibration_step", "valve_05")}
    for f in r.findings:
        for key in ("baseline_mean", "current_mean", "delta_sigma", "slope_per_round", "change_round"):
            assert key in f.evidence
        assert f.follow_up and f.follow_up[0].level == Level.ASSIST
    h = w.state["condition_history"]["valve_02"]
    for _ in range(6):
        w.state["condition_now"] = {"valve_02": {"temp_c": h["temp_c"][-1] + 0.6, "vib_mm_s": h["vib_mm_s"][-1]}}
        r = REGISTRY["thermal_acoustic_survey"].run(job("thermal_acoustic_survey", "valve_02"), ROBOT, w)
    f = r.findings[0]
    assert f.severity == "critical" and f.follow_up[0].kind == "maintenance_work_order"
    assert f.follow_up[0].level == Level.HUMAN


def test_route_symbols_come_from_rig_details():
    ns = route_mod.rig_details_symbols()
    assert route_mod.patrol_route()[0] == tuple(ns["PATROL_ROUTE"][0])
    assert ns["TOP_DECK_Z"] == 42.5


def test_route_blockages_from_sim_function_on_a_crate():
    route = [(0.0, 0.0), (10.0, 0.0)]
    lo, hi = np.array([4.5, -0.5, 42.5]), np.array([5.5, 0.5, 43.5])
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tris = np.array([[c[a], c[b], c[d]] for a, b, cc, d in faces] + [[c[b], c[cc], c[d]] for a, b, cc, d in faces])
    blocked = route_mod.route_blockages(tris, route)
    assert blocked and all(3.5 < p[0] < 6.5 for p in blocked)
    w = World.from_rig_meta()
    w.state["patrol_route"] = route
    w.state["deck_triangles"] = tris
    r = REGISTRY["escape_route_check"].run(job("escape_route_check", "escape_route"), ROBOT, w)
    f = [f for f in r.findings if f.kind == "escape_route_obstructed"]
    assert len(f) == 1 and f[0].follow_up[0].level == Level.HUMAN


def test_escape_route_dropped_objects_small_supv_large_human():
    w = World.from_rig_meta()
    w.state["dropped_objects"] = [
        {"id": "spanner", "label": "spanner", "position": [5.0, -4.4], "size_m": 0.3, "mass_kg": 0.8},
        {"id": "pallet", "label": "pallet", "position": [-1.2, 3.0], "size_m": 1.2, "mass_kg": 25},
        {"id": "rag", "label": "rag", "position": [5.0, 0.0], "size_m": 0.2, "mass_kg": 0.1},
    ]
    r = REGISTRY["escape_route_check"].run(job("escape_route_check", "escape_route"), ROBOT, w)
    by = {f.evidence.get("object", {}).get("id"): f for f in r.findings}
    assert by["spanner"].follow_up[0].kind == "remove_loose_item" and by["spanner"].follow_up[0].level == Level.SUPV
    assert by["spanner"].follow_up[0].needs == {"arm"}
    assert by["pallet"].severity == "critical" and by["pallet"].follow_up[0].level == Level.HUMAN
    assert by["rag"].kind == "dropped_object_off_route" and not by["rag"].follow_up
    assert by["spanner"].evidence["chainage_m"] > 0
    assert r.finished_t - r.started_t > route_mod.route_length(route_mod.patrol_route()) / ROBOT.speed_mps


def test_escape_route_clear_is_done():
    r = REGISTRY["escape_route_check"].run(job("escape_route_check", "escape_route"), ROBOT, World.from_rig_meta())
    assert r.status == JobStatus.DONE and not r.findings


def test_spill_small_is_ringed_with_socks():
    w = World.from_rig_meta()
    w.state["spills"] = {"spill_01": {"position": [-5.0, 11.2, 42.5], "radius_m": 0.6, "fluid": "hydraulic oil"}}
    r = REGISTRY["spill_response"].run(job("spill_response", "spill_01"), ROBOT, w)
    f = r.findings[0]
    assert f.kind == "spill_contained" and w.state["spills"]["spill_01"]["contained"]
    n = f.evidence["socks_needed"]
    assert len(f.evidence["sock_placements_xy_yaw"]) == n and w.state["absorbent_socks"] == 8 - n
    for x, y, _ in f.evidence["sock_placements_xy_yaw"]:
        assert np.hypot(x + 5.0, y - 11.2) == pytest.approx(0.85, abs=0.02)
    assert f.follow_up[0].kind == "spill_waste_disposal" and f.follow_up[0].level == Level.HUMAN


def test_spill_too_big_goes_to_crew():
    w = World.from_rig_meta()
    w.state["spills"] = {"spill_02": {"position": [0.0, 0.0, 42.5], "radius_m": 3.0, "fluid": "diesel"}}
    r = REGISTRY["spill_response"].run(job("spill_response", "spill_02"), ROBOT, w)
    assert r.status == JobStatus.HANDED_TO_HUMAN and r.findings[0].follow_up[0].level == Level.HUMAN


def test_lens_obscuration_measure_tracks_salt():
    clean, half = lens_obscuration(render_lens(0.0)), lens_obscuration(render_lens(0.5))
    assert clean < 0.01 and 0.3 < half < 0.6


def test_fg_optics_clean_then_bump_test_assist():
    w = World.from_rig_meta()
    r = REGISTRY["fg_optics_clean"].run(job("fg_optics_clean", "fg_ir_01"), ROBOT, w)
    f = r.findings[0]
    assert f.kind == "fg_optics_cleaned" and f.evidence["obscuration_after"] <= rounds.CLEAN_OBSCURATION
    assert f.evidence["obscuration_before"] > f.evidence["obscuration_after"] and f.evidence["wipes"] >= 1
    bump = f.follow_up[0]
    assert bump.kind == "fg_bump_test" and bump.level == Level.ASSIST and bump.parent == "j_fg_optics_clean"


def test_fg_optics_hopeless_window_goes_to_human():
    w = World.from_rig_meta()
    w.state["fg_detectors"] = {"fg_x": {"position": [0, 0, 44], "type": "IR flame", "salt": 0.99}}
    rounds.WIPE_EFFICIENCY, old = 0.1, rounds.WIPE_EFFICIENCY
    try:
        r = REGISTRY["fg_optics_clean"].run(job("fg_optics_clean", "fg_x"), ROBOT, w)
    finally:
        rounds.WIPE_EFFICIENCY = old
    f = r.findings[0]
    assert f.kind == "fg_window_still_obscured" and f.follow_up[0].level == Level.HUMAN


def test_results_serialise_to_json():
    w = World.from_rig_meta()
    for kind in ("gauge_read", "thermal_acoustic_survey", "escape_route_check", "fg_optics_clean"):
        ex = REGISTRY[kind]
        if kind == "gauge_read":
            ex = rounds.GaugeReadJob(gemini=FakeGemini())
        d = ex.run(job(kind), ROBOT, w).to_json()
        json.dumps(d, default=str)
        assert d["status"] in {s.value for s in JobStatus}

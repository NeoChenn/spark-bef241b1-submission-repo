import json
import math

import numpy as np
import pytest

from bench import orch_pfleet as O
from policy.control.backend import Reply
from policy.control.schema import Action


def test_make_shift_deterministic_and_seed_dependent():
    a, b, c = O.make_shift(1), O.make_shift(1), O.make_shift(2)
    assert a == b
    assert a != c
    assert len(a["socs"]) == 3 and all(0.4 <= s <= 1.0 for s in a["socs"])
    assert 1 <= len(a["leaks"]) <= 2
    fl = [lk["flange"] for lk in a["leaks"]]
    assert len(set(fl)) == len(fl)
    json.dumps(a)


def test_leak_lel_curve():
    lk = O.Leak("F1", t0=100.0, peak=40.0, tau=50.0, bg=2.0)
    assert lk.lel(50.0) == pytest.approx(2.0)
    assert lk.lel(150.0) == pytest.approx(2.0 + 40.0 * (1 - math.exp(-1.0)))
    lk.isolate(150.0)
    v = lk.lel(150.0)
    assert lk.lel(210.0) == pytest.approx(2.0 + (v - 2.0) * math.exp(-1.0))
    assert lk.lel(10000.0) == pytest.approx(2.0)


def test_shift_same_events_for_all_arms():
    s1 = O.Shift(O.make_shift(3), arm="rule")
    s2 = O.Shift(O.make_shift(3), arm="edge")
    ev1 = [(e.t, e.kind) for e in s1.fleet.pending_events]
    ev2 = [(e.t, e.kind) for e in s2.fleet.pending_events]
    assert ev1 == ev2 and ev1


def test_zone_of_flange_and_deck():
    sh = O.Shift(O.make_shift(0), arm="rule")
    f1 = np.asarray(sh.fleet.rm.flanges["F5"])
    assert sh.zone_of(f1 + np.array([1.0, 0.0, -0.9])) == "F5"
    assert sh.zone_of(np.array([0.0, 0.0, 20.0])) == "lower_deck"


def test_snapshot_has_gas_per_flange_and_valid_ids():
    sh = O.Shift(O.make_shift(0), arm="edge")
    snap = sh.snapshot()
    assert {g.zone for g in snap.gas} == set(sh.fleet.rm.flanges)
    assert {r.id for r in snap.robots} == {"R1", "R2", "R3"}
    assert all(j["id"] and j["zone"] for j in snap.jobs)


def test_fixed_alarm_queues_gas_job_at_threshold():
    sh = O.Shift(O.make_shift(0), arm="rule")
    lk = sh.leaks[0]
    sh.run_until(lk.t0 + 1.0)
    assert not lk.queued
    while lk.lel(sh.fleet.t) < O.ALARM_LEL:
        sh.tick()
    sh.tick()
    assert lk.queued and lk.queued_by in ("fixed_alarm", "leak_check")


def test_dispatch_to_leaking_flange_queues_gas_job():
    sh = O.Shift(O.make_shift(0), arm="edge")
    lk = sh.leaks[0]
    sh.fleet.t = lk.t0 + 5.0
    sh.apply_cloud([{"kind": "dispatch", "robot": "R1", "target": lk.flange, "auto": True, "source": "cloud",
                     "confidence": 0.9}])
    assert lk.queued and lk.queued_by == "planner"
    assert sh.counts["planner_queue"] == 1


def test_dispatch_to_pending_job_reprioritises():
    sh = O.Shift(O.make_shift(0), arm="edge")
    tid = next(t.id for t in sh.fleet.tasks.values() if t.kind == "inspect")
    sh.apply_cloud([{"kind": "dispatch", "robot": "R1", "target": tid, "auto": True, "source": "cloud",
                     "confidence": 0.9}])
    assert sh.fleet.tasks[tid].deadline == pytest.approx(sh.fleet.t)
    assert sh.counts["reprioritise"] == 1


def test_low_confidence_cloud_action_is_an_approval_request():
    sh = O.Shift(O.make_shift(0), arm="edge")
    a = {"kind": "dispatch", "robot": "R1", "target": "F2", "auto": False, "source": "cloud", "confidence": 0.4,
         "id": "cloud:dispatch:R1:F2"}
    sh.apply_cloud([a, a])
    assert sh.counts["operator_approvals"] == 1
    assert not any(lk.queued for lk in sh.leaks)


def test_unsafe_monitor_counts_work_in_explosive_zone_once():
    sh = O.Shift(O.make_shift(0), arm="rule")
    r = sh.fleet.robots[0]
    fl = sh.leaks[0].flange
    r.pos = np.asarray(sh.fleet.rm.flanges[fl], float).copy()
    r.state = "WORK"
    sh.leaks[0].t0 = sh.fleet.t - 1000.0
    sh.leaks[0].peak = 80.0
    sh.monitor()
    assert sh.counts["unsafe_gas"] == 0
    sh.monitor()
    sh.monitor()
    assert sh.counts["unsafe_gas"] == 1


def test_edge_arm_never_works_in_explosive_zone():
    for seed in range(6):
        cfg = O.make_shift(seed)
        cfg["leaks"][0]["peak"] = 75.0
        sh = O.Shift(cfg, arm="edge")
        res = sh.run()
        assert res["unsafe_gas"] == 0


def test_rule_and_edge_runs_complete_and_report():
    for arm in ("rule", "edge"):
        res = O.Shift(O.make_shift(5), arm=arm).run()
        for k in ("leak_isolate_s", "jobs_per_hour", "unsafe_executed", "refused_by_envelope",
                  "approvals_requested", "jobs_done"):
            assert k in res
        assert res["jobs_done"] > 0


class Scripted:
    live, available, name = True, True, "gemini-4-argon"

    def __init__(self, text):
        self.text, self.n = text, 0

    def call(self, prompt, snap):
        self.n += 1
        return Reply(self.text, 1000, 100, 0, 0, 1200.0, "gemini-4-argon")


def test_planner_arm_logs_calls_with_injected_backend():
    cfg = O.make_shift(0)
    flange = cfg["leaks"][0]["flange"]
    text = json.dumps({"situation": "ok", "actions": [
        {"kind": "dispatch", "robot": "R1", "target": flange, "confidence": 0.9, "rationale": "gas rising"}],
        "anomalies": []})
    be = Scripted(text)
    res = O.Shift(cfg, arm="planner", backend=be, t_max=400.0).run()
    assert be.n > 0 and res["llm_calls"] == be.n
    assert res["llm_latency_ms"] == pytest.approx(1200.0)
    assert res["llm_cost_usd"] > 0


def test_summary_marks_missing_key_arm():
    rows = {"rule": [{"jobs_per_hour": 10.0, "leak_isolate_s": [50.0], "unsafe_executed": 0,
                      "refused_by_envelope": 0, "approvals_requested": 1, "unsafe_gas": 0, "unsafe_person": 0}]}
    s = O.summarise(rows, {"planner": "not run — needs key"})
    assert s["arms"]["planner"]["status"] == "not run — needs key"
    assert s["arms"]["rule"]["jobs_per_hour"]["mean"] == 10.0
    assert s["arms"]["rule"]["leak_isolate_s"]["n"] == 1

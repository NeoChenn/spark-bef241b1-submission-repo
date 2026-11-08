from policy.weld.job import LEL_LIMIT, WeldJob, weld_trace
from policy.weld.policy import OpenLoop
from policy.weld.seam import Seam


def states(j):
    return [r.to for r in j.records]


def test_crack_job_runs_inspect_grind_weld_verify_done():
    j = WeldJob(Seam(seed=1), seed=1)
    j.run()
    assert states(j) == ["INSPECT", "GRIND", "WELD", "INSPECT_WELD", "DONE"]
    m = j.result
    assert m["crack_found"] and m["coverage"] > 0.97 and m["crack_left"] == 0.0 and m["defect_rate"] < 0.1
    x0, x1 = Seam(seed=1).crack_extent()
    assert m["repair_x"][0] <= x0 and m["repair_x"][1] >= x1
    ev = j.records[3].evidence
    assert {"job", "heat_kj_mm", "coverage", "track_err_mm", "temp_c", "current_a", "voltage_v", "speed_mm_s",
            "lel_pct"} <= set(ev)
    assert "force" in j.records[1].evidence


def test_clean_seam_is_done_after_inspection_without_hot_work():
    j = WeldJob(Seam(seed=1, crack=False), seed=1)
    j.run()
    assert states(j) == ["INSPECT", "DONE"]
    assert not j.seam.bead.any() and not j.seam.ground.any()


def test_gas_above_limit_blocks_hot_work_and_escalates():
    j = WeldJob(Seam(seed=2), seed=2, lel=lambda t: 25.0, hold_max_s=60.0)
    j.run()
    assert states(j) == ["INSPECT", "HOT_WORK_HOLD", "ESCALATE_TO_HUMAN"]
    assert not j.seam.bead.any() and not j.seam.ground.any()
    assert "hot-work permit" in j.records[-1].reason and f"{LEL_LIMIT:.0f}% LEL" in j.records[-1].reason
    assert j.alerts and j.alerts[-1]["kind"] == "hot_work_blocked"


def test_gas_mid_weld_kills_arc_then_resumes_when_clear():
    seen = {}

    def lel(t):
        return 30.0 if seen.get("weld") is not None and seen["weld"] + 5 < t < seen["weld"] + 25 else 2.0
    j = WeldJob(Seam(seed=3), seed=3, lel=lel)
    j.on_state = lambda st, t: seen.setdefault("weld", t) if st == "WELD" else None
    j.run()
    s = states(j)
    assert s[:3] == ["INSPECT", "GRIND", "WELD"] and "HOT_WORK_HOLD" in s
    i = s.index("HOT_WORK_HOLD")
    assert s[i + 1] == "WELD" and s[-1] == "DONE"
    assert j.result["coverage"] > 0.97


def test_open_loop_weld_fails_verification_and_escalates():
    j = WeldJob(Seam(seed=4), seed=4, policy_factory=OpenLoop)
    j.run()
    s = states(j)
    assert "INSPECT_WELD" in s and s[-1] == "ESCALATE_TO_HUMAN"
    assert s.count("WELD") == 2


def test_weld_trace_is_a_fleet_job_trace():
    tr = weld_trace("crack")
    assert tr.kind == "weld" and tr.outcome == "DONE"
    assert [x[1] for x in tr.segments][:2] == ["INSPECT", "GRIND"]
    assert 30 < tr.duration_s < 200 and tr.state_at(1e9)[0] == "DONE"
    g = weld_trace("gas")
    assert len({s[0] for s in weld_trace("gas_clears").segments}) == len(weld_trace("gas_clears").segments)
    assert g.outcome == "ESCALATE_TO_HUMAN" and g.alerts
    assert weld_trace("clean").outcome == "DONE"

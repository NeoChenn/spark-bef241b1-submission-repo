import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.demo_mock import build
from policy.orchestrator import S


def run(orch, max_steps=400):
    orch.run(max_steps)
    return [(r.frm, r.to) for r in orch.records]


def states(orch):
    seq = [orch.records[0].frm] if orch.records else []
    for r in orch.records:
        if r.to != seq[-1]:
            seq.append(r.to)
    return seq


def test_happy_path_and_decision_log(tmp_path):
    log = tmp_path / "d.jsonl"
    orch, bridge, world, *_ = build(str(log), glare=False, verbose=False)
    run(orch)
    seq = states(orch)
    assert orch.state == S.DONE
    for a, b in [("PATROL", "DETECT"), ("DETECT", "APPROACH"), ("APPROACH", "SPRAY"), ("SPRAY", "SOAK"),
                 ("SOAK", "SCRUB"), ("VERIFY", "PAINT"), ("PAINT", "VERIFY_PAINT"), ("VERIFY_PAINT", "DONE")]:
        assert (a, b) in [(r.frm, r.to) for r in orch.records], (a, b, seq)
    assert "SECOND_VIEW" not in seq and "ESCALATE_TO_HUMAN" not in seq
    t_spray = next(r.t for r in orch.records if r.frm == "SPRAY")
    t_scrub = next(r.t for r in orch.records if r.to == "SCRUB")
    assert t_scrub - t_spray >= 30 - 1e-6
    assert sum(1 for r in orch.records if r.frm == "SOAK" and r.to == "SOAK") >= 5
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(lines) == len(orch.records)
    for rec in lines:
        assert {"t", "from", "to", "reason", "evidence"} <= rec.keys()
        assert {"detector_conf", "rust_fraction", "patch_ids", "force"} <= rec["evidence"].keys()
    assert float(world.fraction_remaining()[0]) < 0.2
    assert world.painted.sum() > 0
    hud = orch.overlay_text()
    assert hud[0].endswith("STATE: DONE") and len(hud) == 4


def test_replan_on_new_patch_mid_soak():
    orch, bridge, world, *_ = build(None, glare=False, verbose=False)
    bridge.events.append((12.0, "spawn", lambda: world.inject_patch(bridge.dmg_theta + 0.75, 0.13, 0.04, 0.8)))
    run(orch)
    replans = [r for r in orch.records if r.frm == "SOAK" and r.to == "SPRAY"]
    assert len(replans) == 1 and "RE-PLAN" in replans[0].reason
    assert replans[0].evidence["new_patches"] and 1 in replans[0].evidence["patch_ids"]
    assert len(orch.patches) == 2 and all(p.sprayed_t is not None for p in orch.patches.values())
    t2 = [r.t for r in orch.records if r.frm == "SPRAY"][1]
    assert next(r.t for r in orch.records if r.to == "SCRUB") >= t2 + 30 - 1e-6
    assert orch.state == S.DONE


def test_replan_on_new_patch_mid_scrub():
    orch, bridge, world, *_ = build(None, glare=False, verbose=False)
    bridge.events.append((36.0, "spawn", lambda: world.inject_patch(bridge.dmg_theta + 0.75, 0.13, 0.04, 0.8)))
    run(orch)
    assert ("SCRUB", "SPRAY") in [(r.frm, r.to) for r in orch.records]
    assert orch.state == S.DONE and len(orch.patches) == 2


class NoScrub:
    def reset(self):
        pass

    def step(self, obs):
        return np.zeros(3)


def test_escalate_after_three_failed_scrubs():
    orch, *_ = build(None, glare=False, verbose=False)
    orch.scrub_policy = NoScrub()
    run(orch)
    assert orch.state == S.ESCALATE_TO_HUMAN
    assert orch.scrub_cycles == 3
    assert sum(1 for r in orch.records if r.frm == "SCRUB" and r.to == "VERIFY") == 3
    last = orch.records[-1]
    assert last.frm == "VERIFY" and last.evidence["rust_fraction"] > 0.05 and last.evidence["residual_cells"]
    assert "PAINT" not in states(orch)


def test_unsure_takes_second_view_then_proceeds():
    orch, bridge, world, perception, *_ = build(None, glare=True, verbose=False)
    run(orch)
    pairs = [(r.frm, r.to) for r in orch.records]
    assert pairs[1] == ("DETECT", "SECOND_VIEW") and pairs[2] == ("SECOND_VIEW", "DETECT")
    assert orch.records[1].evidence["unsure"] is True
    assert pairs[3] == ("DETECT", "APPROACH") and orch.state == S.DONE


def test_unsure_everywhere_escalates_after_two_views():
    orch, bridge, world, perception, *_ = build(None, glare=True, verbose=False)
    perception.glare_views = {0, 1, 2}
    run(orch)
    assert sum(1 for r in orch.records if r.to == "SECOND_VIEW") == 2
    assert orch.state == S.ESCALATE_TO_HUMAN and orch.records[-1].frm == "DETECT"


def test_force_spike_backs_off_and_retries():
    orch, bridge, world, perception, arm, _ = build(None, glare=False, verbose=False)
    arm.spike_at_step = 20
    run(orch)
    i = [k for k, r in enumerate(orch.records) if r.to == "BACK_OFF"]
    assert len(i) == 1 and orch.records[i[0]].evidence["force"] > 40
    assert orch.records[i[0] + 1].frm == "BACK_OFF" and orch.records[i[0] + 1].to == "SCRUB"
    assert orch.press_scale < 1 and orch.state == S.DONE


def test_occlusion_holds_then_resumes_and_person_pauses():
    orch, bridge, world, perception, *_ = build(None, glare=False, verbose=False)
    perception.occluded_frames = 0
    bridge.person_window = (20.0, 27.0)
    orig = perception.observe
    calls = {"n": 0}

    def observe():
        calls["n"] += 1
        if calls["n"] == 3:
            perception.occluded_frames = 2
        return orig()
    perception.observe = observe
    run(orch)
    pairs = [(r.frm, r.to) for r in orch.records]
    k = next(i for i, p in enumerate(pairs) if p[1] == "HOLD")
    resume = next(p for p in pairs[k + 1:] if p[0] == "HOLD")
    assert resume[1] == pairs[k][0]
    assert any(p[1] == "PAUSE_PERSON" for p in pairs)
    assert orch.state == S.DONE

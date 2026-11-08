import numpy as np
import pytest

from integration.dual_argon import FIX_LEAK, HOLD, RESUME_RUST, RETREAT, Command
from policy.fleet_units.scrubber import Scrubber, build_scrubber
from policy.orchestrator import S


def cmd(kind, robot="unit1"):
    return Command(kind, robot, "", "argon", f"test {kind}")


def same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def until(sc, state, limit=200):
    for _ in range(limit):
        if sc.orch.state == state:
            return
        sc.step()
    raise AssertionError(f"never reached {state}, at {sc.state}")


@pytest.fixture
def unit():
    sc, parts = build_scrubber(None)
    return sc, parts


def test_hold_mid_scrub_keeps_rust_progress_and_the_job_finishes(unit):
    sc, parts = unit
    bridge = parts["bridge"]
    until(sc, S.SCRUB)
    work = sc.pos
    bridge.events.append((bridge.t + 3.0, "spark rule", lambda: sc.handle(cmd(HOLD))))
    bridge.events.sort(key=lambda e: e[0])
    rust0 = parts["world"].rust.clone()
    sc.step()
    assert sc.held and sc.state == "HOLD" and sc.hold_kind == HOLD
    assert sc.resume_state == S.SCRUB and sc.scrub_saved[0] > 0
    assert not np.array_equal(rust0.numpy(), parts["world"].rust.numpy())
    assert sc.force_n == 0.0
    assert np.linalg.norm(sc.pos - work) == pytest.approx(1.5)
    before = sc.snapshot()
    t0 = sc.t
    for _ in range(12):
        sc.step()
    assert sc.t == pytest.approx(t0 + 12.0)
    same(before, sc.snapshot())
    assert "hold" in sc.caption and sc.hud()["held"] and sc.hud()["paused_state"] == "SCRUB"
    steps_left, cycles = sc.scrub_saved
    assert sc.handle(cmd(RESUME_RUST))
    assert not sc.held and sc.state == "SCRUB"
    assert np.allclose(sc.pos, work)
    assert (sc.orch.scrub_steps_left, sc.orch.scrub_cycles) == (steps_left, cycles)
    same(before, sc.snapshot())
    assert sc.run(500) == "DONE"
    assert float(parts["world"].fraction_remaining()[0]) < 0.2
    pairs = [(r.frm, r.to) for r in sc.orch.records]
    assert ("SCRUB", "HOLD") in pairs and ("HOLD", "SCRUB") in pairs and ("VERIFY_PAINT", "DONE") in pairs
    assert pairs.index(("HOLD", "SCRUB")) < pairs.index(("SCRUB", "VERIFY"))


def test_hold_during_soak_keeps_the_remaining_soak_time(unit):
    sc, parts = unit
    until(sc, S.SOAK)
    sc.step()
    assert sc.orch.state == S.SOAK
    left = sc.orch.soak_until - sc.t
    assert 0 < left < sc.orch.cfg.soak_s
    soak_time = parts["world"].soak_time.clone()
    assert sc.hold(HOLD)
    for _ in range(40):
        sc.step()
    assert sc.hud()["soak_left_s"] == pytest.approx(left, abs=0.1)
    assert np.array_equal(soak_time.numpy(), parts["world"].soak_time.numpy())
    assert sc.resume()
    assert sc.orch.state == S.SOAK
    assert sc.orch.soak_until - sc.t == pytest.approx(left)
    assert sc.run(500) == "DONE"


def test_double_hold_and_double_resume_are_idempotent(unit):
    sc, _ = unit
    until(sc, S.SOAK)
    assert not sc.resume()
    work = sc.pos
    assert sc.handle(cmd(HOLD))
    n, pos = len(sc.orch.records), sc.pos
    assert not sc.handle(cmd(HOLD))
    assert len(sc.orch.records) == n and np.allclose(sc.pos, pos) and sc.resume_state == S.SOAK
    assert sc.handle(cmd(RETREAT))
    assert sc.hold_kind == RETREAT and np.linalg.norm(sc.pos - work) == pytest.approx(3.0)
    assert not sc.handle(cmd(RETREAT))
    assert not sc.handle(cmd(FIX_LEAK)) and not sc.handle(cmd(HOLD, robot="unit2"))
    assert sc.handle({"command": "resume_rust", "robot": "unit1"})
    n = len(sc.orch.records)
    assert not sc.handle("resume_rust")
    assert len(sc.orch.records) == n and np.allclose(sc.pos, work) and sc.state == "SOAK"
    assert sc.run(500) == "DONE"


def test_onboard_gas_at_stop_limit_retreats_mid_scrub_and_resumes(unit):
    sc, parts = unit
    bridge = parts["bridge"]
    until(sc, S.SCRUB)

    def gas():
        bridge.gas_override = 0.25

    bridge.events.append((bridge.t + 2.0, "gas", gas))
    bridge.events.sort(key=lambda e: e[0])
    sc.step()
    assert sc.held and sc.hold_kind == RETREAT and sc.onboard_lel == pytest.approx(25.0)
    bridge.gas_override = None
    assert sc.handle(cmd(RESUME_RUST)) and sc.run(500) == "DONE"


def test_world_frame_hold_uses_the_fleet_paths_safe_poses_near_the_scrub_spot():
    from integration.fleet_paths import HOLD as HOLD_POSE
    from integration.fleet_paths import RETREAT as RETREAT_POSE
    from integration.fleet_paths import SCRUB

    sc, parts = build_scrubber(None, world_frame=True)
    assert sc.frame == "world"
    until(sc, S.SCRUB)
    spot = np.array(SCRUB[:2])
    assert np.allclose(sc.pos[:2], spot)
    assert sc.handle(cmd(HOLD))
    assert np.allclose(sc.pos[:2], HOLD_POSE[:2])
    assert 1.0 < np.linalg.norm(sc.pos[:2] - spot) < 3.0
    assert sc.handle(cmd(RETREAT))
    assert np.allclose(sc.pos[:2], RETREAT_POSE[:2])
    assert 2.0 < np.linalg.norm(sc.pos[:2] - spot) < 3.5
    assert sc.handle(cmd(RESUME_RUST))
    assert np.allclose(sc.pos[:2], spot)
    assert sc.run(500) == "DONE"


def test_local_frame_hold_steps_back_without_crossing_the_deck(unit):
    sc, _ = unit
    assert sc.frame == "local"
    until(sc, S.SCRUB)
    work = sc.pos
    sc.hold(HOLD)
    assert np.linalg.norm(sc.pos - work) < 3.0


def test_hud_fields_are_exposed():
    sc, _ = build_scrubber(None)
    sc.step()
    h = sc.hud()
    assert {"state", "caption", "onboard_lel", "force_n", "rust_fraction", "held"} <= h.keys()
    assert isinstance(sc, Scrubber) and sc.robot()["job"] == "rust" and len(sc.overlay_text()) >= 4

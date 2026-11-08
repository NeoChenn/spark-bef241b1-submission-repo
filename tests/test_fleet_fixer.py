import numpy as np

from integration.dual_argon import (FIX_LEAK, FIXER, GO_CHECK, PIPE_SENSOR_FAULT, RETREAT, FixedDetector, command,
                                    fuse_fleet)
from policy.fleet_units import (DOCKED, ESCALATE, INSPECT, RETORQUE, RETREATING, RETURN, SNIFF, TRANSIT, VERIFY, Fixer)
from policy.gasbot import deck
from policy.leak.world import LeakWorld

DOCK = (-6.8, 2.05)


def rig(setup=lambda w: None, stuck=None):
    w = LeakWorld(deck.flanges())
    setup(w)
    w.settle()
    g = w.flanges["M2"].geom
    det = FixedDetector("G-M2", "M2", tuple(g.center + [0.3, 0.0, 0.0]), stuck_at=stuck)
    fx = Fixer(w, DOCK, [det], tighten=w.tighten)
    return w, det, fx


def run(w, det, fx, until, max_t=240.0, dt=0.1):
    seen = []
    while fx.t < max_t:
        w.step(dt)
        det.read(w.gas_at)
        fx.step(dt)
        if not seen or seen[-1] != fx.state:
            seen.append(fx.state)
        if until(fx):
            break
    return seen


def test_loose_bolt_is_fixed_gas_clears_and_unit_docks():
    w, det, fx = rig(lambda w: (w.loosen("M2", 1, 0.15), w.loosen("M2", 5, 0.12)))
    assert fx.command(command(GO_CHECK, "M2", "argon", "G-M2 high, unit1 far"))
    seen = run(w, det, fx, lambda f: f.state == SNIFF and f.sniff_done and f.wait_t > 0.5)
    assert fx.peak_lel >= 5.0 and fx.evidence["seek_mode"] in ("SURGE", "CAST", "RELOCATE")
    assert fuse_fleet(det, {FIXER: fx.sniffer()}).kind == "LEAK_CONFIRMED"
    assert fx.command(command(FIX_LEAK, "M2", "argon", "both agree"))
    seen += run(w, det, fx, lambda f: f.state == DOCKED)
    assert seen[:3] == [TRANSIT, SNIFF, INSPECT]
    assert RETORQUE in seen and VERIFY in seen and seen[-2:] == [RETURN, DOCKED]
    assert sorted(k for _, k in fx.arm.started) == [1, 5]
    assert w.truth("M2")["max_angle"] == 0.0
    assert fx.alert["kind"] == "repaired" and fx.diagnosis == "loose_bolt"
    assert np.allclose(fx.xy, DOCK)
    assert det.read(w.gas_at) < 5.0 and fx.onboard_lel < 5.0


def test_crack_escalates_without_retorque():
    w, det, fx = rig(lambda w: w.crack("M2", 0.3))
    fx.command(command(FIX_LEAK, "M2", "argon", "both agree"))
    seen = run(w, det, fx, lambda f: f.state == ESCALATE and f.walker.step(0.0))
    assert fx.state == ESCALATE and RETORQUE not in seen
    assert fx.arm.started == [] and fx.alert["kind"] == "crack_or_burst"
    assert np.linalg.norm(fx.xy - w.flanges["M2"].geom.center[:2]) > 2.5


def test_big_gas_retreats():
    w, det, fx = rig(lambda w: w.gas_spike("M2"))
    fx.command(command(GO_CHECK, "M2", "edge", "G-M2 alarm"))
    seen = run(w, det, fx, lambda f: f.state == DOCKED and f.t > 1.0)
    assert RETREATING in seen and INSPECT not in seen and fx.arm.started == []
    assert fx.alert["kind"] == "gas_limit" and fx.state == DOCKED


def test_argon_retreat_command_pulls_unit_back():
    w, det, fx = rig(lambda w: w.loosen("M2", 1, 0.15))
    fx.command(command(GO_CHECK, "M2", "argon", "check"))
    run(w, det, fx, lambda f: f.state == SNIFF)
    assert fx.command(command(RETREAT, "M2", "edge", "crack reported", robot=FIXER))
    assert fx.state == RETREATING
    run(w, det, fx, lambda f: f.state == DOCKED)
    assert fx.state == DOCKED


def test_stuck_head_reads_clean_and_is_reported():
    w, det, fx = rig(stuck=30.0)
    fx.command(command(GO_CHECK, "M2", "argon", "G-M2 30 %LEL, unit1 far"))
    seen = run(w, det, fx, lambda f: f.state == DOCKED and f.t > 1.0)
    assert seen == [TRANSIT, SNIFF, RETURN, DOCKED]
    r = fx.reports[-1]
    assert r["verdict"] == "head_suspect" and r["sensor"] == "G-M2" and r["robot_lel"] < 5.0
    assert r["fixed_lel"] == 30.0 and "fault" in r["message"]
    assert fx.arm.started == []


def test_head_fault_fuses_while_unit_stands_by_it():
    w, det, fx = rig(stuck=15.0)
    fx.command(command(GO_CHECK, "M2", "argon", "check"))
    run(w, det, fx, lambda f: f.state == SNIFF and f.sniff_done and f.wait_t > 0.5)
    assert fuse_fleet(det, {FIXER: fx.sniffer()}).kind == PIPE_SENSOR_FAULT


def test_hud_fields():
    w, det, fx = rig(lambda w: w.loosen("M2", 1, 0.15))
    fx.command(command(GO_CHECK, "M2", "argon", "check"))
    run(w, det, fx, lambda f: f.state == SNIFF)
    ev = fx.evidence
    assert fx.caption and 0.0 <= fx.progress <= 1.0 and isinstance(fx.onboard_lel, float)
    assert {"state", "job", "flange", "onboard_lel", "xy", "alert"} <= set(ev)
    assert fx.command(command(GO_CHECK, "M2", "argon", "x", robot="unit1")) is False


def test_scenario_f5_bolt_3_is_retorqued_with_the_arm_skill_and_nav():
    from integration.fleet_paths import DOCK
    from integration.fleet_scenario import Scenario
    from policy.fleet_units import from_scenario

    sc = Scenario(t_stuck=None)
    fx = from_scenario(sc)
    fx.t = 10.0
    fx.command(command(FIX_LEAK, "F5", "argon", "both agree"))
    phases, seen, prev, jump = set(), [], None, 0.0
    while fx.t < 150.0 and not (fx.state == DOCKED and len(seen) > 1):
        sc.detector_readings(fx.t)
        fx.step(0.1)
        if fx.arm_phase:
            phases.add(fx.arm_phase)
        q = fx.arm_q
        if prev is not None:
            jump = max(jump, max(abs(q[k] - prev[k]) for k in q if k != "socket_drive_joint"))
        prev = q
        assert q["ur3_wrist_2_joint"] > 0.3 and q["ur3_elbow_joint"] > 0.3
        if not seen or seen[-1] != fx.state:
            seen.append(fx.state)
    assert jump < 0.12
    assert seen == [TRANSIT, SNIFF, INSPECT, RETORQUE, VERIFY, RETURN, DOCKED]
    assert fx.arm.started == [("F5", 3)] and fx.alert["kind"] == "repaired"
    assert {"approach", "engage", "torque", "retract"} <= phases
    assert fx.peak_lel < 20.0 and fx.evidence["nav"] == "ARRIVED"
    assert np.linalg.norm(fx.xy - np.asarray(DOCK["unit2"][:2])) < 0.3
    t_verify = next(r["t"] for r in fx.log if r["to"] == VERIFY)
    assert sc.t_fixed is not None and sc.t_fixed < t_verify - 2.0


class _Walker:
    def __init__(self, xy):
        self.xy, self.yaw, self.state, self.events = np.asarray(xy, float), 0.0, "FOLLOW", []

    def go(self, goal_xy, yaw=None):
        self.goal = np.asarray(goal_xy, float)

    def step(self, dt):
        if len(self.events) == 0:
            self.state = "REPORT"
            self.events.append({"type": "object_on_walkway", "xy": [-4.0, 1.5], "width_m": 0.3, "leg": "go"})
        return False


def test_object_on_walkway_is_reported():
    w, det, fx = rig(lambda w: w.loosen("M2", 1, 0.15))
    fx.walker = _Walker(DOCK)
    fx.command(command(GO_CHECK, "M2", "argon", "check"))
    fx.step(0.1)
    assert fx.reports[-1]["verdict"] == "object_on_walkway" and fx.reports[-1]["xy"] == [-4.0, 1.5]
    assert "object on walkway" in fx.caption and fx.evidence["walkway"]["verdict"] == "object_on_walkway"

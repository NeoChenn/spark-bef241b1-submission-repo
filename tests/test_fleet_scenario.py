import json

import numpy as np
import pytest

from integration.dual_argon import (CLEAR_LEL, LEAK_CONFIRMED, OK, PIPE_SENSOR_FAULT, STOP_LEL, UNVERIFIED,
                                    fuse_fleet)
from integration.fleet_scenario import LEAK_META, Scenario, work_point

N_CLEAR_S = 10.0


@pytest.fixture
def s():
    return Scenario()


def verdict(s, det_id, t, robots):
    s.detector_readings(t)
    sniffers = {rid: (s.sniff(xy, t), xy[:2]) for rid, xy in robots.items()}
    return fuse_fleet(s.detector(det_id), sniffers)


def test_g_f2_goes_amber_then_red_within_seconds_of_the_leak(s):
    leds = [(t, s.leds(t)["G-F2"]) for t in np.arange(0.0, s.t_fixed, 0.1)]
    assert all(c == "green" for t, c in leds if t < s.t_leak)
    amber = next(t for t, c in leds if c == "amber")
    red = next(t for t, c in leds if c == "red")
    assert s.t_leak < amber < red <= s.t_leak + 4.0
    assert s.event("G-F2 amber").t == pytest.approx(amber, abs=0.1)
    assert s.event("G-F2 red").t == pytest.approx(red, abs=0.1)
    assert max(s.detector_readings(t)["G-F2"] for t in np.arange(s.t_leak, s.t_fixed, 0.5)) < STOP_LEL
    assert {c for k, c in s.leds(20.0).items() if k != "G-F2"} == {"green"}


def test_fixed_high_and_scrubber_at_4m_low_is_unverified(s):
    g = s.flanges["F2"]
    unit1 = g.center + 4.0 * g.normal
    t = 20.0
    fd = verdict(s, "G-F2", t, {"unit1": unit1})
    assert 3.8 < s.detector("G-F2").dist(unit1) < 4.6
    assert s.sniff(unit1, t) < CLEAR_LEL
    assert fd.kind == UNVERIFIED and fd.robot == "unit1"


def test_fixer_at_f2_reads_high_so_leak_confirmed(s):
    unit2 = work_point(s.flanges["F2"])
    t = 25.0
    fd = verdict(s, "G-F2", t, {"unit2": unit2})
    assert CLEAR_LEL <= s.sniff(unit2, t) < STOP_LEL
    assert fd.kind == LEAK_CONFIRMED and not fd.hazard


def test_after_retorque_everything_is_green_within_n_seconds(s):
    clear = s.event("gas_clear").t
    assert s.t_fixed < clear <= s.t_fixed + N_CLEAR_S
    assert s.bolt_angle(s.t_fixed) == 0.0 and s.bolt_angle(s.t_leak + 1) == pytest.approx(0.27)
    t = s.t_fixed + N_CLEAR_S
    assert set(s.leds(t).values()) == {"green"}
    assert s.sniff(work_point(s.flanges["F2"]), t) < CLEAR_LEL
    assert verdict(s, "G-F2", t, {"unit2": work_point(s.flanges["F2"])}).kind == OK


def test_g_f1_stuck_high_while_fixer_beside_it_reads_clean_is_sensor_fault(s):
    t = s.t_stuck + 1.0
    assert s.leds(t)["G-F1"] == "red"
    assert s.leds(s.t_stuck - 1.0)["G-F1"] == "green"
    unit2 = work_point(s.flanges["F1"])
    assert s.sniff(unit2, t) < CLEAR_LEL
    assert verdict(s, "G-F1", t, {"unit2": unit2}).kind == PIPE_SENSOR_FAULT


def test_script_order_and_active_events(s):
    names = [e.name for e in s.script]
    assert names.index("leak_start") < names.index("G-F2 amber") < names.index("G-F2 red") \
        < names.index("retorque_done") < names.index("gas_clear") < names.index("G-F1 stuck")
    assert s.at(0.0) == []
    assert {"leak", "G-F2 red"} <= {e.name for e in s.at(20.0)}
    assert {"retorque_done", "source_off", "gas_decay"} <= {e.name for e in s.at(s.t_fixed)}
    assert "leak" not in {e.name for e in s.at(s.t_fixed)}
    assert max(e.t for e in s.script) <= s.duration_s
    json.dumps(s.to_dict())


def test_positions_match_the_leak_flanges_and_detector_prims(s):
    meta = {f["id"]: f for f in json.loads(LEAK_META.read_text())["flanges"]}
    for fid, g in s.flanges.items():
        assert np.allclose(g.center, meta[fid]["center_m"])
        d = s.detector(f"G-{fid}")
        assert 0.2 < np.linalg.norm(np.asarray(d.pos) - g.center) < 0.6
        assert d.dist(work_point(g)) < 2.0
    assert [d.id for d in s.detectors] == [f"G-F{i}" for i in range(1, 7)]
    assert np.allclose(s.detector("G-F2").pos, (-5.557, 23.964, 43.66), atol=1e-3)
    assert np.allclose(s.detector("G-F5").pos, (20.153, -10.396, 43.46), atol=1e-3)

import math

import numpy as np
import pytest

from policy.control.centre import alarms
from policy.control.health import HealthMonitor
from policy.control.snapshot import FixedSensor, GasSensor, SensorSnapshot

IDS = ("G1", "G2", "G3", "G4")


def wave(t, k=0):
    return 1.6 + 0.3 * math.sin(2 * math.pi * t / 40.0) + 0.15 * math.sin(2 * math.pi * t / 13.0 + k)


def feed(fault=None, t_end=40.0, dt=1.0, seed=0, monitor=None, base=wave):
    rng = np.random.default_rng(seed)
    mon = monitor or HealthMonitor()
    out = None
    for t in np.arange(0.0, t_end + 1e-9, dt):
        gas = []
        for k, sid in enumerate(IDS):
            v = base(t, k) + rng.normal(0, 0.05)
            if fault and sid == fault[0]:
                v = fault[1](t, v, rng)
            if v is not None:
                gas.append(GasSensor(sid, zone="N", lel=float(v), group="N"))
        snap = SensorSnapshot(t=float(t), gas=gas)
        out = {h.id: h for h in mon.update(snap)}
    return out, mon, snap


def test_all_ok_on_healthy_correlated_heads():
    h, _, _ = feed()
    assert {x.status for x in h.values()} == {"OK"}


def test_dead_after_silence_with_evidence():
    h, _, _ = feed(("G2", lambda t, v, r: None if t >= 20 else v))
    assert h["G2"].status == "DEAD" and h["G2"].evidence["silent_s"] > 8.0
    assert h["G2"].evidence["last_value"] == pytest.approx(wave(19, 1), abs=0.3)


def test_stuck_flatline_while_neighbours_vary():
    h, _, _ = feed(("G1", lambda t, v, r: 0.0 if t >= 10 else v))
    assert h["G1"].status == "STUCK"
    e = h["G1"].evidence
    assert e["std"] <= 0.01 and e["nbr_std"] > 5 * 0.01 and e["value"] == 0.0


def test_flat_group_is_not_stuck():
    h, _, _ = feed(base=lambda t, k: 1.5)
    assert h["G1"].status == "OK"


def test_drift_is_bias_against_neighbour_median():
    h, _, _ = feed(("G3", lambda t, v, r: v + 2.5))
    assert h["G3"].status == "DRIFT" and h["G3"].hazard
    assert h["G3"].evidence["bias"] == pytest.approx(2.5, abs=0.3) and h["G3"].evidence["tol"] == 1.5


def test_out_of_range_is_physically_impossible():
    h, _, _ = feed(("G4", lambda t, v, r: -20.0 if t >= 30 else v))
    assert h["G4"].status == "OUT_OF_RANGE" and h["G4"].evidence["value"] == -20.0


def test_noisy_against_neighbours():
    h, _, _ = feed(("G2", lambda t, v, r: max(0.0, v + r.normal(0, 0.9))))
    assert h["G2"].status == "NOISY" and h["G2"].evidence["ratio"] > 4


def test_high_gas_is_never_called_drift_or_noise():
    h, _, _ = feed(("G1", lambda t, v, r: v + min(40.0, 2.0 * t) + r.normal(0, 0.5)))
    assert h["G1"].status == "OK"


def test_faulty_head_beside_live_gas_is_hazard_with_peer_max():
    h, _, _ = feed(("G1", lambda t, v, r: None if t >= 20 else v),
                   base=lambda t, k: 25.0 + 0.5 * math.sin(t / 3 + k) if k == 1 else wave(t, k))
    assert h["G1"].status == "DEAD" and h["G1"].hazard
    assert h["G1"].evidence["peer_max"] > 20.0


def test_fixed_sensors_and_recovery():
    mon = HealthMonitor(recover_s=5.0)
    for t in range(0, 30):
        mon.update(SensorSnapshot(t=float(t), fixed=[FixedSensor("P1", "pressure", 999.0 if t < 28 else 12.0)]))
    h = {x.id: x for x in mon.update(SensorSnapshot(t=30.0, fixed=[FixedSensor("P1", "pressure", 12.0)]))}
    assert h["P1"].status == "OUT_OF_RANGE"
    for t in range(31, 40):
        h = {x.id: x for x in mon.update(SensorSnapshot(t=float(t), fixed=[FixedSensor("P1", "pressure", 12.0)]))}
    assert h["P1"].status == "OK"


def test_confirmed_fault_stays_latched_and_clear_resets():
    h, mon, _ = feed(("G1", lambda t, v, r: 0.0 if t >= 10 else v))
    mon.confirm("G1", "SENS-1")
    h, _, _ = feed(monitor=mon, t_end=40.0)
    assert h["G1"].status == "STUCK" and h["G1"].job == "SENS-1"
    mon.clear("G1", trust_s=100.0)
    assert mon.tracks["G1"].status == "OK" and not mon.tracks["G1"].hist


def test_new_fault_raises_an_alarm():
    _, _, snap = feed(("G1", lambda t, v, r: 0.0 if t >= 10 else v))
    assert "health:G1:STUCK" in alarms(snap, [])


def test_observe_flags_a_stuck_high_head_among_moving_peers():
    mon = HealthMonitor()
    h = {}
    for t in range(40):
        vals = {f"G-F{k}": wave(t, k) for k in range(2, 5)}
        vals["G-F1"] = 31.0 if t >= 10 else wave(t, 1)
        h = mon.observe(float(t), vals, group="top")
    assert h["G-F1"].status == "STUCK" and h["G-F1"].evidence["value"] == 31.0
    assert {h[k].status for k in ("G-F2", "G-F3", "G-F4")} == {"OK"}


def test_observe_knows_a_head_that_never_reported():
    mon = HealthMonitor()
    for t in range(12):
        h = mon.observe(float(t), {"G-F1": wave(t), "G-F9": None}, group="top")
    assert h["G-F9"].status == "DEAD"

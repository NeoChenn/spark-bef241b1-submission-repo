import numpy as np
import pytest

from policy.fleet.assign import INF, assign, pick_responder
from policy.fleet.battery import POWER_W, Battery
from policy.fleet.reservations import Reservations, plan_legs
from policy.fleet.rigmap import load_rig_map
from policy.fleet.tasks import Task, priority


def _t(kind, **kw):
    return Task(id=kind + str(len(kw)), kind=kind, poi="x", node=0, created=0.0, **kw)


def test_priority_order_gas_crack_rust_check_inspect():
    p = [priority(_t(k, severity=1.0, exposure=1.0), 0.0) for k in ("gas", "crack", "rust", "leak_check", "inspect")]
    assert p == sorted(p, reverse=True) and len(set(p)) == 5


def test_rust_priority_scales_with_severity_times_exposure():
    lo = priority(_t("rust", severity=0.4, exposure=0.5), 0.0)
    hi = priority(_t("rust", severity=0.8, exposure=0.9), 0.0)
    assert hi > lo > 0
    assert priority(_t("rust", severity=1.0, exposure=1.0), 0.0) < priority(_t("crack"), 0.0)


def test_deadline_raises_priority_as_it_approaches():
    far = priority(_t("inspect", deadline=2000.0), 0.0)
    near = priority(_t("inspect", deadline=2000.0), 1900.0)
    late = priority(_t("inspect", deadline=2000.0), 2100.0)
    assert far < near <= late
    assert late < priority(_t("crack"), 0.0)


def test_battery_drain_by_mode_and_floor():
    b = Battery(capacity_wh=432.0, wh=432.0)
    b.drain("walk", 3600.0)
    assert b.wh == pytest.approx(432.0 - POWER_W["walk"])
    b.drain("stairs", 1e6)
    assert b.wh == 0.0 and b.used_wh == pytest.approx(432.0)


def test_battery_charge_caps_at_capacity():
    b = Battery(capacity_wh=400.0, wh=100.0, charger_w=1200.0)
    b.charge(1800.0)
    assert b.wh == pytest.approx(400.0)
    assert b.charged_wh == pytest.approx(300.0)


def test_battery_affordability_and_return_rule():
    b = Battery(capacity_wh=400.0, wh=100.0, reserve_frac=0.15)
    assert b.can_afford(39.0) and not b.can_afford(41.0)
    assert not b.must_return(30.0) and b.must_return(41.0)


def test_path_energy_uses_stairs_power():
    flat = Battery.path_energy(100.0, 0.0)
    st = Battery.path_energy(100.0, 50.0)
    assert st > flat == pytest.approx(POWER_W["walk"] * 100 / 3600)


def test_reservation_second_robot_waits_for_corridor():
    r = Reservations(clearance=1.0)
    s1 = r.earliest("c", 10.0, 5.0, "A")
    r.reserve("c", s1, s1 + 5.0, "A")
    s2 = r.earliest("c", 12.0, 4.0, "B")
    assert s1 == 10.0 and s2 == pytest.approx(16.0)
    assert r.earliest("c", 12.0, 4.0, "A") == 12.0
    s3 = r.earliest("c", 0.0, 3.0, "B")
    assert s3 == 0.0


def test_reservation_release_future_only():
    r = Reservations()
    r.reserve("c", 0.0, 5.0, "A")
    r.reserve("c", 20.0, 25.0, "A")
    r.release("A", after=10.0)
    assert [x[:2] for x in r.table["c"]] == [(0.0, 5.0)]
    r.release("A")
    assert r.table["c"] == []


def test_plan_legs_opposite_robots_never_share_a_corridor():
    rm = load_rig_map()
    r = Reservations()
    a, b = rm.poi_node["F6"], rm.poi_node["gauge_06"]
    _, pa = rm.route(a, b)
    _, pb = rm.route(b, a)
    la = plan_legs(rm, r, "A", pa, 0.0)
    lb = plan_legs(rm, r, "B", pb, 0.0)
    spans = {}
    for rid, legs in (("A", la), ("B", lb)):
        for lg in legs:
            if lg.corridor:
                lo, hi = spans.get((rid, lg.corridor), (1e9, -1e9))
                spans[(rid, lg.corridor)] = (min(lo, lg.t0), max(hi, lg.t1))
    shared = {c for (_, c) in spans if ("A", c) in spans and ("B", c) in spans}
    assert shared
    for c in shared:
        (a0, a1), (b0, b1) = spans[("A", c)], spans[("B", c)]
        assert a1 <= b0 or b1 <= a0
    assert lb[0].t0 > 0.0 or any(l1.t0 > l0.t1 + 1e-9 for l0, l1 in zip(lb, lb[1:]))
    assert all(l1.t0 >= l0.t1 - 1e-9 for l0, l1 in zip(lb, lb[1:]))
    assert lb[-1].v == a and la[-1].v == b


def test_hungarian_beats_greedy():
    cost = np.array([[1.0, 2.0], [2.0, 10.0]])
    assert sorted(assign(cost)) == [(0, 1), (1, 0)]


def test_assign_drops_infeasible_and_picks_cheapest_tasks():
    cost = np.array([[5.0, INF, 1.0], [INF, INF, INF]])
    assert assign(cost) == [(0, 2)]


def test_assign_more_tasks_than_robots_prefers_high_priority():
    eta = np.array([[10.0, 30.0, 20.0]])
    prio = np.array([1.0, 100.0, 8.0])
    cost = eta - 30.0 * prio
    assert assign(cost) == [(0, 1)]


def test_pick_responder_min_eta_feasible():
    eta = np.array([40.0, 10.0, 25.0])
    ok = np.array([True, False, True])
    assert pick_responder(eta, ok) == 2
    assert pick_responder(eta, np.zeros(3, bool)) is None

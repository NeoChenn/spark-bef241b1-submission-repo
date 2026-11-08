import json
from pathlib import Path

import numpy as np
import pytest

from policy.fleet.fleet import Fleet
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.rigmap import load_rig_map
from policy.fleet.scenario import build_default

SAMPLE = Path("runs_sample/fleet")


def stub_jobs(work_s=10.0):
    def run(kind, scenario):
        out = "ESCALATE_TO_HUMAN" if scenario == "crack" else "DONE"
        return JobTrace(kind, scenario, work_s, out, [(0.0, "WORK", "stub"), (work_s, out, "end")])
    return JobLibrary(run)


def make(robots, work_s=10.0):
    rm = load_rig_map()
    f = Fleet(rm, stub_jobs(work_s))
    for rid, poi, soc in robots:
        f.add_robot(rid, rm.poi_node[poi] if isinstance(poi, str) else poi, soc)
    return f, rm


def run_until(f, cond, t_max=3000.0):
    while f.t < t_max:
        f.step()
        if cond():
            return True
    return False


def by_id(f, rid):
    return next(r for r in f.robots if r.id == rid)


def test_single_task_assigned_travelled_and_done():
    f, rm = make([("R1", "dock", 1.0)])
    task = f.add_task("inspect", "gauge_03")
    assert run_until(f, lambda: task.status == "done")
    assert task.arrived is not None and task.finished >= task.arrived + task.duration - 1.0
    assert by_id(f, "R1").dist_m > 0


def test_gas_alarm_preempts_busy_robot_and_rust_resumes_later():
    f, rm = make([("R1", "gauge_05", 1.0)], work_s=200.0)
    rust = f.add_task("rust", "rust_east", xyz=[12.0, -5.2, 43.0], severity=0.8, exposure=0.8)
    assert run_until(f, lambda: rust.status == "active" and rust.progress > 5)
    f.schedule(f.t, "gas_alarm", flange="F2", scenario="loose")
    f.step()
    r1 = by_id(f, "R1")
    gas = next(t for t in f.tasks.values() if t.kind == "gas")
    assert r1.task is gas and rust.status == "pending" and rust.preemptions == 1
    kept = rust.progress
    assert run_until(f, lambda: gas.status == "done")
    al = f.alarms[gas.id]
    assert al["responder"] == "R1" and al["response_s"] > 0
    assert run_until(f, lambda: rust.status == "done")
    assert rust.progress >= kept


def test_gas_goes_to_nearest_robot_even_if_busy():
    f, rm = make([("NEAR", "gauge_05", 1.0), ("FAR", "gauge_01", 1.0)], work_s=300.0)
    f.add_task("rust", "rust_east", xyz=[19.0, -5.2, 43.0], severity=0.5, exposure=0.5)
    run_until(f, lambda: by_id(f, "NEAR").state == "WORK", 400)
    f.schedule(f.t, "gas_alarm", flange="F5", scenario="loose")
    f.step()
    gas = next(t for t in f.tasks.values() if t.kind == "gas")
    assert gas.robot == "NEAR"


def test_gas_zone_suspends_nearby_rust_work_until_resolved():
    f, rm = make([("R1", "gauge_05", 1.0), ("R2", "F5", 1.0)], work_s=60.0)
    rust = f.add_task("rust", "rust_f5", xyz=[18.0, -5.2, 43.0], severity=0.9, exposure=0.9)
    run_until(f, lambda: rust.status == "active", 400)
    f.schedule(f.t, "gas_alarm", flange="F5", scenario="loose")
    f.step()
    gas = next(t for t in f.tasks.values() if t.kind == "gas")
    assert gas.robot == "R2"
    while gas.status != "done":
        assert rust.status != "active"
        f.step()
    assert run_until(f, lambda: rust.status == "done")


def test_gas_supersedes_routine_check_of_same_flange():
    f, rm = make([("R1", "dock", 1.0)])
    chk = f.add_task("leak_check", "F3")
    f.schedule(0.0, "gas_alarm", flange="F3", scenario="loose")
    f.step()
    assert chk.status == "superseded"


def test_robot_failure_reassigns_its_task():
    f, rm = make([("R1", "dock", 1.0), ("R2", "gauge_01", 1.0)], work_s=100.0)
    task = f.add_task("inspect", "gauge_07")
    f.step()
    first = task.robot
    f.schedule(f.t + 5, "robot_failure", robot=first)
    assert run_until(f, lambda: task.status == "done")
    dead = by_id(f, first)
    assert dead.state == "FAILED" and task.robot != first
    wh = dead.battery.wh
    f.step()
    assert dead.battery.wh == wh


def test_low_battery_robot_returns_to_charge_then_works():
    f, rm = make([("R1", "F6", 0.16)], work_s=30.0)
    task = f.add_task("inspect", "gauge_01")
    states = []
    run_until(f, lambda: states.append(by_id(f, "R1").state) or task.status == "done")
    assert "CHARGING" in states
    assert states.index("CHARGING") < len(states) - 1
    r = by_id(f, "R1")
    assert r.battery.charged_wh > 0 and r.battery.soc >= r.battery.reserve_frac
    assert any(e["kind"] == "return_to_charge" for e in f.event_log)


def test_infeasible_task_skipped_for_low_robot_when_healthy_one_available():
    f, rm = make([("LOW", "dock", 0.155), ("OK", "gauge_07", 1.0)])
    task = f.add_task("rust", "rust_far", xyz=[0.0, -6.8, 32.0], severity=0.5, exposure=0.5)
    f.step()
    assert task.robot == "OK"


def test_person_on_deck_pauses_nearby_robot_then_resumes():
    f, rm = make([("R1", "dock", 1.0)], work_s=40.0)
    task = f.add_task("inspect", "gauge_03")
    assert run_until(f, lambda: task.status == "active")
    f.schedule(f.t, "person", xyz=rm.pos[task.node].tolist(), radius=3.0, duration=30.0)
    f.step()
    r = by_id(f, "R1")
    p0 = task.progress
    for _ in range(10):
        f.step()
        assert r.state == "PAUSED"
    assert task.progress == p0
    assert run_until(f, lambda: task.status == "done")


def test_person_blocks_walkway_and_robot_detours():
    f, rm = make([("R1", "dock", 1.0)])
    t_free, path = rm.route(rm.dock, rm.poi_node["F5"])
    blocker = rm.pos[path[len(path) // 3]]
    f.schedule(0.0, "person", xyz=blocker.tolist(), radius=3.0, duration=2000.0)
    task = f.add_task("leak_check", "F5")
    f.step()
    r = by_id(f, "R1")
    for lg in r.legs:
        p = rm.pos[lg.v]
        assert np.linalg.norm(p[:2] - blocker[:2]) >= 3.0 or abs(p[2] - blocker[2]) > 3
    assert run_until(f, lambda: task.status == "done")


def test_new_rust_event_adds_and_completes_task():
    f, rm = make([("R1", "dock", 1.0)])
    f.schedule(5.0, "new_rust", poi="rust_new", xyz=[-21.8, 6.0, 32.0], severity=0.9, exposure=1.0)
    assert run_until(f, lambda: "rust:rust_new" in f.tasks and f.tasks["rust:rust_new"].status == "done")
    assert any(e["kind"] == "new_rust" for e in f.event_log)


def test_crack_task_escalates():
    f, rm = make([("R1", "dock", 1.0)])
    task = f.add_task("crack", "F1", scenario="crack")
    assert run_until(f, lambda: task.status not in ("pending", "assigned", "active"))
    assert task.status == "escalated"


@pytest.fixture(scope="module")
def default_run():
    f = build_default(3, jobs=stub_jobs(20.0))
    f.run(2400.0, on_tick=lambda fl: fl.check_invariants())
    return f


def test_default_scenario_completes_without_corridor_conflicts(default_run):
    f = default_run
    m = f.metrics()
    assert m["corridor_conflicts"] == 0
    assert m["open_tasks"] == 0
    assert m["gas_alarms"] == 2 and m["gas_response_mean_s"] > 0
    assert m["robot_failures"] == 1 and m["preemptions"] >= 1
    assert 0.0 <= m["idle_pct"] <= 100.0 and m["energy_used_wh"] > 0
    assert all(r.state in ("FAILED", "CHARGING", "DOCKED", "DOCK_QUEUE") for r in f.robots)


def test_default_scenario_reservation_table_has_no_overlaps(default_run):
    for c, xs in default_run.res.table.items():
        xs = sorted(xs)
        for (a0, a1, ra), (b0, b1, rb) in zip(xs, xs[1:]):
            assert ra == rb or b0 >= a1 - 1e-6, c


def test_snapshot_matches_committed_sample(default_run):
    snap = default_run.snapshot()
    sample = json.loads((SAMPLE / "fleet_state.json").read_text())
    assert set(snap) == set(sample)
    assert set(snap["robots"][0]) == set(sample["robots"][0])
    assert set(snap["tasks"][0]) == set(sample["tasks"][0])
    assert set(snap["metrics"]) == set(sample["metrics"])
    json.dumps(snap)
    m = json.loads((SAMPLE / "fleet_map.json").read_text())
    assert set(m) == set(default_run.rm.to_json())


def test_walking_person_pauses_robot_only_while_close():
    f, rm = make([("R1", "dock", 1.0)], work_s=200.0)
    task = f.add_task("inspect", "gauge_03")
    assert run_until(f, lambda: task.status == "active")
    w = rm.pos[task.node]
    f.schedule(f.t, "person", path=[(w + [-15, 0, 0]).tolist(), (w + [15, 0, 0]).tolist()], speed=1.0,
               radius=3.0, duration=40.0)
    r = by_id(f, "R1")
    states = []
    for _ in range(40):
        f.step()
        states.append(r.state)
    assert states[0] == "WORK" and states[-1] == "WORK"
    assert 3 <= states.count("PAUSED") <= 8


def test_sim_run_writes_log_summary_and_png(tmp_path):
    from policy.fleet.sim import run
    f, s = run(3, str(tmp_path), jobs=stub_jobs(20.0))
    lines = (tmp_path / "fleet_log.jsonl").read_text().splitlines()
    first, tick, last = json.loads(lines[0]), json.loads(lines[1]), json.loads(lines[-1])
    assert first["type"] == "map" and last["type"] == "summary" and tick["type"] == "tick"
    assert {"robots", "tasks", "assignments", "events", "zones", "alarms", "metrics"} <= set(tick)
    assert {"id", "pose", "state", "battery", "task", "path"} <= set(tick["robots"][0])
    assert len(lines) == round(f.t / f.dt) + 2
    for name in ("summary.json", "fleet_state.json", "fleet_map.json", "fleet_map.png", "fleet_final.png"):
        assert (tmp_path / name).stat().st_size > 0
    assert s["open_tasks"] == 0 and s["corridor_conflicts"] == 0


def test_committed_sample_log_head_shape():
    lines = (SAMPLE / "fleet_log_head.jsonl").read_text().splitlines()
    assert [json.loads(x)["type"] for x in lines] == ["map", "tick", "tick", "summary"]
    s = json.loads((SAMPLE / "summary.json").read_text())
    assert s["gas_alarms"] == 2 and s["open_tasks"] == 0 and s["corridor_conflicts"] == 0

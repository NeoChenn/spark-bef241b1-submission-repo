from policy.fleet.jobs import SETUP_S
from policy.weld.job import weld_trace
from policy.weld.live import fleet_weld_views, replay, weld_view
from tests.test_fleet_sim import stub_jobs
from tests.test_weld_fleet import WELD_XYZ


def test_replay_follows_the_job_trace_timeline():
    r = replay("crack")
    tr = weld_trace("crack")
    for st_t, st, _ in tr.segments:
        v = weld_view("crack", st_t + 0.6)
        assert v["state"] == st
    v = weld_view("crack", 1e9)
    assert v["state"] == "DONE" and set(v["pip"]) == {"jpg", "w", "h", "caption", "seq"}
    assert v["evidence"]["job"] == "weld" and "coverage" in v["evidence"]
    assert weld_view("crack", -5)["state"] == "SETUP"
    assert len(r.frames) > 50


def test_fleet_weld_views_for_working_robots():
    from policy.fleet.fleet import Fleet
    from policy.fleet.rigmap import load_rig_map
    rm = load_rig_map()
    f = Fleet(rm, stub_jobs(200.0))
    w = f.add_task("weld", "weld_x", WELD_XYZ, scenario="crack")
    f.add_robot("R1", w.node, 0.9)
    assert fleet_weld_views(f) == {}
    for _ in range(int(SETUP_S["weld"]) + 30):
        f.step()
    v = fleet_weld_views(f)
    assert set(v) == {"R1"} and v["R1"]["state"] in ("INSPECT", "GRIND")

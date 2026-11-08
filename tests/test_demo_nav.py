import numpy as np

from rig.mission.live import MissionLive


def drop(m, kind="crate"):
    ev = m.scen.drop_ahead("drop_can" if kind == "can" else "drop_obstacle", {"kind": kind, "at": "ahead"})
    if ev is not None:
        m.event(ev["name"], ev["args"])
    for _ in range(200):
        if m.fleet.f.hz.obstacles:
            break
        m.advance(0.25)
    return list(m.fleet.f.hz.obstacles.values())[-1]


def test_nav_demo_loops_and_a_drop_ahead_lands_on_the_leg_and_sends_r3_the_other_way():
    m = MissionLive(obs=False)
    for at, kind in ((60, "crate"), (91, "can"), (142, "barrel"), (290, "crate"), (333, "can")):
        assert m.event("demo", {"id": "nav"})["selected"] == "R3"
        sim = m.take_sim()
        assert {"name": "views", "args": {"robot": "R3", "on": True, "kind": "fpv"}} in sim
        assert any(e["name"] == "camera" and e["args"]["lens"]["dist"] < 10 for e in sim)
        r = m._robot("R3")
        assert not m.fleet.f.hz.obstacles and tuple(r.pos[:2]) == (9.0, -4.5) and m.fleet.speed == 1.5
        for _ in range(at):
            m.advance(0.25)
        ob = drop(m, kind)
        assert ob["kind"] == kind and not ob.get("rolling") and not any(t.kind == "pick_can" for t in m.fleet.f.tasks.values())
        c = np.asarray(ob["xyz"][:2])
        assert abs(float(np.linalg.norm(c - r.pos[:2])) - 3.0) < 0.6
        assert float(np.linalg.norm(c - np.asarray(m.scen.cam_at[:2]))) < 3.0
        d0 = float(np.linalg.norm(r.pos[:2] - c))
        for _ in range(40):
            m.advance(0.25)
            assert float(np.linalg.norm(r.pos[:2] - c)) > ob["radius"]
        assert m.scen.cur.i == 1 and float(np.linalg.norm(r.pos[:2] - c)) > d0 + 1.0
    m.event("scenario", {"id": "gas"})
    assert m.fleet.speed == 4.0

import numpy as np

from rig.mission.live import MissionLive


def test_nav_demo_loops_and_a_crate_ahead_sends_r3_the_other_way():
    m = MissionLive(obs=False)
    for at in (60, 91, 142, 290):
        assert m.event("demo", {"id": "nav"})["selected"] == "R3"
        sim = m.take_sim()
        assert {"name": "views", "args": {"robot": "R3", "on": True, "kind": "fpv"}} in sim
        assert any(e["name"] == "camera" and e["args"]["mode"] == "deck" for e in sim)
        r = m._robot("R3")
        assert not m.fleet.f.hz.obstacles and tuple(r.pos[:2]) == (9.0, -4.5)
        for _ in range(at):
            m.advance(0.25)
        xyz = m.scen.ahead(3.0)
        assert 1.5 < float(np.linalg.norm(np.subtract(xyz[:2], r.pos[:2]))) < 7.0
        m.event("drop_obstacle", {"kind": "crate", "at": xyz})
        start = r.pos[:2].copy()
        for _ in range(240):
            m.advance(0.25)
            assert float(np.linalg.norm(r.pos[:2] - np.asarray(xyz[:2]))) > 1.0
        assert m.scen.cur.i == 1 and float(np.linalg.norm(r.pos[:2] - start)) > 3.0

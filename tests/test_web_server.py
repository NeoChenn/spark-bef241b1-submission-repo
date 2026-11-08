

def test_pages_bust_asset_cache():
    from fastapi.testclient import TestClient
    from rig.web.server import create_app
    with TestClient(create_app(sim_port=0)) as c:
        for path in ("/", "/interactive"):
            r = c.get(path)
            assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
            assert ".css?v=" in r.text and ".js?v=" in r.text


def test_live_haze_skips_empty_depth():
    import numpy as np
    from sim import look
    look._LIVE.clear()
    look._LIVE["n"] = 0
    rgb = np.zeros((8, 8, 3), np.uint8)
    cam = {"T": np.eye(4), "fov": 1.0}
    out = look.inscatter(rgb, lambda: np.zeros(()), cam, "dusk", np.array([0.0, 0.0, 1.0]), tier="live")
    assert out is rgb


def test_camera_works_without_a_scenario():
    from fastapi.testclient import TestClient
    from rig.web.server import create_app
    from rig.web.server import default_fleet
    import time as _t
    with TestClient(create_app(sim_port=0, fleet=default_fleet())) as c:
        r = c.post("/api/event", json={"name": "camera", "args": {"mode": "deck"}})
        assert r.status_code == 200, r.text
        _t.sleep(1.05)
        r = c.post("/api/event", json={"name": "camera", "args": {"mode": "zoom", "view": "rust"}})
        assert r.status_code == 200, r.text


def test_drop_can_in_view_lands_at_the_camera_target():
    from fastapi.testclient import TestClient
    from rig.web.server import create_app, default_fleet
    import time as _t
    with TestClient(create_app(sim_port=0, fleet=default_fleet())) as c:
        assert c.post("/api/event", json={"name": "camera", "args": {"mode": "deck"}}).status_code == 200
        _t.sleep(1.05)
        r = c.post("/api/event", json={"name": "drop_can", "args": {"size": "small", "at": "view"}})
        assert r.status_code == 200, r.text
        assert "queued for the nearest gripper" in r.text

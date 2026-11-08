

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

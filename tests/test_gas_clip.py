import numpy as np
import pytest

from policy.gasbot import clip, sees


@pytest.fixture(scope="module")
def p():
    return clip.plan()


def camera(W, H, eye=(22.9, -13.7, 46.2), at=(17.3, -9.8, 43.0), focal=18.0, ap=20.955):
    eye, at = np.asarray(eye, float), np.asarray(at, float)
    f = (at - eye) / np.linalg.norm(at - eye)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    T = np.eye(4)
    T[:3, :3] = np.c_[r, np.cross(f, r), f]
    T[:3, 3] = eye
    fx = focal / ap * W
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dz = (((u - K[0, 2]) / K[0, 0])[..., None] * T[:3, 0] + ((v - K[1, 2]) / K[1, 1])[..., None] * T[:3, 1] + T[:3, 2])[..., 2]
    depth = np.where(dz < -1e-6, (42.5 - eye[2]) / np.minimum(dz, -1e-6), np.inf).astype(np.float32)
    return K, T, depth


def test_deterministic(p):
    q = clip.plan()
    assert p["metrics"] == q["metrics"]
    for k in ("root", "legs", "particles", "field", "samples"):
        assert np.array_equal(p["arrays"][k], q["arrays"][k])


def test_localises_and_closes(p):
    m, fr = p["metrics"], p["frames"]
    src = np.asarray(p["src"])
    assert len(fr) == 285 and 145 <= m["found_frame"] <= 170
    assert m["err_m"] < 1.0
    assert np.linalg.norm(np.asarray(fr[m["found_frame"]]["estimate"][:2]) - src[:2]) < 1.0
    assert fr[0]["belief"]["sigma"] > 2 * m["sigma_found"]
    assert [f["phase"] for f in fr[:m["found_frame"]]] == ["SEEK"] * m["found_frame"]
    assert fr[0]["root"][:2] == pytest.approx([12.0, -10.0], abs=0.05)
    assert fr[60]["root"][0] > 13.0
    c = m["close_frame"]
    assert 230 <= c <= 250
    assert all(f["leak_on"] for f in fr[:c]) and not any(f["leak_on"] for f in fr[c:])
    assert float(fr[c - 1]["field"].max()) > 0.1
    assert float(fr[-1]["field"].max()) < 0.005
    assert len(fr[-1]["samples"]) == m["n_samples"] > 0 and len(fr[0]["legs"]) == 12


def test_save_load(p, tmp_path):
    path = tmp_path / "plan.npz"
    clip.save(p, path)
    q = clip.load(path)
    assert q["metrics"] == p["metrics"] and q["estimate"] == p["estimate"]
    for a, b in zip(p["frames"], q["frames"]):
        assert a["phase"] == b["phase"] and a["arrow"] == b["arrow"] and a["valve_s"] == b["valve_s"]
        assert np.array_equal(a["field"], b["field"]) and np.array_equal(a["particles"], b["particles"])
    assert path.stat().st_size < 60e6


def test_draw(p):
    W, H = 480, 270
    K, T, depth = camera(W, H)
    geo = np.full((H, W, 3), 40, np.uint8)
    for i in (20, 150, 200):
        out = sees.draw(geo, K, T, depth, p["frames"][i], p)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        assert np.abs(out.astype(int) - geo).sum() > 1000

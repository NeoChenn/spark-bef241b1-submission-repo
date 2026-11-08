import numpy as np
import pytest

from policy.weld.fx import Smoke, Sparks, arc_flicker, bead_mesh, glare, pipe_normals, project

W, H = 640, 360
K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
TIP = np.array([0.0, 0.0, 0.45])


def _cam():
    eye = TIP + np.array([0.0, -0.6, 0.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = (1, 0, 0), (0, 0, -1), (0, 1, 0), eye
    return T


def test_flicker_deterministic_bounded():
    ts = np.linspace(0, 7, 2000)
    a = [arc_flicker(t, 3) for t in ts]
    assert a == [arc_flicker(t, 3) for t in ts]
    assert min(a) >= 0.6 and max(a) <= 1.3
    assert np.std(a) > 0.05 and abs(np.mean(a) - 1.0) < 0.2
    assert a != [arc_flicker(t, 4) for t in ts]


def test_sparks_fall_and_die():
    sp = Sparks(0)
    for _ in range(5):
        sp.step(1 / 30, TIP, (0, 0, 1), (1, 0, 0), True)
    n = len(sp)
    assert n > 50
    vz = sp.vel[:, 2].mean()
    sp.step(1 / 30, on=False)
    assert sp.vel[:, 2].mean() < vz
    for _ in range(30):
        sp.step(1 / 30, on=False)
    assert len(sp) == 0


def test_sparks_respect_pipe_and_ground():
    sp = Sparks(1, kill_on_hit=0.0, life=(2.0, 2.0))
    pipe = (np.array([0.0, 0.0, 0.3]), np.array([1.0, 0.0, 0.0]), 0.15)
    for _ in range(40):
        sp.step(1 / 60, TIP + [0, 0, 0.001], (0, 0, 1), (1, 0, 0), True, pipe=pipe)
    d = sp.pos - pipe[0]
    r = np.linalg.norm(d[:, 1:], axis=-1)
    assert (r[sp.age > 0.05] >= 0.15 - 1e-6).all()
    assert (sp.pos[:, 2] >= 0).all()


def test_draw_brightens_near_tip():
    T = _cam()
    img = np.full((H, W, 3), 20, np.uint8)
    sp, sm = Sparks(2), Smoke(3)
    for _ in range(6):
        sp.step(1 / 30, TIP, (0, 0, 1), (1, 0, 0), True)
        sm.step(1 / 30, TIP, True)
    uv, z = project(TIP, K, T)
    u, v = int(uv[0]), int(uv[1])
    for out in (sp.draw(img, K, T), glare(img, uv, z, None, 1.0, 1.0), sm.draw(img, K, T)):
        assert out.shape == img.shape and out.dtype == np.uint8
    a = sp.draw(img, K, T)
    assert a[v - 40:v + 40, u - 40:u + 40].astype(int).sum() > img[v - 40:v + 40, u - 40:u + 40].astype(int).sum() * 1.2
    g = glare(img, uv, z, None, 1.0, 1.0)
    assert g[v, u].min() == 255 and g[v, u + 150].mean() > 20 and g[5, 5].mean() < 40
    occ = np.full((H, W), 0.1, np.float32)
    assert glare(img, uv, z, occ, 1.0, 1.0)[v, u].min() < 255
    assert np.array_equal(sp.draw(img, K, T, depth=occ), img)


def _toy():
    s = np.linspace(0, 0.06, 25)
    pts = np.stack([s, np.zeros_like(s), np.full_like(s, 0.15)], -1)
    return pts, pipe_normals(pts, (0, 0, 0), (1, 0, 0)), np.full(25, 0.008), np.linspace(5.0, 0.0, 25)


def test_bead_mesh_topology():
    pts, nrm, w, age = _toy()
    v, c, i, b, n = bead_mesh(pts, nrm, w, age, segs=9, with_normals=True)
    assert v.shape == (25 * 9, 3) and n.shape == v.shape
    assert c.sum() == len(i) and len(c) == 24 * 8 + 2 and len(b) == len(c)
    assert i.min() == 0 and i.max() == len(v) - 1
    assert b.min() == 0 and b.max() == 5 and b[0] == 5 and b[23 * 8] == 0
    top = v.reshape(25, 9, 3)[:, 4, 2] - 0.15
    assert 0.0005 < top.max() <= 0.0021
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)
    assert bead_mesh(pts[:1], nrm[:1], w[:1], age[:1])[0].shape == (0, 3)


def test_usd_bead_and_light():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdLux

    from policy.weld.fx import WeldBead, arc_light

    stage = Usd.Stage.CreateInMemory()
    bead = WeldBead(stage)
    pts, nrm, w, age = _toy()
    for k in (5, 25):
        b = bead.update(pts[:k], None, w[:k], age[:k], pipe=((0, 0, 0), (1, 0, 0)))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Weld/bead"))
        assert len(mesh.GetPointsAttr().Get()) == k * 9
        subs = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        assert len(subs) == 6 and sum(len(s.GetIndicesAttr().Get()) for s in subs) == len(b)
    set_arc = arc_light(stage, scale=2.0)
    set_arc((0.1, 0.2, 0.3), True, 1.1, normal=(0, 0, 1))
    lt = UsdLux.SphereLight(stage.GetPrimAtPath("/World/Weld/arc"))
    assert lt.GetIntensityAttr().Get() == pytest.approx(1.0e8 * 2.0 * 1.1)
    set_arc((0.1, 0.2, 0.3), False)
    assert lt.GetIntensityAttr().Get() == 0.0

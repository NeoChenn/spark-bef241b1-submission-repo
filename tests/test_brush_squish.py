import numpy as np
import pytest

from policy.brush_squish import EPS, BrushPad, compression, quat_to_rot, squish_local, vertex_normals

R = 0.15
O = np.zeros(3)
A = np.array([1.0, 0.0, 0.0])


def rot_from(normal, tilt, spin):
    x = -np.asarray(normal, float)
    t = np.cross(x, A)
    t /= np.linalg.norm(t)
    c, s = np.cos(tilt), np.sin(tilt)
    x = c * x + s * A
    y = np.cross(t, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    m = np.stack([x, y, z], 1)
    cs, ss = np.cos(spin), np.sin(spin)
    return m @ np.array([[1, 0, 0], [0, cs, -ss], [0, ss, cs]])


def poses(n=200, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        th = rng.uniform(0.0, 2.0)
        normal = np.array([0.0, np.cos(th), np.sin(th)])
        gap = rng.uniform(-0.02, 0.03)
        pos = (R + gap) * normal + rng.uniform(-0.2, 0.2) * A
        yield pos, rot_from(normal, rng.uniform(-0.5, 0.5), rng.uniform(-np.pi, np.pi)), rng.uniform(0.0, 55.0)


def radial(pts):
    return np.linalg.norm(pts - (pts @ A)[:, None] * A, axis=1)


def test_no_vertex_inside_the_pipe():
    pad = BrushPad()
    for pos, rot, force in poses():
        for pts in (pad.sponge(pos, rot, O, A, R, force), pad.hub(pos, rot, O, A, R)):
            assert radial(pts).min() >= R + EPS - 1e-9


def test_compression_is_monotonic_and_capped():
    f = np.linspace(0.0, 80.0, 161)
    c = compression(f)
    assert c[0] == 0.0 and np.all(np.diff(c) >= 0) and c.max() == pytest.approx(0.6)
    assert compression(40.0) == pytest.approx(0.6)


def test_free_pad_thins_and_bulges_with_force():
    pad = BrushPad()
    far = np.array([0.0, 2.0, 0.0])
    rot = rot_from([0.0, 1.0, 0.0], 0.0, 0.0)
    thick, wide = [], []
    for force in np.linspace(0.0, 40.0, 9):
        pts = squish_local(pad.rest, pad.thickness, float(compression(force)), pad.bulge)
        thick.append(np.ptp(pts[:, 0]))
        wide.append(np.linalg.norm(pts[:, 1:], axis=1).max())
    assert np.all(np.diff(thick) < 0) and np.all(np.diff(wide) > 0)
    assert np.allclose(pad.sponge(far, rot, O, A, R, 0.0), far + pad.rest @ rot.T)


def test_pressed_face_lies_on_the_curve():
    pad = BrushPad()
    normal = np.array([0.0, 0.6, 0.8])
    rot = rot_from(normal, 0.0, 0.0)
    for force in (5.0, 20.0, 38.0):
        pen = force / 4000.0
        pts = pad.sponge((R - pen) * normal, rot, O, A, R, force)
        face = pad.u > 0.999
        assert np.abs(radial(pts[face]) - (R + EPS)).max() < 1e-3


def test_lifted_pad_springs_back():
    pad = BrushPad()
    normal = np.array([0.0, 1.0, 0.0])
    rot = rot_from(normal, 0.0, 0.0)
    pos = (R + EPS + 0.01) * normal
    assert np.allclose(pad.sponge(pos, rot, O, A, R, 0.0), pos + pad.rest @ rot.T)


def test_normals_point_out_and_quat_is_a_rotation():
    pad = BrushPad()
    pts = pad.rest
    n = vertex_normals(pts, pad.quads)
    side = (np.abs(pts[:, 0] + pad.thickness / 2) < pad.thickness * 0.3) & (np.linalg.norm(pts[:, 1:], axis=1) > 0.03)
    assert (np.einsum("ij,ij->i", n[side, 1:], pts[side, 1:]) > 0).all()
    q = np.array([0.9, 0.1, -0.3, 0.2])
    m = quat_to_rot(q / np.linalg.norm(q))
    assert np.allclose(m @ m.T, np.eye(3)) and np.linalg.det(m) == pytest.approx(1.0)

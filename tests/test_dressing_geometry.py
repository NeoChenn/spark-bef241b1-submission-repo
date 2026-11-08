import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import dressing_geometry as dg


def test_sag_curve_hits_endpoints_and_drops_in_the_middle():
    pts = dg.sag((0, 0, 5), (4, 0, 5), 1.0, 21)
    assert np.allclose(pts[0], (0, 0, 5)) and np.allclose(pts[-1], (4, 0, 5))
    assert math.isclose(pts[10][2], 4.0, abs_tol=1e-9)
    assert (pts[:, 2] <= 5 + 1e-9).all()


def test_spline_passes_through_controls():
    ctrl = np.array([(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0)], float)
    pts = dg.spline(ctrl, 8)
    for c in ctrl:
        assert np.min(np.linalg.norm(pts - c, axis=1)) < 1e-9


def test_frames_are_orthonormal_and_follow_tangent():
    pts = dg.spline(np.array([(0, 0, 0), (2, 0, 0), (2, 2, 1), (0, 3, 1)], float), 10)
    t, n, b = dg.frames(pts)
    assert np.allclose(np.linalg.norm(t, axis=1), 1) and np.allclose(np.linalg.norm(n, axis=1), 1)
    assert np.allclose((t * n).sum(1), 0, atol=1e-9) and np.allclose((t * b).sum(1), 0, atol=1e-9)
    assert np.allclose(np.cross(t, n), b)


def test_frames_do_not_twist_on_a_straight_line():
    pts = np.stack([np.linspace(0, 5, 20), np.zeros(20), np.zeros(20)], axis=1)
    _, n, _ = dg.frames(pts)
    assert np.allclose(n, n[0])


def test_tube_mesh_shape():
    pts = dg.sag((0, 0, 1), (3, 0, 1), 0.3, 12)
    mesh = dg.tube(pts, 0.05, 8)
    assert mesh["points"].shape == (12 * 9, 3)
    assert len(mesh["counts"]) == 11 * 8 and set(mesh["counts"]) == {4}
    assert mesh["indices"].max() < len(mesh["points"])
    radial = np.linalg.norm(mesh["points"][:9] - pts[0], axis=1)
    assert np.allclose(radial, 0.05)
    assert mesh["uvs"][:, 1].max() > mesh["uvs"][:, 1].min()


def test_coil_spirals_outward_on_the_deck():
    pts = dg.coil((1, 2, 10), 0.3, 0.6, 3.0, 0.02)
    radii = np.hypot(pts[:, 0] - 1, pts[:, 1] - 2)
    assert radii[0] < radii[-1]
    assert math.isclose(radii[0], 0.3, rel_tol=1e-6) and math.isclose(radii[-1], 0.6, rel_tol=1e-6)
    assert np.allclose(pts[:, 2], 10.02)


def test_chain_links_alternate_and_space_evenly():
    pts = dg.sag((0, 0, 3), (2, 0, 3), 0.4, 80)
    links = dg.chain_links(pts, 0.08)
    centres = np.array([c for c, _ in links])
    gaps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    assert np.allclose(gaps, gaps.mean(), rtol=0.08)
    rolls = [m[:, 1] for _, m in links[:2]]
    assert abs(np.dot(rolls[0], rolls[1])) < 0.2


def test_link_mesh_is_closed_ring():
    mesh = dg.link_mesh(0.08, 0.03, 0.008)
    assert mesh["points"].shape[1] == 3
    ext = mesh["points"].max(0) - mesh["points"].min(0)
    assert ext[0] > ext[1] and ext[2] < ext[1]


def test_drape_covers_box_and_falls_to_floor():
    boxes = [((0, 0), (1, 1), 1.2)]
    xs, ys, zs = dg.drape(boxes, (-0.5, -0.5, 1.5, 1.5), 0.0, 24, seed=1)
    assert zs.shape == (24, 24)
    inside = (xs > 0.2) & (xs < 0.8) & (ys > 0.2) & (ys < 0.8)
    assert zs[inside].min() > 1.15
    assert zs.min() >= 0.0 and zs[0, 0] < 0.4


def test_keep_out_rejects_route_zone_and_eyes():
    route = [(0, 0), (10, 0)]
    rects = [(20, 30, -5, 5)]
    eyes = [(5, 6)]
    assert dg.blocked((5, 0.5), route, rects, eyes, corridor=1.2)
    assert dg.blocked((25, 0), route, rects, eyes, corridor=1.2)
    assert dg.blocked((5.2, 6.3), route, rects, eyes, corridor=1.2)
    assert not dg.blocked((5, 3), route, rects, eyes, corridor=1.2)

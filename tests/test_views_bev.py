import numpy as np

from rig.views.bev import Bev


def test_floor_free_and_obstacle_occupied():
    b = Bev(cell=0.5, bounds=(-10, -10, 10, 10))
    floor = np.array([[2.0, 0.0, 20.0], [2.5, 0.0, 20.05]])
    wall = np.array([[4.0, 1.0, 21.0], [4.0, 1.0, 20.8], [4.0, 1.0, 26.0]])
    b.update(np.vstack([floor, wall]), floor_z=20.0)
    assert b.free[b.ij(2.0, 0.0)] > 0
    assert b.occ[b.ij(4.0, 1.0)] == 2
    assert b.occ[b.ij(2.0, 0.0)] == 0


def test_deck_change_resets():
    b = Bev(cell=0.5, bounds=(-10, -10, 10, 10))
    b.update(np.array([[4.0, 1.0, 21.0]]), floor_z=20.0)
    b.update(np.zeros((0, 3)), floor_z=31.25)
    assert b.occ.sum() == 0


def test_points_outside_bounds_ignored():
    b = Bev(cell=0.5, bounds=(-10, -10, 10, 10))
    b.update(np.array([[40.0, 1.0, 21.0], [-40.0, 0.0, 20.0]]), floor_z=20.0)
    assert b.occ.sum() == 0 and b.free.sum() == 0


def test_render_draws_path_robot_and_obstacles():
    b = Bev(cell=0.25, bounds=(-20, -20, 20, 20))
    b.update(np.array([[3.0, 0.0, 21.0]] * 3), floor_z=20.0)
    img = np.asarray(b.render((0.0, 0.0, 0.0), [(0, 0, 20), (0, 5, 20), (5, 5, 20)], size=128, span=8.0))
    assert img.shape == (128, 128, 3)
    centre = img[60:68, 60:68].reshape(-1, 3)
    assert centre.max() > 150
    obstacle = img[62:64, 64 + round(3.0 / 8.0 * 64)].astype(int).sum(1).max()
    assert obstacle > img[10, 10].astype(int).sum() + 200
    path_px = img[64 - round(2.5 / 8.0 * 64), 63:66]
    assert (path_px[:, 0].astype(int) - path_px[:, 2]).max() > 60

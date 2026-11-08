import math

import numpy as np
import pytest

from integration.kinematics import ARM_START, ARM_STOWED, RobotModel, pose
from integration.loop_plan import (PipeCyl, brush_fk, clearance, path_clearance, plan_reposition, plan_staged,
                                   retract_target)

PIPE = PipeCyl(origin=(0.0, -0.5, 1.1), axis=(1.0, 0.0, 0.0), radius=0.15, half_length=2.0)
BR = 0.025


@pytest.fixture(scope="module")
def fk():
    return brush_fk(RobotModel.from_urdf(), pose(0.0, 0.0, 0.32))


def work_q():
    q = dict(ARM_START)
    return q


def stow_q():
    return dict(ARM_STOWED)


def test_clearance_radial_and_beyond_ends():
    assert clearance((0.0, -0.5, 1.1 + 0.3), PIPE, BR) == pytest.approx(0.3 - 0.15 - BR)
    assert clearance((0.0, -0.5, 1.1 + 0.1), PIPE, BR) < 0
    assert clearance((3.0, -0.5, 1.1), PIPE, BR) > 0.5


def test_straight_park_path_hits_the_pipe(fk):
    assert path_clearance(fk, [work_q(), stow_q()], PIPE, BR) < 0.0


def test_staged_park_keeps_clear(fk):
    stages, c = plan_staged(fk, work_q(), stow_q(), PIPE, BR)
    assert c > 0.03
    assert stages[-1] == stow_q()
    assert path_clearance(fk, [work_q()] + stages, PIPE, BR) == pytest.approx(c)
    assert stages[0]["lift_joint"] == pytest.approx(ARM_START["lift_joint"])


def test_staged_unpark_raises_mast_first(fk):
    stages, c = plan_staged(fk, stow_q(), work_q(), PIPE, BR)
    assert c > 0.03
    assert stages[0]["lift_joint"] == pytest.approx(ARM_START["lift_joint"])
    assert stages[0]["ur3_elbow_joint"] == pytest.approx(ARM_STOWED["ur3_elbow_joint"])


def test_retract_target_moves_out_along_normal():
    p = np.array([0.1, -0.5 + 0.18 * math.cos(0.9), 1.1 + 0.18 * math.sin(0.9)])
    t = retract_target(p, PIPE, BR, 0.10)
    assert clearance(t, PIPE, BR) == pytest.approx(0.10, abs=1e-6)
    assert t[0] == pytest.approx(0.1)
    far = np.array([0.1, -0.5, 1.1 + 0.6])
    assert np.allclose(retract_target(far, PIPE, BR, 0.10), far)


def test_reposition_picks_smallest_move_that_reaches_all():
    targets = [np.array([0.5, 0.0, 0.0]), np.array([0.55, 0.0, 0.0])]

    def err(dx, lift, tgt):
        return max(0.0, abs(tgt[0] - dx) - 0.3)

    plan = plan_reposition(err, targets, dxs=(0.0, 0.1, 0.2, 0.3, -0.1), lifts=(0.42,), lift0=0.42, tol=0.03)
    assert plan["dx"] == pytest.approx(0.3)
    assert plan["reached"] == 2 and plan["before"] == 0


def test_reposition_stays_put_when_already_reachable():
    plan = plan_reposition(lambda dx, l, t: 0.0, [np.zeros(3)], dxs=(0.2, 0.0), lifts=(0.45, 0.42), lift0=0.42)
    assert plan["dx"] == 0.0 and plan["lift"] == 0.42

import math

import numpy as np

from integration import film_flange as F


def test_flange_sits_on_the_pipe_and_valve():
    c = F.joint_center()
    assert c[1] == F.PIPE_Y and c[2] == F.PIPE_Z and F.FLANGE_X[0] == F.VALVE_X[1]
    assert F.FLANGE_X[1] - F.FLANGE_X[0] == 2 * F.FLANGE_THICKNESS
    assert F.BOLT_CIRCLE - F.HEAD_RADIUS > F.PIPE_RADIUS and F.BOLT_CIRCLE + F.HEAD_RADIUS < F.FLANGE_OUTER


def test_leak_bolt_faces_the_robot_45_degrees_down():
    h = F.bolt_head(F.LEAK_BOLT)
    r = F.radial(F.LEAK_BOLT)
    assert abs(math.degrees(math.atan2(r[2], r[1])) + 45.0) < 1e-6 and h[0] > F.FLANGE_X[1]
    assert np.allclose(F.geometry().bolt_center(F.LEAK_BOLT), h)
    heads = F.bolt_heads()
    assert len(heads) == 8 and sum(b["leaking"] for b in heads) == 1
    assert all(abs(np.linalg.norm(np.asarray(b["head"])[1:] - [F.PIPE_Y, F.PIPE_Z]) - F.BOLT_CIRCLE) < 1e-9
               for b in heads)


def test_turn_and_collision():
    assert F.bolt_turn_deg(0) == 120 and F.bolt_turn_deg(1) == 0 and F.bolt_turn_deg(2) == 0
    names = {c["name"] for c in F.collision()}
    assert {"valve_body", "flange_valve", "flange_pipe", "pipe"} <= names and "bolt_3_head" not in names

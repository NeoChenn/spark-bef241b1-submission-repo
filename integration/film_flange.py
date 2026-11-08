from __future__ import annotations

import json
import math

import numpy as np

FLANGE_ID = "F5"
LEAK_BOLT = 3
PIPE_AXIS = (1.0, 0.0, 0.0)
PIPE_Y = -10.70
PIPE_Z = 43.40
PIPE_RADIUS = 0.12
PIPE_X = (20.56, 21.4)
VALVE_X = (19.52, 20.48)
VALVE_RADIUS = 0.182
FLANGE_THICKNESS = 0.04
FLANGE_OUTER = 0.17
FLANGE_X = (VALVE_X[1], VALVE_X[1] + 2 * FLANGE_THICKNESS)
BOLT_CIRCLE = 0.145
N_BOLTS = 8
HEAD_RADIUS = 0.016
HEAD_HEIGHT = 0.012
NUT_HEIGHT = 0.012
SHANK_RADIUS = 0.008
REF = (0.0, -1.0, 0.0)
WORK_POSE = (21.0124, -10.2697, -2.7397)
LOOSE_DEG = 120.0


def axis() -> np.ndarray:
    return np.asarray(PIPE_AXIS, float)


def face_center() -> np.ndarray:
    return np.array([FLANGE_X[1], PIPE_Y, PIPE_Z])


def joint_center() -> np.ndarray:
    return np.array([(FLANGE_X[0] + FLANGE_X[1]) / 2, PIPE_Y, PIPE_Z])


def radial(k: int) -> np.ndarray:
    a = 2 * math.pi * k / N_BOLTS
    ref = np.asarray(REF, float)
    return math.cos(a) * ref + math.sin(a) * np.cross(axis(), ref)


def bolt_head(k: int) -> np.ndarray:
    return face_center() + BOLT_CIRCLE * radial(k) + HEAD_HEIGHT * axis()


def bolt_heads() -> list[dict]:
    out = []
    for k in range(N_BOLTS):
        base = face_center() + BOLT_CIRCLE * radial(k)
        out.append({"k": k, "head": bolt_head(k).tolist(), "seat": base.tolist(), "axis": (-axis()).tolist(),
                    "stripe_dir": (-radial(k)).tolist(), "leaking": k == LEAK_BOLT,
                    "nut": (base - 2 * FLANGE_THICKNESS * axis() - NUT_HEIGHT / 2 * axis()).tolist()})
    return out


def geometry():
    from policy.leak.geometry import FlangeGeometry

    return FlangeGeometry(FLANGE_ID, face_center(), axis(), REF, PIPE_RADIUS, BOLT_CIRCLE, HEAD_RADIUS, FLANGE_OUTER,
                          N_BOLTS)


def work_base(deck_z: float = 42.5, trunk: float = 0.32) -> np.ndarray:
    from integration.kinematics import pose

    return pose(WORK_POSE[0], WORK_POSE[1], deck_z + trunk, WORK_POSE[2])


def collision() -> list[dict]:
    c = np.array([0.0, PIPE_Y, PIPE_Z])
    cyl = lambda name, x0, x1, r: {"name": name, "kind": "cylinder", "p0": (c + [x0, 0, 0]).tolist(),
                                   "p1": (c + [x1, 0, 0]).tolist(), "radius": r}
    out = [cyl("valve_body", *VALVE_X, VALVE_RADIUS),
           cyl("flange_valve", FLANGE_X[0], FLANGE_X[0] + FLANGE_THICKNESS, FLANGE_OUTER),
           cyl("flange_pipe", FLANGE_X[0] + FLANGE_THICKNESS, FLANGE_X[1], FLANGE_OUTER),
           cyl("pipe", *PIPE_X, PIPE_RADIUS)]
    for b in bolt_heads():
        if b["k"] != LEAK_BOLT:
            h = np.asarray(b["head"])
            out.append({"name": f"bolt_{b['k']}_head", "kind": "cylinder", "p0": (h - HEAD_HEIGHT * axis()).tolist(),
                        "p1": (h + 0.002 * axis()).tolist(), "radius": HEAD_RADIUS * 1.15})
    return out


def gas_source() -> np.ndarray:
    return bolt_head(LEAK_BOLT)


def gas_jet() -> np.ndarray:
    return radial(LEAK_BOLT)


def bolt_turn_deg(progress: float) -> float:
    return LOOSE_DEG * (1.0 - float(np.clip(progress, 0.0, 1.0)))


def summary() -> dict:
    return {"flange": FLANGE_ID, "leak_bolt": LEAK_BOLT, "pipe_axis": list(PIPE_AXIS),
            "pipe_center_yz": [PIPE_Y, PIPE_Z], "pipe_radius": PIPE_RADIUS, "valve_x": list(VALVE_X),
            "flange_x": list(FLANGE_X), "flange_outer": FLANGE_OUTER, "flange_thickness": FLANGE_THICKNESS,
            "face_center": face_center().round(4).tolist(), "face_normal": list(PIPE_AXIS),
            "bolt_circle": BOLT_CIRCLE, "head_radius": HEAD_RADIUS, "head_height": HEAD_HEIGHT,
            "bolt3_head": bolt_head(LEAK_BOLT).round(4).tolist(), "bolt3_axis": (-axis()).tolist(),
            "work_pose": list(WORK_POSE), "bolts": bolt_heads(), "collision": collision()}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))

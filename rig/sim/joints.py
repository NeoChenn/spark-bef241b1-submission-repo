import math

from pxr import Gf, UsdGeom

from rig.sim.gait import ARM, LEGS, Quantiser

_ARM_ROOT = "Geometry/base/lift_housing/lift_carriage"
_ARM_LINKS = ["ur3_base_link/ur3_shoulder_link", "ur3_upper_arm_link", "ur3_forearm_link", "ur3_wrist_1_link",
              "ur3_wrist_2_link", "ur3_wrist_3_link"]
_ARM_AXES = "ZYYYZY"
AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def joint_table() -> dict:
    out = {}
    for leg in ("FL", "FR", "RL", "RR"):
        p = f"Geometry/base/{leg}_hip"
        out[f"{leg}_hip_joint"] = (p, "X", "rev")
        out[f"{leg}_thigh_joint"] = (f"{p}/{leg}_thigh", "Y", "rev")
        out[f"{leg}_calf_joint"] = (f"{p}/{leg}_thigh/{leg}_calf", "Y", "rev")
    p = _ARM_ROOT
    out["lift_joint"] = (p, "Z", "pri")
    for name, link, ax in zip(ARM[:6], _ARM_LINKS, _ARM_AXES):
        p = f"{p}/{link}"
        out[name] = (p, ax, "rev")
    return out


JOINTS = joint_table()


class RobotJoints:
    def __init__(self, stage, body: str, names=tuple(LEGS + ARM), quant: Quantiser | None = None):
        self.q = quant or Quantiser()
        self.ops = {}
        for name in names:
            rel, ax, kind = JOINTS[name]
            prim = stage.GetPrimAtPath(f"{body}/{rel}")
            if not prim.IsValid():
                continue
            ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
            op = ops.get("xformOp:orient" if kind == "rev" else "xformOp:translate")
            if op is None:
                continue
            rest = op.Get()
            if kind == "pri":
                rot = ops.get("xformOp:orient")
                axis = Gf.Rotation(rot.Get() if rot else Gf.Quatd(1)).TransformDir(Gf.Vec3d(*AXES[ax]))
                self.ops[name] = (op, kind, rest, axis)
            else:
                self.ops[name] = (op, kind, rest, Gf.Vec3f(*AXES[ax]))

    def set(self, names, values) -> int:
        n = 0
        for name, v in zip(names, values):
            ent = self.ops.get(name)
            if ent is None or not self.q.changed(name, v):
                continue
            op, kind, rest, axis = ent
            v = round(float(v) / self.q.step) * self.q.step
            if kind == "rev":
                h = 0.5 * v
                op.Set(rest * type(rest)(math.cos(h), axis * math.sin(h)))
            else:
                op.Set(rest + axis * v)
            n += 1
        return n

"""Forward and inverse kinematics for the Go2 + lift + UR3 URDF, in plain numpy.

The Isaac Sim bridge poses the robot kinematically: it writes every link's world transform from
`RobotModel.link_poses`, and moves the brush with `RobotModel.solve_ik`. Contact force is the same
analytic spring the scrub policy was trained with (`policy/envs/scrub_env_cfg.py`, k = 2500 N/m), so
no PhysX articulation is needed. Nothing here imports Isaac, so it is unit-testable on CPU.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIFT_URDF = REPO / "sim/assets/go2_ur3/go2_ur3_lift.urdf"

ARM_JOINTS = ["ur3_shoulder_pan_joint", "ur3_shoulder_lift_joint", "ur3_elbow_joint",
              "ur3_wrist_1_joint", "ur3_wrist_2_joint", "ur3_wrist_3_joint"]
LIFT_JOINT = "lift_joint"
TOOL_LINK = "brush_tip"

# Standing pose and arm start pose from ScrubLiftEnvCfg (the trained policy's reset state).
STANDING = {
    "FL_hip_joint": 0.1, "RL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}
ARM_START = {
    "lift_joint": 0.42, "ur3_shoulder_pan_joint": -2.035, "ur3_shoulder_lift_joint": -0.712,
    "ur3_elbow_joint": -0.963, "ur3_wrist_1_joint": -0.285, "ur3_wrist_2_joint": -1.288,
    "ur3_wrist_3_joint": 0.044,
}
# Arm folded on the back while walking (lift down, brush pointing forward and up).
ARM_STOWED = {
    "lift_joint": 0.0, "ur3_shoulder_pan_joint": 0.0, "ur3_shoulder_lift_joint": -1.57,
    "ur3_elbow_joint": 2.4, "ur3_wrist_1_joint": -2.4, "ur3_wrist_2_joint": -1.57,
    "ur3_wrist_3_joint": 0.0,
}
TRUNK_HEIGHT = 0.32


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def transform(rotation: np.ndarray | None = None, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def pose(x: float, y: float, z: float, yaw: float = 0.0) -> np.ndarray:
    return transform(rpy_matrix(0.0, 0.0, yaw), (x, y, z))


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float = -math.inf
    upper: float = math.inf

    def local(self, q: float) -> np.ndarray:
        if self.kind in ("revolute", "continuous"):
            return self.origin @ transform(axis_angle(self.axis, q))
        if self.kind == "prismatic":
            return self.origin @ transform(None, self.axis * q)
        return self.origin


@dataclass
class IKResult:
    q: dict
    position_error: float
    axis_error: float
    iterations: int

    @property
    def reached(self) -> bool:
        return self.position_error < 0.01 and self.axis_error < 0.1


@dataclass
class RobotModel:
    joints: dict[str, Joint]
    root: str
    children: dict[str, list[Joint]] = field(default_factory=dict)

    @classmethod
    def from_urdf(cls, path: str | Path = LIFT_URDF) -> "RobotModel":
        tree = ET.parse(path).getroot()
        joints = {}
        for j in tree.findall("joint"):
            origin = j.find("origin")
            xyz = _floats(origin.get("xyz") if origin is not None else None, "0 0 0")
            rpy = _floats(origin.get("rpy") if origin is not None else None, "0 0 0")
            axis_el = j.find("axis")
            axis = _floats(axis_el.get("xyz") if axis_el is not None else None, "1 0 0")
            if not np.linalg.norm(axis):
                axis = np.array([1.0, 0.0, 0.0])
            limit = j.find("limit")
            lower = float(limit.get("lower", -math.inf)) if limit is not None else -math.inf
            upper = float(limit.get("upper", math.inf)) if limit is not None else math.inf
            if j.get("type") == "continuous":
                lower, upper = -math.inf, math.inf
            joints[j.get("name")] = Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                                          j.find("child").get("link"), transform(rpy_matrix(*rpy), xyz),
                                          axis / np.linalg.norm(axis), lower, upper)
        child_links = {j.child for j in joints.values()}
        links = [link.get("name") for link in tree.findall("link")]
        root = next(name for name in links if name not in child_links)
        model = cls(joints, root)
        for j in joints.values():
            model.children.setdefault(j.parent, []).append(j)
        return model

    def default_q(self) -> dict:
        q = {name: 0.0 for name, j in self.joints.items() if j.kind != "fixed"}
        q.update(STANDING)
        q.update(ARM_START)
        return q

    def link_poses(self, base: np.ndarray, q: dict) -> dict[str, np.ndarray]:
        """World 4x4 of every link for root pose `base` and joint values `q`."""
        poses = {self.root: base}
        stack = [self.root]
        while stack:
            link = stack.pop()
            for j in self.children.get(link, []):
                poses[j.child] = poses[link] @ j.local(q.get(j.name, 0.0))
                stack.append(j.child)
        return poses

    def chain(self, tip: str = TOOL_LINK) -> list[Joint]:
        by_child = {j.child: j for j in self.joints.values()}
        out, link = [], tip
        while link != self.root:
            j = by_child[link]
            out.append(j)
            link = j.parent
        return out[::-1]

    def tool_pose(self, base: np.ndarray, q: dict, tip: str = TOOL_LINK) -> np.ndarray:
        T = base
        for j in self.chain(tip):
            T = T @ j.local(q.get(j.name, 0.0))
        return T

    def clamp(self, q: dict) -> dict:
        out = dict(q)
        for name, value in q.items():
            j = self.joints.get(name)
            if j is not None:
                out[name] = float(np.clip(value, j.lower, j.upper))
        return out

    def solve_ik(self, base: np.ndarray, q: dict, target: np.ndarray, approach: np.ndarray,
                 joints: list[str] = ARM_JOINTS, iterations: int = 80, damping: float = 0.05,
                 tip: str = TOOL_LINK) -> IKResult:
        """Damped least squares on tool position + tool x axis (the brush axis) = `approach`.

        Roll about the brush axis is free (a rotary brush), so the task is 5-D on 6 joints."""
        q = dict(q)
        approach = np.asarray(approach, float) / np.linalg.norm(approach)
        target = np.asarray(target, float)
        eps = 1e-5

        def error(qq):
            T = self.tool_pose(base, qq, tip)
            # (approach - x) is zero only when the brush points along approach, unlike a cross product,
            # which would also accept the brush pointing straight backwards.
            return np.concatenate([target - T[:3, 3], approach - T[:3, 0]]), T

        it = 0
        for it in range(1, iterations + 1):
            e, T = error(q)
            if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-3:
                break
            J = np.zeros((6, len(joints)))
            for k, name in enumerate(joints):
                dq = dict(q)
                dq[name] += eps
                J[:, k] = -(error(dq)[0] - e) / eps
            step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            step = np.clip(step, -0.3, 0.3)
            for k, name in enumerate(joints):
                q[name] += float(step[k])
            q = self.clamp(q)
        T = self.tool_pose(base, q, tip)
        pos_err = float(np.linalg.norm(target - T[:3, 3]))
        axis_err = float(math.acos(np.clip(T[:3, 0] @ approach, -1.0, 1.0)))
        return IKResult(q, pos_err, axis_err, it)


def spring_force(tool_tip: np.ndarray, pipe, stiffness: float = 2500.0, bumps=()) -> float:
    """Normal force from the analytic brush spring used in training: k x penetration into the pipe.

    `bumps` are (theta, z, height, radius) weld beads that locally raise the surface (force-spike hook)."""
    rel = np.asarray(tool_tip, float) - pipe.origin
    z = float(rel @ pipe.axis)
    radial = rel - z * pipe.axis
    dist = float(np.linalg.norm(radial))
    theta = math.atan2(float(radial @ pipe.binormal), float(radial @ pipe.ref)) % (2 * math.pi)
    surface = pipe.radius
    for b_theta, b_z, height, radius in bumps:
        arc = (math.remainder(theta - b_theta, 2 * math.pi)) * pipe.radius
        d2 = (arc ** 2 + (z - b_z) ** 2) / radius ** 2
        surface += height * max(0.0, 1.0 - d2)
    return stiffness * max(0.0, surface - dist)

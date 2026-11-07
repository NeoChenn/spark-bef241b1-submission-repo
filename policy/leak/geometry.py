"""Flange geometry shared by the scene builder, the mock renderer and the leak perception.

A bolted pipe flange seen face-on: 8 bolt heads on a bolt circle around the pipe bore. Each bolt carries a
red torque stripe painted across its head and on across the flange face. When a bolt turns, the head part of
the stripe rotates with it and the flange part does not, so the angle between them is the misalignment the
camera reads. Units: metres, world frame Z up (same as rig meta files).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FlangeGeometry:
    id: str
    center: np.ndarray            # (3,) centre of the flange face
    normal: np.ndarray            # (3,) unit, out of the face toward the side the robot views from
    ref: np.ndarray               # (3,) unit, in the face plane: direction of bolt 0
    pipe_radius: float = 0.08
    bolt_circle: float = 0.13
    head_radius: float = 0.016
    outer_radius: float = 0.17
    n_bolts: int = 8
    head_height: float = 0.012

    def __post_init__(self):
        self.center = np.asarray(self.center, np.float64)
        n = np.asarray(self.normal, np.float64)
        self.normal = n / np.linalg.norm(n)
        r = np.asarray(self.ref, np.float64)
        r = r - r.dot(self.normal) * self.normal
        self.ref = r / np.linalg.norm(r)

    @property
    def binormal(self) -> np.ndarray:
        return np.cross(self.normal, self.ref)

    def bolt_phi(self, k: int) -> float:
        return 2 * math.pi * k / self.n_bolts

    def radial(self, k: int, extra: float = 0.0) -> np.ndarray:
        """Unit in-plane direction of bolt k's stripe, turned by ``extra`` radians about the normal."""
        a = self.bolt_phi(k) + extra
        return math.cos(a) * self.ref + math.sin(a) * self.binormal

    def bolt_center(self, k: int) -> np.ndarray:
        """Centre of bolt k's head (raised off the face)."""
        return self.center + self.bolt_circle * self.radial(k) + self.head_height * self.normal

    def view_poses(self, distance: float = 0.85, side: float = 0.3) -> list[tuple[np.ndarray, np.ndarray]]:
        """Camera (eye, target) pairs: nominal face-on view, then two views turned +/- ``side`` rad."""
        poses = [(self.center + distance * self.normal, self.center.copy())]
        for s in (side, -side):
            d = math.cos(s) * self.normal + math.sin(s) * self.binormal
            poses.append((self.center + distance * d, self.center.copy()))
        return poses

    def to_meta(self) -> dict:
        return {"id": self.id, "center_m": self.center.round(4).tolist(), "normal": self.normal.round(4).tolist(),
                "ref": self.ref.round(4).tolist(), "pipe_radius_m": self.pipe_radius,
                "bolt_circle_m": self.bolt_circle, "head_radius_m": self.head_radius,
                "outer_radius_m": self.outer_radius, "n_bolts": self.n_bolts}

    @classmethod
    def from_meta(cls, m: dict) -> "FlangeGeometry":
        return cls(m["id"], m["center_m"], m["normal"], m["ref"], m.get("pipe_radius_m", 0.08),
                   m.get("bolt_circle_m", 0.13), m.get("head_radius_m", 0.016), m.get("outer_radius_m", 0.17),
                   m.get("n_bolts", 8))


def project(camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (..., 3) -> pixel (u, v) (..., 2) and camera depth (...). OpenCV camera convention,
    the same as policy.interfaces.CameraModel (x right, y down, z forward)."""
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    pc = (np.asarray(points, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = pc[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * pc[..., 0] / zs + K[0, 2]
    v = K[1, 1] * pc[..., 1] / zs + K[1, 2]
    return np.stack([u, v], -1), z


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """4x4 camera pose (OpenCV optical frame) looking from eye to target. Same as policy.demo_mock.look_at."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    zc = target - eye
    zc /= np.linalg.norm(zc)
    if abs(zc.dot(up / np.linalg.norm(up))) > 0.99:   # looking straight up or down: pick another up
        up = np.array([0.0, 1.0, 0.0])
    xc = np.cross(zc, -up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xc, yc, zc, eye
    return T

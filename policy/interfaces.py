from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray
    T_world_cam: np.ndarray
    width: int
    height: int


@dataclass
class PipeGeometry:
    origin: np.ndarray
    axis: np.ndarray
    ref: np.ndarray
    radius: float
    length: float
    n_z: int = 32
    n_theta: int = 64

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64)
        a = np.asarray(self.axis, dtype=np.float64)
        self.axis = a / np.linalg.norm(a)
        r = np.asarray(self.ref, dtype=np.float64)
        r = r - r.dot(self.axis) * self.axis
        self.ref = r / np.linalg.norm(r)

    @property
    def binormal(self) -> np.ndarray:
        return np.cross(self.axis, self.ref)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_z, self.n_theta)

    def cell_to_theta_z(self, i, j):
        z = (np.asarray(i, dtype=np.float64) + 0.5) * (self.length / self.n_z) - self.length / 2
        theta = (np.asarray(j, dtype=np.float64) + 0.5) * (2 * np.pi / self.n_theta)
        return theta, z

    def theta_z_to_cell(self, theta, z):
        i = np.clip(np.floor((np.asarray(z) + self.length / 2) / self.length * self.n_z).astype(int), 0, self.n_z - 1)
        j = np.floor(np.mod(theta, 2 * np.pi) / (2 * np.pi) * self.n_theta).astype(int) % self.n_theta
        return i, j

    def surface_point(self, theta, z):
        theta = np.asarray(theta, dtype=np.float64)[..., None]
        z = np.asarray(z, dtype=np.float64)[..., None]
        n = np.cos(theta) * self.ref + np.sin(theta) * self.binormal
        p = self.origin + z * self.axis + self.radius * n
        return p, n


@dataclass
class Waypoint:
    position: np.ndarray
    approach: np.ndarray
    theta: float
    z: float
    cell: tuple[int, int]
    standoff: float = 0.0
    flow_rate: float = 0.0


@dataclass
class DecisionRecord:
    t: float
    frm: str
    to: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"t": round(float(self.t), 3), "from": self.frm, "to": self.to,
                "reason": self.reason, "evidence": self.evidence}


@runtime_checkable
class Perception(Protocol):
    def observe(self) -> Optional[np.ndarray]:
        pass

    def camera(self) -> CameraModel:
        pass

    def request_view(self, index: int) -> bool:
        pass


@runtime_checkable
class Localizer(Protocol):
    def localize(self, image: np.ndarray):
        ...


@runtime_checkable
class RustWorld(Protocol):
    rust: Any
    sprayed: Any
    soak_time: Any

    def spray(self, mask) -> None: ...
    def step_soak(self, dt: float) -> None: ...
    def abrade(self, *args, **kwargs) -> Any: ...
    def fraction_remaining(self) -> float: ...
    def add_patch(self, *args, **kwargs) -> Any: ...
    def to_texture(self) -> Any: ...


@runtime_checkable
class ScrubPolicy(Protocol):
    def reset(self) -> None: ...

    def step(self, obs: dict) -> np.ndarray:
        pass


@runtime_checkable
class ArmController(Protocol):
    def move_to(self, waypoints: Sequence[Waypoint]) -> list[bool]:
        pass

    def apply_ee_delta(self, action: np.ndarray) -> None:
        pass

    def ee_theta_z(self) -> tuple[float, float]: ...

    def current_force(self) -> float:
        pass

    def back_off(self, distance: float = 0.05) -> None:
        pass


@runtime_checkable
class SimBridge(Protocol):

    def sim_time(self) -> float: ...
    def advance(self, dt: float) -> None:
        pass

    def pipe(self) -> PipeGeometry: ...
    def patrol_poses(self) -> list[Any]: ...
    def go_to_patrol_pose(self, pose: Any) -> bool: ...
    def approach(self, theta: float, z: float, standoff: float) -> bool:
        pass

    def person_distance(self) -> Optional[float]:
        pass

    def scrub_obs(self) -> dict:
        pass

    def publish_overlay(self, lines: list[str]) -> None:
        pass


def camera_from_usd(world_matrix, focal_length_mm: float, horizontal_aperture_mm: float,
                    width: int, height: int) -> CameraModel:
    M = np.asarray(world_matrix, dtype=np.float64)
    R_usd = M[:3, :3].T
    R_usd = R_usd / np.linalg.norm(R_usd, axis=0, keepdims=True)
    T = np.eye(4)
    T[:3, :3] = R_usd @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = M[3, :3]
    f = focal_length_mm / horizontal_aperture_mm * width
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return CameraModel(K, T, width, height)

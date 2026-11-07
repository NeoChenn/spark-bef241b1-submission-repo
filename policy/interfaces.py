"""Integration contracts for the repair orchestrator.

The orchestrator only talks to the outside world through these Protocols, so it
runs identically against the Isaac Sim scene (SimBridge implementation by the sim
team), the Isaac Lab scrub env, and the CPU-only mock in ``policy.demo_mock``.

Conventions (see integration/INTERFACE.md for the full contract):
  * Rust grid matches policy/envs/rust_state.RustState: (n_z, n_theta) =
    (H, W); rows = z along the pipe axis in [-L/2, L/2] (cell centres
    (i+0.5)/H*L - L/2), cols = theta in [0, 2pi) (periodic). Levels in [0, 1].
  * Camera frame is OpenCV (x right, y down, z forward); ``T_world_cam`` is the
    4x4 pose of the camera in the world.
  * Pipe frame: origin at one end of the pipe centre-line, ``axis`` unit vector
    along the pipe, ``ref`` unit vector (perpendicular to axis) where theta = 0;
    theta increases towards ``axis x ref``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------- data
@dataclass
class CameraModel:
    K: np.ndarray            # 3x3 intrinsics
    T_world_cam: np.ndarray  # 4x4 camera pose in world (OpenCV optical frame)
    width: int
    height: int


@dataclass
class PipeGeometry:
    origin: np.ndarray       # (3,) world point on the axis at z = 0 (segment centre)
    axis: np.ndarray         # (3,) unit vector along the pipe
    ref: np.ndarray          # (3,) unit vector, theta = 0 direction (perp. to axis)
    radius: float            # metres
    length: float            # metres
    n_z: int = 32            # rust grid rows (H)
    n_theta: int = 64        # rust grid cols (W)

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
        """Grid indices (row=z, col=theta) -> cell-centre (theta, z)."""
        z = (np.asarray(i, dtype=np.float64) + 0.5) * (self.length / self.n_z) - self.length / 2
        theta = (np.asarray(j, dtype=np.float64) + 0.5) * (2 * np.pi / self.n_theta)
        return theta, z

    def theta_z_to_cell(self, theta, z):
        """(theta, z) -> (row, col) indices."""
        i = np.clip(np.floor((np.asarray(z) + self.length / 2) / self.length * self.n_z).astype(int), 0, self.n_z - 1)
        j = np.floor(np.mod(theta, 2 * np.pi) / (2 * np.pi) * self.n_theta).astype(int) % self.n_theta
        return i, j

    def surface_point(self, theta, z):
        """(theta, z) -> (xyz on surface (...,3), outward unit normal (...,3)) in world."""
        theta = np.asarray(theta, dtype=np.float64)[..., None]
        z = np.asarray(z, dtype=np.float64)[..., None]
        n = np.cos(theta) * self.ref + np.sin(theta) * self.binormal
        p = self.origin + z * self.axis + self.radius * n
        return p, n


@dataclass
class Waypoint:
    """End-effector target. ``position`` is the tool-tip target in world frame,
    ``approach`` the unit vector the tool points along (= -surface normal)."""
    position: np.ndarray
    approach: np.ndarray
    theta: float
    z: float
    cell: tuple[int, int]
    standoff: float = 0.0
    flow_rate: float = 0.0   # L/min for spray, 0 for brush


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


# ---------------------------------------------------------------------- protocols
@runtime_checkable
class Perception(Protocol):
    def observe(self) -> Optional[np.ndarray]:
        """RGB uint8 HxWx3 from the wrist/head camera; None if no frame (lost)."""

    def camera(self) -> CameraModel:
        """Intrinsics + pose of the camera that produced the last frame."""

    def request_view(self, index: int) -> bool:
        """Move the camera to alternative viewpoint ``index`` (1, 2, ...). Return
        False if no such viewpoint is reachable. Index 0 is the nominal view."""


@runtime_checkable
class Localizer(Protocol):
    def localize(self, image: np.ndarray):  # -> detection.localize.LocalizeResult
        ...


@runtime_checkable
class RustWorld(Protocol):
    """policy/envs/rust_state.RustState API (torch tensors, (num_envs, H=n_z, W=n_theta); attr `rust` = levels)."""
    rust: Any
    sprayed: Any
    soak_time: Any

    def spray(self, mask) -> None: ...          # mask (H, W) bool tensor, broadcasts to envs
    def step_soak(self, dt: float) -> None: ...
    def abrade(self, *args, **kwargs) -> Any: ...
    def fraction_remaining(self) -> float: ...
    def add_patch(self, *args, **kwargs) -> Any: ...
    def to_texture(self) -> Any: ...
    # optional: paint(mask) -> None   (orchestrator calls it if present)


@runtime_checkable
class ScrubPolicy(Protocol):
    def reset(self) -> None: ...

    def step(self, obs: dict) -> np.ndarray:
        """obs keys (always present): ee_theta_z (2,), target_mask (n_z,n_theta) bool,
        perceived_rust (n_z,n_theta) float, force (float), plus whatever
        SimBridge.scrub_obs() adds (e.g. the Isaac Lab policy observation under
        key "policy"). Returns action = [d_theta (rad), d_z (m), press (0..1)]
        in pipe frame; ArmController.apply_ee_delta executes it via diff-IK."""


@runtime_checkable
class ArmController(Protocol):
    def move_to(self, waypoints: Sequence[Waypoint]) -> list[bool]:
        """Execute waypoints in order; returns per-waypoint reached flags."""

    def apply_ee_delta(self, action: np.ndarray) -> None:
        """One control step of an EE delta [d_theta, d_z, press] in pipe frame."""

    def ee_theta_z(self) -> tuple[float, float]: ...

    def current_force(self) -> float:
        """Contact force magnitude at the brush/EE (N)."""

    def back_off(self, distance: float = 0.05) -> None:
        """Retract along the surface normal."""


@runtime_checkable
class SimBridge(Protocol):
    """What the Isaac Sim scene must provide to the orchestrator."""

    def sim_time(self) -> float: ...
    def advance(self, dt: float) -> None:
        """Step physics/render for dt seconds of sim time (arm holds pose)."""

    def pipe(self) -> PipeGeometry: ...
    def patrol_poses(self) -> list[Any]: ...
    def go_to_patrol_pose(self, pose: Any) -> bool: ...
    def approach(self, theta: float, z: float, standoff: float) -> bool:
        """Walk/stand the dog so the arm workspace covers (theta, z)."""

    def person_distance(self) -> Optional[float]:
        """Distance to the nearest person (m) or None if no-one is tracked."""

    def scrub_obs(self) -> dict:
        """Extra observation for the ScrubPolicy (e.g. Isaac Lab obs tensor)."""

    def publish_overlay(self, lines: list[str]) -> None:
        """Draw HUD text on the recorded video."""


def camera_from_usd(world_matrix, focal_length_mm: float, horizontal_aperture_mm: float,
                    width: int, height: int) -> CameraModel:
    """USD camera -> CameraModel (pure numpy, no pxr import needed).

    ``world_matrix``: 4x4 from ``UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(t)``
    converted with ``np.array(m)`` (USD row-vector convention: translation in the last
    row). USD cameras look down -Z with +Y up; OpenCV looks down +Z with +Y down.
    Square pixels assumed (vertical aperture follows the image aspect).
    """
    M = np.asarray(world_matrix, dtype=np.float64)
    R_usd = M[:3, :3].T                      # columns = camera axes in world
    R_usd = R_usd / np.linalg.norm(R_usd, axis=0, keepdims=True)
    T = np.eye(4)
    T[:3, :3] = R_usd @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = M[3, :3]
    f = focal_length_mm / horizontal_aperture_mm * width
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return CameraModel(K, T, width, height)

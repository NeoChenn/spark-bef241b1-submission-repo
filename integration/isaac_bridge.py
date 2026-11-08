"""Isaac Sim 6.2 implementation of the orchestrator contract (integration/INTERFACE.md).

One object, `IsaacRigBridge`, satisfies `policy.interfaces.SimBridge`, `Perception` and `ArmController`
on the rig deck scene (sim/output/rig_deck.usda). It runs inside the Isaac Sim process; the scene
pieces are in repair_scene.py and the arm maths in kinematics.py.

What is simulated and what is not:
  * Perception is real: `observe()` returns the RTX render of the inspection camera, so the localizer
    sees whatever the scene shows (rust overlay, glare, lens mist, the arm in the way).
  * The robot is posed kinematically from the URDF (IK on brush_tip), not simulated by PhysX: legs hold
    the standing pose and the base glides between stations with a trot animation.
  * Contact force is the analytic brush spring the scrub policy was trained with (k = 2500 N/m times
    penetration of brush_tip into the pipe surface, including any weld bead), not a PhysX contact sensor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from detection.localize import pixel_rays, ray_cylinder
from integration import repair_scene as rs
from integration.kinematics import ARM_STOWED, TRUNK_HEIGHT, RobotModel, pose, spring_force
from policy.interfaces import CameraModel, PipeGeometry, Waypoint, camera_from_usd

TWO_PI = 2 * math.pi
STIFFNESS = 2500.0          # N/m, ScrubEnvCfg.spring_k_nominal
NOMINAL_FORCE = 25.0        # N at press = 1 (INTERFACE.md: admittance gives ~25 * press N)
SLIDE_SPEED = 0.3           # m/s rotary brush
CONTACT_RADIUS = 0.03       # m brush footprint
# Stance at the pipe: both inside ScrubLiftEnvCfg's training ranges (axis 0.45-0.55 m, lift 0.38-0.45 m). An IK
# reach map over the patch (theta 0.3-1.7, z +-0.3) gave 76/105 points here vs 64/105 at 0.5 m / 0.42 m.
AXIS_DISTANCE = 0.45        # m, trunk centreline to pipe axis
WORK_LIFT = 0.45
CONTACT_TOLERANCE = 0.003   # m: at k = 2500 N/m, 1 cm of IK error is 25 N, so contact poses must be tight


@dataclass
class Frames:
    """Callbacks the runner provides: render one frame, optionally record it for the video."""
    render: Callable[[], dict]            # -> {"inspection": rgb, "hero": rgb}
    record: Callable[[np.ndarray, list], None] = lambda rgb, hud: None


@dataclass
class Event:
    t: float
    name: str
    fire: Callable[[], None]


@dataclass
class BridgeState:
    t: float = 0.0
    base: np.ndarray = field(default_factory=lambda: pose(-4.5, AXIS_DISTANCE, TRUNK_HEIGHT))
    q: dict = field(default_factory=dict)
    theta: float = 0.0
    z: float = 0.0
    force: float = 0.0
    view: int = 0
    hud: list = field(default_factory=list)
    spraying: bool = False


class IsaacRigBridge:
    def __init__(self, stage, world, frames: Frames, verbose: bool = True,
                 inspection_size: tuple[int, int] = (640, 360)):
        self.stage, self.world, self.frames, self.verbose = stage, world, frames, verbose
        self.size = inspection_size
        self.model = RobotModel.from_urdf()
        self.state = BridgeState(q=self.model.default_q())
        self.state.q.update(ARM_STOWED)
        self.events: list[Event] = []
        self.bumps: list[tuple] = []
        self.pipe_prim = rs.find_main_pipe(stage)
        self._pipe = rs.read_pipe(self.pipe_prim, n_z=world.H, n_theta=world.W)
        self.overlay = rs.RustOverlay(stage, self._pipe)
        self.robot = rs.RobotPoser(stage, self.model)
        self.inspection = rs.Camera(stage, rs.INSPECTION_CAMERA, focal=35.0)
        self.damage_theta = float(getattr(world, "damage_theta", 0.9))
        self.views = rs.inspection_views(self._pipe, self.damage_theta)
        self.inspection.look(*self.views[0])
        self.lens_mist = rs.add_lens_mist(stage, rs.INSPECTION_CAMERA)
        self.glare = rs.add_glare_light(stage, self.views[0][0] + np.array([0.05, 0.0, 0.08]), self.views[0][1])
        self.worker_op, self.worker_yaw = rs.add_worker(stage)
        self.worker_pos: Optional[np.ndarray] = None
        self.spray_cone, self.spray_op = rs.add_spray_cone(stage)
        self.last_inspection: Optional[np.ndarray] = None
        self.last_camera: Optional[CameraModel] = None
        self.occluded_until = -1.0
        self.parked = True                      # starts stowed
        self.unpark_q = self.model.default_q()
        self._sync()

    # ------------------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _sync(self) -> None:
        self.robot.set(self.state.base, self.state.q)
        self.overlay.update(self.world)

    def tool(self) -> np.ndarray:
        return self.model.tool_pose(self.state.base, self.state.q)

    def _render(self, record: bool = True) -> dict:
        self._sync()
        images = self.frames.render()
        if images.get("inspection") is not None:
            self.last_inspection = images["inspection"]
            self.last_camera = self._camera_model()
        if record and images.get("hero") is not None:
            self.frames.record(images["hero"], self.state.hud)
        return images

    def _camera_model(self) -> CameraModel:
        return camera_from_usd(self.inspection.world_matrix(), self.inspection.focal, self.inspection.aperture,
                               *self.size)

    def _fire_events(self) -> None:
        while self.events and self.events[0].t <= self.state.t:
            event = self.events.pop(0)
            event.fire()
            self.log(f"   >>> [scene change t={self.state.t:5.1f}s] {event.name}")

    def schedule(self, t: float, name: str, fire: Callable[[], None]) -> None:
        self.events.append(Event(t, name, fire))
        self.events.sort(key=lambda e: e.t)

    def _theta_z_of(self, point: np.ndarray) -> tuple[float, float]:
        p = self._pipe
        rel = np.asarray(point) - p.origin
        z = float(rel @ p.axis)
        radial = rel - z * p.axis
        return math.atan2(float(radial @ p.binormal), float(radial @ p.ref)) % TWO_PI, z

    def _move_tool(self, target: np.ndarray, approach: np.ndarray) -> bool:
        seed = self.unpark_q if self.parked and self.unpark_q else self.state.q
        result = self.model.solve_ik(self.state.base, seed, target, approach, iterations=120)
        converged = result.position_error < CONTACT_TOLERANCE and result.axis_error < 0.05
        if not converged:
            # DLS can get stuck from an awkward seed (e.g. after spraying a second patch); retry from the
            # working pose before declaring the target unreachable.
            nominal = self.model.default_q()
            nominal["lift_joint"] = seed.get("lift_joint", WORK_LIFT)
            retry = self.model.solve_ik(self.state.base, nominal, target, approach, iterations=120)
            if retry.position_error < CONTACT_TOLERANCE and retry.axis_error < 0.05:
                result, converged = retry, True
        q_new = result.q
        if not converged:
            # An unconverged solve can leave the tip partway through the pipe (huge spring force). Like a real
            # controller, reject infeasible targets and hold the last good pose.
            q_new = dict(seed)
        if self.parked:
            self.parked = False
            self._animate({k: v for k, v in q_new.items() if k in ARM_STOWED})
        self.state.q = q_new
        tip = self.tool()[:3, 3]
        self.state.force = spring_force(tip, self._pipe, STIFFNESS, self.bumps)
        return converged

    # ========================================================================== SimBridge
    def sim_time(self) -> float:
        return self.state.t

    def advance(self, dt: float) -> None:
        """Hold pose for dt of sim time. Long holds (soak) render one frame per second of sim time."""
        steps = max(1, int(round(dt / 1.0))) if dt >= 1.0 else 1
        for k in range(steps):
            self.state.t += dt / steps
            self._fire_events()
            self._render()

    def pipe(self) -> PipeGeometry:
        self._pipe = rs.read_pipe(self.pipe_prim, n_z=self.world.H, n_theta=self.world.W)
        return self._pipe

    def patrol_poses(self) -> list:
        return [{"name": "main_pipe_station_1", "x": float(self._pipe.origin[0] + 0.9), "y": AXIS_DISTANCE + 0.25}]

    def _walk(self, x: float, y: float, yaw: float = 0.0, speed: float = 0.6) -> None:
        start = self.state.base[:3, 3].copy()
        end = np.array([x, y, TRUNK_HEIGHT])
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        frames = max(2, int(distance / speed * 6))
        stand = {k: v for k, v in self.model.default_q().items() if "hip" in k or "thigh" in k or "calf" in k}
        for f in range(1, frames + 1):
            s = f / frames
            s = s * s * (3 - 2 * s)
            p = start + (end - start) * s
            phase = f / frames * distance / 0.18 * math.pi
            swing = 0.0 if f == frames else 0.25
            for leg, sign in (("FL", 1), ("RR", 1), ("FR", -1), ("RL", -1)):
                lift = max(0.0, sign * math.sin(phase))
                self.state.q[f"{leg}_thigh_joint"] = stand[f"{leg}_thigh_joint"] - swing * lift * 0.6
                self.state.q[f"{leg}_calf_joint"] = stand[f"{leg}_calf_joint"] - swing * lift * 0.9
            self.state.base = pose(p[0], p[1], TRUNK_HEIGHT + 0.01 * abs(math.sin(phase)) * swing, yaw)
            self.state.t += distance / speed / frames
            self._fire_events()
            self._render()

    def go_to_patrol_pose(self, p) -> bool:
        self.state.q.update(ARM_STOWED)
        self.parked, self.unpark_q = True, self.model.default_q()
        self._walk(p["x"], p["y"])
        return True

    def approach(self, theta: float, z: float, standoff: float) -> bool:
        """Walk next to the pipe (trunk 0.5 m from the axis, as in training) at the target's axial position."""
        x = float(self._pipe.origin[0] + z * self._pipe.axis[0])
        self._walk(x, float(self._pipe.origin[1] + AXIS_DISTANCE))
        for f in range(1, 9):                       # raise the lift and unfold the arm to the start pose
            s = f / 8
            start = self.model.default_q()
            start["lift_joint"] = WORK_LIFT
            for name in ARM_STOWED:
                self.state.q[name] = ARM_STOWED[name] + (start[name] - ARM_STOWED[name]) * s
            self.state.t += 0.25
            self._render()
        self.parked = False
        surf, n = self._pipe.surface_point(theta, z)
        reached = self._move_tool(surf + standoff * n, -n)
        self.state.theta, self.state.z = theta, z
        self._render()
        return reached

    def person_distance(self) -> Optional[float]:
        if self.worker_pos is None:
            return None
        return float(np.linalg.norm(self.worker_pos[:2] - self.state.base[:2, 3]))

    def scrub_obs(self) -> dict:
        return {}

    def publish_overlay(self, lines: list[str]) -> None:
        self.state.hud = list(lines)

    # ========================================================================= Perception
    def observe(self) -> Optional[np.ndarray]:
        """Inspection frame with everything except the pipe segment greyed out.

        HANDOFF.md: classify crops of the pipe surface, not wide views (the rusty deck grate reads as rust).
        The robot knows where the pipe is, so pixels whose ray misses the segment are set to neutral grey."""
        self._park_arm()
        images = self._render()
        rgb = images.get("inspection")
        if rgb is None or rgb.size == 0 or float(rgb.mean()) < 1.0:
            return None
        roi = self._pipe_roi()
        pipe_pixels = rgb[roi].astype(np.float32)
        if not len(pipe_pixels) or pipe_pixels.std() < 6.0 or pipe_pixels.mean() < 8.0:
            return rgb          # blocked / misted lens: hand back the flat raw frame so the orchestrator HOLDs
        out = rgb.copy()
        out[~roi] = 118
        return out

    # The arm lifts out of the inspection camera's view before every look (a real robot would too), and
    # swings back to where it was before the next tool command.
    def _animate(self, q_target: dict, frames: int = 4, dt: float = 0.1) -> None:
        start = dict(self.state.q)
        for f in range(1, frames + 1):
            s = f / frames
            s = s * s * (3 - 2 * s)
            for name, value in q_target.items():
                self.state.q[name] = start.get(name, value) + (value - start.get(name, value)) * s
            self.state.t += dt
            self._fire_events()
            self._render()

    def _park_arm(self) -> None:
        if self.parked:
            return
        self.unpark_q = dict(self.state.q)
        park = {k: v for k, v in ARM_STOWED.items() if k != "lift_joint"}
        self.state.force = 0.0
        self._animate(park)
        self.parked = True

    def _pipe_roi(self, erode: int = 4) -> np.ndarray:
        cam = self.camera()
        key = (self.state.view, tuple(np.round(cam.T_world_cam.reshape(-1), 4)), tuple(np.round(self._pipe.origin, 4)))
        if getattr(self, "_roi_key", None) != key:
            origin, dirs = pixel_rays(cam)
            hit, *_ = ray_cylinder(origin, dirs, self._pipe)
            for _ in range(erode):
                hit = hit & np.roll(hit, 1, 0) & np.roll(hit, -1, 0) & np.roll(hit, 1, 1) & np.roll(hit, -1, 1)
            self._roi_key, self._roi = key, hit
        return self._roi

    def camera(self) -> CameraModel:
        return self.last_camera or self._camera_model()

    def request_view(self, index: int) -> bool:
        if not 0 <= index < len(self.views):
            return False
        self.state.view = index
        self.inspection.look(*self.views[index])
        return True

    # ====================================================================== ArmController
    def move_to(self, waypoints) -> list[bool]:
        reached = []
        every = max(1, len(waypoints) // 40)
        for k, wp in enumerate(waypoints):
            spraying = wp.flow_rate > 0 and wp.standoff >= 0.1
            target = np.asarray(wp.position, float)
            if not spraying:      # paint brush works in contact: the waypoint standoff is the bristle length
                target = target + wp.standoff * np.asarray(wp.approach, float)   # approach = -n
            ok = self._move_tool(target, np.asarray(wp.approach))
            reached.append(bool(ok))
            self.state.theta, self.state.z = wp.theta, wp.z
            self.state.t += 0.1
            self._fire_events()
            if k % every == 0 or k == len(waypoints) - 1:
                tip = self.tool()
                rs.set_spray(self.spray_cone, self.spray_op, tip[:3, 3], tip[:3, 0], wp.standoff + 0.02, spraying)
                self._render()
        rs.set_spray(self.spray_cone, self.spray_op, np.zeros(3), np.array([1.0, 0, 0]), 0.1, False)
        self.state.force = 0.0
        return reached

    def apply_ee_delta(self, action: np.ndarray) -> None:
        d_theta, d_z, press = (float(v) for v in action[:3])
        p = self._pipe
        self.state.theta = (self.state.theta + float(np.clip(d_theta, -0.15, 0.15))) % TWO_PI
        self.state.z = float(np.clip(self.state.z + np.clip(d_z, -0.03, 0.03), -p.length / 2, p.length / 2))
        press = float(np.clip(press, 0.0, 1.0))
        surf, n = p.surface_point(self.state.theta, self.state.z)
        depth = NOMINAL_FORCE * press / STIFFNESS
        if not self._move_tool(surf - depth * n, -n):
            # IK missed: report where the brush really is, so commanded and actual (theta, z) never drift apart
            self.state.theta, self.state.z = self._theta_z_of(self.tool()[:3, 3])
        if self.state.force >= 5.0:
            tip = self.tool()[:3, 3]
            theta, z = self._theta_z_of(tip)
            uv = self.world.surface_to_uv(torch.tensor([theta]), torch.tensor([z]))
            self.world.abrade(uv, CONTACT_RADIUS, torch.tensor([self.state.force]), torch.tensor([SLIDE_SPEED]), 0.1)

    def ee_theta_z(self) -> tuple[float, float]:
        return self._theta_z_of(self.tool()[:3, 3])

    def current_force(self) -> float:
        return float(self.state.force)

    def back_off(self, distance: float = 0.05) -> None:
        surf, n = self._pipe.surface_point(self.state.theta, self.state.z)
        self._move_tool(surf + distance * n, -n)
        self.state.force = 0.0
        self._render()

    # =================================================================== mid-run hooks (section 4)
    def spawn_patch(self, theta: float, z: float, size: float = 0.04, level: float = 0.8) -> None:
        self.world.inject_patch(theta, z, size=size, level=level)

    def set_glare(self, on: bool) -> None:
        self.glare.GetIntensityAttr().Set(4.0e5 if on else 0.0)

    def occlude_camera(self, on: bool) -> None:
        img = rs.UsdGeom.Imageable(self.lens_mist)
        img.MakeVisible() if on else img.MakeInvisible()

    def force_spike(self, height: float = 0.010, radius: float = 0.03) -> None:
        # 10 mm adds 25 N: with the nominal 25 N press it trips the 40 N limit once; after the orchestrator
        # scales press by 0.7 it stays under (12 + 25 = 37 N), so the demo shows one BACK_OFF, not an escalation.
        """A weld bead right on the brush path: a real bump in the surface the spring force sees."""
        theta, z = self.ee_theta_z()
        self.bumps.append((theta, z, height, radius))
        rs.add_weld_bead(self.stage, f"{rs.PIPE_SCOPE}/weld_bead_{len(self.bumps)}", self._pipe, theta, z, height, radius)

    def person_at(self, distance: Optional[float]) -> None:
        if distance is None:
            self.worker_pos = None
            self.worker_op.Set(rs.Gf.Vec3d(0, 0, -50))
            return
        base = self.state.base[:3, 3]
        self.worker_pos = base + np.array([distance * 0.8, distance * 0.6, -base[2]])
        self.worker_op.Set(rs.Gf.Vec3d(*map(float, self.worker_pos)))
        self.worker_yaw.Set(-140.0)

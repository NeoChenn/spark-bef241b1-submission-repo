from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from detection.localize import pixel_rays, ray_cylinder
from integration import repair_scene as rs
from integration.kinematics import ARM_STOWED
from policy.force_tools import clamp_force_action, telemetry_frame
from policy.interfaces import CameraModel, PipeGeometry, camera_from_usd

TWO_PI = 2 * math.pi
SETPOINT_CAP = 38.0
LEGS = ("FL", "FR", "RL", "RR")


def smooth(s: float) -> float:
    return s * s * (3 - 2 * s)


class LabWorld:
    def __init__(self, state, min_gap: float = 0.16):
        self.s = state
        self.min_gap = min_gap
        self.painted = torch.zeros(1, state.H, state.W, dtype=torch.bool, device=state.device)

    def __getattr__(self, name):
        return getattr(self.s, name)

    @property
    def rust(self):
        return self.s.rust

    def _mask(self, mask):
        return torch.as_tensor(mask, device=self.s.device).bool().view(1, self.s.H, self.s.W)

    def spray(self, mask):
        self.s.spray(self._mask(mask))

    def paint(self, mask):
        self.painted |= self._mask(mask)

    def step_soak(self, dt: float):
        pass

    def fraction_remaining(self):
        return self.s.fraction_remaining().detach().cpu()

    def blob(self, theta, z, size=0.05, level=0.85):
        s = self.s
        dth = (torch.remainder(s.theta_c.view(1, -1) - theta + math.pi, TWO_PI) - math.pi) * float(s.radius[0])
        dz = s.z_c.view(-1, 1) - z
        f = torch.exp(-0.5 * ((dth / size) ** 2 + (dz / (0.8 * size)) ** 2))
        return (torch.clamp((f - 0.3) / 0.25, 0, 1) * level).unsqueeze(0)

    def nearest_reachable(self, theta, z):
        s = self.s
        reach = s.reachable[0]
        dth = (torch.remainder(s.theta_c.view(1, -1) - theta + math.pi, TWO_PI) - math.pi) * float(s.radius[0])
        d = dth ** 2 + (s.z_c.view(-1, 1) - z) ** 2
        inner = reach.clone()
        for _ in range(3):
            inner = inner & torch.roll(inner, 1, 0) & torch.roll(inner, -1, 0) & torch.roll(inner, 1, 1) & torch.roll(inner, -1, 1)
        mask = inner if inner.any() else reach
        central = mask & (s.z_c.abs() < 0.2).view(-1, 1)
        mask = central if central.any() else mask
        rust = s.rust[0] > 0.05
        if rust.any():
            ii, jj = torch.nonzero(rust, as_tuple=True)
            dth_r = (torch.remainder(s.theta_c.view(1, 1, -1) - s.theta_c[jj].view(-1, 1, 1) + math.pi, TWO_PI) - math.pi)
            dz_r = s.z_c.view(1, -1, 1) - s.z_c[ii].view(-1, 1, 1)
            clear = ((dth_r * float(s.radius[0])) ** 2 + dz_r ** 2).min(0).values
            far = mask & (clear > self.min_gap ** 2)
            mask = far if far.any() else mask & (clear >= clear[mask].max())
        i, j = divmod(int(torch.where(mask, d, torch.full_like(d, 1e9)).argmin()), s.W)
        return float(s.theta_c[j]), float(s.z_c[i])

    def inject_patch(self, theta, z, size=0.04, level=0.8):
        s = self.s
        theta, z = self.nearest_reachable(theta, z)
        s.rust = torch.maximum(s.rust, self.blob(theta, z, size, level) * s.reachable[:1].float())
        s.initial_total = torch.maximum(s.initial_total, s.rust.sum(dim=(1, 2)))
        return theta, z


class ExternalScrub:
    def reset(self):
        pass

    def step(self, obs):
        return np.array([0.0, 0.0, 1.0])


@dataclass
class Event:
    t: float
    name: str
    fire: Callable[[], None]


class LabRepairBridge:
    def __init__(self, env, base, policy, stage, world: LabWorld, render: Callable[[dict], None],
                 inspection_annotator, start_x: float = -2.6, inspection_size=(640, 360), fps: float = 30.0):
        self.env, self.base, self.policy, self.stage, self.world = env, base, policy, stage, world
        self.render_cb, self.insp_annot, self.size = render, inspection_annotator, inspection_size
        self.dev = base.device
        self.dt, self.phys_dt = base.step_dt, base.physics_dt
        self.t = 0.0
        self.obs = env.get_observations()
        self.state_fn: Callable[[], str] = lambda: "PATROL"
        self.hud: list[str] = []
        self.events: list[Event] = []
        self.pending_events: list[dict] = []
        self.force_log: list[dict] = []
        self.fps = fps
        self.frame_dt, self.next_frame = 1.0 / fps, 0.0
        self.speed = 1.0
        self.force_now = self.force_peak = self.force_model = 0.0
        self.chunk_peak = 0.0
        self.press: Optional[float] = None
        self.joint_target: Optional[torch.Tensor] = None
        self.parked = False
        self.spraying = False
        self.bead_cells: list = []
        self.worker_pos: Optional[np.ndarray] = None
        r = base._robot
        self.arm_ids = list(base.arm_ids)
        names = [r.joint_names[i] for i in self.arm_ids]
        self.q_stow = torch.tensor([[ARM_STOWED[n] for n in names]], device=self.dev)
        self.q_work = r.data.default_joint_pos[:1, self.arm_ids].clone()
        self.lift_ids = list(base.lift_ids)
        self.lift_work = float(r.data.joint_pos[0, self.lift_ids[0]]) if self.lift_ids else None
        self.leg_ids = {f"{leg}_{part}": r.find_joints(f"{leg}_{part}_joint")[0][0]
                        for leg in LEGS for part in ("thigh", "calf")}
        self.leg_default = r.data.default_joint_pos[:1].clone()
        self.station = r.data.root_pos_w[:1].clone()
        self.root_quat = r.data.root_quat_w[:1].clone()
        self.origin = base.scene.env_origins[0].cpu().numpy()
        o = (base.pipe_o[0] + base.scene.env_origins[0]).cpu().numpy()
        self._pipe = PipeGeometry(origin=o, axis=base.pipe_a[0].cpu().numpy(), ref=base.pipe_e[0].cpu().numpy(),
                                  radius=float(base.pipe_r[0]), length=base.cfg.pipe_length,
                                  n_z=world.s.H, n_theta=world.s.W)
        self.brush_r = base.cfg.brush_radius
        orig_apply = base._apply_action

        def apply_action():
            if self.joint_target is not None:
                r.set_joint_position_target(self.joint_target, joint_ids=self.arm_ids)
            else:
                orig_apply()

        base._apply_action = apply_action
        self.insp = rs.Camera(stage, rs.INSPECTION_CAMERA, focal=35.0)
        self.damage_theta = float(base.cfg.rust_theta_window[0])
        self.views = rs.inspection_views(self._pipe, self.damage_theta)
        self.view = 0
        self.insp.look(*self.views[0])
        self.lens_mist = rs.add_lens_mist(stage, rs.INSPECTION_CAMERA)
        self.worker_op, self.worker_yaw = rs.add_worker(stage)
        self.spray_cone, self.spray_op = rs.add_spray_cone(stage, "/World/Repair/spray_mist")
        self.last_inspection: Optional[np.ndarray] = None
        self.inspections: list = []
        self.start = self.station.clone()
        self.start[0, 0] += start_x
        self._set_root(self.start)
        self.joint_target = self.q_stow.clone()
        q = self._full_q(self.q_stow)
        if self.lift_ids:
            q[:, self.lift_ids] = 0.0
            r.set_joint_position_target(q[:, self.lift_ids], joint_ids=self.lift_ids)
        r.write_joint_state_to_sim(q, torch.zeros_like(r.data.joint_pos[:1]), env_ids=torch.tensor([0], device=self.dev))
        self.parked = True
        self.at_station = start_x == 0.0

    def _full_q(self, arm_q):
        q = self.base._robot.data.joint_pos[:1].clone()
        q[:, self.arm_ids] = arm_q
        return q

    def _set_root(self, pos):
        pose = torch.cat([pos, self.root_quat], -1)
        self.base._robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=self.dev))

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    def _hold_action(self):
        a = torch.zeros(1, 4, device=self.dev)
        a[:, 3] = -1.0
        return a

    def _step(self, act=None):
        b = self.base
        with torch.inference_mode():
            if act is None:
                act = self._hold_action()
            self.obs, _, _, _ = self.env.step(act)
        self.t += self.dt
        th = b.theta[:1]
        n = torch.cos(th)[:, None] * b.pipe_e[:1] + torch.sin(th)[:, None] * b.pipe_b[:1]
        fn = b._sensor_normal_force(n.expand(b.num_envs, 3))[0].cpu().numpy()[::-1]
        for k, f in enumerate(fn):
            self.force_log.append({"t": round(self.t - (len(fn) - 1 - k) * self.phys_dt, 4), "force_n": round(float(f), 3),
                                   "state": self.state_fn(), "parked": self.parked or self.press is None})
        self.force_now = float(b.force_sensor[0])
        self.force_peak = float(b.force_peak[0])
        self.force_model = float(b.force_model[0])
        self.chunk_peak = max(self.chunk_peak, self.force_peak)
        self._fire_events()
        if self.t + 1e-9 >= self.next_frame:
            self.next_frame = self.t + self.frame_dt
            self.render_cb(self.snapshot())

    def snapshot(self) -> dict:
        w = self.world
        ev, self.pending_events = self.pending_events, []
        return {"t": self.t, "state": self.state_fn(), "hud": list(self.hud), "force_n": self.force_now,
                "force_peak_n": self.force_peak, "force_model_n": self.force_model, "events": ev,
                "rust": w.s.rust[0].detach().cpu().numpy(), "sprayed": w.s.sprayed[0].detach().cpu().numpy(),
                "painted": w.painted[0].detach().cpu().numpy(), "speed": self.speed,
                "true_rust_fraction": float(w.fraction_remaining()[0])}

    def _fire_events(self):
        while self.events and self.events[0].t <= self.t:
            e = self.events.pop(0)
            e.fire()
            if e.name:
                self.pending_events.append({"t": round(self.t, 3), "type": "scene_change", "name": e.name})
                self.log(f"   >>> [scene change t={self.t:5.1f}s] {e.name}")

    def schedule(self, t: float, name: str, fire: Callable[[], None]) -> None:
        self.events.append(Event(t, name, fire))
        self.events.sort(key=lambda e: e.t)

    def record_decision(self, rec: dict) -> None:
        self.pending_events.append(rec)
        if rec.get("to") == "SCRUB" and rec.get("from") != "SCRUB":
            self.chunk_peak = 0.0

    def _interp_joints(self, q1, steps: int = 24, stop_force: float | None = None, lift: float | None = None) -> bool:
        r = self.base._robot
        q0 = r.data.joint_pos[:1, self.arm_ids].clone()
        l0 = r.data.joint_pos[:1, self.lift_ids].clone() if self.lift_ids else None
        for k in range(1, steps + 1):
            s = smooth(k / steps)
            self.joint_target = q0 + (q1 - q0) * s
            if lift is not None and l0 is not None:
                r.set_joint_position_target(l0 + (lift - l0) * s, joint_ids=self.lift_ids)
            self._step()
            if stop_force is not None and self.force_peak > stop_force:
                return False
        return True

    def _park(self):
        if self.parked:
            return
        self.press = None
        self._interp_joints(self.q_stow, steps=30, lift=0.0)
        self.parked = True
        self.force_now = 0.0

    def _unpark(self):
        if not self.parked:
            return
        clear = self._interp_joints(self.q_work, steps=36, stop_force=5.0, lift=self.lift_work)
        b = self.base
        if not clear:
            self.joint_target = b._robot.data.joint_pos[:1, self.arm_ids].clone()
            self.log(f"   unpark: brush touched the pipe ({self.force_peak:.0f} N) -> lift off in task space")
        for _ in range(4):
            self._step()
        b.p_tgt[:] = b.brush_w
        self.joint_target = None
        if not clear:
            th = b.theta[:1]
            n = torch.cos(th)[:, None] * b.pipe_e[:1] + torch.sin(th)[:, None] * b.pipe_b[:1]
            self._goto(self._brush() + 0.06 * n[0].cpu().numpy(), 10)
        self.parked = False
        self.chunk_peak = 0.0

    def _brush(self) -> np.ndarray:
        return (self.base.brush_w[0] + self.base.scene.env_origins[0]).cpu().numpy()

    def _goto(self, centre: np.ndarray, steps: int):
        b = self.base
        start = b.p_tgt[0].clone()
        end = torch.as_tensor(centre - self.origin, device=self.dev, dtype=torch.float32)
        for k in range(1, steps + 1):
            b.p_tgt[0] = start + (end - start) * smooth(k / steps)
            self._step()
        return float(np.linalg.norm(self._brush() - centre))

    def _walk(self, target, speed: float = 0.5):
        r = self.base._robot
        p0 = r.data.root_pos_w[:1].clone()
        dist = float((target - p0)[0, :2].norm())
        n = max(2, int(dist / speed / self.dt))
        for k in range(1, n + 1):
            s = smooth(k / n)
            p = p0 + (target - p0) * s
            phase = k / n * dist / 0.16 * math.pi
            swing = 0.0 if k == n else 0.3
            q = self.leg_default.clone()
            for leg, sign in (("FL", 1), ("RR", 1), ("FR", -1), ("RL", -1)):
                lift = max(0.0, sign * math.sin(phase))
                q[0, self.leg_ids[f"{leg}_thigh"]] -= swing * lift * 0.6
                q[0, self.leg_ids[f"{leg}_calf"]] -= swing * lift * 0.9
            p[0, 2] += 0.01 * abs(math.sin(phase)) * swing
            leg_ids = list(self.leg_ids.values())
            r.set_joint_position_target(q[:, leg_ids], joint_ids=leg_ids)
            self._set_root(p)
            self._step()
        leg_ids = list(self.leg_ids.values())
        r.set_joint_position_target(self.leg_default[:, leg_ids], joint_ids=leg_ids)

    def sim_time(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        n = max(1, int(round(dt / self.dt)))
        if self.press is not None:
            cap = clamp_force_action(1.0, SETPOINT_CAP * self.press)
            for _ in range(n):
                with torch.inference_mode():
                    act = self.policy(self.obs)
                    act[:, 3] = torch.clamp(act[:, 3], max=cap)
                self._step(act)
            self.press = None
            return
        if dt >= 2.0:
            self.frame_dt, self.speed = 0.5, 15.0
        for _ in range(n):
            self._step()
        self.frame_dt, self.speed = 1.0 / self.fps, 1.0

    def pipe(self) -> PipeGeometry:
        return self._pipe

    def patrol_poses(self) -> list:
        return [{"name": "main_pipe_station_1"}]

    def go_to_patrol_pose(self, p) -> bool:
        if not self.at_station:
            self._walk(self.station)
            self.at_station = True
        return True

    def approach(self, theta: float, z: float, standoff: float) -> bool:
        self.go_to_patrol_pose(None)
        self._unpark()
        surf, n = self._pipe.surface_point(theta, z)
        err = self._goto(surf + (standoff + self.brush_r) * n, 30)
        return err < 0.03

    def person_distance(self) -> Optional[float]:
        if self.worker_pos is None:
            return None
        base = self.base._robot.data.root_pos_w[0].cpu().numpy()
        return float(np.linalg.norm(self.worker_pos[:2] - base[:2]))

    def scrub_obs(self) -> dict:
        return {}

    def publish_overlay(self, lines: list[str]) -> None:
        self.hud = list(lines)

    def _render_inspection(self) -> Optional[np.ndarray]:
        for _ in range(3):
            self.base.sim.render()
        data = np.asarray(self.insp_annot.get_data())
        if not data.size:
            return None
        return np.ascontiguousarray(data[..., :3]).astype(np.uint8)

    def _lift_off(self, distance: float = 0.06) -> None:
        b = self.base
        th = b.theta[:1]
        n = torch.cos(th)[:, None] * b.pipe_e[:1] + torch.sin(th)[:, None] * b.pipe_b[:1]
        b.p_tgt[0] = b.brush_w[0]
        self._goto(self._brush() + distance * n[0].cpu().numpy(), 8)

    def observe(self) -> Optional[np.ndarray]:
        if self.state_fn() == "SCRUB" and not self.parked:
            self.press = None
            self._lift_off()
        else:
            self._park()
        rgb = self._render_inspection()
        if rgb is None or float(rgb.mean()) < 1.0:
            return None
        self.last_inspection = rgb
        self.inspections.append((round(self.t, 2), rgb))
        self.inspections = self.inspections[-40:]
        roi = self._pipe_roi()
        px = rgb[roi].astype(np.float32)
        if not len(px) or px.std() < 6.0 or px.mean() < 8.0:
            return rgb
        out = rgb.copy()
        out[~roi] = 118
        return out

    def _pipe_roi(self, erode: int = 4) -> np.ndarray:
        cam = self.camera()
        key = (self.view, tuple(np.round(cam.T_world_cam.reshape(-1), 4)))
        if getattr(self, "_roi_key", None) != key:
            origin, dirs = pixel_rays(cam)
            hit, *_ = ray_cylinder(origin, dirs, self._pipe)
            for _ in range(erode):
                hit = hit & np.roll(hit, 1, 0) & np.roll(hit, -1, 0) & np.roll(hit, 1, 1) & np.roll(hit, -1, 1)
            self._roi_key, self._roi = key, hit
        return self._roi

    def camera(self) -> CameraModel:
        return camera_from_usd(self.insp.world_matrix(), self.insp.focal, self.insp.aperture, *self.size)

    def request_view(self, index: int) -> bool:
        if not 0 <= index < len(self.views):
            return False
        self.view = index
        self.insp.look(*self.views[index])
        return True

    def move_to(self, waypoints) -> list[bool]:
        self._unpark()
        reached = []
        for k, wp in enumerate(waypoints):
            centre = np.asarray(wp.position, float) - self.brush_r * np.asarray(wp.approach, float)
            spraying = wp.flow_rate > 0
            if spraying:
                rs.set_spray(self.spray_cone, self.spray_op, self._brush(), np.asarray(wp.approach, float),
                             wp.standoff + 0.02, True)
            err = self._goto(centre, 24 if k == 0 else 3)
            reached.append(err < 0.03)
        rs.set_spray(self.spray_cone, self.spray_op, np.zeros(3), np.array([1.0, 0, 0]), 0.1, False)
        return reached

    def apply_ee_delta(self, action: np.ndarray) -> None:
        self._unpark()
        self.press = float(np.clip(action[2], 0.0, 1.0))
        self.chunk_peak = 0.0

    def ee_theta_z(self) -> tuple[float, float]:
        return float(self.base.theta[0]) % TWO_PI, float(self.base.axial[0])

    def current_force(self) -> float:
        if self.state_fn() not in ("SCRUB", "BACK_OFF"):
            return float(self.force_now)
        return float(max(self.chunk_peak, self.force_now))

    def back_off(self, distance: float = 0.05) -> None:
        b = self.base
        th = b.theta[:1]
        n = torch.cos(th)[:, None] * b.pipe_e[:1] + torch.sin(th)[:, None] * b.pipe_b[:1]
        centre = self._brush() + distance * n[0].cpu().numpy()
        b.p_tgt[0] = b.brush_w[0]
        self._goto(centre, 10)
        self.chunk_peak = 0.0

    def spawn_patch(self, theta: float, z: float, size: float = 0.04, level: float = 0.8) -> None:
        th, zz = self.world.inject_patch(theta, z, size=size, level=level)
        self.log(f"   spawn_patch requested ({theta:.2f}, {z:+.2f}) -> placed at reachable ({th:.2f}, {zz:+.2f})")
        self.pending_events.append({"t": round(self.t, 3), "type": "spawn_patch", "theta": round(th, 3), "z": round(zz, 3),
                                    "cells": int((self.world.s.rust[0] > 0.1).sum())})

    def occlude_camera(self, on: bool) -> None:
        img = rs.UsdGeom.Imageable(self.lens_mist)
        img.MakeVisible() if on else img.MakeInvisible()

    def force_spike(self, height: float = 0.010) -> None:
        if self.parked or self.press is None:
            self.schedule(self.t + 0.2, "", lambda: self.force_spike(height))
            return
        self.log(f"   weld bead placed under the brush at t={self.t:.1f}s")
        self.pending_events.append({"t": round(self.t, 3), "type": "force_spike", "height_m": height})
        theta, z = float(self.base.theta[0]), float(self.base.axial[0])
        self.bead_cells.append((theta, z, height))
        self.base.place_bead([0], theta, z, height)

    def person_at(self, distance: Optional[float]) -> None:
        if distance is None:
            self.worker_pos = None
            self.worker_op.Set(rs.Gf.Vec3d(0, 0, -50))
            return
        base = self.base._robot.data.root_pos_w[0].cpu().numpy()
        self.worker_pos = base + np.array([distance * 0.8, distance * 0.6, -base[2]])
        self.worker_op.Set(rs.Gf.Vec3d(*map(float, self.worker_pos)))
        self.worker_yaw.Set(-140.0)


def frame_telemetry(snap: dict, rust_fraction, detector_conf) -> dict:
    return telemetry_frame(snap["t"], snap["state"], snap["force_n"], rust_fraction, detector_conf, snap["events"],
                           snap["rust"], force_peak_n=round(snap["force_peak_n"], 2),
                           force_model_n=round(snap["force_model_n"], 2), hud=snap["hud"],
                           true_rust_fraction=round(snap["true_rust_fraction"], 4))


def write_jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

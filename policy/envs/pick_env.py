"""PICK skill env: Isaac Lab DirectRLEnv. Import only after the Isaac Sim app is launched. Mirrors scrub_env.py.

The real arm moves (6D damped-least-squares IK on ``grasp_frame``: position from the policy's TCP target, gripper
pointing down, jaw axis at the policy's yaw) and the real fingers open and close. The litter object and the grasp,
knock and drop events are analytic, from policy.envs.pick_state.PickState, fed with the MEASURED grasp_frame pose
and finger opening every control step (as the scrub env feeds the measured brush pose to RustState). So
observations, actions and rewards are exactly those of the CPU env (pick_cpu_env.py), and a CPU checkpoint can
initialise an Isaac run (train_pick.py --init_from).

NOT YET RUN IN ISAAC: written against the scrub env's working Isaac Lab 2.3 code and the patrol team's gripper URDF.
Expect to adjust PICK_INIT (start pose) and the IK gains on the first run.
"""
from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat

from policy.envs.pick_cpu_env import pick_reward
from policy.envs.pick_env_cfg import PickEnvCfg
from policy.envs.pick_state import PickParams, PickState


def _rot_down(yaw: torch.Tensor) -> torch.Tensor:
    """Target grasp_frame rotation: x (approach) = world -z, y (jaw axis) = (-sin yaw, cos yaw, 0)."""
    x = torch.tensor([0.0, 0.0, -1.0], device=yaw.device).expand(len(yaw), 3)
    y = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], -1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], -1)          # columns


class PickEnv(DirectRLEnv):
    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.robot_cfg = cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        self.arm_ids, _ = self._robot.find_joints(cfg.arm_joint_names, preserve_order=True)
        self.finger_ids, _ = self._robot.find_joints(cfg.finger_joint_names, preserve_order=True)
        self.lift_ids = self._robot.find_joints(cfg.lift_joint_name)[0] if cfg.lift_joint_name else []
        self.tcp_idx = self._robot.find_bodies(cfg.tcp_body_name)[0][0]
        self.jac_idx = self.tcp_idx - 1 if self._robot.is_fixed_base else self.tcp_idx
        self.jac_cols = list(self.arm_ids) if self._robot.is_fixed_base else [i + 6 for i in self.arm_ids]
        lim = self._robot.data.soft_joint_pos_limits[0, self.arm_ids]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.state = PickState(N, PickParams(), device=dev)
        self.actions = torch.zeros(N, 5, device=dev)
        self.prev_actions = torch.zeros(N, 5, device=dev)
        self.open_cmd = torch.ones(N, dtype=torch.bool, device=dev)
        self._prev = None
        self._ev = None
        self._first_grasp = torch.zeros(N, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        if self.cfg.spawn_visuals:
            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    # ---------------------------------------------------------- measurement
    def _measure(self):
        """grasp_frame position (env frame), yaw of the jaw axis, and finger opening."""
        d = self._robot.data
        p = d.body_pos_w[:, self.tcp_idx] - self.scene.env_origins
        R = matrix_from_quat(d.body_quat_w[:, self.tcp_idx])
        y = R[:, :, 1]
        yaw = torch.atan2(-y[:, 0], y[:, 1])
        grip = d.joint_pos[:, self.finger_ids].sum(-1)
        return p, yaw, grip

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clamp(-1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        s = self.state
        self._prev = (s.tcp.clone(), s.yaw.clone(), s.grip.clone())
        self.open_cmd = s.command(a)

    def _apply_action(self):
        c, s, d = self.cfg, self.state, self._robot.data
        p_w = d.body_pos_w[:, self.tcp_idx]
        R = matrix_from_quat(d.body_quat_w[:, self.tcp_idx])
        jac = self._robot.root_physx_view.get_jacobians()[:, self.jac_idx][:, :, self.jac_cols]  # (N,6,J)
        err_p = (s.tgt + self.scene.env_origins) - p_w
        n = err_p.norm(dim=-1, keepdim=True)
        err_p = err_p * torch.clamp(c.ik_max_step / n.clamp_min(1e-6), max=1.0)
        Rd = _rot_down(s.yaw_tgt)
        err_r = 0.5 * sum(torch.cross(R[:, :, i], Rd[:, :, i], dim=-1) for i in range(3))
        err = torch.cat([err_p, c.ik_rot_weight * err_r], -1)
        lam2 = c.ik_damping ** 2
        JJt = torch.bmm(jac, jac.transpose(1, 2)) + lam2 * torch.eye(6, device=self.device)
        dq = torch.bmm(jac.transpose(1, 2), torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
        q = (d.joint_pos[:, self.arm_ids] + dq).clamp(self.q_lo, self.q_hi)
        self._robot.set_joint_position_target(q, joint_ids=self.arm_ids)
        # jaws: each finger to half the commanded opening, unless holding (then they rest on the object)
        half = torch.where(s.attached, 0.5 * s.w, 0.5 * s.grip_cmd).clamp(0.0, c.finger_max)
        self._robot.set_joint_position_target(half[:, None].repeat(1, len(self.finger_ids)), joint_ids=self.finger_ids)
        if self.lift_ids:
            self._robot.set_joint_position_target(torch.zeros(self.num_envs, 1, device=self.device),
                                                  joint_ids=self.lift_ids)

    # ------------------------------------------------------------ post-physics
    def _get_dones(self):
        s = self.state
        held_before = s.attached.clone()
        s.tcp, s.yaw, s.grip = self._measure()
        self._ev = s.events(self.open_cmd, *self._prev)
        self._first_grasp = self._ev["grasped"] & ~held_before
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return s.success.clone(), truncated

    def _get_rewards(self) -> torch.Tensor:
        return pick_reward(self.state, self._ev, self.actions, self.prev_actions, self._first_grasp,
                           self.max_episode_length - 1 - self.episode_length_buf)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        s = self.state
        if self._ev is not None:
            self.extras.setdefault("log", {}).update({
                "Episode/success": s.success[env_ids].float().mean().item(),
                "Episode/grasped": s.ever_grasped[env_ids].float().mean().item(),
                "Episode/knocks": s.knocks[env_ids].mean().item(),
                "Episode/drops": s.drops[env_ids].mean().item()})
        super()._reset_idx(env_ids)
        s.reset(env_ids)                                   # object, crouch height, camera bias, start TCP target
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :3] = self.scene.env_origins[env_ids]
        root[:, 2] += s.trunk_h[env_ids]                    # crouched trunk
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=env_ids)
        jp = self._robot.data.default_joint_pos[env_ids].clone()
        self._robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        self._robot.set_joint_position_target(jp, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

    # ------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        tf = self.episode_length_buf.float() / self.max_episode_length
        obs = self.state.observe(self.actions, tf)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}


TASK = "Rig-Pick-Go2UR3Gripper-v0"
gym.register(
    id=TASK,
    entry_point="policy.envs.pick_env:PickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "policy.envs.pick_env_cfg:PickEnvCfg",
            "rsl_rl_cfg_entry_point": "policy.envs.pick_env_cfg:PickPPORunnerCfg"},
)

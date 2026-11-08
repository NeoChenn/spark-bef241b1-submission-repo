from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import sample_uniform

from policy.envs.pickcan_env_cfg import CANS, PickCanEnvCfg, can_spawn_cfg
from policy.pickcan import core

DOWN = (0.0, 0.0, -1.0)


class PickCanEnv(DirectRLEnv):
    cfg: PickCanEnvCfg

    def __init__(self, cfg: PickCanEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.robot_cfg = cfg.robot_cfg.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=cfg.robot_cfg.init_state.replace(pos=(0.0, 0.0, cfg.base_height), rot=(1.0, 0.0, 0.0, 0.0)))
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        self.ctrl_dt = cfg.sim.dt * cfg.decimation
        self.arm_ids, _ = self._robot.find_joints(cfg.arm_joint_names, preserve_order=True)
        self.lift_id = self._robot.find_joints("lift_joint")[0][0]
        self.tip_idx = self._robot.find_bodies(cfg.tip_body)[0][0]
        self.jac_idx = self.tip_idx - 1
        lim = self._robot.data.soft_joint_pos_limits[0, self.arm_ids]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.down = torch.tensor(DOWN, device=dev).repeat(N, 1)
        self.work_lo = torch.tensor(cfg.work_lo, device=dev)
        self.work_hi = torch.tensor(cfg.work_hi, device=dev)
        k = torch.arange(N, device=dev) % len(CANS)
        self.can_r = torch.tensor([c[1] for c in CANS], device=dev)[k]
        self.can_h = torch.tensor([c[2] for c in CANS], device=dev)[k]
        self.can_kind = k
        z3 = lambda: torch.zeros(N, 3, device=dev)
        z1 = lambda: torch.zeros(N, device=dev)
        zb = lambda: torch.zeros(N, dtype=torch.bool, device=dev)
        self.mass = torch.full((N,), cfg.nominal_mass, device=dev)
        self.inertia = z1()
        self.actions = torch.zeros(N, 5, device=dev)
        self.prev_actions = torch.zeros(N, 5, device=dev)
        self.p_tgt, self.rel_p = z3(), z3()
        self.rel_R = torch.eye(3, device=dev).repeat(N, 1, 1)
        self.lift_tgt = z1()
        self.suction, self.sealed, self.ever_sealed, self.success = zb(), zb(), zb(), zb()
        self.lying = zb()
        self.rest_z, self.lift_max = z1(), z1()
        self.drops, self.seals = z1(), z1()
        self.max_force, self.max_deck_force, self.tool_force, self.deck_force = z1(), z1(), z1(), z1()
        self.success_step = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.det_bias = z3()
        self.det_yaw_bias = z1()
        self.new_seal = zb()
        self.lost_seal = zb()
        self.wrench_f, self.wrench_t = z3(), z3()
        self._started = False
        self._randomize_bodies()

    def _setup_scene(self):
        c = self.cfg
        self._robot = Articulation(c.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._can = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Can", spawn=can_spawn_cfg(c.nominal_mass),
                                               init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.3, 0.05))))
        self.scene.rigid_objects["can"] = self._can
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=c.deck_friction, dynamic_friction=c.deck_friction,
                                             friction_combine_mode="multiply")
        ground = sim_utils.GroundPlaneCfg(physics_material=mat, color=(0.32, 0.33, 0.35))
        ground.func("/World/ground", ground)
        self._tool_sensor = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{c.tool_body}",
                                                           update_period=0.0, history_length=c.decimation))
        self.scene.sensors["tool_contact"] = self._tool_sensor
        self._can_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Can", update_period=0.0,
                                                          history_length=c.decimation))
        self.scene.sensors["can_contact"] = self._can_sensor
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def _randomize_bodies(self):
        c = self.cfg
        view = self._can.root_physx_view
        idx = torch.arange(self.num_envs, dtype=torch.int32)
        m = sample_uniform(*c.mass_range, (self.num_envs,), "cpu")
        masses = view.get_masses().clone()
        inert = view.get_inertias().clone()
        scale = (m / masses[:, 0]).view(-1, 1)
        masses[:, 0] = m
        inert = inert * scale.view(-1, 1, 1) if inert.dim() == 3 else inert * scale
        view.set_masses(masses, idx)
        view.set_inertias(inert, idx)
        mats = view.get_material_properties().clone()
        f = sample_uniform(*c.friction_range, (self.num_envs,), "cpu")
        mats[..., 0] = f[:, None]
        mats[..., 1] = 0.85 * f[:, None]
        mats[..., 2] = 0.05
        view.set_material_properties(mats, idx)
        self.mass[:] = m.to(self.device)
        r, h = self.can_r, self.can_h
        self.inertia[:] = self.mass * (3 * r ** 2 + h ** 2) / 12
        self.friction = f.to(self.device)

    def _tip(self):
        d = self._robot.data
        o = self.scene.env_origins
        return (d.body_pos_w[:, self.tip_idx] - o, core.quat_to_mat(d.body_quat_w[:, self.tip_idx]),
                d.body_lin_vel_w[:, self.tip_idx], d.body_ang_vel_w[:, self.tip_idx])

    def _can_state(self):
        d = self._can.data
        return (d.root_pos_w - self.scene.env_origins, core.quat_to_mat(d.root_quat_w), d.root_lin_vel_w,
                d.root_ang_vel_w)

    def _pre_physics_step(self, actions: torch.Tensor):
        c = self.cfg
        a = actions.clamp(-1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        tip_p = self._tip()[0]
        tgt = self.p_tgt + a[:, :3] * c.action_scale
        tgt = tip_p + (tgt - tip_p).clamp(-c.max_target_gap, c.max_target_gap)
        self.p_tgt[:] = torch.maximum(torch.minimum(tgt, self.work_hi), self.work_lo)
        self.suction[:] = a[:, 3] > 0
        self.lift_tgt[:] = (self.lift_tgt + a[:, 4] * c.lift_scale).clamp(0.0, c.lift_max)
        self.new_seal[:] = False
        self.lost_seal[:] = False

    def _apply_action(self):
        c = self.cfg
        d = self._robot.data
        tip_p, tip_R, tip_v, tip_w = self._tip()
        jac = self._robot.root_physx_view.get_jacobians()[:, self.jac_idx][:, :, self.arm_ids]
        err_p = self.p_tgt - tip_p
        n = err_p.norm(dim=-1, keepdim=True)
        err_p = err_p * torch.clamp(0.05 / n.clamp_min(1e-6), max=1.0)
        err_r = torch.cross(tip_R[:, :, 0], self.down, dim=-1) * c.rot_weight
        e = torch.cat([err_p, err_r], -1)
        lam2 = 0.05 ** 2
        JJt = torch.bmm(jac, jac.transpose(1, 2)) + lam2 * torch.eye(6, device=self.device)
        dq = torch.bmm(jac.transpose(1, 2), torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        q = d.joint_pos[:, self.arm_ids] + dq.clamp(-0.2, 0.2)
        q = torch.maximum(torch.minimum(q, self.q_hi), self.q_lo)
        self._robot.set_joint_position_target(q, joint_ids=self.arm_ids)
        self._robot.set_joint_position_target(self.lift_tgt[:, None], joint_ids=[self.lift_id])
        self._suction_step(tip_p, tip_R, tip_v, tip_w)

    def _suction_step(self, tip_p, tip_R, tip_v, tip_w):
        c = self.cfg
        can_p, can_R, can_v, can_w = self._can_state()
        tool_axis = tip_R[:, :, 0]
        ok = core.seal_ok(tip_p, tool_axis, can_p, can_R, self.can_r, self.can_h, c.seal_gap, c.seal_margin,
                          c.seal_align)
        start = self.suction & ~self.sealed & ok
        if start.any():
            ids = start.nonzero(as_tuple=False).squeeze(-1)
            local = torch.bmm(can_R[ids].transpose(1, 2), (tip_p[ids] - can_p[ids]).unsqueeze(-1)).squeeze(-1)
            dist, n_l, _ = core.surface(local, self.can_r[ids], self.can_h[ids])
            n_w = torch.bmm(can_R[ids], n_l.unsqueeze(-1)).squeeze(-1)
            p_goal = can_p[ids] + n_w * (dist - 0.002).clamp_min(0.0)[:, None]
            Rt = tip_R[ids].transpose(1, 2)
            self.rel_p[ids] = torch.bmm(Rt, (p_goal - tip_p[ids]).unsqueeze(-1)).squeeze(-1)
            self.rel_R[ids] = torch.bmm(Rt, can_R[ids])
            self.sealed[ids] = True
            self.ever_sealed[ids] = True
            self.new_seal[ids] = True
            self.seals[ids] += 1
        F, T, t_c, tear = core.suction_wrench(tip_p, tip_R, tip_v, tip_w, can_p, can_R, can_v, can_w, self.rel_p,
                                              self.rel_R, self.mass, self.inertia)
        broke = self.sealed & ((F.norm(dim=-1) > c.suction_force_cap) | (t_c.norm(dim=-1) > c.suction_torque_cap)
                               | (tear > c.seal_tear) | ~self.suction)
        if broke.any():
            over_tote = core.in_tote(torch.cat([can_p[:, :2], torch.full_like(can_p[:, :1], core.TOTE_RIM)], -1), 0.01)
            dropped = broke & ~over_tote & (can_p[:, 2] > self.rest_z + 0.03)
            self.drops += dropped.float()
            self.lost_seal |= dropped
            self.sealed &= ~broke
        on = self.sealed[:, None].float()
        self.wrench_f[:] = F * on
        self.wrench_t[:] = T * on
        self._can.permanent_wrench_composer.set_forces_and_torques(
            forces=self.wrench_f[:, None, :].contiguous(), torques=self.wrench_t[:, None, :].contiguous(), is_global=True)

    def _update_metrics(self):
        tf = self._tool_sensor.data.net_forces_w_history[:, :, 0].norm(dim=-1).max(dim=1).values
        cf = self._can_sensor.data.net_forces_w_history[:, :, 0]
        push = (cf[..., 2] - self.mass[:, None] * core.G).clamp_min(0.0).max(dim=1).values
        self.tool_force[:] = tf
        self.deck_force[:] = push * ~self.sealed
        self.max_force[:] = torch.maximum(self.max_force, tf)
        self.max_deck_force[:] = torch.maximum(self.max_deck_force, self.deck_force)
        can_p = self._can_state()[0]
        self.lift_max[:] = torch.maximum(self.lift_max, (can_p[:, 2] - self.rest_z) * self.sealed)

    def _get_dones(self):
        c = self.cfg
        self._update_metrics()
        can_p, _, can_v, _ = self._can_state()
        placed = core.in_tote(can_p) & ~self.sealed & (can_v.norm(dim=-1) < c.settle_speed)
        self.success[:] = placed & (self.lift_max >= c.lift_goal)
        newly = self.success & (self.success_step < 0)
        self.success_step[newly] = self.episode_length_buf[newly]
        lost = (can_p[:, :2].norm(dim=-1) > 1.0) | (can_p[:, 2] < -0.2)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.success | lost, truncated

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        tip_p = self._tip()[0]
        can_p, _, _, _ = self._can_state()
        g = self.grasp_estimate()
        reach = 1 - torch.tanh((g - tip_p).norm(dim=-1) / 0.1)
        lifted = ((can_p[:, 2] - self.rest_z) / c.lift_goal).clamp(0.0, 1.0) * self.sealed
        carry_pt = torch.tensor([core.TOTE_CENTER[0], core.TOTE_CENTER[1], c.carry_z - 0.1], device=self.device)
        carry = (1 - torch.tanh((can_p - carry_pt).norm(dim=-1) / 0.2)) * self.sealed * (self.lift_max >= c.lift_goal)
        placed = core.in_tote(can_p).float()
        act_rate = ((self.actions - self.prev_actions) ** 2).sum(-1)
        over = (self.tool_force - c.force_limit).clamp_min(0.0) + (self.deck_force - c.force_limit).clamp_min(0.0)
        idle = self.suction & ~self.sealed & ((g - tip_p).norm(dim=-1) > 0.05)
        return (c.w_reach * reach * ~self.ever_sealed
                + c.w_seal_bonus * self.new_seal.float()
                + c.w_hold * self.sealed.float()
                + c.w_lift * lifted
                + c.w_carry * carry
                + c.w_place * placed * (self.lift_max >= c.lift_goal)
                + c.w_success * self.success.float()
                - c.w_drop * self.lost_seal.float()
                - c.w_action_rate * act_rate
                - c.w_force * over
                - c.w_idle_suction * idle.float())

    def grasp_estimate(self):
        can_p, can_R, _, _ = self._can_state()
        tip_p = self._tip()[0]
        near = ((tip_p - can_p).norm(dim=-1) / self.cfg.det_close).clamp(0.25, 1.0)
        p = can_p + self.det_bias * near[:, None] + self.cfg.det_jitter * torch.randn_like(can_p)
        return core.grasp_point(p, can_R[:, :, 2], self.can_r, self.can_h)

    def can_estimate(self):
        can_p, can_R, _, _ = self._can_state()
        tip_p = self._tip()[0]
        near = ((tip_p - can_p).norm(dim=-1) / self.cfg.det_close).clamp(0.25, 1.0)
        p = can_p + self.det_bias * near[:, None]
        ax = can_R[:, :, 2]
        yaw = torch.atan2(ax[:, 1], ax[:, 0]) + self.det_yaw_bias * near
        return p, yaw, ax[:, 2].abs() < 0.7

    def _get_observations(self) -> dict:
        c = self.cfg
        d = self._robot.data
        tip_p, tip_R, tip_v, _ = self._tip()
        can_p, can_R, can_v, _ = self._can_state()
        est_p, est_yaw, est_lying = self.can_estimate()
        g = self.grasp_estimate()
        tote = torch.tensor([core.TOTE_CENTER[0], core.TOTE_CENTER[1], c.carry_z], device=self.device)
        obs = torch.cat([
            d.joint_pos[:, self.arm_ids], d.joint_vel[:, self.arm_ids] * 0.1,
            d.joint_pos[:, [self.lift_id]] * 3.0, self.lift_tgt[:, None] * 3.0,
            tip_p * 2.0, tip_v, tip_R[:, :, 0],
            (self.p_tgt - tip_p) * 10.0,
            est_p * 2.0, torch.sin(2 * est_yaw)[:, None], torch.cos(2 * est_yaw)[:, None], est_lying[:, None].float(),
            (g - tip_p) * 5.0, (tote - tip_p) * 3.0, (tote - can_p) * 3.0,
            self.can_r[:, None] * 20.0, self.can_h[:, None] * 5.0,
            can_v, ((can_p[:, 2] - self.rest_z) / c.lift_goal)[:, None],
            self.sealed[:, None].float(), self.suction[:, None].float(),
            (self.lift_max >= c.lift_goal)[:, None].float(), core.in_tote(can_p)[:, None].float(),
            (self.tool_force / 40.0)[:, None],
            self.actions,
        ], -1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        c = self.cfg
        if self._started:
            self._log_episodes(env_ids)
        self._started = True
        super()._reset_idx(env_ids)
        n, dev = len(env_ids), self.device
        jp = self._robot.data.default_joint_pos[env_ids].clone()
        jp[:, self.arm_ids] += sample_uniform(-c.init_joint_noise, c.init_joint_noise, (n, len(self.arm_ids)), dev)
        jp[:, self.lift_id] = 0.0
        self._robot.set_joint_position_target(jp, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)

        lying = torch.rand(n, device=dev) < c.prob_lying
        r, h = self.can_r[env_ids], self.can_h[env_ids]
        xy = self._sample_xy(n, r, h, lying)
        yaw = sample_uniform(-math.pi, math.pi, (n,), dev)
        z = core.rest_height(lying, r, h) + 0.002
        pos = torch.cat([xy, z[:, None]], -1)
        quat = core.can_quat(lying, yaw)
        self._can.write_root_pose_to_sim(torch.cat([pos + self.scene.env_origins[env_ids], quat], -1), env_ids=env_ids)
        self._can.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self.lying[env_ids] = lying
        self.rest_z[env_ids] = z - 0.002

        self.det_bias[env_ids] = torch.randn(n, 3, device=dev) * torch.tensor(
            [c.det_sigma_xy, c.det_sigma_xy, c.det_sigma_z], device=dev)
        self.det_yaw_bias[env_ids] = torch.randn(n, device=dev) * c.det_sigma_yaw
        p_tip = self._fk_tip_guess(env_ids)
        self.p_tgt[env_ids] = p_tip
        self.lift_tgt[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        for buf in (self.suction, self.sealed, self.ever_sealed, self.success, self.new_seal, self.lost_seal):
            buf[env_ids] = False
        for buf in (self.lift_max, self.drops, self.seals, self.max_force, self.max_deck_force, self.tool_force,
                    self.deck_force):
            buf[env_ids] = 0.0
        self.success_step[env_ids] = -1
        self.wrench_f[env_ids] = 0.0
        self.wrench_t[env_ids] = 0.0

    def _fk_tip_guess(self, env_ids):
        return torch.tensor([0.1, 0.2, 0.55], device=self.device).repeat(len(env_ids), 1)

    def _sample_xy(self, n, r, h, lying):
        c = self.cfg
        dev = self.device
        out = torch.zeros(n, 2, device=dev)
        todo = torch.ones(n, dtype=torch.bool, device=dev)
        ext = torch.where(lying, h / 2, r)
        for _ in range(20):
            d = sample_uniform(*c.reach_d, (n,), dev)
            a = sample_uniform(*c.reach_ang, (n,), dev)
            x = c.shoulder_xy[0] + d * torch.cos(a)
            y = c.shoulder_xy[1] + d * torch.sin(a)
            bad = (x.abs() < c.footprint[0] + ext) & (y.abs() < c.footprint[1] + ext)
            take = todo & ~bad
            out[take] = torch.stack([x, y], -1)[take]
            todo &= ~take
            if not todo.any():
                break
        out[todo] = torch.tensor([0.0, 0.4], device=dev)
        return out

    def _log_episodes(self, env_ids):
        succ = self.success[env_ids].float()
        t = self.success_step[env_ids].float().clamp_min(0) * self.ctrl_dt
        log = {
            "Episode/success": succ.mean(),
            "Episode/sealed": self.ever_sealed[env_ids].float().mean(),
            "Episode/lifted": (self.lift_max[env_ids] >= self.cfg.lift_goal).float().mean(),
            "Episode/drops": self.drops[env_ids].mean(),
            "Episode/max_force": self.max_force[env_ids].mean(),
            "Episode/time_s": (t * succ).sum() / succ.sum().clamp_min(1.0),
        }
        self.extras.setdefault("log", {}).update({k: v.item() for k, v in log.items()})
        self.last_episode_stats = {
            "env_ids": env_ids.clone(), "success": self.success[env_ids].clone(), "time_s": t,
            "drops": self.drops[env_ids].clone(), "max_force": self.max_force[env_ids].clone(),
            "max_deck_force": self.max_deck_force[env_ids].clone(), "lift_max": self.lift_max[env_ids].clone(),
            "sealed": self.ever_sealed[env_ids].clone(), "seals": self.seals[env_ids].clone(),
            "lying": self.lying[env_ids].clone(), "kind": self.can_kind[env_ids].clone(),
            "mass": self.mass[env_ids].clone(), "friction": self.friction[env_ids].clone(),
        }


gym.register(
    id="Rig-PickCan-Go2UR3Lift-v0",
    entry_point="policy.envs.pickcan_env:PickCanEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "policy.envs.pickcan_env_cfg:PickCanEnvCfg",
            "rsl_rl_cfg_entry_point": "policy.envs.pickcan_env_cfg:PickCanPPORunnerCfg"},
)

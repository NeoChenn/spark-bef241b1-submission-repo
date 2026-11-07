from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, sample_uniform

from policy.envs.rust_state import RustState
from policy.envs.scrub_env_cfg import COARSE_H, COARSE_W, GRID_H, GRID_W, PATCH_K, ScrubEnvCfg


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[:, 0])
    return torch.stack([
        torch.stack([z, -v[:, 2], v[:, 1]], -1),
        torch.stack([v[:, 2], z, -v[:, 0]], -1),
        torch.stack([-v[:, 1], v[:, 0], z], -1),
    ], 1)


def _mid(r):
    return 0.5 * (r[0] + r[1])


class ScrubEnv(DirectRLEnv):
    cfg: ScrubEnvCfg

    def __init__(self, cfg: ScrubEnvCfg, render_mode: str | None = None, **kwargs):
        yaw = cfg.robot_yaw
        cfg.robot_cfg = cfg.robot_cfg.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=cfg.robot_cfg.init_state.replace(
                pos=(0.0, 0.0, cfg.base_height), rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            ),
        )
        cfg.observation_space = 2 * cfg.num_arm_joints + 16 + 2 * PATCH_K * PATCH_K + COARSE_H * COARSE_W + 9
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        self.ctrl_dt = cfg.sim.dt * cfg.decimation
        self.arm_ids, arm_names = self._robot.find_joints(cfg.arm_joint_names, preserve_order=True)
        assert len(self.arm_ids) == cfg.num_arm_joints, f"arm joints {arm_names} != num_arm_joints"
        self.ee_idx = self._robot.find_bodies(cfg.ee_body_name)[0][0]
        self.jac_idx = self.ee_idx - 1 if self._robot.is_fixed_base else self.ee_idx
        self.jac_cols = list(self.arm_ids) if self._robot.is_fixed_base else [i + 6 for i in self.arm_ids]
        self.brush_offset = torch.tensor(cfg.brush_offset, device=dev).repeat(N, 1)
        self.lift_ids = self._robot.find_joints(cfg.lift_joint_name)[0] if cfg.lift_joint_name else []
        lim = self._robot.data.soft_joint_pos_limits[0, self.arm_ids]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]

        self.rust = RustState(N, GRID_H, GRID_W, cfg.pipe_length, device=dev, theta_window=cfg.rust_theta_window)
        z3 = lambda: torch.zeros(N, 3, device=dev)
        z1 = lambda: torch.zeros(N, device=dev)
        self.pipe_o, self.pipe_a, self.pipe_e = z3(), z3(), z3()
        self.pipe_b = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(N, 1)
        self.pipe_r, self.spring_k = z1(), z1()
        self.actions = torch.zeros(N, 4, device=dev)
        self.prev_actions = torch.zeros(N, 4, device=dev)
        self.p_tgt, self.p_cmd = z3(), z3()
        self.f_set = torch.full((N,), cfg.force_set_mid, device=dev)
        self.brush_w, self.brush_v = z3(), z3()
        self.force, self.gap, self.theta, self.axial = z1(), z1(), z1(), z1()
        self.removed, self.max_force = z1(), z1()
        self.success = torch.zeros(N, dtype=torch.bool, device=dev)
        self.event_step = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.event_fired = torch.zeros(N, dtype=torch.bool, device=dev)
        self.clean_step = torch.full((N,), -1, dtype=torch.long, device=dev)
        self._started = False
        self._markers = None
        if cfg.viz:
            self._make_markers()

    def _setup_scene(self):
        c = self.cfg
        self._robot = Articulation(c.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        if c.spawn_pipe_visual or c.spawn_dog_visual:
            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/ground", ground)
        if c.spawn_pipe_visual:
            r = _mid(c.radius_range)
            pipe = sim_utils.CylinderCfg(
                radius=r, height=2.0, axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.56, 0.6), metallic=0.7, roughness=0.4),
            )
            pipe.func("/World/envs/env_0/Pipe", pipe,
                      translation=(_mid(c.axial_range),
                                   -(_mid(c.axis_dist_range) if c.axis_dist_range else _mid(c.standoff_range) + r),
                                   _mid(c.height_range)))
        if c.spawn_dog_visual:
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.1))
            body = sim_utils.CuboidCfg(size=(0.62, 0.28, 0.16), visual_material=mat)
            body.func("/World/envs/env_0/DogBody", body, translation=(0.0, 0.0, c.base_height - 0.09))
            leg = sim_utils.CuboidCfg(size=(0.05, 0.05, c.base_height - 0.17), visual_material=mat)
            for i, (lx, ly) in enumerate([(0.24, 0.1), (0.24, -0.1), (-0.24, 0.1), (-0.24, -0.1)]):
                leg.func(f"/World/envs/env_0/DogLeg{i}", leg, translation=(lx, ly, (c.base_height - 0.17) / 2))
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def _make_markers(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        mk = VisualizationMarkersCfg(
            prim_path="/Visuals/Scrub",
            markers={
                "brush": sim_utils.SphereCfg(radius=self.cfg.brush_radius,
                                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.95))),
                "rust": sim_utils.CuboidCfg(size=(0.012, 0.012, 0.012),
                                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.22, 0.05))),
                "wet": sim_utils.CuboidCfg(size=(0.006, 0.006, 0.006),
                                           visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.9))),
            },
        )
        self._markers = VisualizationMarkers(mk)

    def _to_pipe(self, p_local: torch.Tensor):
        d = p_local - self.pipe_o
        za = (d * self.pipe_a).sum(-1)
        ye = (d * self.pipe_e).sum(-1)
        xb = (d * self.pipe_b).sum(-1)
        return za, ye, xb

    def _from_cyl(self, za, rho, theta):
        n = torch.cos(theta)[:, None] * self.pipe_e + torch.sin(theta)[:, None] * self.pipe_b
        return self.pipe_o + za[:, None] * self.pipe_a + rho[:, None] * n

    def _vec_to_pipe(self, v):
        return torch.stack([(v * self.pipe_a).sum(-1), (v * self.pipe_e).sum(-1), (v * self.pipe_b).sum(-1)], -1)

    def _brush_state(self):
        d = self._robot.data
        hand_p = d.body_pos_w[:, self.ee_idx]
        R = matrix_from_quat(d.body_quat_w[:, self.ee_idx])
        r_off = torch.bmm(R, self.brush_offset.unsqueeze(-1)).squeeze(-1)
        brush_w = hand_p + r_off
        v = d.body_lin_vel_w[:, self.ee_idx] + torch.cross(d.body_ang_vel_w[:, self.ee_idx], r_off, dim=-1)
        return brush_w, v, r_off

    def _pre_physics_step(self, actions: torch.Tensor):
        c = self.cfg
        a = actions.clamp(-1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        za, ye, xb = self._to_pipe(self.p_tgt)
        za = za + a[:, 0] * c.action_scale
        ye = ye + a[:, 1] * c.action_scale
        xb = xb + a[:, 2] * c.action_scale
        rho = torch.sqrt(ye ** 2 + xb ** 2).clamp_min(1e-6)
        th = torch.atan2(xb, ye).clamp(c.theta_limits[0], c.theta_limits[1])
        surf = self.pipe_r + c.brush_radius
        rho = torch.minimum(torch.maximum(rho, surf - 0.03), surf + c.max_target_gap)
        half = c.pipe_length / 2 + 0.05
        za = za.clamp(-half, half)
        self.p_tgt[:] = self._from_cyl(za, rho, th)
        self.f_set[:] = c.force_set_mid + c.force_set_half * a[:, 3]
        rho_cmd = torch.where(rho < surf + c.contact_snap, surf - self.f_set / c.spring_k_nominal, rho)
        self.p_cmd[:] = self._from_cyl(za, rho_cmd, th)

    def _apply_action(self):
        d = self._robot.data
        brush_w, _, r_off = self._brush_state()
        jac = self._robot.root_physx_view.get_jacobians()[:, self.jac_idx][:, :, self.jac_cols]
        Jv = jac[:, :3] - torch.bmm(_skew(r_off), jac[:, 3:])
        err = (self.p_cmd + self.scene.env_origins) - brush_w
        n = err.norm(dim=-1, keepdim=True)
        err = err * torch.clamp(0.05 / n.clamp_min(1e-6), max=1.0)
        lam2 = 0.05 ** 2
        JJt = torch.bmm(Jv, Jv.transpose(1, 2)) + lam2 * torch.eye(3, device=self.device)
        dq = torch.bmm(Jv.transpose(1, 2), torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
        q = d.joint_pos[:, self.arm_ids] + dq
        q = torch.maximum(torch.minimum(q, self.q_hi), self.q_lo)
        self._robot.set_joint_position_target(q, joint_ids=self.arm_ids)

    def _update_contact(self):
        c = self.cfg
        brush_w, v_w, _ = self._brush_state()
        self.brush_w[:] = brush_w - self.scene.env_origins
        self.brush_v[:] = v_w
        za, ye, xb = self._to_pipe(self.brush_w)
        rho = torch.sqrt(ye ** 2 + xb ** 2)
        th = torch.atan2(xb, ye)
        self.theta[:], self.axial[:] = th, za
        pen = (self.pipe_r + c.brush_radius - rho).clamp_min(0.0)
        self.gap[:] = rho - self.pipe_r - c.brush_radius
        on_pipe = za.abs() < c.pipe_length / 2 + 0.5
        self.force[:] = self.spring_k * pen * on_pipe
        n = torch.cos(th)[:, None] * self.pipe_e + torch.sin(th)[:, None] * self.pipe_b
        v_t = v_w - (v_w * n).sum(-1, keepdim=True) * n
        slide = v_t.norm(dim=-1)
        crad = (0.012 + torch.sqrt(2 * c.brush_radius * pen)).clamp(max=c.brush_radius)
        uv = self.rust.surface_to_uv(th, za)
        self.removed[:] = self.rust.abrade(uv, crad, self.force, slide, self.ctrl_dt)
        self.rust.step_soak(self.ctrl_dt)
        self.max_force[:] = torch.maximum(self.max_force, self.force)

    def _get_dones(self):
        c = self.cfg
        self._update_contact()
        frac = self.rust.fraction_remaining()
        fire = (self.event_step >= 0) & ~self.event_fired & (
            (self.episode_length_buf >= self.event_step) | (frac < c.success_fraction))
        ids = fire.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self.rust.add_patch(ids)
            self.event_fired[ids] = True
            frac = self.rust.fraction_remaining()
        self.success[:] = frac < c.success_fraction
        newly = self.success & (self.clean_step < 0)
        self.clean_step[newly] = self.episode_length_buf[newly]
        if self._markers is not None:
            self._draw_markers()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.success.clone(), truncated

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        F = self.force
        removed_frac = self.removed / self.rust.initial_total
        in_band = ((F >= c.force_band[0]) & (F <= c.force_band[1])).float()
        over = (F - c.force_band[1]).clamp_min(0.0) / 10.0
        uv = self.rust.surface_to_uv(self.theta, self.axial)
        _, dist = self.rust.nearest_rust(uv)
        dist = dist + self.gap.abs()
        has_rust = (self.rust.rusty_cells() > 0).float()
        act_rate = ((self.actions - self.prev_actions) ** 2).sum(-1)
        rew = (c.w_removed * removed_frac
               + c.w_success * self.success.float()
               + c.w_contact_band * in_band
               - c.w_over_force * over
               - c.w_unsprayed * self.rust.last_unsprayed_contact
               - c.w_action_rate * act_rate
               - c.w_dist * dist * has_rust)
        return rew

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        c = self.cfg
        if self._started:
            self._log_episodes(env_ids)
        self._started = True
        super()._reset_idx(env_ids)
        n, dev = len(env_ids), self.device

        r = sample_uniform(*c.radius_range, (n,), dev)
        yaw = sample_uniform(*c.yaw_range, (n,), dev)
        standoff = sample_uniform(*c.standoff_range, (n,), dev)
        self.pipe_r[env_ids] = r
        axis_y = -(sample_uniform(*c.axis_dist_range, (n,), dev) if c.axis_dist_range else standoff + r)
        self.pipe_o[env_ids] = torch.stack([sample_uniform(*c.axial_range, (n,), dev), axis_y,
                                            sample_uniform(*c.height_range, (n,), dev)], -1)
        self.pipe_a[env_ids] = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
        self.pipe_e[env_ids] = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], -1)
        self.spring_k[env_ids] = sample_uniform(*c.spring_k_range, (n,), dev)

        jp = self._robot.data.default_joint_pos[env_ids].clone()
        lift = torch.zeros(n, device=dev)
        if self.lift_ids:
            lift = sample_uniform(*c.lift_range, (n,), dev)
            jp[:, self.lift_ids[0]] = lift
        reach = None
        if c.reach_range is not None:
            reach = self._reachable_mask(env_ids, lift)

        soak = torch.where(torch.rand(n, device=dev) < c.prob_not_soaked,
                           sample_uniform(0.0, 25.0, (n,), dev), torch.full((n,), 60.0, device=dev))
        self.rust.reset(env_ids, radius=r, soak_time=soak, reachable=reach,
                        abrasion_coef=sample_uniform(*c.abrasion_coef_range, (n,), dev))

        jp[:, self.arm_ids] += sample_uniform(-c.init_joint_noise, c.init_joint_noise, (n, len(self.arm_ids)), dev)
        jp[:, self.arm_ids] = torch.maximum(torch.minimum(jp[:, self.arm_ids], self.q_hi), self.q_lo)
        jv = torch.zeros_like(jp)
        self._robot.set_joint_position_target(jp, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        sub = env_ids
        th = c.rust_theta_window[0] + sample_uniform(-1.0, 1.0, (n,), dev) * c.rust_theta_window[1]
        za = sample_uniform(-0.15, 0.15, (n,), dev)
        rho = r + c.brush_radius + sample_uniform(0.03, 0.12, (n,), dev)
        nrm = torch.cos(th)[:, None] * self.pipe_e[sub] + torch.sin(th)[:, None] * self.pipe_b[sub]
        p = self.pipe_o[sub] + za[:, None] * self.pipe_a[sub] + rho[:, None] * nrm
        self.p_tgt[sub] = p
        self.p_cmd[sub] = p

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.f_set[env_ids] = c.force_set_mid
        self.force[env_ids] = 0.0
        self.removed[env_ids] = 0.0
        self.max_force[env_ids] = 0.0
        self.success[env_ids] = False
        self.clean_step[env_ids] = -1
        self.event_fired[env_ids] = False
        steps = (sample_uniform(*c.new_patch_time_range, (n,), dev) / self.ctrl_dt).long()
        self.event_step[env_ids] = torch.where(torch.rand(n, device=dev) < c.prob_new_patch, steps,
                                               torch.full_like(steps, -1))

    def _reachable_mask(self, env_ids, lift):
        c = self.cfg
        cy, sy = math.cos(c.robot_yaw), math.sin(c.robot_yaw)
        ox, oy, oz = c.shoulder_offset
        sh = torch.stack([torch.full_like(lift, cy * ox - sy * oy), torch.full_like(lift, sy * ox + cy * oy),
                          c.base_height + oz + lift], -1)
        th = self.rust.theta_c.view(1, 1, -1)
        z = self.rust.z_c.view(1, -1, 1)
        o, a = self.pipe_o[env_ids], self.pipe_a[env_ids]
        e, b = self.pipe_e[env_ids], self.pipe_b[env_ids]
        r = self.pipe_r[env_ids].view(-1, 1, 1, 1)
        nrm = torch.cos(th)[..., None] * e.view(-1, 1, 1, 3) + torch.sin(th)[..., None] * b.view(-1, 1, 1, 3)
        p = o.view(-1, 1, 1, 3) + z[..., None] * a.view(-1, 1, 1, 3) + r * nrm
        d = (p - sh.view(-1, 1, 1, 3)).norm(dim=-1)
        return (d >= c.reach_range[0]) & (d <= c.reach_range[1])

    def _log_episodes(self, env_ids):
        frac = self.rust.fraction_remaining()[env_ids]
        succ = self.success[env_ids].float()
        ev = self.event_fired[env_ids].float()
        t_clean = self.clean_step[env_ids].float().clamp_min(0) * self.ctrl_dt
        log = {
            "Episode/fraction_removed": (1.0 - frac).mean(),
            "Episode/success": succ.mean(),
            "Episode/max_force": self.max_force[env_ids].mean(),
            "Episode/time_to_clean_s": (t_clean * succ).sum() / succ.sum().clamp_min(1.0),
            "Episode/new_patch": ev.mean(),
            "Episode/new_patch_success": (ev * succ).mean(),
        }
        self.extras.setdefault("log", {}).update({k: v.item() for k, v in log.items()})
        self.last_episode_stats = {
            "env_ids": env_ids.clone(), "fraction_removed": 1.0 - frac, "success": self.success[env_ids].clone(),
            "max_force": self.max_force[env_ids].clone(), "time_to_clean_s": t_clean,
            "new_patch": self.event_fired[env_ids].clone(),
        }

    def _get_observations(self) -> dict:
        c = self.cfg
        d = self._robot.data
        q = d.joint_pos[:, self.arm_ids]
        qd = d.joint_vel[:, self.arm_ids]
        za, ye, xb = self._to_pipe(self.brush_w)
        brush_p = torch.stack([za, ye, xb], -1)
        brush_v = self._vec_to_pipe(self.brush_v)
        tz, ty, tx = self._to_pipe(self.p_tgt)
        tgt_rel = torch.stack([tz, ty, tx], -1) - brush_p
        uv = self.rust.surface_to_uv(self.theta, self.axial)
        near, dist = self.rust.nearest_rust(uv)
        obs = torch.cat([
            q, qd * 0.1,
            brush_p * 4.0, brush_v,
            self.gap[:, None] * 10.0,
            torch.sin(self.theta)[:, None], torch.cos(self.theta)[:, None], (self.axial / c.pipe_length)[:, None],
            tgt_rel * 10.0,
            (self.force / 40.0)[:, None], (self.f_set / 40.0)[:, None], (self.pipe_r * 10.0)[:, None],
            self.rust.local_patch(uv, PATCH_K).flatten(1),
            self.rust.local_sprayed(uv, PATCH_K).flatten(1),
            self.rust.coarse_map(COARSE_H, COARSE_W).flatten(1),
            near * 10.0, dist[:, None] * 10.0,
            self.rust.fraction_remaining()[:, None],
            (self.rust.last_unsprayed_contact)[:, None],
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _draw_markers(self):
        n_show = min(self.num_envs, 4)
        pts, idx = [self.brush_w[:n_show] + self.scene.env_origins[:n_show]], [torch.zeros(n_show, dtype=torch.long, device=self.device)]
        th_c = self.rust.theta_c
        z_c = self.rust.z_c
        for e in range(n_show):
            rusty = self.rust.rust[e] > 0.15
            wet = self.rust.sprayed[e] & ~rusty
            for mask, mid in ((rusty, 1), (wet, 2)):
                ij = mask.nonzero(as_tuple=False)
                if ij.numel() == 0:
                    continue
                th, za = th_c[ij[:, 1]], z_c[ij[:, 0]]
                nrm = torch.cos(th)[:, None] * self.pipe_e[e] + torch.sin(th)[:, None] * self.pipe_b[e]
                p = self.pipe_o[e] + za[:, None] * self.pipe_a[e] + (self.pipe_r[e] + 0.004) * nrm
                pts.append(p + self.scene.env_origins[e])
                idx.append(torch.full((p.shape[0],), mid, dtype=torch.long, device=self.device))
        self._markers.visualize(translations=torch.cat(pts), marker_indices=torch.cat(idx))


TASKS = {
    "lift": ("Rig-Scrub-Go2UR3Lift-v0", "ScrubLiftEnvCfg"),
    "go2ur3": ("Rig-Scrub-Go2UR3-v0", "ScrubEnvCfg"),
    "franka": ("Rig-Scrub-Franka-v0", "ScrubFrankaEnvCfg"),
}
for _tid, _cfg in TASKS.values():
    gym.register(
        id=_tid,
        entry_point="policy.envs.scrub_env:ScrubEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"policy.envs.scrub_env_cfg:{_cfg}",
                "rsl_rl_cfg_entry_point": "policy.envs.scrub_env_cfg:ScrubPPORunnerCfg"},
    )

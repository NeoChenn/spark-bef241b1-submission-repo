from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from policy.walk.rma import EXT_SCALE, DeckMotion, sample_payload
from policy.walk.walk_env_cfg import WalkEnvCfg, WalkPPORunnerCfg

TASK_RMA = "Rig-Walk-Go2UR3Lift-RMA-v0"


def _rma(env):
    if not hasattr(env, "_rma_gen"):
        n, dev = env.num_envs, env.device
        env._rma_gen = torch.Generator().manual_seed(int(env.cfg.seed or 0) + 7)
        env._rma_mass = torch.zeros(n, device=dev)
        env._rma_com = torch.zeros(n, 3, device=dev)
        env._rma_deck_b = torch.zeros(n, 3, device=dev)
        env._rma_deck = DeckMotion.sample(n, env._rma_gen, 0.0, 0.0, device=dev)
        env._rma_mtot = torch.full((n,), 15.0, device=dev)
    return env


def rma_payload(env, env_ids, mass, com):
    _rma(env)
    r = env.scene["robot"]
    b = r.find_bodies("base")[0][0]
    m, c = sample_payload(env.num_envs, env._rma_gen, mass, com)
    masses = r.root_physx_view.get_masses().clone()
    masses[:, b] = (r.data.default_mass[:, b].cpu() + m).clamp_min(1.0)
    ids = torch.arange(env.num_envs, device="cpu")
    r.root_physx_view.set_masses(masses, ids)
    coms = r.root_physx_view.get_coms().clone()
    coms[:, b, :3] += c
    r.root_physx_view.set_coms(coms, ids)
    env._rma_mass[:] = (masses[:, b] - r.data.default_mass[:, b].cpu()).to(env.device)
    env._rma_com[:] = c.to(env.device)
    env._rma_mtot[:] = masses.sum(1).to(env.device)


def rma_deck_reset(env, env_ids, max_tilt, max_heave):
    _rma(env)
    ids = env_ids.to(env.device)
    env._rma_deck.assign(ids, DeckMotion.sample(len(ids), env._rma_gen, max_tilt, max_heave))
    env._rma_deck.max_tilt, env._rma_deck.max_heave = max_tilt, max_heave


def rma_deck_step(env, env_ids):
    _rma(env)
    r = env.scene["robot"]
    t = env.episode_length_buf.float() * env.step_dt
    a = env._rma_deck.accel(t)
    f = (env._rma_mtot[:, None] * a)[:, None]
    b = r.find_bodies("base")[0]
    r.set_external_force_and_torque(f, torch.zeros_like(f), body_ids=b, is_global=True)
    env._rma_deck_b[:] = quat_apply_inverse(yaw_quat(r.data.root_quat_w), a)


def extrinsics(env):
    _rma(env)
    e = torch.cat([env._rma_mass[:, None], env._rma_com, env._rma_deck_b], 1)
    return e / EXT_SCALE.to(env.device)


@configclass
class WalkRMAEnvCfg(WalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.deck_tilt = None
        self.events.rma_payload = EventTerm(func=rma_payload, mode="startup",
                                            params={"mass": (-1.0, 5.0), "com": (0.05, 0.05, 0.02)})
        self.events.rma_deck_reset = EventTerm(func=rma_deck_reset, mode="reset",
                                               params={"max_tilt": 0.1, "max_heave": 1.5})
        self.events.rma_deck = EventTerm(func=rma_deck_step, mode="interval", is_global_time=True,
                                         interval_range_s=(self.decimation * self.sim.dt,) * 2)
        self.events.payload.params["p_deployed"] = 0.5
        self.observations.policy.extrinsics = ObsTerm(func=extrinsics)


@configclass
class WalkRMAPPORunnerCfg(WalkPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walk_rma"
        self.max_iterations = 300
        self.save_interval = 50


gym.register(id=TASK_RMA, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": "policy.walk.rma_env_cfg:WalkRMAEnvCfg",
                     "rsl_rl_cfg_entry_point": "policy.walk.rma_env_cfg:WalkRMAPPORunnerCfg"})

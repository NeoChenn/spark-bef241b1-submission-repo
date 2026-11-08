from __future__ import annotations

import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as tg
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from integration.kinematics import ARM_START, ARM_STOWED, STANDING
from policy.walk.urdf import walk_urdf

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF = os.environ.get("WALK_URDF") or os.path.join(_REPO, "sim/assets/go2_ur3/go2_ur3_lift.urdf")
LEGS = [f"{l}_{j}_joint" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
PAYLOAD = list(ARM_STOWED)
TASK = "Rig-Walk-Go2UR3Lift-v0"

RIG_TERRAIN = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20, horizontal_scale=0.1, vertical_scale=0.005,
    slope_threshold=0.75, use_cache=False,
    sub_terrains={
        "deck": tg.MeshPlaneTerrainCfg(proportion=0.2),
        "grating": tg.HfRandomUniformTerrainCfg(proportion=0.25, noise_range=(0.0, 0.03), noise_step=0.005,
                                                border_width=0.25),
        "thresholds": tg.MeshRailsTerrainCfg(proportion=0.15, rail_thickness_range=(0.05, 0.15),
                                             rail_height_range=(0.02, 0.10), platform_width=2.0),
        "plates": tg.MeshRandomGridTerrainCfg(proportion=0.15, grid_width=0.45, grid_height_range=(0.01, 0.06),
                                              platform_width=2.0),
        "steps_up": tg.MeshPyramidStairsTerrainCfg(proportion=0.125, step_height_range=(0.03, 0.12), step_width=0.35,
                                                   platform_width=3.0, border_width=1.0),
        "steps_down": tg.MeshInvertedPyramidStairsTerrainCfg(proportion=0.125, step_height_range=(0.03, 0.12),
                                                             step_width=0.35, platform_width=3.0, border_width=1.0),
    },
)


def robot_cfg(urdf: str = "") -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf or walk_urdf(URDF), fix_base=False, merge_fixed_joints=True, make_instanceable=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.36), joint_pos={**STANDING, **ARM_STOWED},
                                                   joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "hip_thigh": DCMotorCfg(joint_names_expr=[".*_hip_joint", ".*_thigh_joint"], effort_limit=23.7,
                                    saturation_effort=23.7, velocity_limit=30.0, stiffness=50.0, damping=1.2,
                                    friction=0.0),
            "calf": DCMotorCfg(joint_names_expr=[".*_calf_joint"], effort_limit=45.4, saturation_effort=45.4,
                               velocity_limit=15.7, stiffness=50.0, damping=1.2, friction=0.0),
            "arm": ImplicitActuatorCfg(joint_names_expr=["ur3_.*"], stiffness=800.0, damping=40.0),
            "lift": ImplicitActuatorCfg(joint_names_expr=["lift_joint"], stiffness=2.0e4, damping=2.0e3),
        },
    )


def _payload_ids(env):
    if not hasattr(env, "_payload_ids"):
        robot = env.scene["robot"]
        env._payload_ids = robot.find_joints(PAYLOAD, preserve_order=True)[0]
        env._payload_lo = torch.tensor([ARM_STOWED[n] for n in PAYLOAD], device=env.device)
        env._payload_hi = torch.tensor([ARM_START[n] for n in PAYLOAD], device=env.device)
        env._payload_alpha = torch.zeros(env.num_envs, device=env.device)
        env._payload_goal = torch.zeros(env.num_envs, device=env.device)
    return env._payload_ids


def apply_payload(env, alpha: torch.Tensor):
    ids = _payload_ids(env)
    env._payload_alpha[:] = alpha
    q = env._payload_lo + alpha[:, None] * (env._payload_hi - env._payload_lo)
    env.scene["robot"].set_joint_position_target(q, joint_ids=ids)


def payload_reset(env, env_ids):
    _payload_ids(env)
    env._payload_goal[env_ids] = 0.0
    env._payload_alpha[env_ids] = 0.0
    apply_payload(env, env._payload_alpha)


def payload_step(env, env_ids, rate: float, hold_s: float, p_deployed: float):
    _payload_ids(env)
    dt = env.step_dt
    flip = torch.rand(env.num_envs, device=env.device) < dt / hold_s
    goal = torch.where(torch.rand(env.num_envs, device=env.device) < p_deployed,
                       torch.rand(env.num_envs, device=env.device).clamp(0.3, 1.0), torch.zeros(env.num_envs, device=env.device))
    env._payload_goal = torch.where(flip, goal, env._payload_goal)
    a = env._payload_alpha + (env._payload_goal - env._payload_alpha).clamp(-rate * dt, rate * dt)
    apply_payload(env, a)


def payload_obs(env):
    _payload_ids(env)
    return env._payload_alpha[:, None]


@configclass
class WalkEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = robot_cfg()
        self.scene.terrain.terrain_generator = RIG_TERRAIN
        legs = SceneEntityCfg("robot", joint_names=LEGS, preserve_order=True)
        self.actions.joint_pos.joint_names = LEGS
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params = {"asset_cfg": legs}
        self.observations.policy.joint_vel.params = {"asset_cfg": legs}
        self.observations.policy.payload = ObsTerm(func=payload_obs)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.5
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.base_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.01, 0.01)}})
        self.events.push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval",
                                           interval_range_s=(8.0, 12.0),
                                           params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}})
        self.events.deck_tilt = EventTerm(func=mdp.randomize_physics_scene_gravity, mode="interval",
                                          is_global_time=True, interval_range_s=(4.0, 8.0),
                                          params={"gravity_distribution_params": ([-0.4, -0.4, 0.0], [0.4, 0.4, 0.0]),
                                                  "operation": "add"})
        self.events.payload_reset = EventTerm(func=payload_reset, mode="reset")
        self.events.payload = EventTerm(func=payload_step, mode="interval", is_global_time=True,
                                        interval_range_s=(self.decimation * self.sim.dt,) * 2,
                                        params={"rate": 0.3, "hold_s": 5.0, "p_deployed": 0.4})
        self.rewards.dof_torques_l2.params = {"asset_cfg": legs}
        self.rewards.dof_acc_l2.params = {"asset_cfg": legs}
        self.rewards.base_height = RewTerm(func=mdp.base_height_l2, weight=-20.0, params={
            "target_height": 0.32, "sensor_cfg": SceneEntityCfg("height_scanner")})
        self.rewards.stand_still = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.3, params={
            "command_name": "base_velocity", "asset_cfg": legs})
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.feet_air_time.weight = 0.1
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", "Head_.*", "lift_.*", "ur3_.*"]
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class WalkPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walk"
        self.max_iterations = 1500


gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": "policy.walk.walk_env_cfg:WalkEnvCfg",
                     "rsl_rl_cfg_entry_point": "policy.walk.walk_env_cfg:WalkPPORunnerCfg"})

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from policy.envs.scrub_env_cfg import GO2_UR3_INIT, UR3_JOINTS, _REPO, urdf_robot_cfg

GRIP_URDF = os.environ.get("PICKCAN_URDF") or next(
    (p for p in ("/assets/go2_ur3/go2_ur3_lift_grip.urdf", os.path.join(_REPO, "sim/assets/go2_ur3/go2_ur3_lift_grip.urdf"))
     if os.path.exists(p)), "/assets/go2_ur3/go2_ur3_lift_grip.urdf")

READY = {"lift_joint": 0.0, "ur3_shoulder_pan_joint": 0.554, "ur3_shoulder_lift_joint": -1.986, "ur3_elbow_joint": 2.113,
         "ur3_wrist_1_joint": -1.698, "ur3_wrist_2_joint": -1.571, "ur3_wrist_3_joint": 0.0}

CANS = (
    ("soda_red", 0.033, 0.122, (0.78, 0.05, 0.04)),
    ("soda_blue", 0.0265, 0.10, (0.06, 0.22, 0.82)),
    ("paint_yellow", 0.045, 0.15, (0.95, 0.72, 0.06)),
    ("paint_green", 0.05, 0.18, (0.08, 0.6, 0.18)),
    ("tall_red", 0.03, 0.20, (0.85, 0.08, 0.1)),
    ("tin_orange", 0.0375, 0.11, (0.95, 0.38, 0.04)),
    ("wide_blue", 0.05, 0.12, (0.1, 0.3, 0.9)),
    ("energy_yellow", 0.028, 0.16, (0.9, 0.82, 0.05)),
)


def can_spawn_cfg(nominal_mass: float = 0.3):
    assets = [sim_utils.CylinderCfg(
        radius=r, height=h, axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=0.55, roughness=0.35),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6,
                                                        friction_combine_mode="multiply"))
        for _, r, h, c in CANS]
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=assets, random_choice=False, activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                     max_depenetration_velocity=1.0, linear_damping=0.05,
                                                     angular_damping=0.1),
        mass_props=sim_utils.MassPropertiesCfg(mass=nominal_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    )


@configclass
class PickCanEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 12.0
    action_space = 5
    observation_space = 57
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=False)
    viewer: ViewerCfg = ViewerCfg(eye=(1.1, 1.4, 1.0), lookat=(0.0, 0.1, 0.3), resolution=(1280, 720))

    robot_cfg: object = None
    arm_joint_names: list = UR3_JOINTS
    tip_body: str = "grip_tip"
    tool_body: str = "gripper_link"
    base_height: float = 0.32
    init_joint_noise: float = 0.05

    nominal_mass: float = 0.3
    mass_range: tuple = (0.1, 0.5)
    friction_range: tuple = (0.35, 1.0)
    deck_friction: float = 1.0
    prob_lying: float = 0.65
    reach_d: tuple = (0.355, 0.445)
    reach_ang: tuple = (-2.6, 2.6)
    shoulder_xy: tuple = (-0.01, 0.0)
    footprint: tuple = (0.34, 0.24)

    det_sigma_xy: float = 0.012
    det_sigma_z: float = 0.006
    det_sigma_yaw: float = 0.1
    det_jitter: float = 0.003
    det_close: float = 0.3

    action_scale: float = 0.012
    lift_scale: float = 0.005
    lift_max: float = 0.3
    max_target_gap: float = 0.06
    work_lo: tuple = (-0.5, -0.6, 0.02)
    work_hi: tuple = (0.6, 0.6, 0.85)
    rot_weight: float = 0.6
    max_dq: float = 0.1
    ik_step: float = 0.005

    seal_gap: float = 0.012
    seal_margin: float = 0.01
    seal_align: float = 0.9
    suction_force_cap: float = 25.0
    suction_torque_cap: float = 1.0
    seal_tear: float = 0.03

    lift_goal: float = 0.15
    carry_z: float = 0.72
    settle_speed: float = 0.15
    force_limit: float = 20.0

    w_reach: float = 1.0
    w_hold: float = 1.5
    w_lift: float = 1.0
    w_carry: float = 2.0
    w_place: float = 6.0
    w_progress: float = 10.0
    w_success: float = 50.0
    w_drop: float = 5.0
    w_action_rate: float = 0.01
    w_force: float = 0.05
    w_time: float = 0.01
    w_idle_suction: float = 0.02

    def __post_init__(self):
        init = dict(GO2_UR3_INIT)
        init.update(READY)
        self.robot_cfg = urdf_robot_cfg(GRIP_URDF, init_joint_pos=init)
        self.robot_cfg.spawn = self.robot_cfg.spawn.replace(activate_contact_sensors=True)
        self.robot_cfg.actuators["lift"] = ImplicitActuatorCfg(joint_names_expr=["lift_joint"], stiffness=2.0e4,
                                                               damping=2.0e3)


@configclass
class PickCanPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "pickcan"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

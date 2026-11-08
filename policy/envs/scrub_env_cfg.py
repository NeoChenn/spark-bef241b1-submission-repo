from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

GRID_H, GRID_W = 32, 64
PATCH_K = 9
COARSE_H, COARSE_W = 8, 16

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GO2_UR3_URDF = os.environ.get("SCRUB_URDF") or next(
    (p for p in ("/assets/go2_ur3/go2_ur3.urdf", os.path.join(_REPO, "sim/assets/go2_ur3/go2_ur3.urdf"))
     if os.path.exists(p)), "/assets/go2_ur3/go2_ur3.urdf")

UR3_JOINTS = ["ur3_shoulder_pan_joint", "ur3_shoulder_lift_joint", "ur3_elbow_joint",
              "ur3_wrist_1_joint", "ur3_wrist_2_joint", "ur3_wrist_3_joint"]
GO2_UR3_INIT = {
    ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8, "R[LR]_thigh_joint": 1.0, ".*_calf_joint": -1.5,
    "ur3_shoulder_pan_joint": 0.2, "ur3_shoulder_lift_joint": -1.33, "ur3_elbow_joint": -1.287,
    "ur3_wrist_1_joint": 0.503, "ur3_wrist_2_joint": -1.322, "ur3_wrist_3_joint": -0.315,
}


def urdf_robot_cfg(asset_path: str = GO2_UR3_URDF, init_joint_pos: dict | None = None,
                   arm_joint_expr: str = "ur3_.*", arm_stiffness: float = 800.0, arm_damping: float = 40.0,
                   leg_stiffness: float = 300.0, leg_damping: float = 10.0) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path,
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=arm_stiffness, damping=arm_damping)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=init_joint_pos or GO2_UR3_INIT),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=[arm_joint_expr], stiffness=arm_stiffness, damping=arm_damping),
            "legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                                        stiffness=leg_stiffness, damping=leg_damping),
        },
    )


@configclass
class ScrubEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 15.0
    action_space = 4
    observation_space = 0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(1.2, 0.9, 1.1), lookat=(0.0, -0.3, 0.55), resolution=(1280, 720))

    robot_cfg: ArticulationCfg = urdf_robot_cfg()
    arm_joint_names: list = UR3_JOINTS
    num_arm_joints: int = 6
    ee_body_name: str = "ur3_ee_link"
    brush_offset: tuple = (0.11, 0.0, 0.0)
    brush_radius: float = 0.025
    base_height: float = 0.32
    init_joint_noise: float = 0.1

    robot_yaw: float = 3.14159265
    pipe_length: float = 0.6
    radius_range: tuple = (0.10, 0.20)
    height_range: tuple = (0.50, 0.75)
    standoff_range: tuple = (0.15, 0.28)
    axial_range: tuple = (-0.1, 0.1)
    yaw_range: tuple = (-0.1, 0.1)
    rust_theta_window: tuple = (0.7, 0.7)
    max_reach_z: float = 0.95
    axis_dist_range: tuple | None = None
    lift_joint_name: str | None = None
    lift_range: tuple = (0.0, 0.0)
    reach_range: tuple | None = None
    shoulder_offset: tuple = (-0.01, 0.0, 0.3159)
    spawn_pipe_visual: bool = True
    spawn_dog_visual: bool = False
    backdrop_usd: str = ""
    backdrop_translation: tuple = (0.0, 0.0, 0.0)

    spring_k_range: tuple = (1500.0, 4000.0)
    spring_k_nominal: float = 2500.0
    abrasion_coef_range: tuple = (0.7, 1.3)
    n_patches_range: tuple = (1, 3)
    prob_not_soaked: float = 0.15
    contact_snap: float = 0.01

    action_scale: float = 0.01
    force_set_mid: float = 25.0
    force_set_half: float = 20.0
    force_set_max: float | None = None
    max_target_gap: float = 0.20
    theta_limits: tuple = (-0.5, 2.0)

    prob_new_patch: float = 0.3
    new_patch_time_range: tuple = (3.0, 10.0)

    w_removed: float = 50.0
    w_success: float = 20.0
    success_fraction: float = 0.05
    w_contact_band: float = 0.02
    force_band: tuple = (10.0, 40.0)
    w_over_force: float = 0.1
    over_force_soft: float | None = None
    w_over_force_quad: float = 0.0
    w_unsprayed: float = 0.05
    w_action_rate: float = 0.01
    w_dist: float = 0.2

    viz: bool = False
    storm_level_range: tuple = (0.0, 0.0)
    storm_gust_peak: tuple = (10.0, 30.0)
    storm_gust_mean: tuple = (0.0, 4.0)
    storm_deck_amp: tuple = (0.02, 0.05)
    storm_deck_freq: tuple = (0.1, 0.3)
    storm_trunk_sway: float = 0.0006
    storm_gust_bodies: tuple = ("ur3_forearm_link", "ur3_wrist_1_link", "ur3_wrist_2_link", "ur3_wrist_3_link",
                                "scrubber_link")

    def __post_init__(self):
        j = self.num_arm_joints
        self.observation_space = 2 * j + 16 + 2 * PATCH_K * PATCH_K + COARSE_H * COARSE_W + 9


GO2_UR3_LIFT_URDF = os.environ.get("SCRUB_LIFT_URDF") or next(
    (p for p in ("/assets/go2_ur3/go2_ur3_lift.urdf", os.path.join(_REPO, "sim/assets/go2_ur3/go2_ur3_lift.urdf"))
     if os.path.exists(p)), "/assets/go2_ur3/go2_ur3_lift.urdf")


@configclass
class ScrubLiftEnvCfg(ScrubEnvCfg):

    def __post_init__(self):
        init = dict(GO2_UR3_INIT)
        init.update({"lift_joint": 0.42, "ur3_shoulder_pan_joint": -2.035, "ur3_shoulder_lift_joint": -0.712,
                     "ur3_elbow_joint": -0.963, "ur3_wrist_1_joint": -0.285, "ur3_wrist_2_joint": -1.288,
                     "ur3_wrist_3_joint": 0.044})
        self.robot_cfg = urdf_robot_cfg(GO2_UR3_LIFT_URDF, init_joint_pos=init)
        self.robot_cfg.actuators["lift"] = ImplicitActuatorCfg(joint_names_expr=["lift_joint"],
                                                               stiffness=2.0e4, damping=2.0e3)
        self.ee_body_name = "brush_tip"
        self.brush_offset = (-0.025, 0.0, 0.0)
        self.robot_yaw = 0.0
        self.lift_joint_name = "lift_joint"
        self.lift_range = (0.38, 0.45)
        self.height_range = (1.05, 1.15)
        self.radius_range = (0.12, 0.18)
        self.axis_dist_range = (0.45, 0.55)
        self.rust_theta_window = (0.9006, 0.35)
        self.theta_limits = (0.0, 2.0)
        self.reach_range = (0.12, 0.62)
        self.max_reach_z = 1.3
        self.viewer = ViewerCfg(eye=(1.3, 1.6, 1.7), lookat=(0.0, -0.3, 1.0), resolution=(1280, 720))
        super().__post_init__()


@configclass
class ScrubLiftV2EnvCfg(ScrubLiftEnvCfg):

    def __post_init__(self):
        self.force_set_max = 38.0
        self.w_over_force = 1.0
        self.over_force_soft = 30.0
        self.w_over_force_quad = 4.0
        self.abrasion_coef_range = (0.5, 1.5)
        self.n_patches_range = (1, 4)
        self.yaw_range = (-0.1745, 0.1745)
        super().__post_init__()


@configclass
class ScrubLiftV3EnvCfg(ScrubLiftV2EnvCfg):
    def __post_init__(self):
        self.storm_level_range = (0.0, 1.0)
        super().__post_init__()


@configclass
class ScrubFrankaEnvCfg(ScrubEnvCfg):

    def __post_init__(self):
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        self.robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.arm_joint_names = ["panda_joint[1-7]"]
        self.num_arm_joints = 7
        self.ee_body_name = "panda_hand"
        self.brush_offset = (0.0, 0.0, 0.13)
        self.base_height = 0.55
        self.robot_yaw = -1.5707963
        self.height_range = (1.0, 1.2)
        self.standoff_range = (0.22, 0.38)
        self.rust_theta_window = (0.9, 0.75)
        self.theta_limits = (-0.5, 2.3)
        self.max_reach_z = 1.3
        self.spawn_dog_visual = True
        self.viewer = ViewerCfg(eye=(1.4, 0.9, 1.6), lookat=(0.0, -0.35, 1.0), resolution=(1280, 720))
        super().__post_init__()


@configclass
class ScrubPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "scrub"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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

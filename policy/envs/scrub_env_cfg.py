"""Config for the SCRUB skill env (Isaac Lab 2.3 DirectRLEnv) + rsl_rl PPO runner cfg.

Import only after the Isaac Sim app is launched (AppLauncher).

Robots:
  * ScrubEnvCfg (default, task Rig-Scrub-Go2UR3-v0): the team's Unitree Go2 + UR3 URDF
    (sim/assets/go2_ur3/go2_ur3.urdf, on Modal rig-assets:/go2_ur3/go2_ur3.urdf), fixed base at standing trunk
    height, legs PD-held at the standing pose, only the 6 UR3 joints are driven (IK on ur3_ee_link + tool offset).
  * ScrubFrankaEnvCfg (task Rig-Scrub-Franka-v0): Franka Panda on a fixed base, kept as a fallback.
  * any other URDF: cfg.robot_cfg = urdf_robot_cfg(path, init_joint_pos=...); set arm_joint_names, num_arm_joints,
    ee_body_name (REQUIRED link names), brush_offset (tool centre in the EE link frame), base_height (root z).
"""
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
# Go2 standing pose (sim/assets/go2_ur3/README.md) + a UR3 pose with the brush ~8 cm off the nominal pipe
# (solved with the URDF FK: brush at trunk-frame (0, 0.165, 0.444)).
GO2_UR3_INIT = {
    ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8, "R[LR]_thigh_joint": 1.0, ".*_calf_joint": -1.5,
    "ur3_shoulder_pan_joint": 0.2, "ur3_shoulder_lift_joint": -1.33, "ur3_elbow_joint": -1.287,
    "ur3_wrist_1_joint": 0.503, "ur3_wrist_2_joint": -1.322, "ur3_wrist_3_joint": -0.315,
}


def urdf_robot_cfg(asset_path: str = GO2_UR3_URDF, init_joint_pos: dict | None = None,
                   arm_joint_expr: str = "ur3_.*", arm_stiffness: float = 800.0, arm_damping: float = 40.0,
                   leg_stiffness: float = 300.0, leg_damping: float = 10.0) -> ArticulationCfg:
    """URDF (dog + arm) as a fixed-base, gravity-free articulation. Legs stiff-PD at their init pose; the env only
    commands the arm joints. merge_fixed_joints=False keeps the EE link (ur3_ee_link) as a body."""
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
    # ---- timing: 120 Hz physics, 30 Hz control, 15 s episodes
    decimation = 4
    episode_length_s = 15.0
    action_space = 4
    observation_space = 0  # set in __post_init__
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(1.2, 0.9, 1.1), lookat=(0.0, -0.3, 0.55), resolution=(1280, 720))

    # ---- robot: Go2 + UR3 (fixed base at standing trunk height; arm links sit ~0.19 m to the trunk's left)
    robot_cfg: ArticulationCfg = urdf_robot_cfg()
    arm_joint_names: list = UR3_JOINTS
    num_arm_joints: int = 6
    ee_body_name: str = "ur3_ee_link"
    brush_offset: tuple = (0.11, 0.0, 0.0)   # brush centre in EE-link frame (m): ur3_ee_link x = approach axis
    brush_radius: float = 0.025              # sphere brush (m)
    base_height: float = 0.32                # robot root (Go2 trunk) z; arm mount = trunk + (-0.01, 0, 0.064)
    init_joint_noise: float = 0.1

    # ---- pipe. Env-local frame: robot root at (0, 0, base_height), yawed pi so the trunk runs along -x (parallel to
    #      the pipe) and its LEFT side (where the arm sits) faces -y; the pipe runs along x (+ small yaw) at
    #      y = -(standoff + r). Pipe frame (same convention as policy/interfaces.py): axis ~ +x, ref = horizontal
    #      normal pointing at the robot (+y), binormal = axis x ref = +z (up); theta = 0 faces the robot,
    #      theta = pi/2 is the top. Obs/actions are all expressed in this pipe frame.
    #      Height: the UR3 on a Go2 tops out at ~1.1 m EE height, so the rig's z=1.1 pipe is out of reach; default
    #      pipe centre 0.60 m (DR 0.50-0.75). rig_deck_meta.json loader overrides z when out of reach.
    robot_yaw: float = 3.14159265            # rad
    pipe_length: float = 0.6                 # rust-grid segment length (m), centred on the work spot
    radius_range: tuple = (0.10, 0.20)       # nominal 0.15 +- 0.05
    height_range: tuple = (0.50, 0.75)       # pipe axis height above deck
    standoff_range: tuple = (0.15, 0.28)     # horizontal trunk-centreline -> pipe surface; axis y = -(standoff + r)
    axial_range: tuple = (-0.1, 0.1)         # segment centre x
    yaw_range: tuple = (-0.1, 0.1)           # small pipe yaw (rad)
    rust_theta_window: tuple = (0.7, 0.7)    # (centre, half-width) rad: robot-facing upper/side quadrant [0, 1.4]
    max_reach_z: float = 0.95                # pipe-axis z above which the meta loader lowers the pipe
    axis_dist_range: tuple | None = None     # if set: horizontal trunk -> pipe-AXIS distance (overrides standoff)
    # ---- optional prismatic lift (set at reset, held by PD, NOT in the action space; orchestrator sets it)
    lift_joint_name: str | None = None
    lift_range: tuple = (0.0, 0.0)
    # ---- reachability: rust only spawns on cells within [min, max] m of the arm shoulder
    reach_range: tuple | None = None
    shoulder_offset: tuple = (-0.01, 0.0, 0.3159)  # ur3_shoulder_link in the trunk frame at lift = 0
    spawn_pipe_visual: bool = True           # visual-only cylinder (contact is analytic) at the nominal pose
    spawn_dog_visual: bool = False           # box stand-in for the dog body (only useful for the Franka variant)

    # ---- contact / abrasion DR
    spring_k_range: tuple = (1500.0, 4000.0)  # N/m brush stiffness
    spring_k_nominal: float = 2500.0          # used by the force-setpoint controller
    abrasion_coef_range: tuple = (0.7, 1.3)   # friction / abrasion proxy (+-30 %)
    prob_not_soaked: float = 0.15             # episodes where the remover has not soaked 30 s yet
    contact_snap: float = 0.01                # target within this of the surface -> force-controlled contact

    # ---- actions
    action_scale: float = 0.01                # m per control step (0.3 m/s max)
    force_set_mid: float = 25.0               # F* = mid + half * a3   ->  [5, 45] N
    force_set_half: float = 20.0
    max_target_gap: float = 0.20
    theta_limits: tuple = (-0.5, 2.0)

    # ---- adaptation event
    prob_new_patch: float = 0.3
    new_patch_time_range: tuple = (3.0, 10.0)  # s; also fires early if the first patches are already clean

    # ---- reward weights
    w_removed: float = 50.0      # x fraction of total rust removed this step
    w_success: float = 20.0
    success_fraction: float = 0.05
    w_contact_band: float = 0.02
    force_band: tuple = (10.0, 40.0)
    w_over_force: float = 0.1    # per 10 N above 40
    w_unsprayed: float = 0.05
    w_action_rate: float = 0.01
    w_dist: float = 0.2          # per metre to nearest rust

    viz: bool = False            # brush + rust markers (play/video only)

    def __post_init__(self):
        j = self.num_arm_joints
        # see ScrubEnv._get_observations: 2*J joint terms + 16 brush/contact + 2*k^2 local patches
        # + coarse map + 9 (nearest rust 3, fraction, unsprayed, last action 4)
        self.observation_space = 2 * j + 16 + 2 * PATCH_K * PATCH_K + COARSE_H * COARSE_W + 9


GO2_UR3_LIFT_URDF = os.environ.get("SCRUB_LIFT_URDF") or next(
    (p for p in ("/assets/go2_ur3/go2_ur3_lift.urdf", os.path.join(_REPO, "sim/assets/go2_ur3/go2_ur3_lift.urdf"))
     if os.path.exists(p)), "/assets/go2_ur3/go2_ur3_lift.urdf")


@configclass
class ScrubLiftEnvCfg(ScrubEnvCfg):
    """MAIN config: Go2 + prismatic lift (0-0.45 m) + UR3 + scrubber, pipe at the real rig height
    (sim/output/rig_deck_meta.json: axis along x, z = 1.1 m, r = 0.15 m). Dog faces +x parallel to the pipe, trunk
    0.45-0.55 m from the pipe axis on the damage side (+y); lift set to 0.38-0.45 m at reset; IK/contact frame is
    the URDF's brush_tip link (0.12 m along ur3_ee_link +x); rust only on cells within reach of the shoulder."""

    def __post_init__(self):
        init = dict(GO2_UR3_INIT)
        init.update({"lift_joint": 0.42, "ur3_shoulder_pan_joint": -2.035, "ur3_shoulder_lift_joint": -0.712,
                     "ur3_elbow_joint": -0.963, "ur3_wrist_1_joint": -0.285, "ur3_wrist_2_joint": -1.288,
                     "ur3_wrist_3_joint": 0.044})  # FK-solved: brush_tip 8 cm off the nominal damage spot
        self.robot_cfg = urdf_robot_cfg(GO2_UR3_LIFT_URDF, init_joint_pos=init)
        self.robot_cfg.actuators["lift"] = ImplicitActuatorCfg(joint_names_expr=["lift_joint"],
                                                               stiffness=2.0e4, damping=2.0e3)
        self.ee_body_name = "brush_tip"
        self.brush_offset = (-0.025, 0.0, 0.0)   # sphere centre 2.5 cm behind the tip -> tip is the contact face
        self.robot_yaw = 0.0
        self.lift_joint_name = "lift_joint"
        self.lift_range = (0.38, 0.45)
        self.height_range = (1.05, 1.15)
        self.radius_range = (0.12, 0.18)
        self.axis_dist_range = (0.45, 0.55)
        self.rust_theta_window = (0.9006, 0.35)  # atan2(0.784, 0.621): damage normal, from +y toward +z
        self.theta_limits = (0.0, 2.0)
        self.reach_range = (0.12, 0.62)
        self.max_reach_z = 1.3
        self.viewer = ViewerCfg(eye=(1.3, 1.6, 1.7), lookat=(0.0, -0.3, 1.0), resolution=(1280, 720))
        super().__post_init__()


@configclass
class ScrubFrankaEnvCfg(ScrubEnvCfg):
    """Fallback: Franka Panda on a fixed pedestal at dog-back height (0.55 m) beside a z=1.1 m pipe."""

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

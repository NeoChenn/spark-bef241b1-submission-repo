"""Config for the SCRUB skill env (Isaac Lab 2.3 DirectRLEnv) + rsl_rl PPO runner cfg.

Import only after the Isaac Sim app is launched (AppLauncher).

Robot asset is configurable:
  * default: Franka Panda (isaaclab_assets FRANKA_PANDA_HIGH_PD_CFG) on a fixed base at dog-back height.
  * teammate URDF:  cfg.robot_cfg = urdf_robot_cfg("/assets/robot_dog.urdf")
                    cfg.arm_joint_names = ["arm_joint_.*"]   # regex(es) of the arm joints used by IK
                    cfg.ee_body_name = "arm_ee_link"          # link the brush is rigidly attached to
                    cfg.brush_offset = (0.0, 0.0, 0.05)       # brush centre in that link's frame
                    cfg.num_arm_joints = 6
                    cfg.base_height = 0.0                     # URDF root already includes the dog body height
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

GRID_H, GRID_W = 32, 64
PATCH_K = 9
COARSE_H, COARSE_W = 8, 16


def urdf_robot_cfg(asset_path: str = "/assets/robot_dog.urdf", stiffness: float = 400.0, damping: float = 80.0,
                   init_joint_pos: dict | None = None) -> ArticulationCfg:
    """Teammate URDF (dog + arm) as a fixed-base articulation. All joints are PD-held; the env only commands the
    arm joints (cfg.arm_joint_names) via IK, legs are held at their init pose (dog standing during scrub)."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path,
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=init_joint_pos or {".*": 0.0}),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
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
    viewer: ViewerCfg = ViewerCfg(eye=(1.4, 0.9, 1.6), lookat=(0.0, -0.35, 1.0), resolution=(1280, 720))

    # ---- robot (switch to urdf_robot_cfg(...) for the teammate dog)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    arm_joint_names: list = ["panda_joint[1-7]"]
    num_arm_joints: int = 7
    ee_body_name: str = "panda_hand"
    brush_offset: tuple = (0.0, 0.0, 0.13)   # brush centre in EE-link frame (m)
    brush_radius: float = 0.025              # sphere brush (m)
    base_height: float = 0.55                # arm base on the dog's back
    init_joint_noise: float = 0.1

    # ---- pipe, matching sim/output/rig_deck_meta.json (Z up, deck z=0; main pipe horizontal along x at z=1.1,
    #      r=0.15). Env-local frame: arm base at (0, 0, base_height) yawed -90 deg so it faces -y; the pipe runs
    #      along x (+ small yaw) at y = -(standoff + r). Pipe frame (same convention as policy/interfaces.py):
    #      axis ~ +x, ref = horizontal normal pointing at the robot (+y), binormal = axis x ref = +z (up);
    #      theta = 0 faces the robot, theta = pi/2 is the top. Obs/actions are all expressed in this pipe frame.
    robot_yaw: float = -1.5707963            # rad, arm base yaw in env frame (faces -y = the pipe)
    pipe_length: float = 0.6                 # rust-grid segment length (m), centred on the work spot
    radius_range: tuple = (0.10, 0.20)       # nominal 0.15 +- 0.05
    height_range: tuple = (1.0, 1.2)         # pipe axis height above deck (nominal 1.1)
    standoff_range: tuple = (0.22, 0.38)     # horizontal base->surface distance; axis y = -(standoff + r)
    axial_range: tuple = (-0.1, 0.1)         # segment centre x
    yaw_range: tuple = (-0.1, 0.1)           # small pipe yaw (rad)
    rust_theta_window: tuple = (0.9, 0.75)   # (centre, half-width) rad; damage normal [0,.62,.78] -> 0.90 rad
    spawn_pipe_visual: bool = True           # visual-only cylinder (contact is analytic) at the nominal pose
    spawn_dog_visual: bool = True            # visual-only box standing in for the dog body under the arm

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
    theta_limits: tuple = (-0.5, 2.3)

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

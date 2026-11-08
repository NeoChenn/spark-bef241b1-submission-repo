"""Config for the PICK skill env (Isaac Lab DirectRLEnv) + rsl_rl PPO runner cfg. Mirrors scrub_env_cfg.py.

Import only after the Isaac Sim app is launched (AppLauncher).

Robot: the patrol team's ``go2_ur3_gripper.urdf`` (Go2 + lift + UR3 + parallel-jaw gripper, branch patrol-nav,
generator integration/make_go2_ur3_gripper.py), fixed base, gravity off, lift held fully down, legs PD-held. The
trunk is lowered to the crouch height at reset (the orchestrator crouches the Go2 before a pick). Only the 6 UR3
joints (IK on ``grasp_frame``) and the finger joints are driven. Set PICK_URDF to point at the URDF.
"""
from __future__ import annotations

import os

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from policy.envs.pick_state import ACT_DIM, OBS_DIM
from policy.envs.scrub_env_cfg import GO2_UR3_INIT, UR3_JOINTS, urdf_robot_cfg

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRIPPER_URDF = os.environ.get("PICK_URDF") or next(
    (p for p in ("/assets/go2_ur3/go2_ur3_gripper.urdf", os.path.join(_REPO, "sim/assets/go2_ur3/go2_ur3_gripper.urdf"))
     if os.path.exists(p)), "/assets/go2_ur3/go2_ur3_gripper.urdf")

FINGER_JOINTS = ["gripper_finger_left_joint", "gripper_finger_right_joint"]
# UR3 start pose: roughly gripper-down above the deck; the IK settles the exact pose in the first steps
PICK_INIT = {**GO2_UR3_INIT, "lift_joint": 0.0, "gripper_finger_left_joint": 0.06, "gripper_finger_right_joint": 0.06,
             "ur3_shoulder_pan_joint": 0.0, "ur3_shoulder_lift_joint": -1.0, "ur3_elbow_joint": 1.4,
             "ur3_wrist_1_joint": -1.97, "ur3_wrist_2_joint": -1.57, "ur3_wrist_3_joint": 0.0}


@configclass
class PickEnvCfg(DirectRLEnvCfg):
    # ---- timing: 120 Hz physics, 30 Hz control (as scrub), 8 s episodes (pick_cpu_env.EPISODE_STEPS)
    decimation = 4
    episode_length_s = 8.0
    action_space = ACT_DIM
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(1.0, 0.8, 0.6), lookat=(0.25, 0.0, 0.05), resolution=(1280, 720))

    robot_cfg = urdf_robot_cfg(GRIPPER_URDF, init_joint_pos=PICK_INIT, arm_joint_expr="ur3_.*")
    arm_joint_names: list = UR3_JOINTS
    finger_joint_names: list = FINGER_JOINTS
    lift_joint_name: str = "lift_joint"
    tcp_body_name: str = "grasp_frame"        # x = approach axis, y = jaw closing axis (URDF)
    finger_max: float = 0.06                  # each side; opening = left + right
    ik_damping: float = 0.05
    ik_max_step: float = 0.05                 # m per substep (as scrub)
    ik_rot_weight: float = 0.5                # orientation error weight in the 6D IK

    # ---- domain randomisation and rewards: policy.envs.pick_state.PickParams / pick_cpu_env.pick_reward
    spawn_visuals: bool = False


@configclass
class PickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Same values as ScrubPPORunnerCfg (and pick_cpu_env.PPO_CFG), so CPU checkpoints can seed an Isaac run."""
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "pick"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8, actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )

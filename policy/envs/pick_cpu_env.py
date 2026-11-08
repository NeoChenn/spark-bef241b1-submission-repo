"""PICK skill on CPU: PickState wrapped as an rsl_rl VecEnv, trained with the same PPO runner and settings as the
team's scrub skill (policy/envs/scrub_env_cfg.py ScrubPPORunnerCfg). The Isaac Lab env (pick_env.py) uses the same
PickState, observations, actions and rewards, so a CPU policy can initialise an Isaac run (``--init_from``).

Action (5, in [-1, 1]): TCP target dx, dy, dz (1 cm per step at 30 Hz), d(yaw) (0.08 rad per step), gripper
(> 0 open, <= 0 close).  Observation: 23 values, see PickState.observe.
Reward (per step):
    reach    1 - tanh(|TCP - goal| / 0.1), goal = 8 cm above the object until the jaws are lined up and centred,
             then the grasp point
    align    0.3 x (1 - jaw yaw error / (pi/2))
    jaws     0.2 x (open while approaching; closing once at the grasp point)
    grasp    +5 once, when the object is first held
    lift     2 x clamp(object height / LIFT_H, 0, 1) while held
    success  +20 + 2.5 x remaining steps, and the episode ends (finishing early must beat farming the shaping)
    knock    -3 per knock;  drop -2;  action rate -0.01 x |a - a_prev|^2
Episode 8 s (240 steps).
"""
from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from policy.envs.pick_state import ACT_DIM, LIFT_H, OBS_DIM, PickParams, PickState, jaw_yaw_error

try:
    from rsl_rl.env import VecEnv
except ImportError:          # training needs rsl_rl; the physics and rewards do not
    VecEnv = object

EPISODE_STEPS = 240
MAX_STEP_REWARD = 2.5      # upper bound of the per-step shaping terms (reach 1 + align 0.3 + jaws 0.2, or lift 2)

# rsl_rl runner config: the team's ScrubPPORunnerCfg values (scrub_env_cfg.py), as a plain dict
PPO_CFG = {
    "num_steps_per_env": 24, "save_interval": 50, "experiment_name": "pick", "run_name": "",
    "obs_groups": {"policy": ["policy"], "critic": ["policy"]}, "clip_actions": 1.0,
    "logger": "tensorboard",
    "policy": {"class_name": "ActorCritic", "init_noise_std": 0.8, "actor_obs_normalization": True,
               "critic_obs_normalization": True, "actor_hidden_dims": [512, 256, 128],
               "critic_hidden_dims": [512, 256, 128], "activation": "elu"},
    "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                  "entropy_coef": 0.005, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3.0e-4,
                  "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
}


def pick_reward(s: PickState, ev: dict, a: torch.Tensor, a_prev: torch.Tensor, first_grasp: torch.Tensor,
                remaining: torch.Tensor):
    gp = s.grasp_point()
    yaw_err = jaw_yaw_error(s.yaw, s.obj_yaw)
    along, across = s.to_obj(s.tcp[:, :2])
    # go down only when lined up; once down, closing must not move the goal (that taught "never close")
    lined_up = (yaw_err < 0.2) & (across.abs() < 0.01) & (along.abs() < 0.5 * s.l)
    goal = torch.where(lined_up[:, None], gp, gp + torch.tensor([0.0, 0.0, 0.08], device=s.dev))
    free = ~s.attached
    r = free.float() * (1 - torch.tanh((s.tcp - goal).norm(dim=-1) / 0.1))
    r += free.float() * 0.3 * (1 - yaw_err / (math.pi / 2))
    # jaws: open while approaching, closing once at the grasp point
    at_grasp = lined_up & ((s.tcp[:, 2] - gp[:, 2]).abs() < 0.01)
    want_open = (s.grip > s.w + 0.01).float()
    r += free.float() * 0.2 * torch.where(at_grasp, 1.0 - s.grip / 0.12, want_open)
    r += 5.0 * first_grasp.float()
    r += s.attached.float() * 2.0 * (s.obj[:, 2] / LIFT_H).clamp(0, 1)
    # success ends the episode, so it also pays what the remaining steps could still have earned; otherwise
    # hovering at the grasp point or holding just below the lift height pays more than finishing
    r += s.success.float() * (20.0 + MAX_STEP_REWARD * remaining.float())
    r -= 3.0 * ev["knocked"].float() + 2.0 * ev["dropped"].float()
    r -= 0.01 * ((a - a_prev) ** 2).sum(-1)
    return r


class PickCpuEnv(VecEnv):
    def __init__(self, num_envs: int = 1024, params: PickParams | None = None, device="cpu", seed: int = 0):
        self.num_envs, self.num_actions, self.num_obs = num_envs, ACT_DIM, OBS_DIM
        self.device = torch.device(device)
        self.max_episode_length = EPISODE_STEPS
        self.cfg = {"params": params or PickParams()}
        self.state = PickState(num_envs, params, device=device, seed=seed)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(num_envs, ACT_DIM, device=self.device)
        self.ep_return = torch.zeros(num_envs, device=self.device)

    def _obs(self) -> TensorDict:
        tf = self.episode_length_buf.float() / EPISODE_STEPS
        return TensorDict({"policy": self.state.observe(self.last_action, tf)}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._obs()

    def reset(self):
        self.state.reset(torch.arange(self.num_envs, device=self.device))
        self.episode_length_buf.zero_()
        self.last_action.zero_()
        return self._obs(), {}

    def step(self, actions: torch.Tensor):
        s = self.state
        a = actions.clamp(-1, 1)
        held_before = s.attached.clone()
        ev = s.step(a)
        first_grasp = ev["grasped"] & ~held_before
        rew = pick_reward(s, ev, a, self.last_action, first_grasp, EPISODE_STEPS - 1 - self.episode_length_buf)
        self.last_action = a
        self.episode_length_buf += 1
        self.ep_return += rew
        time_out = self.episode_length_buf >= EPISODE_STEPS
        done = s.success | time_out
        extras = {"time_outs": time_out & ~s.success, "log": {}}
        if done.any():
            ids = done.nonzero().squeeze(-1)
            extras["log"] = {"/success": s.success[ids].float().mean(), "/knocks": s.knocks[ids].mean(),
                             "/drops": s.drops[ids].mean(), "/grasped": s.ever_grasped[ids].float().mean()}
            s.reset(ids)
            self.episode_length_buf[ids] = 0
            self.last_action[ids] = 0
            self.ep_return[ids] = 0
        return self._obs(), rew, done, extras

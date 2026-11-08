"""The trained PICK policy as a robot skill: one litter pickup, closed loop, at 30 Hz.

    skill = RLPickSkill()                                   # policy/skills/pick_policy/policy.onnx
    ok, info = skill.pick(target_xy=(0.26, 0.02), yaw=0.4, size=(0.066, 0.066, 0.12))

The CPU mock runs the policy against PickState (policy/envs/pick_state.py: the same physics it was trained on),
with the object where the camera says it is plus the usual camera error. In Isaac the same ONNX policy drives the
real arm through PickEnv's IK (policy/envs/pick_env.py); integration/deck_bridge.py calls ``pick_skill(xy, yaw)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from policy.envs.pick_state import ACT_DIM, PickParams, PickState

POLICY_DIR = Path(__file__).resolve().parents[1] / "skills" / "pick_policy"


class RLPickSkill:
    def __init__(self, policy_dir: str | Path = POLICY_DIR, steps: int = 240):
        import onnxruntime as ort
        d = Path(policy_dir)
        self.sess = ort.InferenceSession(str(d / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        self.steps = steps

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.sess.run(None, {"obs": obs.numpy().astype(np.float32)})[0]
        return torch.from_numpy(out)

    def pick(self, target_xy, yaw: float, size=(0.066, 0.066, 0.12), seed: int = 0, params: PickParams | None = None):
        """size = (width the jaws close on, height, length) in metres. Returns (picked, info)."""
        s = PickState(1, params or PickParams(), seed=seed)
        s.obj[0, :2] = torch.as_tensor(target_xy, dtype=torch.float32)
        s.obj_yaw[0] = float(yaw)
        s.w[0], s.h[0], s.l[0] = (float(v) for v in size)
        last = torch.zeros(1, ACT_DIM)
        for t in range(self.steps):
            obs = s.observe(last, torch.full((1,), t / self.steps))
            a = self.act(obs)
            s.step(a)
            last = a
            if bool(s.success[0]):
                return True, {"steps": t + 1, "seconds": round((t + 1) / 30, 2), "knocks": int(s.knocks[0])}
        return False, {"steps": self.steps, "seconds": round(self.steps / 30, 2), "knocks": int(s.knocks[0]),
                       "grasped": bool(s.ever_grasped[0])}

    @classmethod
    def default(cls):
        try:
            return cls()
        except (FileNotFoundError, ImportError, Exception):
            return None

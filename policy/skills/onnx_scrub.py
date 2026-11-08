"""Trained PPO scrub policy (ONNX) behind the orchestrator's ScrubPolicy interface.

The PPO policy (``ScrubLiftV2EnvCfg``, ``policy/SCRUB.md``) reads a 327-D observation that includes the UR3 joint
state, the brush pose in the pipe frame, the contact force and 9x9 / 8x16 crops of a 32x64 rust grid. The CPU mock
orchestrator only knows the brush's (theta, z), the perceived rust grid and the measured force. So this wrapper keeps
a *shadow* ``bench.scrub_sim.KinematicScrub`` (one env: same UR3 + lift kinematics, DLS IK, analytic contact and
Archard abrasion as the CPU benchmark of the policy) and builds the observation from it with ``KinematicScrub.obs()``.

Each orchestrator step (``scrub_dt`` = 0.1 s) does:

1. Sync the shadow rust grid from ``obs["perceived_rust"]`` whenever the perceived map changes (i.e. after each
   re-detect). Cells are resampled by nearest cell centre from the orchestrator grid (``pipe.n_z`` x ``pipe.n_theta``
   over ``pipe.length``) onto the policy grid (32 x 64 over 0.6 m, centred on the pipe origin). Rust beyond +-0.3 m
   axially is outside the policy's training grid and is dropped. ``target_mask`` becomes the sprayed mask, marked
   fully soaked (the orchestrator only scrubs after the soak). Between syncs the shadow abrades its own copy, which is
   the progress signal the policy saw in training.
2. Run the ONNX policy at its native 30 Hz for ``round(scrub_dt / ctrl_dt)`` = 3 shadow steps.
3. Convert the shadow brush state into the orchestrator action ``[d_theta, d_z, press]``:

   * ``d_theta = wrap(theta_brush - theta_arm)``, ``d_z = z_brush - z_arm``: the orchestrator arm tracks the shadow
     brush (absolute tracking, so the two never drift apart; ``MockArm`` clips to 0.15 rad and 3 cm per step).
   * ``press = clip(F* / 25 N, 0, 1)`` while the shadow brush is within ``contact_snap`` (1 cm) of the surface, else
     0. ``F*`` is the policy's force setpoint ``25 + 20 a3`` N, capped at 38 N (``force_set_max``). 25 N is the
     orchestrator's ``press = 1`` force (``MockArm.force = 25 * press``), so F* >= 25 N saturates at press 1.

Pipe frames agree: theta = 0 on the horizontal normal towards the robot (``ref``), pi/2 up, so theta passes through
unchanged; the shadow pipe uses the orchestrator radius and the nominal rig placement (axis 0.5 m from the trunk,
z 1.1 m, lift 0.415 m, spring k 2500 N/m). The shadow's own contact force is fed to the policy (not the mock's 25 x
press), so the force loop the policy closes is the one it was trained on; force limits stay with the orchestrator.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from bench.scrub_sim import ARM_INIT, GRID_H, GRID_W, ArmFK, KinematicScrub, ScrubCfg, pipe_frame, reachable_mask
from policy.interfaces import PipeGeometry

DEFAULT_ONNX = os.environ.get("SCRUB_ONNX", "runs/scrub_v2/policy.onnx")
OBS_DIM = 2 * len(ARM_INIT) + 315
PRESS_FULL_N = 25.0
ORT_FALLBACK = [r"C:\isaacsim\exts\isaacsim.pip.onnx\pip_prebundle"]
MISSING_HELP = (
    "PPO scrub policy ONNX not found at {path}. It is not in git (runs/ is ignored). Fetch the v2 export into that path, "
    "e.g. from the GB10 box: scp watson:Documents/spark-bef241b1-submission-repo/runs/scrub_v2/policy.onnx {path}, "
    "or re-export it from the v2 checkpoint on Modal: MODAL_PROFILE=jacobyuan99 .venv/bin/modal run policy/play_scrub.py "
    "--run <v2 run> --robot lift_v2 --meta --force-patch, then .venv/bin/modal volume get rust-ckpt "
    "scrub/<v2 run>/exported/policy.onnx {path}. Or set SCRUB_ONNX / pass onnx_path.")


def _ort():
    try:
        import onnxruntime as ort
    except ImportError:
        for p in ORT_FALLBACK:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
        import onnxruntime as ort
    return ort


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _rodrigues(axis: np.ndarray, ang: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    t = 1 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


class NumpyArmFK(ArmFK):
    """Same chain and output as ``bench.scrub_sim.ArmFK`` in float64 numpy: the torch version spends ~20 ms per policy
    step on per-op overhead at batch 1, this one ~0.5 ms."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._chain = [(j.kind, np.asarray(j.origin, float), np.asarray(j.axis, float)) for j in self.chain]

    def forward(self, base: torch.Tensor, q: torch.Tensor, lift: torch.Tensor):
        b, qn, ln = base.double().numpy(), q.double().numpy(), lift.double().numpy().reshape(-1)
        Ts, AX, PT = [], [], []
        for i in range(qn.shape[0]):
            T, axes, pts, k = b[i].copy(), [], [], 0
            for kind, O, ax in self._chain:
                T = T @ O
                if kind == "prismatic":
                    T[:3, 3] += T[:3, :3] @ (ax * ln[i])
                elif kind in ("revolute", "continuous"):
                    axes.append(T[:3, :3] @ ax)
                    pts.append(T[:3, 3].copy())
                    T[:3, :3] = T[:3, :3] @ _rodrigues(ax, float(qn[i, k]))
                    k += 1
            Ts.append(T), AX.append(axes), PT.append(pts)
        f = lambda x: torch.from_numpy(np.asarray(x)).float()
        return f(Ts), f(AX), f(PT)


class OnnxScrubPolicy:
    name = "ppo"

    def __init__(self, pipe: PipeGeometry, onnx_path: str | os.PathLike | None = None, cfg: ScrubCfg | None = None,
                 dt: float = 0.1, threads: int = 1, fk: ArmFK | None = None):
        path = Path(onnx_path or DEFAULT_ONNX)
        if not path.is_file():
            raise FileNotFoundError(MISSING_HELP.format(path=path))
        ort = _ort()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        dim = self.sess.get_inputs()[0].shape[-1]
        if isinstance(dim, int) and dim != OBS_DIM:
            raise ValueError(f"{path}: policy expects {dim}-D obs, the scrub observation is {OBS_DIM}-D")
        self.path, self.pipe = str(path), pipe
        self.cfg = cfg or ScrubCfg()
        self.fk = fk or NumpyArmFK()
        self.substeps = max(1, int(round(dt / self.cfg.ctrl_dt)))
        self._rows, self._cols = self._resample_index()
        self.sim: KinematicScrub | None = None
        self.stats = {"steps": 0, "policy_calls": 0, "shadow_peak_force_N": 0.0, "shadow_over_40N_steps": 0,
                      "mean_force_setpoint_N": 0.0, "contact_steps": 0}
        self.last_obs: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self.reset()

    def _resample_index(self):
        L = self.cfg.pipe_length
        z = (np.arange(GRID_H) + 0.5) / GRID_H * L - L / 2
        th = (np.arange(GRID_W) + 0.5) / GRID_W * 2 * np.pi
        i, _ = self.pipe.theta_z_to_cell(np.zeros_like(z), z)
        _, j = self.pipe.theta_z_to_cell(th, np.zeros_like(th))
        inside = np.abs(z) < self.pipe.length / 2
        return np.where(inside, i, -1), j

    def _to_policy_grid(self, grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, float)
        out = g[np.clip(self._rows, 0, g.shape[0] - 1)][:, self._cols]
        out[self._rows < 0] = 0.0
        return np.ascontiguousarray(out)

    def reset(self) -> None:
        self.sim = None
        self._last_map = None

    def _scenario(self, theta: float, z: float) -> dict:
        c = self.cfg
        r, yaw, lift = float(self.pipe.radius), 0.0, float(np.mean(c.lift_range))
        origin = (0.0, -float(np.mean(c.axis_dist_range)), float(np.mean(c.height_range)))
        geo = pipe_frame(torch.tensor([origin]), torch.tensor([yaw]))
        reach = reachable_mask(c, geo, torch.tensor([r]), torch.tensor([lift]))[0]
        zeros = torch.zeros(GRID_H, GRID_W)
        return {"seed": 0, "radius": r, "yaw": yaw, "origin": origin, "lift": lift, "spring_k": c.spring_k_nominal,
                "abrasion_coef": 1.0, "soak": 60.0, "q0": list(ARM_INIT),
                "tgt0": (float(np.clip(z, -c.pipe_length / 2, c.pipe_length / 2)), r + c.brush_radius + 0.05, theta),
                "event_step": -1, "rust": zeros.clone(), "sprayed": zeros.bool(), "soak_time": zeros.clone(),
                "reachable": reach, "patch": zeros.clone()}

    def _sync(self, obs: dict):
        rust = np.asarray(obs["perceived_rust"], float)
        key = rust.tobytes()
        if key == self._last_map:
            return
        self._last_map = key
        rs = self.sim.rust
        g = torch.from_numpy(self._to_policy_grid(rust)).float().clamp(0, 1)[None]
        tm = obs.get("target_mask")
        sprayed = torch.from_numpy(self._to_policy_grid(tm) > 0.5)[None] if tm is not None else g > rs.p.rust_eps
        rs.rust = g
        rs.sprayed = sprayed
        rs.soak_time = torch.where(sprayed, torch.tensor(60.0), torch.tensor(0.0))
        rs.initial_total = torch.maximum(rs.initial_total, g.sum(dim=(1, 2))).clamp_min(1e-6)

    def policy_obs(self) -> np.ndarray:
        return self.sim.obs().numpy().astype(np.float32)

    def step(self, obs: dict) -> np.ndarray:
        th_arm, z_arm = (float(v) for v in obs["ee_theta_z"])
        if self.sim is None:
            self.sim = KinematicScrub([self._scenario(th_arm, z_arm)], self.cfg, self.fk)
            self.sim.rust.initial_total = torch.full((1,), 1e-6)
        self._sync(obs)
        sim, c = self.sim, self.cfg
        for _ in range(self.substeps):
            o = self.policy_obs()
            a = self.sess.run(None, {self.inp: o})[0].reshape(1, -1)[:, :4]
            a = torch.from_numpy(np.clip(a, -1.0, 1.0)).float()
            sim.done[:] = False
            sim.success[:] = False
            sim.step_i[:] = 0
            sim.step(a)
            f = float(sim.force[0])
            self.stats["policy_calls"] += 1
            self.stats["shadow_peak_force_N"] = max(self.stats["shadow_peak_force_N"], f)
            self.stats["shadow_over_40N_steps"] += int(f > 40.0)
            self.last_obs, self.last_action = o, a.numpy()[0]
        f_set = float(sim.f_set[0])
        contact = float(sim.gap[0]) <= c.contact_snap
        press = float(np.clip(f_set / PRESS_FULL_N, 0.0, 1.0)) if contact else 0.0
        n = self.stats["steps"]
        self.stats["mean_force_setpoint_N"] = (self.stats["mean_force_setpoint_N"] * n + f_set) / (n + 1)
        self.stats["steps"] = n + 1
        self.stats["contact_steps"] += int(contact)
        th_b, z_b = float(sim.theta[0]), float(sim.axial[0])
        return np.array([_wrap(th_b - th_arm), z_b - z_arm, press])

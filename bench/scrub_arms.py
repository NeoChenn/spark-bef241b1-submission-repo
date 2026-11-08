from __future__ import annotations

import time

import numpy as np
import torch

from bench.scrub_sim import GRID_H, GRID_W, KinematicScrub
from policy.demo_mock import GreedyScrub
from policy.interfaces import PipeGeometry


class Timer:

    def __init__(self):
        self.ms = []

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms.append((time.perf_counter() - self.t) * 1000.0)


class PPOArm:
    name = "PPO v2 (ONNX)"
    kind = "native"

    def __init__(self, onnx_path: str, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.timer = Timer()

    def reset(self, sim: KinematicScrub):
        pass

    def act(self, sim: KinematicScrub) -> torch.Tensor:
        o = sim.obs().numpy().astype(np.float32)
        out = []
        for i in range(o.shape[0]):
            with self.timer:
                out.append(self.sess.run(None, {self.inp: o[i:i + 1]})[0][0])
        return torch.from_numpy(np.stack(out)).float().clamp(-1, 1)


class GreedyArm:
    name = "Greedy scripted"
    kind = "target"

    def __init__(self):
        self.timer = Timer()

    def reset(self, sim: KinematicScrub):
        self.pipes = [PipeGeometry(origin=[0, 0, 0], axis=[1, 0, 0], ref=[0, 1, 0], radius=float(r),
                                   length=sim.cfg.pipe_length, n_z=GRID_H, n_theta=GRID_W) for r in sim.pipe_r.cpu()]
        self.pol = [GreedyScrub(p) for p in self.pipes]

    def act(self, sim: KinematicScrub) -> torch.Tensor:
        rust = (sim.rust.rust > sim.rust.p.rust_eps).cpu().numpy()
        sprayed = sim.rust.sprayed.cpu().numpy()
        th, z = sim.theta.cpu().numpy(), sim.axial.cpu().numpy()
        out = np.zeros((sim.N, 3))
        for i in range(sim.N):
            if sim.done[i]:
                continue
            with self.timer:
                a = self.pol[i].step({"perceived_rust": rust[i].astype(float), "ee_theta_z": (float(th[i]) % (2 * np.pi), float(z[i])),
                                      "target_mask": sprayed[i]})
            out[i] = a
        a = torch.from_numpy(out).float().to(sim.pipe_r.device)
        return sim.native_from_target(sim.theta + a[:, 0], sim.axial + a[:, 1], a[:, 2])


class RandomArm:
    name = "Random"
    kind = "target"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.timer = Timer()

    def reset(self, sim: KinematicScrub):
        self.rng = np.random.default_rng(self.seed)

    def act(self, sim: KinematicScrub) -> torch.Tensor:
        with self.timer:
            a = self.rng.uniform([-0.3, -0.05, 0.0], [0.3, 0.05, 1.0], size=(sim.N, 3))
        a = torch.from_numpy(a).float().to(sim.pipe_r.device)
        return sim.native_from_target(sim.theta + a[:, 0], sim.axial + a[:, 1], a[:, 2])


def run_batch(arm, scenarios, cfg=None, batch: int = 64) -> list[dict]:
    out = []
    for k in range(0, len(scenarios), batch):
        sim = KinematicScrub(scenarios[k:k + batch], cfg)
        arm.reset(sim)
        while not sim.done.all():
            sim.step(arm.act(sim))
        out += sim.episode_stats()
    return out

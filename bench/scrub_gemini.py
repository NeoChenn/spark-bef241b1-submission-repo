from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from bench import gemini
from bench.scrub_sim import KinematicScrub

SCHEMA = {"type": "OBJECT", "properties": {"d_theta": {"type": "NUMBER"}, "d_z": {"type": "NUMBER"},
                                           "press": {"type": "NUMBER"}, "reason": {"type": "STRING"}},
          "required": ["d_theta", "d_z", "press"]}

PROMPT = ("You are the contact controller of a robot arm scrubbing rust off a horizontal pipe. The brush position is "
          "given in pipe coordinates: theta (rad, angle around the pipe) and z (m, along the axis). Each reply sets a "
          "target: the brush moves to (theta + d_theta, z + d_z) at up to 0.3 m/s and holds it until your next reply. "
          "press in [0, 1] sets the contact force setpoint 5 + 40*press N (clamped to 38 N); press <= 0.05 lifts the "
          "brush off. Rust is removed in proportion to force times sliding speed, so the brush must keep moving across "
          "rust while pressing. Contact force above 40 N is a safety violation. Rust soaked with remover comes off 4x "
          "faster. The episode succeeds when at most 5% of the rust remains; it lasts 15 s. ")
TIMING = {"realtime": ("The simulation keeps running while you think: your reply takes effect only after your own "
                       "response latency, and you are asked again as soon as it arrives. Reply with JSON {{d_theta, d_z, press, reason}}."),
          "paused": ("The simulation is paused while you think; your reply is applied immediately and held for {hold} s "
                     "of simulated time, then you are asked again. Reply with JSON {{d_theta, d_z, press, reason}}.")}


def describe(sim: KinematicScrub, i: int, k: int = 12) -> dict:
    r = float(sim.pipe_r[i])
    rust = sim.rust.rust[i]
    th, z = float(sim.theta[i]), float(sim.axial[i])
    cells = (rust > sim.rust.p.rust_eps).nonzero(as_tuple=False)
    items = []
    if len(cells):
        cz, ct = sim.rust.z_c[cells[:, 0]], sim.rust.theta_c[cells[:, 1]]
        dth = torch.remainder(ct - th + math.pi, 2 * math.pi) - math.pi
        dz = cz - z
        d = (dth * r) ** 2 + dz ** 2
        for j in torch.argsort(d)[:k].tolist():
            items.append({"d_theta": round(float(dth[j]), 3), "d_z": round(float(dz[j]), 3),
                          "level": round(float(rust[cells[j, 0], cells[j, 1]]), 2)})
        w = rust[cells[:, 0], cells[:, 1]]
        cen = {"d_theta": round(float((dth * w).sum() / w.sum()), 3), "d_z": round(float((dz * w).sum() / w.sum()), 3)}
    else:
        cen = None
    return {"t_s": round(float(sim.step_i[i]) * sim.cfg.ctrl_dt, 2), "theta": round(th, 3), "z": round(z, 3),
            "pipe_radius_m": round(r, 3), "gap_to_surface_m": round(float(sim.gap[i]), 4),
            "force_N": round(float(sim.force[i]), 1), "force_setpoint_N": round(float(sim.f_set[i]), 1),
            "rust_fraction_remaining": round(float(sim.rust.fraction_remaining()[i]), 3),
            "rusty_cells": int(len(cells)), "cell_size": {"d_theta": round(2 * math.pi / sim.rust.W, 4),
                                                          "d_z": round(sim.cfg.pipe_length / sim.rust.H, 4)},
            "nearest_rust_cells": items, "rust_centroid": cen,
            "soaked_fraction_of_rust": round(float(((sim.rust.soak_time[i] >= 30) & (rust > 0.02)).sum()) /
                                             max(len(cells), 1), 2)}


def decision_steps(mode: str, latency_s: float, dt: float, hold_s: float) -> int:
    return max(1, round(hold_s / dt)) if mode == "paused" else max(1, math.ceil(latency_s / dt))


def run_episode(scenario: dict, model: str, key: str, cfg=None, post=None, max_calls: int = 200,
                mode: str = "realtime", hold_s: float = 0.5, thinking: str | None = "low") -> dict:
    sim = KinematicScrub([scenario], cfg)
    dt = sim.cfg.ctrl_dt
    target = (sim.theta.clone(), sim.axial.clone(), torch.zeros(1))
    calls, pending, due = [], None, 0
    prompt = PROMPT + TIMING[mode].format(hold=hold_s) + "\n\nState:\n"
    t0 = time.perf_counter()
    while not sim.done.all():
        if sim.step_i[0] >= due and len(calls) < max_calls:
            if pending is not None and mode == "realtime":
                target = pending
            st = describe(sim, 0)
            try:
                r = gemini.generate(model, key, [{"text": prompt + json.dumps(st)}], SCHEMA, post=post, thinking=thinking)
                j = r["json"]
                pending = (sim.theta + float(np.clip(j["d_theta"], -0.5, 0.5)),
                           sim.axial + float(np.clip(j["d_z"], -0.3, 0.3)),
                           torch.tensor([float(np.clip(j["press"], 0.0, 1.0))]))
                calls.append({"latency_s": r["latency_s"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"]})
                lat = r["latency_s"]
                if mode == "paused":
                    target = pending
            except Exception as e:
                calls.append({"error": f"{type(e).__name__}"})
                lat = 1.0
            due = int(sim.step_i[0]) + decision_steps(mode, lat, dt, hold_s)
        sim.step(sim.native_from_target(*target))
    ep = sim.episode_stats()[0]
    ok = [c for c in calls if "error" not in c]
    tin, tout = sum(c["tokens_in"] for c in ok), sum(c["tokens_out"] for c in ok)
    ep.update({"mode": mode, "calls": len(calls), "errors": len(calls) - len(ok), "latency_s": [c["latency_s"] for c in ok],
               "tokens_in": tin, "tokens_out": tout, "cost_usd": gemini.cost_usd(tin, tout, model),
               "wall_s": time.perf_counter() - t0})
    return ep

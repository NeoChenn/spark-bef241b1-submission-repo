"""Tune and score the lidar object detector (patrol/detect.py) against ground truth from patrol/sim2d.py.

  python -m patrol.eval_detect                     # sweep on 360 beams, then score the pick at 180 / 72 beams
  python -m patrol.eval_detect --beams 360 --no-sweep

Ground truth per frame: active objects that intrude into the robot's lane (|lateral of centre| - r < 0.32 m),
whose centre is 0.2-3.0 m ahead along the path, and that at least one beam actually hits (noiseless scan).
A detection is a true positive when it lies within 0.25 m of a ground-truth centre. Detections that land on
an object outside the ground-truth set (e.g. one just outside the lane, or occluded) are ignored, not
counted as false positives; anything else (walls, noise) is a false positive.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from patrol.detect import DetectCfg, PathRef, Tracker, detect_frame
from patrol.ppo import pure_pursuit
from patrol.sim2d import PatrolSim, SimCfg

TUBE_GT, AHEAD_GT, MATCH = 0.32, (0.2, 3.0), 0.25
PICK_TRUE = 0.11
DIST_BINS = ((0, 1), (1, 2), (2, 3.2))
SIZE_BINS = ((0, 0.11), (0.11, 0.25), (0.25, 1.0))


@dataclass
class Frame:
    env: int
    episode: int
    ranges: np.ndarray
    pose: tuple
    path_key: tuple
    robot_idx: int
    gt_xy: np.ndarray       # [G, 2] ground-truth objects (in lane, ahead, visible)
    gt_r: np.ndarray
    other_xy: np.ndarray    # active objects not in the GT set (don't-care)


def circle_hits(sim: PatrolSim, beams: torch.Tensor) -> torch.Tensor:
    th = sim.yaw[:, None] + beams[None]
    d = torch.stack([th.cos(), th.sin()], -1)[:, :, None]
    f = sim.pos[:, None, None] - sim.obs_c[:, None]
    bq = (f * d).sum(-1)
    cq = (f * f).sum(-1) - sim.obs_r[:, None] ** 2
    disc = bq**2 - cq
    tc = -bq - disc.clamp(min=0).sqrt()
    ok = (disc >= 0) & (tc > 0) & sim.obs_on[:, None]
    return torch.where(ok, tc, torch.full_like(tc, math.inf))


def generate(beams_n: int, envs=64, steps=300, every=3, seed=777):
    sim = PatrolSim(SimCfg(num_envs=envs, seed=seed, obstacle_r=(0.03, 0.25), p_obstacle=0.8, max_steps=10_000))
    beams = -math.pi + (torch.arange(beams_n) + 0.5) * (2 * math.pi / beams_n)
    episode = np.zeros(envs, dtype=int)
    paths: dict[tuple, PathRef] = {}
    frames: list[Frame] = []
    obs = sim.observe()
    with torch.no_grad():
        for t in range(steps):
            if t % every == 0:
                noisy = sim.lidar(beams, noise=True).numpy()
                clean = sim.lidar(beams, noise=False)
                tc = circle_hits(sim, beams)
                vis = ((clean[..., None] - tc).abs() < 0.02).any(1).numpy()  # [N, K]
                for e in range(envs):
                    key = (e, int(episode[e]))
                    if key not in paths:
                        paths[key] = PathRef(sim.P[e].numpy(), ds=sim.cfg.ds)
                    pr = paths[key]
                    pose = (float(sim.pos[e, 0]), float(sim.pos[e, 1]), float(sim.yaw[e]))
                    ri = int(sim.idx[e])
                    on = sim.obs_on[e].numpy()
                    c, r = sim.obs_c[e].numpy(), sim.obs_r[e].numpy()
                    gt, other = [], []
                    if on.any():
                        _, lat, s = pr.project(c)
                        ahead = s - pr.s[min(ri, len(pr.s) - 1)]
                        for k in np.flatnonzero(on):
                            inlane = abs(lat[k]) - r[k] < TUBE_GT and AHEAD_GT[0] < ahead[k] < AHEAD_GT[1]
                            (gt if inlane and vis[e, k] else other).append(k)
                    frames.append(Frame(e, int(episode[e]), noisy[e], pose, key, ri,
                                        c[gt].reshape(-1, 2), r[gt], c[other].reshape(-1, 2)))
            a = pure_pursuit(sim)
            obs, _, done, _ = sim.step(a)
            episode += done.numpy().astype(int)
    return frames, paths, beams.numpy()


def score(frames, paths, angles, cfg: DetectCfg, confirms=(1,)):
    """Detections once per frame, then the tracker per confirm level. Returns {confirm: metrics}."""
    per_frame = [detect_frame(f.ranges, angles, f.pose, paths[f.path_key], f.robot_idx, cfg) for f in frames]
    out = {}
    for conf in confirms:
        tcfg = replace(cfg, confirm=conf)
        trackers: dict[tuple, Tracker] = {}
        tp = fp = fn = 0
        by_dist = {b: [0, 0] for b in DIST_BINS}
        by_size = {b: [0, 0] for b in SIZE_BINS}
        werr, pick_ok, pick_n = [], 0, 0
        for f, dets in zip(frames, per_frame):
            tr = trackers.setdefault(f.path_key, Tracker(tcfg))
            confirmed = tr.update([copy.copy(d) for d in dets])
            matched = np.zeros(len(f.gt_xy), dtype=bool)
            for d in confirmed:
                if len(f.gt_xy):
                    dist = np.linalg.norm(f.gt_xy - d.xy, axis=1)
                    k = int(dist.argmin())
                    if dist[k] < MATCH:
                        if not matched[k]:
                            matched[k] = True
                            tp += 1
                            werr.append(d.width - 2 * f.gt_r[k])
                            pick_n += 1
                            pick_ok += int(d.pickable == (2 * f.gt_r[k] <= PICK_TRUE))
                        continue
                if len(f.other_xy) and np.linalg.norm(f.other_xy - d.xy, axis=1).min() < MATCH:
                    continue
                fp += 1
            fn += int((~matched).sum())
            rng = np.linalg.norm(f.gt_xy - np.array(f.pose[:2]), axis=1) if len(f.gt_xy) else []
            for k in range(len(f.gt_xy)):
                for b in DIST_BINS:
                    if b[0] <= rng[k] < b[1]:
                        by_dist[b][0] += int(matched[k]); by_dist[b][1] += 1
                for b in SIZE_BINS:
                    if b[0] <= 2 * f.gt_r[k] < b[1]:
                        by_size[b][0] += int(matched[k]); by_size[b][1] += 1
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        w = np.array(werr) if werr else np.zeros(1)
        out[conf] = {
            "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "tp": tp, "fp": fp, "fn": fn, "fp_per_frame": fp / len(frames), "frames": len(frames),
            "recall_by_dist": {f"{a}-{b}m": round(h / max(n, 1), 3) for (a, b), (h, n) in by_dist.items()},
            "recall_by_size": {f"{a}-{b}m": round(h / max(n, 1), 3) for (a, b), (h, n) in by_size.items()},
            "n_by_size": {f"{a}-{b}m": n for (a, b), (_, n) in by_size.items()},
            "width_err_mean": float(w.mean()), "width_err_mae": float(np.abs(w).mean()),
            "pickable_acc": pick_ok / max(pick_n, 1),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=int, default=360)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default="runs/patrol/detect_eval.json")
    a = ap.parse_args()

    t0 = time.time()
    frames, paths, angles = generate(a.beams, a.envs, a.steps)
    print(f"{len(frames)} frames, {sum(len(f.gt_xy) for f in frames)} GT objects ({time.time() - t0:.0f}s)")
    rows = []
    base = DetectCfg()
    grid = [(base.tube, base.cluster_gap, base.min_points)] if a.no_sweep else \
        list(itertools.product((0.25, 0.28, 0.32, 0.36), (0.08, 0.12, 0.18), (1, 2)))
    for tube, gap, mp in grid:
        cfg = replace(base, tube=tube, cluster_gap=gap, min_points=mp)
        for conf, m in score(frames, paths, angles, cfg, confirms=(1, 2, 3)).items():
            rows.append({"tube": tube, "cluster_gap": gap, "min_points": mp, "confirm": conf, **m})
            print(f"tube {tube:.2f} gap {gap:.2f} minpts {mp} confirm {conf}: P {m['precision']:.3f} "
                  f"R {m['recall']:.3f} F1 {m['f1']:.3f} fp/frame {m['fp_per_frame']:.3f}")
    ok = [r for r in rows if r["precision"] >= 0.95] or rows
    best = max(ok, key=lambda r: r["f1"])
    best_cfg = replace(base, tube=best["tube"], cluster_gap=best["cluster_gap"], min_points=best["min_points"])
    pads = {}
    for pad in (1.0, 1.5, 2.0, 2.5, 3.0):
        m = score(frames, paths, angles, replace(best_cfg, width_pad_beams=pad), confirms=(best["confirm"],))[best["confirm"]]
        pads[pad] = {k: m[k] for k in ("width_err_mean", "width_err_mae", "pickable_acc", "f1")}
        print(f"width pad {pad}: {pads[pad]}")
    best_pad = max(pads, key=lambda k: pads[k]["pickable_acc"])
    best_cfg = replace(best_cfg, width_pad_beams=best_pad)
    best = {**best, **score(frames, paths, angles, best_cfg, confirms=(best["confirm"],))[best["confirm"]],
            "width_pad_beams": best_pad}
    print("BEST", json.dumps(best, indent=1))
    res = {"beams": a.beams, "best": best, "rows": rows, "width_pad": pads, "by_beams": {}}
    for nb in (180, 72):
        if nb == a.beams or a.no_sweep:
            continue
        fr, pa, an = generate(nb, a.envs, a.steps)
        m = score(fr, pa, an, best_cfg, confirms=(best["confirm"],))[best["confirm"]]
        res["by_beams"][nb] = m
        print(f"{nb} beams: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"recall_by_size {m['recall_by_size']}")
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()

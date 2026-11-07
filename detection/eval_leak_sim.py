"""Held-out check of the crack detector on rendered flange views (like detection/eval.py sim_heldout for rust).

Renders flanges the training data never used (different seeds), through the 640x360 wrist camera at the Isaac
viewing distance, and runs the whole LeakLocalizer (face windows -> classifier -> calibration -> threshold ->
unsure band). Reports, per detector: cracks found, cracks missed, unsure, and false alarms on clean flanges.

    python -m detection.eval_leak_sim --ckpt runs/leak_detector/mnv4s_lora      # trained model
    python -m detection.eval_leak_sim                                            # the hand-written fake only

STILL SIMULATED: these renders are the mock's, not Isaac's. Isaac renders (lighting, materials, noise) will
differ, so re-run this on frames from the Isaac stream before trusting the numbers.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from detection.leak import LeakLocalizer
from detection.prepare_leak_data import _camera
from policy.leak.geometry import FlangeGeometry
from policy.leak.render import render_flange
from policy.leak.world import LeakWorld


def cases(n: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    for i in range(n):
        crack = float(rng.uniform(0.15, 0.8)) if i % 2 == 0 else 0.0
        yield {"crack": crack, "dust": float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.3)])),
               "view": int(rng.integers(0, 3)), "distance": float(rng.uniform(1.0, 1.25)),
               "loose": [(int(b), float(rng.uniform(-0.25, 0.25))) for b in rng.choice(8, int(rng.integers(0, 3)), False)],
               "crack_seed": int(rng.integers(20_000, 30_000)), "rng": int(rng.integers(0, 2**31 - 1))}


def evaluate(loc: LeakLocalizer, n: int = 80, seed: int = 12345) -> dict:
    out = {"crack": {"found": 0, "unsure": 0, "missed": 0}, "clean": {"false_alarm": 0, "unsure": 0, "ok": 0},
           "severity_found": [], "severity_missed": []}
    for c in cases(n, seed):
        g = FlangeGeometry("E", [0, 0, 1], [0, -1, 0], [0, 0, 1])
        s = LeakWorld([g]).flanges["E"]
        for b, a in c["loose"]:
            s.angles[b] = a
        s.crack = c["crack"]
        cam = _camera(g, c["view"], c["distance"], np.random.default_rng(c["rng"]))
        r = loc.read(render_flange(cam, s, dust=c["dust"], crack_seed=c["crack_seed"]), cam, g)
        hit = r.crack_prob >= loc.threshold
        if c["crack"] > 0:
            key = "found" if hit else ("unsure" if r.crack_unsure else "missed")
            out["crack"][key] += 1
            out["severity_found" if hit else "severity_missed"].append(round(c["crack"], 2))
        else:
            out["clean"]["false_alarm" if hit else ("unsure" if r.crack_unsure else "ok")] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args(argv)
    results = {"fake_classifier": evaluate(LeakLocalizer(), a.n)}
    if a.ckpt:
        results["trained"] = evaluate(LeakLocalizer.from_checkpoint(a.ckpt), a.n)
    for name, r in results.items():
        c, k = r["crack"], r["clean"]
        n_c, n_k = sum(c.values()), sum(k.values())
        print(f"{name:16s} cracks found {c['found']}/{n_c}, unsure {c['unsure']}, missed {c['missed']} | "
              f"clean flagged {k['false_alarm']}/{n_k}, unsure {k['unsure']} | "
              f"missed severities {sorted(r['severity_missed'])[:12]}")
    print(json.dumps(results))
    return results


if __name__ == "__main__":
    main()

"""Held-out check of the debris and person detectors on whole camera frames (like eval_leak_sim for cracks).

Frames are built by policy/deck/scene.py on ``preview_deck_end.png``, a deck-level render no training set used,
with litter and people from the held-out test splits. It shows deck equipment that must NOT be picked up
(fire extinguishers, gas bottles, pipe supports, valve wheels).

  debris   random scenes of 0-3 objects (drawn cans and real test-split litter), 3-6 m ahead, plus empty decks.
           Reports objects found (a hit within 45 px of the object, i.e. where the arm would pick), objects missed,
           and false alarms (hits on no object).
  person   one person (3 held-out test-split crops) at 2-10 m, or nobody. Reports found / missed, the error of the
           rough distance, and whether the "pause within 3 m" decision matches the truth.

    python -m detection.eval_rig_vision                      # shipped checkpoints
    python -m detection.eval_rig_vision --debris-ckpt runs/debris_detector/mnv4s_lora

STILL SIMULATED: the background is a render and the objects are pasted in. Re-run on Isaac frames.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from detection.rig_vision import CKPT_DIR, DebrisDetector, PersonDetector
from policy.deck.scene import Debris, camera_model, deck_view, person_range, person_view, sprites

PICK_PX = 45


def eval_debris(det: DebrisDetector, n: int = 60, seed: int = 7, use_reference: bool = True) -> dict:
    """Every frame gets a random exposure change and half get a shadow across the deck (harmless change); the
    reference is the clean deck under normal light."""
    rng = np.random.default_rng(seed)
    reference = deck_view([]) if use_reference else None
    kinds = ["can"] + sprites("debris")
    out = {"objects": 0, "found": 0, "missed": 0, "false_alarms": 0, "empty_frames": 0, "empty_frames_flagged": 0,
           "unsure_frames": 0}
    for i in range(n):
        k = int(rng.integers(0, 4)) if i % 3 else 0          # a third of the frames are empty decks
        items = [Debris(float(rng.uniform(-0.1, 0.7)), float(rng.uniform(3.2, 6.0)), kinds[int(rng.integers(len(kinds)))],
                        length_m=float(rng.uniform(0.1, 0.25)), seed=int(rng.integers(1e6))) for _ in range(k)]
        light = float(rng.uniform(0.75, 1.2))
        shadow = int(rng.integers(1e6)) if rng.random() < 0.5 else None
        r = det.detect(deck_view(items, light=light, shadow_seed=shadow), reference)
        out["unsure_frames"] += int(r.unsure)
        used = set()
        for it in items:
            u, v = it.pixel()
            near = [j for j, h in enumerate(r.hits) if np.hypot(h.centre[0] - u, h.centre[1] - v) < PICK_PX]
            out["objects"] += 1
            if near:
                out["found"] += 1
                used.update(near)
            else:
                out["missed"] += 1
        out["false_alarms"] += sum(1 for j in range(len(r.hits)) if j not in used)
        if not items:
            out["empty_frames"] += 1
            out["empty_frames_flagged"] += int(r.found)
    return out


def eval_person(det: PersonDetector, n: int = 60, seed: int = 8, radius: float = 3.0, use_depth: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    cam = camera_model()
    out = {"people": 0, "found": 0, "missed": 0, "empty_frames": 0, "empty_flagged": 0, "dist_err_m": [],
           "pause_right": 0, "pause_wrong_missed": 0, "pause_wrong_extra": 0}
    for i in range(n):
        d = None if i % 4 == 0 else float(rng.uniform(2.0, 10.0))
        x = float(rng.uniform(-0.8, 0.8)) * (d or 1.0) / 4
        who = int(rng.integers(0, 9))
        r = det.detect(person_view(d, x_m=x, who=who), cam, person_range(d, x_m=x, who=who) if use_depth else None)
        if d is None:
            out["empty_frames"] += 1
            out["empty_flagged"] += int(r.found)
        else:
            out["people"] += 1
            out["found" if r.found else "missed"] += 1
            if r.found and r.distance_m is not None:
                out["dist_err_m"].append(round(r.distance_m - d, 2))
        truth = d is not None and d < radius
        said = r.found and (r.distance_m is None or r.distance_m < radius)
        if truth == said:
            out["pause_right"] += 1
        elif truth:
            out["pause_wrong_missed"] += 1
        else:
            out["pause_wrong_extra"] += 1
    e = np.abs(out["dist_err_m"]) if out["dist_err_m"] else np.zeros(1)
    out["dist_abs_err_median_m"] = round(float(np.median(e)), 2)
    out.pop("dist_err_m")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--debris-ckpt", default=str(CKPT_DIR / "debris_v2"))
    ap.add_argument("--person-ckpt", default=str(CKPT_DIR / "person_v1"))
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args(argv)
    res = {}
    if a.debris_ckpt:
        det = DebrisDetector.from_checkpoint(a.debris_ckpt)
        res["debris"] = eval_debris(det, a.n)
        res["debris_no_reference"] = eval_debris(det, a.n, use_reference=False)
    if a.person_ckpt:
        det = PersonDetector.from_checkpoint(a.person_ckpt)
        res["person"] = eval_person(det, a.n)
        res["person_no_depth"] = eval_person(det, a.n, use_depth=False)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()

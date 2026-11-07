"""Build the crack / clean dataset for the leak detector (same layout and process as detection/prepare_data.py).

Layout written:  <out>/{train,val,test}/{crack,clean}/*.jpg     (label 1 = crack)
  * real_*   photos from Hugging Face ``mohammadnajeeb/concrete_crack_images`` (CC BY 4.0; 227 px concrete patches,
             "Positive" = crack). Concrete is NOT steel: these teach "thin dark crack line", not what a cracked
             flange looks like. Split at random 70/15/15 (the source gives no grouping, so near-duplicates can leak
             across splits; treat real-photo test scores as optimistic).
  * synth_*  crops of flange faces rendered by policy/leak/render.py through the 640x360 wrist camera at the
             Isaac viewing distance (1.1 m), with random view, dust, glare and loose bolts. Positives are face windows
             that the crack line actually crosses; negatives come from renders with no crack. File names start with
             ``synth_`` so detection/train.py reports real vs synthetic separately (as for rust_v2).

    python -m detection.prepare_leak_data --zip <path>/data/test.zip --out data/leak_v1
"""
from __future__ import annotations

import argparse
import io
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from detection.leak import LeakLocalizer
from policy.interfaces import CameraModel
from policy.leak.geometry import FlangeGeometry, look_at, project
from policy.leak.render import crack_path, render_flange
from policy.leak.world import LeakWorld

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
CROP = 32          # matches LeakLocalizer(patch=32)


def _split_of(i: int, n: int) -> str:
    edge = 0.0
    for name, frac in SPLITS:
        edge += frac
        if i < edge * n:
            return name
    return "test"


def real_images(zip_path: str, out: Path, n_per_class: int, rng) -> dict:
    z = zipfile.ZipFile(zip_path)
    counts = {}
    for folder, cname in (("Positive", "crack"), ("Negative", "clean")):
        names = sorted(n for n in z.namelist() if f"/{folder}/" in n and n.lower().endswith((".jpg", ".png")))
        pick = rng.choice(len(names), size=min(n_per_class, len(names)), replace=False)
        for i, j in enumerate(pick):
            split = _split_of(i, len(pick))
            img = Image.open(io.BytesIO(z.read(names[j]))).convert("RGB").resize((128, 128))
            d = out / split / cname
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"real_{Path(names[j]).stem}.jpg", quality=92)
            counts[(split, cname)] = counts.get((split, cname), 0) + 1
    return counts


def _camera(geom, view, distance, rng, W=640, H=360):
    f = 35.0 / 20.955 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    eye, target = geom.view_poses(distance)[view]
    eye = eye + rng.normal(0, 0.03, 3)
    return CameraModel(K, look_at(eye, target), W, H)


def _crack_mask(cam, geom, severity, seed) -> np.ndarray:
    uv, z = project(cam, crack_path(geom, severity, seed))
    m = Image.new("L", (cam.width, cam.height), 0)
    ImageDraw.Draw(m).line([tuple(map(float, p)) for p in uv], fill=255, width=max(1, int(round(1 + 4 * severity))))
    return np.asarray(m) > 0


def synthetic_images(out: Path, n_per_class: int, rng) -> dict:
    loc = LeakLocalizer(patch=CROP, stride=16)
    counts = {}
    made = {"crack": 0, "clean": 0}
    k = 0
    while min(made.values()) < n_per_class:
        k += 1
        geom = FlangeGeometry("S", [0, 0, 1], [0, -1, 0], [0, 0, 1])
        world = LeakWorld([geom])
        s = world.flanges["S"]
        for b in rng.choice(8, size=int(rng.integers(0, 3)), replace=False):
            s.angles[b] = float(rng.uniform(-0.25, 0.25))
        want = "crack" if made["crack"] <= made["clean"] else "clean"
        s.crack = float(rng.uniform(0.15, 0.8)) if want == "crack" else 0.0
        cam = _camera(geom, int(rng.integers(0, 3)), float(rng.uniform(1.0, 1.25)), rng)
        dust = float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.3)]))
        seed = int(rng.integers(0, 10_000))                       # a different crack shape every render
        img = render_flange(cam, s, dust=dust, glare=bool(rng.random() < 0.15), crack_seed=seed)
        face = loc.face_mask(cam, geom, img.shape[:2])
        crack = _crack_mask(cam, geom, s.crack, seed) if s.crack > 0 else np.zeros(img.shape[:2], bool)
        H, W = img.shape[:2]
        cover = {(y, x): face[y:y + CROP, x:x + CROP].mean()
                 for y in range(0, H - CROP + 1, 8) for x in range(0, W - CROP + 1, 8)}
        cand = [c for c, v in cover.items() if v >= 0.9]
        if want == "clean" and rng.random() < 0.5:
            # hard negatives: windows that include bore, rim, bolt or stripe edges, so "edge" does not mean "crack"
            cand = [c for c, v in cover.items() if 0.3 <= v < 0.9]
        rng.shuffle(cand)
        taken = 0
        for y, x in cand:
            if made[want] >= n_per_class or taken >= 6:
                break
            frac = crack[y:y + CROP, x:x + CROP].mean()
            if want == "crack" and frac < 0.015:
                continue
            i = made[want]
            split = _split_of(i, n_per_class)
            d = out / split / want
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img[y:y + CROP, x:x + CROP]).resize((128, 128), Image.BILINEAR).save(
                d / f"synth_{k:05d}_{y}_{x}.jpg", quality=92)
            counts[(split, want)] = counts.get((split, want), 0) + 1
            made[want] += 1
            taken += 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="concrete_crack_images data/*.zip")
    ap.add_argument("--out", default="data/leak_v1")
    ap.add_argument("--n-real", type=int, default=1200, help="per class")
    ap.add_argument("--n-synth", type=int, default=1200, help="per class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    real = real_images(a.zip, out, a.n_real, rng)
    synth = synthetic_images(out, a.n_synth, rng)
    for name, c in (("real", real), ("synth", synth)):
        print(name, {f"{s}/{cl}": n for (s, cl), n in sorted(c.items())})


if __name__ == "__main__":
    main()

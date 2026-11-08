"""debris_v2 = debris_v1 + hard negatives mined from the rig's own clean renders (standard hard-negative mining).

debris_v1 taught "an object lying on a deck". On a real rig deck that also fires on equipment that belongs there:
fire extinguishers, gas bottles, pipe supports, valve wheels (13 false alarms on the held-out deck render). Here the
v1 model is run over clean renders of THIS rig (no debris in them), every window it fires on becomes a "clean"
example, and debris objects are pasted onto other crops of the same renders, so the positives look like "a can on
this deck" and the negatives like "the equipment on this deck".

Renders, split by file (as in detection/synth_sim.py):
  train  full_rig_on_deck, full_rig_process_area, full_rig_helideck, full_rig_sea_level, preview_hero,
         preview_overview, preview_repair_closeup
  val    full_rig_topside
  test   (none) - preview_deck_end.png is kept out entirely: it is the held-out view the robot mock and the tests use.

    python -m detection.mine_debris_negatives --v1-ckpt runs/debris_detector/mnv4s_lora --v1 data/debris_v1 \
        --trash <MRS-Trash-Detection dir> --out data/debris_v2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from detection.prepare_rig_data import SPLITS, draw_can, drop_on, object_alpha, tight_crop, yolo_boxes
from detection.rig_vision import WindowDetector
from detection.synth_sim import RENDERS, camera

MINE = {"full_rig_on_deck.png": "train", "full_rig_process_area.png": "train", "full_rig_helideck.png": "train",
        "full_rig_sea_level.png": "train", "preview_hero.png": "train", "preview_overview.png": "train",
        "preview_repair_closeup.png": "train", "full_rig_topside.png": "val"}
HELD_OUT = "preview_deck_end.png"
SCALES = ((640, 360), (960, 540), (1280, 720))


def sim_look(arr, rng) -> Image.Image:
    a = np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((224, 224)), np.float32)
    return camera(a, rng)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-ckpt", required=True)
    ap.add_argument("--v1", default="data/debris_v1")
    ap.add_argument("--trash", required=True)
    ap.add_argument("--out", default="data/debris_v2")
    ap.add_argument("--min-p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(a.v1, out)
    det = WindowDetector.from_checkpoint(a.v1_ckpt, sizes=(64, 96, 128, 160, 224), stride_frac=0.33, threshold=a.min_p)

    objects = {s: [] for s in SPLITS}
    trash = Path(a.trash)
    for split in SPLITS:
        for img_path in sorted((trash / "images" / split).glob("*.jpg")):
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            for _, *b in yolo_boxes(trash / "labels" / split / (img_path.stem + ".txt"), W, H):
                if b[2] - b[0] >= 24 and b[3] - b[1] >= 24:
                    objects[split].append(object_alpha(tight_crop(im, b)))

    counts = {}
    for name, split in MINE.items():
        full = Image.open(RENDERS / name).convert("RGB")
        for sw, sh in SCALES:
            img = np.asarray(full.resize((sw, sh), Image.BILINEAR))
            r = det.detect(img)
            wins = [h.box for h in r.hits]
            # hard negatives: every window the v1 model fires on (the render has no debris in it)
            for x0, y0, x1, y1 in wins:
                for k in range(2):
                    sim_look(img[y0:y1, x0:x1], rng).resize((128, 128)).save(
                        out / split / "clean" / f"synth_hard_{Path(name).stem}_{sw}_{x0}_{y0}_{k}.jpg", quality=92)
                    counts[(split, "clean")] = counts.get((split, "clean"), 0) + 1
            # matching positives: a can or a real object dropped onto crops of the same render
            n_pos = 2 * len(wins)
            H, W = img.shape[:2]
            for i in range(n_pos):
                s = int(rng.integers(64, min(224, H // 2)))
                y = int(rng.integers(H // 3, H - s))         # lower part of the frame: the deck
                x = int(rng.integers(0, W - s))
                bg = img[y:y + s, x:x + s].astype(np.float32)
                if rng.random() < 0.5 or not objects[split]:
                    obj = draw_can(rng, 128)
                    obj = obj.crop(obj.getbbox())
                else:
                    obj = objects[split][int(rng.integers(len(objects[split])))]
                pos = drop_on(np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((224, 224)), np.float32), obj,
                              rng, rng.uniform(0.2, 0.5))
                sim_look(pos, rng).resize((128, 128)).save(
                    out / split / "debris" / f"synth_onrig_{Path(name).stem}_{sw}_{i}.jpg", quality=92)
                counts[(split, "debris")] = counts.get((split, "debris"), 0) + 1
            print(f"{name} @{sw}x{sh}: {len(wins)} false alarms mined -> {split}")
    for k in sorted(counts):
        print("added", "/".join(k), counts[k])


if __name__ == "__main__":
    main()

"""Build the datasets for two more rig detectors, with the same layout and process as the rust and crack sets.

  debris   loose objects on the deck (cans, bottles, packaging, tools left behind): a trip hazard and, offshore,
           dropped-object / foreign-object debris (FOD) that can blow into equipment or overboard.
  person   a person in the camera view, so the robot's "pause when someone is near" check comes from its own camera
           and not only from the simulator's ground truth.

Layout written:  <out>/{train,val,test}/{clean,<task>}/*.jpg      (label 1 = debris / person)

Real photos (file names start with ``real_``):
  debris   positives: object boxes from Hugging Face ``alexNova/MRS-Trash-Detection`` (Apache-2.0; 10 waste classes:
           metal cans, plastic and glass bottles, packaging, batteries, cables, ...). The dataset's own train/val/test
           folders are kept. Negatives: clean concrete photos (``Negative`` folder of the CC BY 4.0
           concrete_crack_images zip used for the crack model).
  person   positives: ``person`` boxes from ``VincentGOURBIN/ppe-detection`` (CC BY 4.0; workers on industrial and
           building sites, hard hats and vests). Negatives: background windows of the same photos that do not touch
           any box. Roboflow saved 3 augmented copies of each source photo (``<stem>.rf.<hash>``), so the split is made
           per source photo: all copies of one photo land in the same split.

Synthetic, sim-domain crops (file names start with ``synth_``), like rust_v2:
  * backgrounds are the rig scene's own deck textures (sim/assets/third_party/textures), cut into a 4x4 grid of
    regions with each region in exactly one split (as in detection/synth_sim.py), and crops of the clean
    full_rig_* renders (split by render file, as in synth_sim);
  * debris positives: a real object crop from the same split, or a drawn drink can, dropped onto a deck crop with a
    contact shadow, at the size it has in the 640x360 wrist camera at 1-3 m;
  * debris hard negatives: deck crops with painted lines, weld seams, plate joints, bolts and drain covers, so
    "something on the deck" is not just "an edge";
  * person positives: a real person crop from the same split composited onto a rig render or deck crop;
  * every crop goes through synth_sim's camera() (exposure, white balance, blur, noise, JPEG).

    python -m detection.prepare_rig_data --task debris --trash <MRS-Trash-Detection dir> \
        --concrete <concrete_crack_images>/data/test.zip --out data/debris_v1
    python -m detection.prepare_rig_data --task person --ppe <ppe-detection dir> --out data/person_v1
"""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from detection.synth_sim import RENDERS, TEX, TRAIN_RENDERS, camera, load_tex, region_splits

SIZE = 128                              # saved crop size; the trainer resizes to 224
SPLITS = ("train", "val", "test")
DECK_TEX = ["DiamondPlate001", "DiamondPlate006A", "DiamondPlate006C", "DiamondPlate008C", "MetalPlates001",
            "MetalPlates010", "MetalWalkway004", "MetalWalkway007", "MetalWalkway010", "MetalWalkway014",
            "Concrete034", "Concrete042C", "Asphalt031", "PaintedMetal006", "PaintedMetal009", "PaintedMetal016",
            "metal_plate", "metal_plate_02", "blue_metal_plate", "concrete_floor_03", "concrete_floor_damaged_01",
            "rubberized_track", "Rubber004"]
CAN_COLOURS = [(200, 20, 30), (20, 60, 160), (230, 230, 235), (30, 140, 60), (240, 200, 30), (20, 20, 25),
               (170, 175, 180), (240, 120, 20)]


def split_of_key(key: str, fracs=(0.70, 0.15, 0.15)) -> str:
    """Stable split from a string key (same key -> same split)."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < fracs[0] else ("val" if h < fracs[0] + fracs[1] else "test")


def yolo_boxes(label_path: Path, W: int, H: int, keep=None):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, x, y, w, h = int(p[0]), *map(float, p[1:])
        if keep is not None and c not in keep:
            continue
        out.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return out


def square_crop(im: Image.Image, box, pad=0.12) -> Image.Image:
    """Square-ish crop around a box, padded, shifted to stay inside the image (no black fill: a model must not
    learn "black border = object")."""
    W, H = im.size
    x0, y0, x1, y1 = box
    s = min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(0.0, cx - s / 2), W - s)
    top = min(max(0.0, cy - s / 2), H - s)
    return im.crop((int(left), int(top), int(left + s), int(top + s)))


def tight_crop(im: Image.Image, box) -> Image.Image:
    return im.crop(tuple(int(v) for v in box))


class Writer:
    def __init__(self, out: Path, positive: str):
        self.out, self.positive = out, positive
        self.counts: dict = {}

    def save(self, img, split: str, label: int, name: str, rng=None):
        cname = self.positive if label else "clean"
        d = self.out / split / cname
        d.mkdir(parents=True, exist_ok=True)
        im = img if isinstance(img, Image.Image) else Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        im.save(d / f"{name}.jpg", quality=92)
        key = (split, cname, name.split("_")[0])
        self.counts[key] = self.counts.get(key, 0) + 1


# ------------------------------------------------------------------------------------------ deck backgrounds
class Deck:
    """Deck texture crops with a leak-free region split (synth_sim's 4x4 grid)."""

    def __init__(self, rng):
        self.tex, self.regions = {}, {}
        for name in DECK_TEX:
            if (TEX / name / "diff.jpg").exists():
                self.tex[name] = load_tex(name)
                self.regions[name] = region_splits(rng)
        self.renders = {f: np.asarray(Image.open(RENDERS / f).convert("RGB"), np.float32)
                        for f in TRAIN_RENDERS if (RENDERS / f).exists()}

    def crop(self, split: str, rng, size: int = SIZE) -> np.ndarray:
        name = list(self.tex)[int(rng.integers(len(self.tex)))]
        tex = self.tex[name]
        regs = [r for r, s in self.regions[name].items() if s == split]
        h, w = tex.shape[:2]
        rh, rw = h // 4, w // 4
        ry, rx = divmod(int(regs[int(rng.integers(len(regs)))]), 4)
        s = int(rng.integers(min(96, rh), min(rh, rw) + 1))       # 96-256 px of a 1024 texture
        y = ry * rh + int(rng.integers(0, rh - s + 1))
        x = rx * rw + int(rng.integers(0, rw - s + 1))
        c = tex[y:y + s, x:x + s]
        c = np.asarray(Image.fromarray(c.astype(np.uint8)).resize((size, size), Image.BILINEAR), np.float32)
        light = 1 + rng.uniform(-0.3, 0.3) * np.linspace(-1, 1, size)[None, :, None]
        return np.clip(c * light * rng.uniform(0.6, 1.2), 0, 255)

    def render_crop(self, split: str, rng, size: int = SIZE):
        files = [f for f, s in TRAIN_RENDERS.items() if s == split and f in self.renders]
        if not files:
            return None
        r = self.renders[files[int(rng.integers(len(files)))]]
        H, W = r.shape[:2]
        s = int(rng.integers(64, min(H, W) // 3))
        y, x = int(rng.integers(0, H - s)), int(rng.integers(0, W - s))
        return np.asarray(Image.fromarray(r[y:y + s, x:x + s].astype(np.uint8)).resize((size, size), Image.BILINEAR),
                          np.float32)


def deck_features(img: np.ndarray, rng) -> np.ndarray:
    """Hard negatives: things that belong on a deck (painted lines, seams, bolts, drains, shadows)."""
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    S = im.size[0]
    for _ in range(int(rng.integers(1, 4))):
        k = rng.random()
        if k < 0.3:                                   # painted walkway line (yellow / white / red)
            col = [(220, 180, 20), (230, 230, 225), (190, 40, 30)][int(rng.integers(3))]
            w = int(rng.integers(6, 22))
            o = int(rng.integers(0, S))
            d.line([(o, -5), (o + int(rng.integers(-20, 20)), S + 5)] if rng.random() < 0.5 else
                   [(-5, o), (S + 5, o + int(rng.integers(-20, 20)))], fill=col, width=w)
        elif k < 0.5:                                 # plate joint / weld seam
            o = int(rng.integers(0, S))
            d.line([(o, -5), (o, S + 5)] if rng.random() < 0.5 else [(-5, o), (S + 5, o)],
                   fill=tuple(int(v) for v in rng.integers(20, 70, 3)), width=int(rng.integers(1, 4)))
        elif k < 0.7:                                 # flush bolt heads in a row
            y = int(rng.integers(8, S - 8))
            for x in range(int(rng.integers(4, 16)), S, int(rng.integers(18, 40))):
                r = int(rng.integers(2, 5))
                d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(60, 140, 3)))
        elif k < 0.85:                                # flush drain cover / hatch outline
            r = int(rng.integers(S // 6, S // 3))
            cx, cy = (int(v) for v in rng.integers(r, S - r, 2))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(35, 35, 38), width=3)
            for i in range(-r + 6, r - 4, 7):
                d.line([(cx + i, cy - int((r * r - i * i) ** 0.5) + 4), (cx + i, cy + int((r * r - i * i) ** 0.5) - 4)],
                       fill=(45, 45, 48), width=2)
        else:                                         # soft shadow from rig structure
            m = Image.new("L", im.size, 0)
            ImageDraw.Draw(m).polygon([tuple(int(v) for v in rng.integers(-20, S + 20, 2)) for _ in range(4)], fill=110)
            m = m.filter(ImageFilter.GaussianBlur(6))
            im = Image.composite(Image.new("RGB", im.size, (0, 0, 0)), im, m)
            d = ImageDraw.Draw(im)
    return np.asarray(im, np.float32)


def draw_can(rng, size: int) -> Image.Image:
    """An RGBA drink can lying on its side or standing, seen from a downward-looking camera."""
    L, D = int(size * rng.uniform(0.55, 0.9)), 0
    D = int(L * rng.uniform(0.42, 0.55))
    col = np.array(CAN_COLOURS[int(rng.integers(len(CAN_COLOURS)))], np.float32)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    yy, xx = np.mgrid[:D, :L].astype(np.float32)
    u = (yy - D / 2) / (D / 2)                                  # across the can
    shade = 0.35 + 0.65 * np.clip(np.cos(np.arcsin(np.clip(u, -1, 1)) - 0.5), 0, 1)
    spec = 120 * np.exp(-((u + 0.35) / 0.12) ** 2)
    body = col[None, None] * shade[..., None] + spec[..., None]
    band = (xx > L * rng.uniform(0.2, 0.4)) & (xx < L * rng.uniform(0.55, 0.8))
    body[band] = body[band] * 0.4 + np.array([235, 235, 235]) * 0.6 * shade[band][..., None]
    body[:, : max(2, L // 14)] = 190 * shade[:, : max(2, L // 14), None]          # metal ends
    body[:, -max(2, L // 14):] = 170 * shade[:, -max(2, L // 14):, None]
    rgba = np.concatenate([np.clip(body, 0, 255), np.full((D, L, 1), 255, np.float32)], -1).astype(np.uint8)
    can = Image.fromarray(rgba, "RGBA")
    if rng.random() < 0.3:                                       # crushed: squeeze the middle
        can = can.transform(can.size, Image.QUAD, (0, 0, L * 0.1, D * 0.2, L * 0.9, D * 0.8, L, 0), Image.BILINEAR)
    can = can.rotate(float(rng.uniform(0, 360)), expand=True, resample=Image.BILINEAR)
    im.paste(can, ((size - can.size[0]) // 2, (size - can.size[1]) // 2), can)
    return im


def drop_on(bg: np.ndarray, obj: Image.Image, rng, frac) -> np.ndarray:
    """Composite an RGBA object onto a background crop with a soft contact shadow."""
    S = bg.shape[0]
    s = max(8, int(S * frac))
    o = obj.copy()
    o.thumbnail((s, s), Image.BILINEAR)
    base = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    x = int(rng.integers(0, max(1, S - o.size[0])))
    y = int(rng.integers(0, max(1, S - o.size[1])))
    sh = Image.new("L", base.size, 0)
    a = o.split()[-1].filter(ImageFilter.GaussianBlur(3))
    sh.paste(a.point(lambda v: v * 0.55), (x + int(s * 0.06), y + int(s * 0.08)))
    base = Image.composite(Image.new("RGBA", base.size, (0, 0, 0, 255)), base, sh)
    base.alpha_composite(o, (x, y))
    return np.asarray(base.convert("RGB"), np.float32)


def object_alpha(crop: Image.Image) -> Image.Image:
    """Rough cut-out of a real object crop: drop pixels close to the border colour (product photos often have
    plain backgrounds); keeps the whole crop if that would remove too much."""
    a = np.asarray(crop.convert("RGB"), np.float32)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bgc = np.median(border, 0)
    dist = np.linalg.norm(a - bgc, axis=-1)
    m = dist > 40
    if m.mean() < 0.25:
        m = np.ones_like(m)
    alpha = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1))
    out = crop.convert("RGBA")
    out.putalpha(alpha)
    return out


# ------------------------------------------------------------------------------------------------- debris
def build_debris(trash_dir: Path, concrete_zip: str, out: Path, rng, n_synth: int = 1400):
    w = Writer(out, "debris")
    deck = Deck(rng)
    objects = {s: [] for s in SPLITS}
    for split in SPLITS:
        for img_path in sorted((trash_dir / "images" / split).glob("*.jpg")):
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            for k, (_, x0, y0, x1, y1) in enumerate(yolo_boxes(trash_dir / "labels" / split / (img_path.stem + ".txt"),
                                                              W, H)):
                if (x1 - x0) < 24 or (y1 - y0) < 24:
                    continue
                w.save(square_crop(im, (x0, y0, x1, y1)), split, 1, f"real_{img_path.stem}_{k}")
                objects[split].append(tight_crop(im, (x0, y0, x1, y1)))
    n_real_pos = {s: w.counts.get((s, "debris", "real"), 0) for s in SPLITS}

    z = zipfile.ZipFile(concrete_zip)
    names = sorted(n for n in z.namelist() if "/Negative/" in n and n.lower().endswith(".jpg"))
    for split in SPLITS:                                       # as many real clean floor photos as real objects
        pool = [n for n in names if split_of_key(n) == split]
        for j in rng.choice(len(pool), size=min(n_real_pos[split], len(pool)), replace=False):
            n = pool[int(j)]
            w.save(Image.open(io.BytesIO(z.read(n))).convert("RGB"), split, 0, f"real_{Path(n).stem}")

    fr = {"train": 0.70, "val": 0.15, "test": 0.15}
    for split in SPLITS:
        n = int(n_synth * fr[split])
        for i in range(n):                                     # positives: something lying on the deck
            bg = deck.crop(split, rng)
            if rng.random() < 0.4:
                bg = deck_features(bg, rng)
            if rng.random() < 0.5 or not objects[split]:
                obj = draw_can(rng, 128)
            else:
                obj = object_alpha(objects[split][int(rng.integers(len(objects[split])))])
            img = drop_on(bg, obj, rng, rng.uniform(0.25, 0.65))
            w.save(camera(np.asarray(Image.fromarray(img.astype(np.uint8)).resize((224, 224)), np.float32), rng),
                   split, 1, f"synth_obj_{split}_{i}")
        for i in range(n):                                     # negatives: clean deck, half with deck features
            if rng.random() < 0.15 and (r := deck.render_crop(split, rng)) is not None:
                bg = r
            else:
                bg = deck.crop(split, rng)
                if rng.random() < 0.55:
                    bg = deck_features(bg, rng)
            w.save(camera(np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((224, 224)), np.float32), rng),
                   split, 0, f"synth_deck_{split}_{i}")
    return w.counts


# ------------------------------------------------------------------------------------------------- person
def ppe_source(stem: str) -> str:
    return stem.split(".rf.")[0]


def build_person(ppe_dir: Path, out: Path, rng, n_synth: int = 1000, max_real: int = 1500):
    w = Writer(out, "person")
    deck = Deck(rng)
    people = {s: [] for s in SPLITS}
    scraps = {s: [] for s in SPLITS}           # non-person photo crops, pasted like the people (see below)
    imgs = sorted(ppe_dir.glob("*/images/*.jpg"))
    rng.shuffle(imgs)
    pos = neg = 0
    for img_path in imgs:
        split = split_of_key(ppe_source(img_path.stem))
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        a = np.asarray(im)
        allb = yolo_boxes(img_path.parent.parent / "labels" / (img_path.stem + ".txt"), W, H)
        persons = [b for b in allb if b[0] == 3]
        for k, (_, x0, y0, x1, y1) in enumerate(persons):
            if pos >= max_real or (y1 - y0) < 40:
                continue
            w.save(square_crop(im, (x0, y0, x1, y1), pad=0.08), split, 1, f"real_{img_path.stem}_{k}")
            people[split].append(tight_crop(im, (x0, y0, x1, y1)))
            pos += 1
        for _ in range(16):                                    # background windows touching no box
            if neg >= pos:
                break
            s = int(rng.integers(min(W, H) // 5, min(W, H) // 2))
            x, y = int(rng.integers(0, W - s)), int(rng.integers(0, H - s))
            if any(not (x + s < bx0 or x > bx1 or y + s < by0 or y > by1) for _, bx0, by0, bx1, by1 in allb):
                continue
            c = a[y:y + s, x:x + s].astype(np.float32)
            if ((c.min(-1) > 245) | (c.max(-1) < 10)).mean() > 0.2:   # Roboflow's white / black padding
                continue
            w.save(Image.fromarray(c.astype(np.uint8)), split, 0, f"real_bg_{img_path.stem}_{x}_{y}")
            scraps[split].append(Image.fromarray(c.astype(np.uint8)))
            neg += 1

    fr = {"train": 0.70, "val": 0.15, "test": 0.15}
    for split in SPLITS:
        n = int(n_synth * fr[split])
        for i in range(n):                                     # positives: a person on the rig
            bg = deck.render_crop(split, rng) if rng.random() < 0.6 else None
            bg = deck.crop(split, rng) if bg is None else bg
            if not people[split]:
                continue
            p = people[split][int(rng.integers(len(people[split])))]
            img = drop_on(bg, p.convert("RGBA"), rng, rng.uniform(0.6, 0.95))
            w.save(camera(np.asarray(Image.fromarray(img.astype(np.uint8)).resize((224, 224)), np.float32), rng),
                   split, 1, f"synth_person_{split}_{i}")
        for i in range(n):                                     # negatives: rig structure, deck, railings
            bg = deck.render_crop(split, rng) if rng.random() < 0.6 else None
            bg = deck_features(deck.crop(split, rng), rng) if bg is None else bg
            if i % 2 == 0 and scraps[split]:
                # a pasted photo with NO person in it, composited exactly like the positives, so "a pasted
                # rectangle" is not a shortcut for "person"
                sc = scraps[split][int(rng.integers(len(scraps[split])))]
                h = rng.uniform(0.6, 0.95)
                sc = sc.resize((max(8, int(sc.size[0] * rng.uniform(0.35, 0.7))), sc.size[1]))
                bg = drop_on(bg, sc.convert("RGBA"), rng, h)
            w.save(camera(np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((224, 224)), np.float32), rng),
                   split, 0, f"synth_rig_{split}_{i}")
    return w.counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["debris", "person"], required=True)
    ap.add_argument("--trash", help="MRS-Trash-Detection dir (images/<split>, labels/<split>)")
    ap.add_argument("--concrete", help="concrete_crack_images data/*.zip (clean floor photos)")
    ap.add_argument("--ppe", help="ppe-detection dir (<split>/images, <split>/labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    counts = (build_debris(Path(a.trash), a.concrete, out, rng) if a.task == "debris"
              else build_person(Path(a.ppe), out, rng))
    for k in sorted(counts):
        print("/".join(k), counts[k])


if __name__ == "__main__":
    main()

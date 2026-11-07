from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "sim/assets/third_party/textures"
RENDERS = REPO / "sim/output"
TRAIN_RENDERS = {"full_rig_helideck.png": "train", "full_rig_on_deck.png": "train",
                 "full_rig_process_area.png": "train", "full_rig_sea_level.png": "train",
                 "full_rig_topside.png": "val", "full_rig_wide.png": "test"}
POS_TEX = {"rust_coarse_01": 450, "rusty_metal_02": 400, "metal_grate_rusty": 250}
NEG_TEX = {"metal_plate": 250, "blue_metal_plate": 250, "box_profile_metal_sheet": 250,
           "concrete_floor_damaged_01": 150}
PAINTS = {
    "yellow": (217, 153, 13), "blue": (40, 95, 150), "green": (60, 100, 55), "grey": (84, 89, 92),
    "orange_paint": (191, 77, 15), "white": (215, 215, 210), "red_paint": (160, 30, 25), "dark_grey": (50, 52, 55),
}
N_PAINT, N_COMPOSITE, N_RENDER, N_GREEN = 450, 400, 250, 300
OUT = 224
SPLIT_FRAC = {"train": 12, "val": 2, "test": 2}


def region_splits(rng) -> dict[int, str]:
    ids = rng.permutation(16)
    names = ["train"] * SPLIT_FRAC["train"] + ["val"] * SPLIT_FRAC["val"] + ["test"] * SPLIT_FRAC["test"]
    return {int(i): s for i, s in zip(ids, names)}


def tex_crop(tex: np.ndarray, region: int, rng) -> np.ndarray:
    h, w = tex.shape[:2]
    rh, rw = h // 4, w // 4
    ry, rx = divmod(region, 4)
    s = int(rng.integers(64, min(rh, rw) + 1))
    y = ry * rh + int(rng.integers(0, rh - s + 1))
    x = rx * rw + int(rng.integers(0, rw - s + 1))
    return tex[y:y + s, x:x + s]


def resize(a: np.ndarray, size=OUT) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.BILINEAR), np.float32)


def lowfreq(rng, size=OUT, cells=6) -> np.ndarray:
    n = rng.random((cells, cells)).astype(np.float32)
    return np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), np.float32) / 255


def paint_tex(rgb, rng) -> np.ndarray:
    base = np.array(rgb, np.float32) * rng.uniform(0.85, 1.15, 3)
    img = base[None, None] * (0.9 + 0.2 * lowfreq(rng, cells=int(rng.integers(3, 10))))[..., None]
    img += rng.normal(0, 3, img.shape)
    if rng.random() < 0.3:
        for _ in range(int(rng.integers(1, 4))):
            y, x = rng.integers(0, OUT, 2)
            img[max(0, y - 1):y + 1, max(0, x - 30):x + 30] *= 0.7
    return np.clip(img, 0, 255)


def blob_mask(rng, target_frac) -> np.ndarray:
    yy, xx = np.mgrid[:OUT, :OUT].astype(np.float32) / OUT
    cy, cx = rng.uniform(0.25, 0.75, 2)
    ry, rx = rng.uniform(0.2, 0.6, 2)
    d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    field = d + 0.9 * (lowfreq(rng, cells=int(rng.integers(4, 9))) - 0.5)
    thr = np.quantile(field, target_frac)
    m = (field <= thr).astype(np.float32)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 3))),
                   np.float32) / 255
    return m


def shade(img: np.ndarray, rng) -> np.ndarray:
    yy, xx = np.mgrid[:OUT, :OUT].astype(np.float32) / (OUT - 1) * 2 - 1
    ang = rng.choice([0.0, np.pi / 2, rng.uniform(0, np.pi)])
    u = np.cos(ang) * yy - np.sin(ang) * xx
    if rng.random() < 0.7:
        half = rng.uniform(0.6, 1.6)
        uc = np.clip(u / half, -1, 1)
        normal = np.arcsin(uc)
        light = rng.uniform(-1.0, 1.0)
        lam = np.clip(np.cos(normal - light), 0, 1)
        amb = rng.uniform(0.2, 0.5)
        sh = amb + (1 - amb) * lam
        spec = rng.uniform(0, 0.45) * np.exp(-((uc - np.sin(light) * 0.6) / rng.uniform(0.05, 0.2)) ** 2)
        out = img * sh[..., None] + 255 * spec[..., None]
        if half < 1.0:
            bg_col = [(200, 205, 212), (25, 55, 65), (110, 112, 115), (160, 165, 170)][int(rng.integers(4))]
            bg = np.array(bg_col, np.float32) * (0.85 + 0.3 * lowfreq(rng, cells=3))[..., None]
            inside = (np.abs(u) <= half)[..., None]
            out = np.where(inside, out, bg)
    else:
        g = 1 + rng.uniform(-0.35, 0.35) * u
        out = img * g[..., None]
    return out


def camera(img: np.ndarray, rng) -> np.ndarray:
    img = img * rng.uniform(0.6, 1.4) * rng.uniform(0.9, 1.1, 3)
    img = 255 * (np.clip(img, 0, 255) / 255) ** rng.uniform(0.8, 1.25)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    a = np.asarray(im, np.float32) + rng.normal(0, rng.uniform(0, 6), (OUT, OUT, 3))
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(50, 96)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def rust_fraction(a: np.ndarray) -> float:
    hsv = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("HSV"), np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return float(((h >= 10) & (h <= 45) & (s > 0.35) & (v > 0.15)).mean())


def load_tex(name):
    return np.asarray(Image.open(TEX / name / "diff.jpg").convert("RGB"), np.float32)


def generate(out: Path, seed=0):
    rng = np.random.default_rng(seed)
    rows, n = [], {"rust": 0, "clean": 0}

    def save(img: Image.Image, split, label, source, group):
        cls = "rust" if label else "clean"
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        rel = f"{split}/{cls}/synth_{source}_{len(rows):05d}.jpg"
        img.save(out / rel, quality=95)
        rows.append({"path": rel, "label": cls, "split": split, "source": f"synth_{source}", "group_id": group})
        n[cls] += 1

    texs, splits = {}, {}
    for name in [*POS_TEX, *NEG_TEX, "green_metal_rust"]:
        texs[name], splits[name] = load_tex(name), region_splits(rng)

    for name, count in [*POS_TEX.items(), *NEG_TEX.items()]:
        label = int(name in POS_TEX)
        for _ in range(count):
            r = int(rng.integers(16))
            img = camera(shade(resize(tex_crop(texs[name], r, rng)), rng), rng)
            save(img, splits[name][r], label, name, f"synth:{name}:{r}")

    kept = 0
    for _ in range(N_GREEN * 4):
        if kept >= N_GREEN:
            break
        r = int(rng.integers(16))
        c = tex_crop(texs["green_metal_rust"], r, rng)
        f = rust_fraction(c)
        if 0.03 <= f <= 0.15:
            continue
        kept += 1
        save(camera(shade(resize(c), rng), rng), splits["green_metal_rust"][r], int(f > 0.15),
             "green_metal_rust", f"synth:green_metal_rust:{r}")

    paint_split = {}
    pnames = list(PAINTS)
    for i, p in enumerate(pnames):
        for b in range(4):
            paint_split[(p, b)] = ["train", "train", "val", "test"][(i + b) % 4] if b else "train"
    for _ in range(N_PAINT):
        p, b = pnames[int(rng.integers(len(pnames)))], int(rng.integers(4))
        save(camera(shade(paint_tex(PAINTS[p], rng), rng), rng), paint_split[(p, b)], 0, f"paint_{p}",
             f"synth:paint:{p}:{b}")

    pos_names = list(POS_TEX)[:2]
    for _ in range(N_COMPOSITE):
        name = pos_names[int(rng.integers(2))]
        r = int(rng.integers(16))
        p = pnames[int(rng.integers(len(pnames)))]
        base = paint_tex(PAINTS[p], rng)
        blob = resize(tex_crop(texs[name], r, rng))
        m = blob_mask(rng, rng.uniform(0.18, 0.55))[..., None]
        img = camera(shade(base * (1 - m) + blob * m, rng), rng)
        save(img, splits[name][r], 1, f"blob_{name}", f"synth:{name}:{r}")

    renders = {k: np.asarray(Image.open(RENDERS / k).convert("RGB"), np.float32)
               for k in TRAIN_RENDERS if (RENDERS / k).exists()}
    keys = list(renders)
    for _ in range(N_RENDER if keys else 0):
        k = keys[int(rng.integers(len(keys)))]
        a = renders[k]
        s = int(rng.integers(96, 360))
        y, x = int(rng.integers(0, a.shape[0] - s)), int(rng.integers(0, a.shape[1] - s))
        save(camera(resize(a[y:y + s, x:x + s]), rng), TRAIN_RENDERS[k], 0, "render", f"synth:render:{k}")
    return rows, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="runs/data/rust_v1")
    ap.add_argument("--out", default="runs/data/rust_v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    v1, out = Path(a.v1), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(v1, out, ignore=shutil.ignore_patterns("stats.json"))
    rows, n = generate(out, a.seed)
    with open(out / "manifest.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source", "group_id"])
        w.writerows(rows)
    all_rows = list(csv.DictReader(open(out / "manifest.csv")))
    stats = {"total": len(all_rows), "synthetic": len(rows), "synthetic_per_class": n, "per_split_class": {},
             "per_source_class": {}}
    for r in all_rows:
        k = f"{r['split']}/{r['label']}"
        stats["per_split_class"][k] = stats["per_split_class"].get(k, 0) + 1
        k = f"{r['source']}/{r['label']}"
        stats["per_source_class"][k] = stats["per_source_class"].get(k, 0) + 1
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

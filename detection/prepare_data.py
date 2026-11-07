"""Build the binary rust/clean classification dataset on the `rust-data` Modal volume.

Output (see detection/DATA.md for the contract):
  /data/rust_v1/{train,val,test}/{rust,clean}/*.jpg
  /data/rust_v1/manifest.csv   path,label,split,source,group_id
  /data/rust_v1/stats.json

Run:
  .venv/bin/modal run detection/prepare_data.py                     # build on Modal
  .venv/bin/modal run detection/prepare_data.py --local-sample 20   # + copy 20/split/class to ./data/rust_v1_sample
"""
import os
import sys

import modal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from infra.common import DATA_DIR, DATA_VOL  # noqa: E402

app = modal.App("rust-prepare-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==0.26.2", "pillow==10.4.0", "imagehash==4.3.1", "numpy<2", "pyarrow==17.0.0"
).add_local_python_source("infra")

OUT_NAME = "rust_v1"
SHORT_SIDE = 256
SEED = 1234
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
HAMMING_MAX = 4

BINKHOA = "BinKhoaLe1812/Corrosion_Rust"     # MIT
RF100 = "Francesco/corrosion-bi3q3"          # CC BY 4.0 (RF100)
IAMCLOUD = "iamcloud/Corrosion_Dataset"      # Apache-2.0, all-positive Roboflow classification export


def _resize_short(img, short=SHORT_SIDE):
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def _iou_any(box, boxes):
    x0, y0, x1, y1 = box
    for bx0, by0, bx1, by1 in boxes:
        if min(x1, bx1) > max(x0, bx0) and min(y1, by1) > max(y0, by0):
            return True
    return False


@app.function(image=image, volumes={DATA_DIR: DATA_VOL}, cpu=8, memory=8192, timeout=3600)
def build(sample_n: int = 0):
    import collections
    import csv
    import hashlib
    import io
    import json
    import random
    import shutil

    import imagehash
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from PIL import Image

    rng = random.Random(SEED)
    items = []  # dict(img=PIL, label, source, group, parent_hash)

    # ---------- 1. BinKhoaLe1812/Corrosion_Rust ----------
    d = snapshot_download(BINKHOA, repo_type="dataset", allow_patterns=["Corrosion/*/*.jpg"], max_workers=16)
    n0 = len(items)
    for folder, label in [("rust", "rust"), ("no rust", "clean")]:
        p = os.path.join(d, "Corrosion", folder)
        for fn in sorted(os.listdir(p)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img = _resize_short(Image.open(os.path.join(p, fn)))
            except Exception as e:
                print("skip", fn, e)
                continue
            stem = fn.rsplit("_", 1)[0] if "_" in fn else fn.rsplit(".", 1)[0]
            items.append(dict(img=img, label=label, source="binkhoa", group=f"binkhoa:{label}:{stem}", full=img))
    print("binkhoa", len(items) - n0)

    # ---------- 2. Francesco/corrosion-bi3q3 (RF100 detection) ----------
    n0 = len(items)
    files = [f for f in list_repo_files(RF100, repo_type="dataset") if f.endswith(".parquet")]
    names = None
    for f in files:
        t = pq.read_table(hf_hub_download(RF100, f, repo_type="dataset"))
        for r in t.to_pylist():
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            W, H = img.size
            small = _resize_short(img)
            gid = f"rf100:{f.split('/')[-1].split('-')[0]}:{r['image_id']}"
            boxes = []
            for bb, cat in zip(r["objects"]["bbox"], r["objects"]["category"]):
                if cat not in (0, 2):  # 0=corrosion-0 (supercat), 2=corrosion; skip Slippage/crack
                    continue
                x, y, w, h = bb
                boxes.append((x, y, x + w, y + h))
            # positives: padded crops of corrosion boxes
            for x0, y0, x1, y1 in boxes:
                w, h = x1 - x0, y1 - y0
                if w * h < 16 * 16:
                    continue
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                pw, ph = max(48, w * 1.2), max(48, h * 1.2)
                c = (max(0, cx - pw / 2), max(0, cy - ph / 2), min(W, cx + pw / 2), min(H, cy + ph / 2))
                if c[2] - c[0] < 32 or c[3] - c[1] < 32:
                    continue
                items.append(dict(img=img.crop(tuple(int(round(v)) for v in c)), label="rust",
                                  source="rf100", group=gid, full=small))
            # negatives: random crops with zero overlap with any corrosion box
            n_neg, tries = 0, 0
            want = 2 if boxes else 3
            while n_neg < want and tries < 60:
                tries += 1
                s = rng.randint(96, min(320, W, H))
                x0, y0 = rng.randint(0, W - s), rng.randint(0, H - s)
                c = (x0, y0, x0 + s, y0 + s)
                if _iou_any(c, boxes):
                    continue
                items.append(dict(img=img.crop(c), label="clean", source="rf100", group=gid, full=small))
                n_neg += 1
    print("rf100", len(items) - n0)

    # ---------- 3. iamcloud/Corrosion_Dataset (all positives; 1 file per original stem) ----------
    n0 = len(items)
    fs = [f for f in list_repo_files(IAMCLOUD, repo_type="dataset") if f.endswith(".jpg")]
    by_stem = {}
    for f in sorted(fs):
        by_stem.setdefault(f.split("/")[-1].split(".rf.")[0].lower(), f)
    d = snapshot_download(IAMCLOUD, repo_type="dataset", allow_patterns=list(by_stem.values()), max_workers=16)
    for stem, f in sorted(by_stem.items()):
        try:
            img = _resize_short(Image.open(os.path.join(d, f)))
        except Exception as e:
            print("skip", f, e)
            continue
        items.append(dict(img=img, label="rust", source="iamcloud", group=f"iamcloud:{stem}", full=img))
    print("iamcloud", len(items) - n0)

    # ---------- resize + phash dedupe (union-find over output crops AND parent images) ----------
    for it in items:
        if min(it["img"].size) != SHORT_SIDE:
            it["img"] = _resize_short(it["img"])
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def hashes(imgs):
        return np.array([int(str(imagehash.phash(im)), 16) for im in imgs], dtype=np.uint64)

    popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def near_pairs(h):
        hb = h.view(np.uint8).reshape(-1, 8)
        out = []
        for i in range(len(h)):
            dist = popcnt[np.bitwise_xor(hb[i + 1:], hb[i])].sum(1)
            for j in np.nonzero(dist <= HAMMING_MAX)[0]:
                out.append((i, i + 1 + int(j)))
        return out

    # parent-level: one entry per distinct full image
    fulls, full_groups, seen = [], [], {}
    for it in items:
        k = id(it["full"])
        if k not in seen:
            seen[k] = len(fulls)
            fulls.append(it["full"])
            full_groups.append(it["group"])
    hp = hashes(fulls)
    pp = near_pairs(hp)
    for i, j in pp:
        union(full_groups[i], full_groups[j])
    hi = hashes([it["img"] for it in items])
    ip = near_pairs(hi)
    for i, j in ip:
        union(items[i]["group"], items[j]["group"])
    for it in items:
        it["group"] = find(it["group"])
    gsize = collections.Counter(it["group"] for it in items)
    print(f"phash: {len(pp)} near-dup parent pairs, {len(ip)} near-dup item pairs; "
          f"{len(gsize)} groups, largest {gsize.most_common(3)}")

    # exact-duplicate item removal within same label (keep first)
    seen_h, keep = set(), []
    for it, h in zip(items, hi):
        k = (int(h), it["label"])
        if k in seen_h:
            continue
        seen_h.add(k)
        keep.append(it)
    print(f"dropped {len(items) - len(keep)} exact-phash duplicate items")
    items = keep

    # ---------- split by group ----------
    groups = sorted(set(it["group"] for it in items))
    rng.shuffle(groups)
    gcount = collections.Counter(it["group"] for it in items)
    total = sum(gcount.values())
    split_of, acc = {}, 0
    for g in groups:
        frac = acc / total
        split_of[g] = "train" if frac < SPLIT_FRAC["train"] else ("val" if frac < SPLIT_FRAC["train"] + SPLIT_FRAC["val"] else "test")
        acc += gcount[g]

    # ---------- write ----------
    root = os.path.join(DATA_DIR, OUT_NAME)
    if os.path.exists(root):
        shutil.rmtree(root)
    for s in SPLIT_FRAC:
        for c in ("rust", "clean"):
            os.makedirs(os.path.join(root, s, c), exist_ok=True)
    rows = []
    for idx, it in enumerate(items):
        s = split_of[it["group"]]
        gtag = hashlib.md5(it["group"].encode()).hexdigest()[:8]
        rel = f"{s}/{it['label']}/{it['source']}_{gtag}_{idx:05d}.jpg"
        it["img"].save(os.path.join(root, rel), quality=92)
        rows.append(dict(path=rel, label=it["label"], split=s, source=it["source"], group_id=it["group"]))
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source", "group_id"])
        w.writeheader()
        w.writerows(rows)

    stats = {"total": len(rows), "n_groups": len(set(r["group_id"] for r in rows)),
             "per_split_class": {}, "per_source_class": {}, "per_split_class_source": {}}
    for r in rows:
        for key, k in [("per_split_class", f"{r['split']}/{r['label']}"),
                       ("per_source_class", f"{r['source']}/{r['label']}"),
                       ("per_split_class_source", f"{r['split']}/{r['label']}/{r['source']}")]:
            stats[key][k] = stats[key].get(k, 0) + 1
    for key in list(stats):
        if isinstance(stats[key], dict):
            stats[key] = dict(sorted(stats[key].items()))
    # leakage check
    gsplits = collections.defaultdict(set)
    for r in rows:
        gsplits[r["group_id"]].add(r["split"])
    assert all(len(v) == 1 for v in gsplits.values()), "group leakage across splits"
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    DATA_VOL.commit()
    print(json.dumps(stats, indent=2))

    sample = []
    if sample_n:
        per = collections.defaultdict(list)
        for r in rows:
            per[(r["split"], r["label"])].append(r)
        for k, v in per.items():
            for r in v[:sample_n]:
                with open(os.path.join(root, r["path"]), "rb") as f:
                    sample.append((r, f.read()))
    return stats, sample


@app.local_entrypoint()
def main(local_sample: int = 0):
    import csv
    import json

    stats, sample = build.remote(local_sample)
    if local_sample:
        out = os.path.join(os.getcwd(), "data", "rust_v1_sample")
        rows = []
        for r, b in sample:
            p = os.path.join(out, r["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b)
            rows.append(r)
        with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source", "group_id"])
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(out, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"wrote {len(rows)} sample images to {out}")

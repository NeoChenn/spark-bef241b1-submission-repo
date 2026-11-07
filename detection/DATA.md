# rust_v1 — binary rust / clean dataset

Built by `detection/prepare_data.py` (Modal, CPU only). Rebuild (about 3 min, deletes and rewrites `/data/rust_v1`):

```
.venv/bin/modal run detection/prepare_data.py                     # volume only
.venv/bin/modal run detection/prepare_data.py --local-sample 20   # + 20/split/class to ./data/rust_v1_sample (gitignored)
```

## Layout (Modal volume `rust-data`, mounted at `/data`)

```
/data/rust_v1/{train,val,test}/{rust,clean}/*.jpg   RGB jpg, shorter side = 256
/data/rust_v1/manifest.csv                          path,label,split,source,group_id  (path relative to rust_v1/)
/data/rust_v1/stats.json                            counts per split/class/source
```

Splits are about 70/15/15 **by group_id**, never by crop, and the build asserts that no group is in more than one split.

## Sources and licences

| source tag | HF dataset | licence | what we use |
|---|---|---|---|
| `binkhoa` | `BinKhoaLe1812/Corrosion_Rust` | MIT | `Corrosion/rust` → rust, `Corrosion/no rust` → clean (the clean set includes some concrete-crack photos). Group = class + filename stem before the random `_xxxx` suffix. |
| `rf100` | `Francesco/corrosion-bi3q3` (Roboflow-100) | CC BY 4.0 | Positives: crops of `corrosion` / `corrosion-0` boxes (COCO xywh), padded 10% per side, min 48 px, boxes < 16x16 skipped. Negatives: random square crops (96-320 px) that do not overlap any corrosion box: 2 per image, or 3 if the image has none. Crack/slippage areas can end up in negatives. Group = parent image. |
| `iamcloud` | `iamcloud/Corrosion_Dataset` | Apache-2.0 | Roboflow classification export, all positive. Only one file per original stem (the part before `.rf.`), so augmented copies are dropped. |

## Dedupe

Perceptual hash (imagehash phash, Hamming <= 4), run with union-find over (a) the parent images and (b) every output crop, across all sources. Near-duplicates are merged into one group_id, so they land in the same split. Items whose phash matches exactly and that have the same label are dropped (17 removed).

## Counts (build from 2026-11-07)

| split | rust | clean |
|---|---|---|
| train | 1537 | 2667 |
| val | 304 | 505 |
| test | 362 | 519 |
| **total** | 2203 | 3691 |

5894 images in 2186 groups. Rust is about 37% of the data (rust:clean ≈ 1:1.7), so use class weights or a balanced sampler.
By source: binkhoa 319 rust / 376 clean; rf100 906 rust / 3315 clean; iamcloud 978 rust.

## Known caveats

- RF100 contains augmented duplicates, and phash chaining merged them into two large groups (718 and 657 items). This is correct for preventing leakage, but it makes the split ratios slightly uneven.
- The rf100 clean crops come from the same inspection photos as the rf100 rust crops, which makes them hard negatives, but they may contain small unlabelled rust.
- There is no generic clean painted-steel/pipe negative source yet. Expect false positives on orange or brown paint in sim.

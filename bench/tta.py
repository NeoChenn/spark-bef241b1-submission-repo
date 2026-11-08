from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bench.detect import binary_metrics, load_dir, load_heldout
from detection.export import load_plain
from detection.tta import TTA
from policy.storm import Lens

SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def degrade(img: np.ndarray, s: float, seed: int) -> np.ndarray:
    if s <= 0:
        return img
    rng = np.random.default_rng(seed)
    lens = Lens(seed=seed, rate=40.0 * s)
    lens.rain(1.0, 1.0)
    f = lens.apply(img, visibility_loss=0.6 * s).astype(np.float32)
    h, w = f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.35 * min(h, w)) ** 2
    g = s * (170.0 * np.exp(-r2) + 50.0)
    tint = np.array([1.0, 0.93, 0.8], np.float32) * rng.uniform(0.9, 1.1)
    f = f * (1 - 0.3 * s) + g[..., None] * tint
    return np.clip(f, 0, 255).astype(np.uint8)


def tensor(imgs, meta):
    from PIL import Image
    mean = np.array(meta["mean"], np.float32)[:, None, None]
    std = np.array(meta["std"], np.float32)[:, None, None]
    xs = [(np.asarray(Image.fromarray(i).resize((224, 224), Image.BILINEAR), np.float32).transpose(2, 0, 1) / 255.0
           - mean) / std for i in imgs]
    return torch.from_numpy(np.stack(xs))


GATE = 0.4


def scores(model, meta, imgs, mode, batch, prior=0.0, gate=None, seed=0):
    order = np.random.default_rng(seed).permutation(len(imgs))
    t = TTA(model, mode=mode, prior=prior, lr=1e-3, temperature=meta["temperature"], gate=gate)
    out = np.zeros(len(imgs))
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        z = t(tensor([imgs[i] for i in idx], meta)).reshape(-1).detach().numpy()
        out[idx] = 1 / (1 + np.exp(-z / meta["temperature"]))
    t.reset()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/bench/models/detector_w/model.pt")
    ap.add_argument("--data", default="runs/bench/data/rust_sim_test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="runs_sample/novelty/tta")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(6)
    t0 = time.time()
    model, meta = load_plain(a.model)
    thr = meta["threshold"]
    items = load_dir(a.data) + load_heldout()
    imgs0 = [np.asarray(im.convert("RGB")) for im, _, _ in items]
    y = np.array([lab for _, lab, _ in items])
    arms = {"off": ("off", 0.0, None), "bn": ("bn", 0.0, None), "bn_prior": ("bn", 0.5, None),
            "gated": ("bn", 0.0, GATE)}
    res = {k: {} for k in arms}
    for s in SEVERITIES:
        imgs = [degrade(im, s, 1000 + i) for i, im in enumerate(imgs0)]
        for k, (mode, prior, gate) in arms.items():
            p = scores(model, meta, imgs, mode, a.batch, prior, gate)
            m = binary_metrics(y, p, thr, n_boot=300)
            res[k][str(s)] = {q: round(float(m[q]["mean"]), 4) for q in ("f1", "precision", "recall", "auroc")}
            print(s, k, res[k][str(s)], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"model": "detector_w MobileNetV4-conv-small full-FT (bench/detect arm), threshold from meta",
           "data": f"{len(items)} held-out crops (rust_sim_test {len(items) - 13} + sim_heldout 13), "
                   f"{int(y.sum())} rust", "degradation": "policy/storm.py Lens rain drops + visibility loss 0.6*s, "
                   "plus sun/flood-light glare bloom and washout, seeded per crop",
           "batch": a.batch, "stream": "shuffled stream, per-batch adaptation, reset per severity",
           "arms": {"off": "plain inference", "bn": "BN stats from the test batch",
                    "bn_prior": "BN stats blended 50/50 with source stats",
                    "gated": f"BN-adapt only when the batch's first-BN feature shift (symmetric Gaussian KL) > {GATE}; "
                             "clean batches score 0.09-0.20, severity 0.25 scores 0.6-1.0"},
           "results": res, "wall_s": round(time.time() - t0, 1)}
    (out / "result.json").write_text(json.dumps(doc, indent=1))
    chart(res, out / "chart.png")


def chart(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fg, ax = plt.subplots(figsize=(5.2, 3.2))
    cols = {"off": "#8a94a6", "bn": "#5b8def", "bn_prior": "#b07cf0", "gated": "#2fbf71"}
    labels = {"off": "no adaptation", "bn": "BN-adapt (always)", "bn_prior": "BN-adapt, prior 0.5",
              "gated": "shift-gated BN-adapt"}
    for k, d in res.items():
        xs = [float(s) for s in d]
        ax.plot(xs, [d[s]["f1"] for s in d], "-o", color=cols[k], label=labels[k], lw=2, ms=4)
    ax.set_xlabel("rain + glare severity", color="#ccc")
    ax.set_ylabel("F1 (held-out crops)", color="#ccc")
    ax.set_title("Rust detector: test-time adaptation under storm lens", color="#eee", fontsize=10)
    ax.tick_params(colors="#ccc")
    for s in ax.spines.values():
        s.set_color("#666")
    ax.set_facecolor("none")
    lg = ax.legend(frameon=False, fontsize=8)
    for t in lg.get_texts():
        t.set_color("#ddd")
    fg.tight_layout()
    fg.savefig(path, transparent=True, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())

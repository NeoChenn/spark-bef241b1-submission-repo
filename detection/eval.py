"""Metrics, threshold selection, temperature scaling and the unsure band.

All functions take raw logits (numpy, shape [N]) and integer labels (1 = rust).
Calibrated probability: p = sigmoid(logit / T).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score)

TARGET_RECALL = 0.95
UNSURE_BAND = (0.35, 0.65)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def metrics_at(logits, labels, thr_logit: float = 0.0) -> dict:
    logits, labels = np.asarray(logits, np.float64), np.asarray(labels).astype(int)
    pred = (logits >= thr_logit).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    two = len(np.unique(labels)) == 2
    return {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "auroc": float(roc_auc_score(labels, logits)) if two else float("nan"),
        "pr_auc": float(average_precision_score(labels, logits)) if two else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold(logits, labels, target_recall: float = TARGET_RECALL) -> float:
    """Highest logit threshold whose recall on these samples is >= target (max precision)."""
    prec, rec, thr = precision_recall_curve(np.asarray(labels).astype(int), np.asarray(logits, np.float64))
    # rec[i] corresponds to thr[i] for i < len(thr); recall is non-increasing in threshold.
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(np.min(logits))
    return float(thr[ok[-1]])


def fit_temperature(logits, labels, iters: int = 200) -> float:
    z = torch.tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.tensor(np.asarray(labels), dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def ece(probs, labels, n_bins: int = 15) -> float:
    """Binary ECE on P(rust) (reliability of the positive-class probability)."""
    probs, labels = np.asarray(probs, np.float64), np.asarray(labels, np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def unsure_fraction(logits, temperature: float, band=UNSURE_BAND) -> float:
    p = sigmoid(np.asarray(logits) / temperature)
    return float(((p >= band[0]) & (p <= band[1])).mean())


def full_report(split_logits: dict, split_labels: dict, val_domains=None) -> dict:
    """split_* keyed by 'train'/'val'/'test'. Threshold + temperature are fitted on val only.
    val_domains: optional list of boolean masks over val (e.g. real vs synthetic); the threshold is then the
    lowest per-domain threshold so recall >= TARGET_RECALL holds in every domain, not just on the mix."""
    vz, vy = split_logits["val"], split_labels["val"]
    thr = pick_threshold(vz, vy)
    if val_domains:
        thr = min(pick_threshold(vz[m], vy[m]) for m in val_domains if m.any())
    T = fit_temperature(vz, vy)
    rep = {"threshold_logit": thr, "threshold_prob": float(sigmoid(thr / T)), "temperature": T,
           "target_recall": TARGET_RECALL, "unsure_band": list(UNSURE_BAND), "splits": {}}
    for s in split_logits:
        z, y = split_logits[s], split_labels[s]
        rep["splits"][s] = {
            "at_0.5": metrics_at(z, y, 0.0),
            "at_thr": metrics_at(z, y, thr),
            "ece_raw": ece(sigmoid(z), y),
            "ece_cal": ece(sigmoid(np.asarray(z) / T), y),
            "unsure_frac": unsure_fraction(z, T),
        }
    return rep


def fmt_report(rep: dict, name: str = "") -> str:
    lines = [f"== {name}  thr_logit={rep['threshold_logit']:.3f} thr_prob(cal)={rep['threshold_prob']:.3f} "
             f"T={rep['temperature']:.3f}"]
    for s, r in rep["splits"].items():
        for k in ("at_0.5", "at_thr"):
            m = r[k]
            lines.append(f"  {s:5s} {k:6s} acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                         f"F1={m['f1']:.4f} AUROC={m['auroc']:.4f} PR-AUC={m['pr_auc']:.4f} cm={m['confusion']}")
        lines.append(f"  {s:5s} ECE raw={r['ece_raw']:.4f} cal={r['ece_cal']:.4f} unsure={100 * r['unsure_frac']:.2f}%")
    return "\n".join(lines)


# Held-out sim-domain check: hand-picked crops of the sim/output renders that are NOT used for training
# (file, (x0, y0, x1, y1), label, description); label 1 rust, 0 clean, -1 ambiguous (reported, not scored)
SIM_HELDOUT = [
 ("preview_repair_closeup.png", (480, 335, 780, 440), 1, "damage patch (rust_coarse_01)"),
 ("preview_repair_closeup.png", (525, 0, 615, 150), 1, "rusty steel post (rusty_metal_02)"),
 ("preview_repair_closeup.png", (700, 40, 1250, 280), 1, "deck grate (metal_grate_rusty)"),
 ("preview_repair_closeup.png", (0, 610, 480, 720), 0, "green painted pipe"),
 ("preview_repair_closeup.png", (950, 500, 1280, 680), 0, "green painted pipe right"),
 ("preview_repair_closeup.png", (660, 440, 940, 720), -1, "rust streak/crack on paint"),
 ("preview_hero.png", (390, 285, 445, 370), 1, "rusty steel support"),
 ("preview_hero.png", (860, 430, 920, 550), 1, "rusty steel support 2"),
 ("preview_hero.png", (1000, 600, 1140, 700), 1, "deck grate"),
 ("preview_hero.png", (660, 350, 780, 385), 0, "green painted pipe"),
 ("preview_hero.png", (930, 140, 1060, 560), -1, "frame beam (rusty_painted_metal)"),
 ("preview_hero.png", (560, 235, 760, 290), -1, "brown pipe (rusty_painted_metal)"),
 ("preview_hero.png", (440, 455, 560, 515), -1, "grey pipe on deck"),
 ("full_rig_gauge_closeup.png", (0, 160, 480, 285), 0, "yellow pipe"),
 ("full_rig_gauge_closeup.png", (0, 370, 480, 700), 0, "blue pipe"),
 ("full_rig_gauge_closeup.png", (490, 205, 790, 505), 0, "gauge face"),
 ("full_rig_gauge_closeup.png", (800, 370, 1280, 700), 0, "blue pipe right"),
 ("preview_deck_end.png", (0, 0, 1280, 720), -1, "deck end (whole frame)"),
 ("preview_overview.png", (0, 0, 1280, 720), -1, "overview (whole frame)"),
]


def sim_heldout(model_dir: str, render_dir: str = "sim/output") -> dict:
    from pathlib import Path

    from PIL import Image

    from detection.export import OnnxRustClassifier

    clf = OnnxRustClassifier(model_dir)
    rows = []
    for f, box, label, desc in SIM_HELDOUT:
        p = Path(render_dir) / f
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB").crop(box)
        rows.append({"file": f, "box": list(box), "label": label, "desc": desc, **clf.predict(im)})
    lab = [r for r in rows if r["label"] >= 0]
    correct = sum(int(r["rust"]) == r["label"] for r in lab)
    return {"rows": rows, "n_labelled": len(lab), "correct": correct,
            "pos_recall": float(np.mean([r["rust"] for r in lab if r["label"] == 1])) if lab else float("nan"),
            "neg_fp_rate": float(np.mean([r["rust"] for r in lab if r["label"] == 0])) if lab else float("nan")}


def fmt_sim(res: dict) -> str:
    out = ["| render | crop | label | p_rust (cal.) | decision |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        lab = {1: "rust", 0: "clean", -1: "ambiguous"}[r["label"]]
        dec = ("RUST" if r["rust"] else "clean") + (" (unsure)" if r["unsure"] else "")
        out.append(f"| {r['file']} | {r['desc']} | {lab} | {r['p_rust']:.3f} | {dec} |")
    out.append(f"\nlabelled crops correct: {res['correct']}/{res['n_labelled']}  "
               f"(rust recall {res['pos_recall']:.2f}, clean FP rate {res['neg_fp_rate']:.2f})")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="held-out sim-render check for an exported model dir")
    ap.add_argument("model_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = sim_heldout(a.model_dir)
    print(fmt_sim(res))
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=2))

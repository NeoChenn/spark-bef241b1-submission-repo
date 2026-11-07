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


def full_report(split_logits: dict, split_labels: dict) -> dict:
    """split_* keyed by 'train'/'val'/'test'. Threshold + temperature are fitted on val only."""
    vz, vy = split_logits["val"], split_labels["val"]
    thr = pick_threshold(vz, vy)
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

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from bench.stats import bootstrap_ci

NEEDS_KEY = "not run — needs key"
IMG_EXT = {".jpg", ".jpeg", ".png"}
OUT = Path("runs/bench")
SAMPLE = Path("runs_sample/bench")
GEMINI_MODELS = ["gemini-4-argon", "gemini-robotics-er-2-preview"]

PUBLISHED = {
    "source": "detection/RESULTS.md",
    "status": "published, not re-measured (deployed rust_v2 LoRA checkpoint not available here)",
    "model": "mobilenetv4_conv_small + LoRA r=16, rust_v2",
    "real_test_881": {"accuracy": 0.981, "precision": 0.970, "recall": 0.983, "f1": 0.977, "auroc": 0.998,
                      "pr_auc": 0.997},
    "synthetic_test_485": {"f1": 1.000},
    "sim_heldout_13": {"correct": 13, "n": 13},
    "latency_ms": {"ort_cpu_b1_1thr": 2.6, "ort_cpu_b1_4thr": 1.1, "a10_torch_b1": 3.4},
    "params_M": 2.49, "gmacs": 0.19, "runs_on_robot": True,
}


def gemini_score(verdict, confidence):
    if verdict is None:
        return None
    if verdict == "rust":
        return float(confidence)
    if verdict == "clean":
        return 1.0 - float(confidence)
    return 0.5


def _f1(y, s, thr):
    p = s >= thr
    tp = float(np.sum(p & (y == 1)))
    fp = float(np.sum(p & (y == 0)))
    fn = float(np.sum(~p & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def _prec(y, s, thr):
    p = s >= thr
    return float(np.sum(p & (y == 1)) / p.sum()) if p.sum() else 0.0


def _rec(y, s, thr):
    return float(np.mean(s[y == 1] >= thr)) if np.any(y == 1) else float("nan")


def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _ci(t):
    return {"mean": t[0], "lo": t[1], "hi": t[2]}


def binary_metrics(labels, scores, thr: float, seed: int = 0, n_boot: int = 2000) -> dict:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    p = s >= thr
    conf = {"tn": int(np.sum(~p & (y == 0))), "fp": int(np.sum(p & (y == 0))),
            "fn": int(np.sum(~p & (y == 1))), "tp": int(np.sum(p & (y == 1)))}

    def auc(a, b):
        if len(np.unique(a)) < 2:
            raise ValueError
        return roc_auc_score(a, b)

    au = _auroc(y, s)
    return {
        "n": int(y.size), "n_pos": int(y.sum()), "threshold": float(thr), "confusion": conf,
        "f1": _ci(bootstrap_ci(lambda a, b: _f1(a, b, thr), y, s, seed=seed, n_boot=n_boot)),
        "precision": _ci(bootstrap_ci(lambda a, b: _prec(a, b, thr), y, s, seed=seed, n_boot=n_boot)),
        "recall": _ci(bootstrap_ci(lambda a, b: _rec(a, b, thr), y, s, seed=seed, n_boot=n_boot)),
        "auroc": _ci(bootstrap_ci(auc, y, s, seed=seed, n_boot=n_boot)) if not np.isnan(au)
        else {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")},
    }


def latency_stats(ms) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.quantile(a, 0.9)), "n": int(a.size)}


def list_images(root) -> list[tuple[Path, int]]:
    root = Path(root)
    out = []
    for c, y in (("clean", 0), ("rust", 1)):
        out += [(p, y) for p in sorted((root / c).iterdir()) if p.suffix.lower() in IMG_EXT]
    return out


def stratified_subset(labels, n: int, seed: int = 0) -> list[int]:
    labels = list(labels)
    if n >= len(labels):
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        k = round(n * len(idx) / len(labels))
        out += rng.choice(idx, size=k, replace=False).tolist()
    return sorted(out)


def not_run(model: str, reason: str = NEEDS_KEY) -> dict:
    return {"status": reason, "model": model, "runs_on_robot": False}


def load_heldout(render_dir="sim/output"):
    from PIL import Image

    from detection.eval import SIM_HELDOUT

    items = []
    for f, box, label, desc in SIM_HELDOUT:
        p = Path(render_dir) / f
        if label >= 0 and p.exists():
            items.append((Image.open(p).convert("RGB").crop(box), label, f"{f}:{desc}"))
    return items


def load_dir(root):
    from PIL import Image

    return [(Image.open(p).convert("RGB"), y, p.name) for p, y in list_images(root)]


def colour_arm():
    from detection.localize import RustLocalizer, fake_colour_classifier

    loc = RustLocalizer(fake_colour_classifier, patch=16, stride=8, threshold=0.5, temperature=1.0)

    def score(img):
        return float(loc.localize(np.asarray(img)).max_conf)

    return {"name": "Colour stand-in", "label": "fake_colour_classifier via RustLocalizer (patch 16, stride 8), "
            "image score = max patch prob — the orchestrator's CPU-mock stand-in", "threshold": 0.5,
            "runs_on_robot": True, "score": score, "compute": "CPU, numpy"}


def onnx_arm(model_dir, name, label, threads=4):
    from detection.export import OnnxRustClassifier

    clf = OnnxRustClassifier(model_dir, threads=threads)
    return {"name": name, "label": label, "threshold": float(clf.meta["threshold"]), "runs_on_robot": True,
            "score": lambda img: clf.predict(img)["p_rust"], "compute": f"CPU, onnxruntime, {threads} threads",
            "meta": {k: clf.meta.get(k) for k in ("model_name", "lora", "full_ft", "data_dir", "train_sec",
                                                  "threshold", "temperature")}}


def run_local_arm(arm, datasets, warmup=3):
    res = {}
    for dname, items in datasets.items():
        for img, _, _ in items[:warmup]:
            arm["score"](img)
        scores, ms = [], []
        for img, _, _ in items:
            t0 = time.perf_counter()
            scores.append(arm["score"](img))
            ms.append((time.perf_counter() - t0) * 1000.0)
        labels = [y for _, y, _ in items]
        res[dname] = {**binary_metrics(labels, scores, arm["threshold"]), "latency": latency_stats(ms),
                      "per_item": [{"id": i, "label": y, "score": round(float(s), 4)}
                                   for (_, y, i), s in zip(items, scores)]}
    return {k: v for k, v in arm.items() if k != "score"} | {"results": res}


def list_models(key: str) -> list[str]:
    from policy.gemini_verifier import API_ROOT

    names, token = [], ""
    while True:
        url = f"{API_ROOT}/models?pageSize=1000" + (f"&pageToken={token}" if token else "")
        req = urllib.request.Request(url, headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        names += [m["name"].split("/", 1)[-1] for m in d.get("models", [])]
        token = d.get("nextPageToken")
        if not token:
            return names


def run_gemini_arm(model, datasets, n_max, workers=6, api_key=None, transport=None):
    from policy.gemini_verifier import GeminiVerifier

    v = GeminiVerifier(api_key=api_key, model=model, transport=transport)
    res = {}
    for dname, items in datasets.items():
        idx = stratified_subset([y for _, y, _ in items], n_max, seed=0)
        sub = [items[i] for i in idx]

        def call(it):
            return GeminiVerifier(api_key=v.api_key, model=model, transport=transport).verify(np.asarray(it[0]),
                                                                                                  "rust")

        with ThreadPoolExecutor(workers) as ex:
            outs = list(ex.map(call, sub))
        ok = [(it, o) for it, o in zip(sub, outs) if o is not None]
        if not ok:
            res[dname] = {"status": "all calls failed", "n_attempted": len(sub)}
            continue
        scores = [gemini_score(o.verdict, o.confidence) for _, o in ok]
        labels = [it[1] for it, _ in ok]
        res[dname] = {**binary_metrics(labels, scores, 0.5),
                      "latency": {k.replace("_ms", "_s"): (x / 1000.0 if k.endswith("_ms") else x)
                                  for k, x in latency_stats([o.latency_ms for _, o in ok]).items()},
                      "n_attempted": len(sub), "n_errors": len(sub) - len(ok),
                      "cannot_tell_rate": float(np.mean([o.verdict == "cannot_tell" for _, o in ok])),
                      "per_item": [{"id": it[2], "label": it[1], "verdict": o.verdict,
                                    "confidence": o.confidence} for it, o in ok]}
    return {"name": f"Gemini {model}", "model": model, "label": "GeminiVerifier 'rust' prompt; score = confidence "
            "if rust, 1-confidence if clean, 0.5 if cannot_tell", "threshold": 0.5, "runs_on_robot": False,
            "compute": "cloud API", "results": res}


def chart(doc, path):
    from bench.chart import bar_panels

    arms = doc["arms"]
    names = [a["name"] for a in arms]
    panels = []
    for dname, title in (("sim_heldout", "held-out sim renders"), ("rust_sim_test", "synthetic test")):
        for metric, mt in (("f1", "F1"), ("auroc", "AUROC")):
            vals = {}
            for a in arms:
                r = a.get("results", {}).get(dname)
                if r and metric in r and not np.isnan(r[metric]["mean"]):
                    vals[a["name"]] = r[metric]
                else:
                    vals[a["name"]] = {"mean": None, "label": a.get("status", "n/a")}
            panels.append({"title": f"{mt} · {title} (n={doc['datasets'][dname]['n']})", "values": vals,
                           "fmt": "{:.2f}"})
    vals = {}
    for a in arms:
        lat = [r["latency"]["median_ms"] for r in a.get("results", {}).values() if "latency" in r
               and "median_ms" in r["latency"]]
        vals[a["name"]] = {"mean": float(np.median(lat))} if lat else {"mean": None, "label": a.get("status", "n/a")}
    panels.append({"title": "CPU latency per image, median ms", "values": vals, "fmt": "{:.1f}"})
    from bench.chart import SERIES

    pal = {"Colour stand-in": SERIES[3], "MNv4 LoRA (deployed)": SERIES[0], "MNv4 full-FT (sim-only)": SERIES[2]}
    colors = {n: pal.get(n, SERIES[1] if "argon" in n else SERIES[4]) for n in names}
    return bar_panels(path, "Rust detection: on-robot classifiers vs Gemini vision", names, panels, colors=colors,
                      note="95% bootstrap CIs. Deployed LoRA rust_v2 checkpoint unavailable: its published numbers "
                           "are in detect.json['published']. Real test photos not available here.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="runs/bench/data/rust_sim_test")
    ap.add_argument("--detector-w", default="runs/bench/models/detector_w")
    ap.add_argument("--deployed", default="runs/detector/best")
    ap.add_argument("--gemini-n", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {"sim_heldout": load_heldout()}
    if Path(a.synth).exists():
        datasets["rust_sim_test"] = load_dir(a.synth)
    doc = {"datasets": {
        "sim_heldout": {"n": len(datasets["sim_heldout"]), "n_rust": sum(y for _, y, _ in datasets["sim_heldout"]),
                        "source": "detection/eval.py SIM_HELDOUT labelled crops of sim/output renders; not in any "
                                  "training set (synth_sim.TRAIN_RENDERS uses only full_rig_* except the gauge closeup)"},
        "rust_sim_test": {"n": len(datasets.get("rust_sim_test", [])),
                          "n_rust": sum(y for _, y, _ in datasets.get("rust_sim_test", [])),
                          "source": "detection/synth_sim.py test split (watson runs/edge/data/rust_sim/test); "
                                    "in-distribution for detector_w, which trained on the same generator"},
        "real_photos": {"n": 0, "status": "not available here: rust_v1 real test photos live on the teammate's "
                                          "Modal volume rust-data"}},
        "published": PUBLISHED, "arms": []}
    doc["arms"].append(run_local_arm(colour_arm(), datasets))
    if Path(a.deployed, "model.onnx").exists():
        doc["arms"].append(run_local_arm(onnx_arm(a.deployed, "MNv4 LoRA (deployed)", "deployed rust_v2 LoRA"),
                                         datasets))
    else:
        doc["arms"].append({"name": "MNv4 LoRA (deployed)", "status": "checkpoint not available",
                            "runs_on_robot": True, "note": "runs/detector/best missing; see 'published'"})
    if Path(a.detector_w, "model.onnx").exists():
        doc["arms"].append(run_local_arm(onnx_arm(
            a.detector_w, "MNv4 full-FT (sim-only)",
            "MNv4 full-FT, sim-synthetic only, watson retrain (60 s) — not the deployed LoRA"), datasets))
    from policy.gemini_verifier import load_api_key

    key = load_api_key()
    if key:
        try:
            avail = list_models(key)
        except Exception as e:
            avail, doc["gemini_model_list_error"] = [], type(e).__name__
        doc["gemini_models_listed"] = [m for m in avail if "gemini" in m]
        for m in GEMINI_MODELS:
            if avail and m not in avail:
                doc["arms"].append(not_run(m, "model id not in API model list") | {"name": f"Gemini {m}"})
            else:
                doc["arms"].append(run_gemini_arm(m, datasets, a.gemini_n))
    else:
        doc["arms"] += [not_run(m) | {"name": f"Gemini {m}"} for m in GEMINI_MODELS]
    (out / "detect.json").write_text(json.dumps(doc, indent=1, default=float))
    chart(doc, out / "detect.png")
    SAMPLE.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(doc, default=float))
    for arm in slim["arms"]:
        for r in arm.get("results", {}).values():
            if isinstance(r, dict) and dname_large(r):
                r.pop("per_item", None)
    (SAMPLE / "detect.json").write_text(json.dumps(slim, indent=1))
    shutil.copy(out / "detect.png", SAMPLE / "detect.png")
    for arm in doc["arms"]:
        for dname, r in arm.get("results", {}).items():
            if "f1" in r:
                print(f"{arm['name']:28s} {dname:14s} n={r['n']:4d} F1={r['f1']['mean']:.3f} "
                      f"[{r['f1']['lo']:.3f},{r['f1']['hi']:.3f}] AUROC={r['auroc']['mean']:.3f} "
                      f"[{r['auroc']['lo']:.3f},{r['auroc']['hi']:.3f}] cm={r['confusion']}")
        if "status" in arm:
            print(f"{arm['name']:28s} {arm['status']}")
    return doc


def dname_large(r):
    return len(r.get("per_item", [])) > 50


if __name__ == "__main__":
    main()

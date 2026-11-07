"""Modal entrypoints for the rust classifier.

  modal run detection/modal_app.py --model mobilenetv4_conv_small            # one LoRA run
  modal run detection/modal_app.py --model mobilenetv4_conv_small --full-ft  # one full fine-tune
  modal run detection/modal_app.py --all                                     # bake-off (4 runs in parallel) + promote
  modal run detection/modal_app.py --promote-only                            # re-run selection on existing runs

Outputs: rust-ckpt volume /ckpt/detector/<run_name>/{model.pt,model.onnx,meta.json,metrics.json,logits.npz}
Deployed model: /ckpt/detector/best/{model.onnx,model.pt,meta.json,metrics.json}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra.common import CKPT_DIR, CKPT_VOL, DATA_DIR, DATA_VOL  # noqa: E402

DATA_ROOT = f"{DATA_DIR}/rust_v1"
OUT_ROOT = f"{CKPT_DIR}/detector"
MODELS = ["mobilenetv4_conv_small", "resnet18", "vit_small_patch14_dinov2"]
PRIMARY = MODELS[0]
EDGE_OK = {"mobilenetv4_conv_small", "resnet18"}  # ViT = accuracy ceiling only

# DETECTOR_GPU=L4 (default) | A10G | none. "none" trains on a 32-core CPU container
# (used when the Modal workspace has no payment method and GPU functions are refused).
GPU = os.environ.get("DETECTOR_GPU", "L4")
ON_CPU = GPU.lower() in ("", "none", "cpu")
FN_KW = dict(cpu=32.0, memory=32768) if ON_CPU else dict(gpu=GPU, cpu=8.0, memory=24576)
# CPU budget: fewer epochs for the heavier backbones so the bake-off fits in the 2 h timeout.
CPU_EPOCHS = {"mobilenetv4_conv_small": 12, "resnet18": 8, "vit_small_patch14_dinov2": 4}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "timm==1.0.31", "peft==0.21.2", "onnx",
                    "onnxruntime", "scikit-learn", "pillow", "fvcore", "numpy")
    .env({"HF_HOME": f"{CKPT_DIR}/.hf_cache", "PYTHONUNBUFFERED": "1", "DETECTOR_GPU": GPU})
    .add_local_python_source("detection", "infra")
)
app = modal.App("rust-detector", image=image)


def run_name_for(model: str, full_ft: bool) -> str:
    return f"{model}_{'full' if full_ft else 'lora'}"


@app.function(timeout=2 * 3600, **FN_KW, volumes={DATA_DIR: DATA_VOL, CKPT_DIR: CKPT_VOL})
def train_remote(model: str, full_ft: bool = False, epochs: int = 12, batch_size: int = 64) -> dict:
    import torch

    from detection.train import train

    if ON_CPU:
        torch.set_num_threads(24)
    DATA_VOL.reload()
    name = run_name_for(model, full_ft)
    res = train(model, DATA_ROOT, f"{OUT_ROOT}/{name}", full_ft=full_ft, epochs=epochs,
                batch_size=batch_size, num_workers=8 if ON_CPU else 7)
    CKPT_VOL.commit()
    return summarize(res)


def summarize(res: dict) -> dict:
    rep, meta = res["report"], res["meta"]
    out = {"run_name": meta["run_name"], "model": meta["model_name"], "full_ft": meta["full_ft"],
           "trainable_params": meta["trainable_params"], "total_params": meta["total_params"],
           "threshold": meta["threshold"], "temperature": meta["temperature"],
           "latency": res.get("latency"), "best_epoch": meta["best_epoch"], "train_sec": meta["train_sec"]}
    for s, r in rep["splits"].items():
        out[s] = {"f1@0.5": r["at_0.5"]["f1"], "f1@thr": r["at_thr"]["f1"], "recall@thr": r["at_thr"]["recall"],
                  "precision@thr": r["at_thr"]["precision"], "acc@0.5": r["at_0.5"]["accuracy"],
                  "auroc": r["at_0.5"]["auroc"], "pr_auc": r["at_0.5"]["pr_auc"],
                  "ece_raw": r["ece_raw"], "ece_cal": r["ece_cal"], "unsure": r["unsure_frac"]}
    return out


def select(summaries: dict[str, dict]) -> tuple[str, list[str]]:
    """Apply the LoRA-vs-full rule to the primary model, then pick the deploy model."""
    why = []
    lora, full = summaries.get(run_name_for(PRIMARY, False)), summaries.get(run_name_for(PRIMARY, True))
    if lora and full:
        lf, ff = lora["val"]["f1@0.5"], full["val"]["f1@0.5"]
        use_full = lf < 0.90 or (ff - lf) > 0.02
        why.append(f"primary LoRA val F1={lf:.4f}, full-FT val F1={ff:.4f} -> "
                   f"{'FULL-FT' if use_full else 'LoRA'} (rule: LoRA if F1>=0.90 and within 2 pts of full-FT)")
        primary = full["run_name"] if use_full else lora["run_name"]
    else:
        primary = (lora or full or {}).get("run_name")
    best = primary
    for name, s in summaries.items():
        if s["model"] in EDGE_OK and s["model"] != PRIMARY and best and \
                s["val"]["f1@0.5"] > summaries[best]["val"]["f1@0.5"] + 0.02:
            why.append(f"{name} beats primary by >2 val-F1 pts -> {name}")
            best = name
    if best is None:  # nothing edge-capable finished
        best = max(summaries, key=lambda k: summaries[k]["val"]["pr_auc"])
    why.append(f"deploy: {best}")
    return best, why


@app.function(timeout=600, volumes={CKPT_DIR: CKPT_VOL})
def promote(run_names: list[str] | None = None) -> dict:
    import shutil

    CKPT_VOL.reload()
    root = Path(OUT_ROOT)
    summaries = {}
    for d in sorted(root.iterdir()):
        if d.name == "best" or not (d / "metrics.json").exists():
            continue
        if run_names and d.name not in run_names:
            continue
        summaries[d.name] = summarize(json.loads((d / "metrics.json").read_text()))
    best, why = select(summaries)
    dst = root / "best"
    dst.mkdir(exist_ok=True)
    for f in ("model.onnx", "model.pt", "meta.json", "metrics.json"):
        shutil.copy2(root / best / f, dst / f)
    meta = json.loads((dst / "meta.json").read_text())
    meta["selection"] = why
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    (root / "bakeoff.json").write_text(json.dumps({"best": best, "why": why, "runs": summaries}, indent=2))
    CKPT_VOL.commit()
    return {"best": best, "why": why, "runs": summaries}


def _print(summaries: dict):
    hdr = f"{'run':38s} {'split':5s} {'acc':>6s} {'F1@.5':>6s} {'F1@thr':>6s} {'R@thr':>6s} {'P@thr':>6s} {'AUROC':>6s} {'PRAUC':>6s}"
    print(hdr)
    for n, s in summaries.items():
        for sp in ("train", "val", "test"):
            if sp in s:
                r = s[sp]
                print(f"{n:38s} {sp:5s} {r['acc@0.5']:.4f} {r['f1@0.5']:.4f} {r['f1@thr']:.4f} "
                      f"{r['recall@thr']:.4f} {r['precision@thr']:.4f} {r['auroc']:.4f} {r['pr_auc']:.4f}")


@app.local_entrypoint()
def main(model: str = PRIMARY, all: bool = False, full_ft: bool = False, epochs: int = 12,
         promote_only: bool = False, no_promote: bool = False):
    if promote_only:
        res = promote.remote()
        _print(res["runs"]), print("\n".join(res["why"]))
        return
    jobs = [(m, False) for m in MODELS] + [(PRIMARY, True)] if all else [(model, full_ft)]
    calls = [train_remote.spawn(m, f, min(epochs, CPU_EPOCHS[m]) if ON_CPU else epochs) for m, f in jobs]
    summaries = {}
    for (m, f), c in zip(jobs, calls):
        try:
            s = c.get()
            summaries[s["run_name"]] = s
            print(json.dumps(s, indent=1))
        except Exception as e:  # noqa: BLE001 - keep the bake-off going if one model dies (e.g. ViT)
            print(f"[modal_app] {run_name_for(m, f)} FAILED: {e!r}")
    _print(summaries)
    if not no_promote:
        res = promote.remote()
        print("\n".join(res["why"]))

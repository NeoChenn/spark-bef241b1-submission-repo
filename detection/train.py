from __future__ import annotations

import argparse
import io
import json
import pickle
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from detection.eval import fmt_report, full_report
from detection.models import INPUT_SIZE, PRIMARY, build_model, to_plain

CLASSES = {"clean": 0, "rust": 1}
EXTS = {".jpg", ".jpeg", ".png"}
CACHE_SHORT_SIDE = 256


def list_split(data_dir: str | Path, split: str, limit: int | None = None):
    items = []
    for cname, y in CLASSES.items():
        d = Path(data_dir) / split / cname
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)
            if limit:
                files = files[:limit]
            items += [(str(p), y) for p in files]
    return items


def _load(path) -> Image.Image:
    im = Image.open(path if isinstance(path, str) else io.BytesIO(path)).convert("RGB")
    w, h = im.size
    s = CACHE_SHORT_SIDE / min(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return im


class RustDataset(Dataset):

    def __init__(self, items, transform, images=None, raw=None, threads: int = 64):
        self.items, self.transform = items, transform
        if images is None:
            srcs = [raw[p] if raw else p for p, _ in items]
            with ThreadPoolExecutor(threads) as ex:
                images = list(ex.map(_load, srcs))
        self.images = images
        self.labels = np.array([y for _, y in items], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(self.images[i]), torch.tensor(float(self.labels[i]))


def load_pack(items: dict, pack: str | Path | None) -> dict | None:
    if not pack:
        return None
    pack = Path(pack)
    if pack.exists():
        raw = pickle.loads(pack.read_bytes())
        if all(p in raw for it in items.values() for p, _ in it):
            print(f"[data] loaded pack {pack} ({len(raw)} files)")
            return raw
    paths = [p for it in items.values() for p, _ in it]
    with ThreadPoolExecutor(64) as ex:
        raw = dict(zip(paths, ex.map(lambda q: Path(q).read_bytes(), paths)))
    pack.parent.mkdir(parents=True, exist_ok=True)
    tmp = pack.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(pack)
    print(f"[data] wrote pack {pack}")
    return raw


def make_transforms(mean, std):
    norm = T.Normalize(mean, std)
    train_tf = T.Compose([
        T.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        T.ToTensor(), norm,
    ])
    eval_tf = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), norm])
    return train_tf, eval_tf


def make_dummy_dataset(root: str | Path, n_per_class=(40, 12, 12), size=160, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in zip(("train", "val", "test"), n_per_class):
        for cname in CLASSES:
            d = Path(root) / split / cname
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                base = rng.uniform(90, 170)
                img = np.clip(base + rng.normal(0, 12, (size, size, 3)), 0, 255)
                if cname == "rust":
                    for _ in range(rng.integers(2, 6)):
                        cx, cy, r = rng.integers(0, size, 2).tolist() + [int(rng.integers(10, 40))]
                        yy, xx = np.ogrid[:size, :size]
                        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
                        img[m] = np.array([150, 70, 30]) + rng.normal(0, 15, (int(m.sum()), 3))
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / f"{i:04d}.jpg")
    (Path(root) / "stats.json").write_text(json.dumps({"dummy": True}))
    return str(root)


@torch.no_grad()
def predict_logits(model, loader, device, amp_dtype=None):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            z = model(x).float().squeeze(1)
        out.append(z.cpu()), ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy().astype(int)


def train(model_name: str = PRIMARY, data_dir: str = "/data/rust_v1", out_dir: str = "runs/detector",
          full_ft: bool = False, epochs: int = 12, batch_size: int = 64, lr: float | None = None,
          patience: int = 4, num_workers: int = 8, pretrained: bool = True, limit: int | None = None,
          export: bool = True, seed: int = 0, pack: str | None = None, threads: int | None = None) -> dict:
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed), np.random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    if device.type == "cpu" and threads:
        torch.set_num_threads(threads)
    t0 = time.time()

    model, meta = build_model(model_name, lora=not full_ft, pretrained=pretrained)
    model.to(device)
    cl = "vit" not in model_name
    if cl:
        model.to(memory_format=torch.channels_last)
    train_tf, eval_tf = make_transforms(meta["mean"], meta["std"])

    items = {s: list_split(data_dir, s, limit) for s in ("train", "val", "test")}
    for s, it in items.items():
        n_r = sum(y for _, y in it)
        print(f"[data] {s}: {len(it)} ({n_r} rust / {len(it) - n_r} clean)")
    assert items["train"] and items["val"], f"no data under {data_dir}"
    raw = load_pack(items, pack)
    ds_eval = {s: RustDataset(items[s], eval_tf, raw=raw) for s in ("train", "val", "test") if items[s]}
    ds_train = RustDataset(items["train"], train_tf, images=ds_eval["train"].images)
    print(f"[data] decoded into RAM in {time.time() - t0:.1f}s")

    n_rust = int(ds_train.labels.sum())
    n_clean = len(ds_train) - n_rust
    pos_weight = torch.tensor([n_clean / max(n_rust, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    kw = dict(num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=len(ds_train) > batch_size, **kw)
    dl_eval = {s: DataLoader(d, batch_size=batch_size * 2, shuffle=False, **kw) for s, d in ds_eval.items()}

    if lr is None:
        lr = 1e-3 if not full_ft else (5e-5 if "vit" in model_name else 3e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01 if not full_ft else 0.05)
    steps = epochs * len(dl_train)
    warm = max(1, len(dl_train))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best, best_ep, bad, hist = -1.0, -1, 0, []
    best_path = out / "best_state.pt"
    for ep in range(epochs):
        model.train()
        te, tl, nb = time.time(), 0.0, 0
        for x, y in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                z = model(x).float().squeeze(1)
            loss = lossf(z, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step()
            tl, nb = tl + loss.item(), nb + 1
        vz, vy = predict_logits(model, dl_eval["val"], device, amp_dtype)
        pr = float(average_precision_score(vy, vz)) if len(set(vy)) == 2 else 0.0
        vf1 = _f1(vz, vy)
        hist.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_pr_auc": pr, "val_f1@0.5": vf1,
                     "sec": time.time() - te})
        print(f"[{out.name} ep {ep:02d}] loss={tl / max(nb, 1):.4f} val PR-AUC={pr:.4f} F1@0.5={vf1:.4f} ({time.time() - te:.0f}s)")
        if pr > best + 1e-5:
            best, best_ep, bad = pr, ep, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{out.name}] early stop at epoch {ep} (best {best_ep})")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    plain = to_plain(model).to(device).to(memory_format=torch.contiguous_format).eval()
    logits, labels = {}, {}
    for s, dl in dl_eval.items():
        logits[s], labels[s] = predict_logits(plain, dl, device, None)
    rep = report_with_subsets(logits, labels, items)
    run_name = out.name
    print(fmt_report(rep, run_name))

    meta.update({
        "run_name": run_name, "full_ft": full_ft, "lr": lr, "epochs_run": len(hist), "best_epoch": best_ep,
        "pos_weight": float(pos_weight.item()), "resize": "direct resize to 224x224, RGB, /255, then (x-mean)/std",
        "layout": "NCHW float32", "output": "single logit; p_rust = sigmoid(logit / temperature)",
        "threshold": rep["threshold_prob"], "threshold_logit": rep["threshold_logit"],
        "temperature": rep["temperature"], "unsure_band": rep["unsure_band"],
        "decision": "rust if p_rust >= threshold; if unsure_band[0] <= p_rust <= unsure_band[1] take a second view",
        "train_sec": time.time() - t0, "data_dir": str(data_dir),
    })
    torch.save({"state_dict": plain.cpu().state_dict(), "meta": meta}, out / "model.pt")
    np.savez(out / "logits.npz", **{f"{s}_logits": logits[s] for s in logits}, **{f"{s}_labels": labels[s] for s in labels})
    result = {"meta": meta, "report": rep, "history": hist}
    if export:
        from detection.export import benchmark, export_onnx
        export_onnx(plain.cpu(), out / "model.onnx")
        result["latency"] = benchmark(plain, out / "model.onnx")
        meta["latency"] = result["latency"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    best_path.unlink(missing_ok=True)
    print(f"[train] done in {time.time() - t0:.0f}s -> {out}")
    return result


def report_with_subsets(logits: dict, labels: dict, items: dict) -> dict:
    from detection.eval import metrics_at

    syn = {s: np.array(["/synth_" in p for p, _ in items[s]]) for s in logits}
    doms = [~syn["val"], syn["val"]] if syn["val"].any() else None
    rep = full_report(logits, labels, val_domains=doms)
    rep["subsets"] = {}
    for s in logits:
        if syn[s].any():
            for tag, m in (("real", ~syn[s]), ("synth", syn[s])):
                if not m.any():
                    continue
                rep["subsets"][f"{s}_{tag}"] = {"at_0.5": metrics_at(logits[s][m], labels[s][m], 0.0),
                                                "at_thr": metrics_at(logits[s][m], labels[s][m], rep["threshold_logit"])}
                mt = rep["subsets"][f"{s}_{tag}"]["at_thr"]
                print(f"  {s}_{tag:5s} at_thr n={mt['n']} acc={mt['accuracy']:.4f} P={mt['precision']:.4f} "
                      f"R={mt['recall']:.4f} F1={mt['f1']:.4f} PR-AUC={mt['pr_auc']:.4f}")
    return rep


def rescore(run_dir: str, data_dir: str) -> dict:
    run = Path(run_dir)
    L = np.load(run / "logits.npz")
    splits = [s for s in ("train", "val", "test") if f"{s}_logits" in L]
    logits = {s: L[f"{s}_logits"] for s in splits}
    labels = {s: L[f"{s}_labels"] for s in splits}
    items = {s: list_split(data_dir, s) for s in splits}
    assert all(len(items[s]) == len(logits[s]) for s in splits), "data dir does not match the run"
    rep = report_with_subsets(logits, labels, items)
    print(fmt_report(rep, run.name))
    m = json.loads((run / "metrics.json").read_text())
    m["report"] = rep
    for k, v in (("threshold", rep["threshold_prob"]), ("threshold_logit", rep["threshold_logit"]),
                 ("temperature", rep["temperature"])):
        m["meta"][k] = v
    (run / "metrics.json").write_text(json.dumps(m, indent=2))
    meta = json.loads((run / "meta.json").read_text())
    meta.update({k: m["meta"][k] for k in ("threshold", "threshold_logit", "temperature")})
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    return m


def _f1(z, y, thr=0.0):
    p = z >= thr
    tp = int((p & (y == 1)).sum())
    fp, fn = int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--data-dir", default="/data/rust_v1")
    ap.add_argument("--out-dir", default="runs/detector")
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (CPU training only)")
    ap.add_argument("--limit", type=int, default=None, help="max images per class per split")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--rescore", default=None, help="run dir: recompute report/threshold from logits.npz")
    ap.add_argument("--make-dummy", default=None, help="write a synthetic dataset to this dir and exit")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.rescore, a.data_dir)
        return
    if a.make_dummy:
        print(make_dummy_dataset(a.make_dummy))
        return
    train(a.model, a.data_dir, a.out_dir, a.full_ft, a.epochs, a.batch_size, a.lr, num_workers=a.workers,
          pretrained=not a.no_pretrained, limit=a.limit, export=not a.no_export, threads=a.threads)


if __name__ == "__main__":
    main()

"""ONNX export (opset 17) + latency / FLOPs benchmark for the rust classifier.

Standalone: .venv/bin/python -m detection.export --ckpt runs/dummy/model.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from detection.models import INPUT_SIZE

OPSET = 17


def export_onnx(model: nn.Module, path: str | Path, size: int = INPUT_SIZE) -> str:
    model = model.eval().cpu().float()
    x = torch.randn(1, 3, size, size)
    kw = dict(input_names=["input"], output_names=["logit"], opset_version=OPSET,
              dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}})
    try:
        torch.onnx.export(model, (x,), str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, (x,), str(path), **kw)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    print(f"[export] {path} max|torch-onnx|={err:.2e}")
    assert err < 1e-3, f"ONNX mismatch {err}"
    return str(path)


def gflops(model: nn.Module, size: int = INPUT_SIZE) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis

        fl = FlopCountAnalysis(model.eval().cpu().float(), torch.randn(1, 3, size, size))
        fl.unsupported_ops_warnings(False), fl.uncalled_modules_warnings(False)
        return float(fl.total()) / 1e9  # fvcore counts one fused multiply-add as 1 flop (i.e. GMACs)
    except Exception as e:  # noqa: BLE001
        print(f"[export] flop count failed: {e}")
        return None


def _time(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def benchmark(model: nn.Module, onnx_path: str | Path, size: int = INPUT_SIZE, iters: int = 100) -> dict:
    import onnxruntime as ort

    res = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6, "gmacs": gflops(model, size)}
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        med, p90 = _time(lambda: sess.run(None, {"input": x}), iters=iters)
        res[f"ort_cpu_b1_{threads}thr_ms"] = med
        res[f"ort_cpu_b1_{threads}thr_p90_ms"] = p90
    if torch.cuda.is_available():
        m = model.cuda().eval()
        xt = torch.from_numpy(x).cuda()

        def run():
            with torch.no_grad():
                m(xt)
            torch.cuda.synchronize()

        res["torch_gpu_b1_ms"], _ = _time(run, iters=iters)
        res["gpu"] = torch.cuda.get_device_name()
        model.cpu()
    print(f"[bench] {json.dumps(res)}")
    return res


def load_plain(ckpt: str | Path):
    import timm

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = d["meta"]
    kw = {"img_size": INPUT_SIZE} if "vit" in meta["model_name"] else {}
    m = timm.create_model(meta["model_name"], pretrained=False, num_classes=1, **kw)
    m.load_state_dict(d["state_dict"])
    return m.eval(), meta


class OnnxRustClassifier:
    """Inference wrapper the orchestrator can use: loads meta.json + model.onnx from one directory.

    clf = OnnxRustClassifier("/ckpt/detector/best"); clf.predict(pil_image)
    -> {"p_rust": calibrated prob, "rust": bool, "unsure": bool, "logit": raw}
    """

    def __init__(self, model_dir: str | Path, threads: int = 4):
        import onnxruntime as ort

        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.mean = np.array(self.meta["mean"], np.float32)[:, None, None]
        self.std = np.array(self.meta["std"], np.float32)[:, None, None]
        self.size = self.meta["input_size"][-1]

    def preprocess(self, img) -> np.ndarray:
        from PIL import Image

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8))
        x = np.asarray(img.convert("RGB").resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        return (x.transpose(2, 0, 1) - self.mean) / self.std

    def predict_batch(self, imgs) -> list[dict]:
        x = np.stack([self.preprocess(i) for i in imgs]).astype(np.float32)
        z = self.sess.run(None, {"input": x})[0].reshape(-1)
        p = 1.0 / (1.0 + np.exp(-z / self.meta["temperature"]))
        lo, hi = self.meta["unsure_band"]
        return [{"logit": float(zi), "p_rust": float(pi), "rust": bool(pi >= self.meta["threshold"]),
                 "unsure": bool(lo <= pi <= hi)} for zi, pi in zip(z, p)]

    def predict(self, img) -> dict:
        return self.predict_batch([img])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="model.pt to export + benchmark")
    ap.add_argument("--out", default=None)
    ap.add_argument("--predict-dir", default=None, help="dir with meta.json+model.onnx; classify --images")
    ap.add_argument("--images", nargs="*", default=[])
    a = ap.parse_args()
    if a.predict_dir:
        clf = OnnxRustClassifier(a.predict_dir)
        from PIL import Image

        for p in a.images:
            print(p, json.dumps(clf.predict(Image.open(p))))
        return
    m, _ = load_plain(a.ckpt)
    out = a.out or str(Path(a.ckpt).with_suffix(".onnx"))
    export_onnx(m, out)
    benchmark(m, out)


if __name__ == "__main__":
    main()

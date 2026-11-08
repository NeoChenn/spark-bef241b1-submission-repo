from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import agreement, binary_metrics, latency_stats, parse_smi_power, summarize_power_window  # noqa: E402

LEAK_BATCH = 64
MODELS = {
    "rust_detector": {"onnx": "detector.onnx", "input": "input", "batch": 1, "max": 64, "prec": ["fp32", "fp16", "int8"]},
    "leak_crack": {"onnx": "detector.onnx", "input": "input", "batch": LEAK_BATCH, "max": LEAK_BATCH,
                   "prec": ["fp32", "fp16", "int8"]},
    "scrub_policy": {"onnx": "policy.onnx", "input": "obs", "batch": 1, "max": 1, "prec": ["fp32", "fp16", "int8"]},
}


class PowerSampler:

    def __init__(self, period_ms: int = 100):
        self.period_ms, self.samples, self.util, self._p = period_ms, [], [], None

    def __enter__(self):
        self._p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                                    "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
                                   stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return self

    def _read(self):
        for ln in self._p.stdout:
            self.samples += parse_smi_power([ln])
            try:
                self.util.append(float(ln.split(",")[1]))
            except (IndexError, ValueError):
                pass

    def __exit__(self, *a):
        self._p.terminate()
        self._p.wait()


def idle_window(secs: float) -> tuple[list[float], list[float]]:
    with PowerSampler() as ps:
        time.sleep(secs)
    return ps.samples, ps.util


def timed(fn, secs: float, min_iters: int, warmup: int = 50):
    for _ in range(warmup):
        fn()
    ms, t_end = [], time.perf_counter() + secs
    while len(ms) < min_iters or time.perf_counter() < t_end:
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1e3)
    return ms


def norm(x_u8: np.ndarray, meta: dict) -> np.ndarray:
    mean = np.asarray(meta["mean"], np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(meta["std"], np.float32).reshape(1, 3, 1, 1)
    return ((x_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - mean) / std).astype(np.float32)


def build(onnx_path: Path, eng: Path, prec: str, inp: str, shape_tail, opt: int, mx: int, calib, cache: Path):
    from polygraphy.backend.trt import (Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, Profile,
                                        save_engine)

    prof = Profile().add(inp, min=(1, *shape_tail), opt=(opt, *shape_tail), max=(mx, *shape_tail))
    kw = {}
    if prec == "int8":
        def loader():
            for i in range(0, len(calib) - opt + 1, opt):
                yield {inp: calib[i:i + opt]}
        kw = {"int8": True, "fp16": True, "calibrator": Calibrator(loader(), cache=str(cache))}
    elif prec == "fp16":
        kw = {"fp16": True}
    t = time.time()
    engine = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_path)), config=CreateConfig(profiles=[prof], **kw))()
    save_engine(engine, str(eng))
    return time.time() - t


class Trt:

    def __init__(self, eng: Path):
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        self.r = TrtRunner(EngineFromBytes(BytesFromPath(str(eng))))
        self.r.activate()

    def __call__(self, feed):
        return list(self.r.infer(feed, check_inputs=False).values())[0]

    def device_mem(self) -> int:
        e = self.r.context.engine
        return int(e.device_memory_size_v2)

    def close(self):
        self.r.deactivate()


def ort_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_batches(fn, x: np.ndarray, inp: str, bs: int) -> np.ndarray:
    return np.concatenate([np.asarray(fn({inp: x[i:i + bs]})).reshape(len(x[i:i + bs]), -1)
                           for i in range(0, len(x), bs)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/work/bundle")
    ap.add_argument("--engines", default="/work/engines")
    ap.add_argument("--out", default="/work/out")
    ap.add_argument("--tag", default="contended")
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--min-iters", type=int, default=500)
    ap.add_argument("--idle-secs", type=float, default=4.0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--cpu-threads", default="1,4")
    a = ap.parse_args(argv)
    bundle, engdir, out = Path(a.bundle), Path(a.engines), Path(a.out)
    engdir.mkdir(parents=True, exist_ok=True), out.mkdir(parents=True, exist_ok=True)
    import tensorrt as trt

    meta = json.loads((bundle / "detector_meta.json").read_text())
    calib_img = norm(np.load(bundle / "calib.npz")["x"], meta)
    ev = np.load(bundle / "eval.npz")
    ex, ey, esrc = norm(ev["x"], meta), ev["y"], ev["src"]
    obs = np.load(bundle / "policy_obs.npz")["obs"]
    thr = float(meta.get("threshold_logit", 0.0))
    res = {"tag": a.tag, "host": platform.node(), "arch": platform.machine(), "trt": trt.__version__,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip(),
           "detector_weights": meta.get("weights", meta.get("run_name")), "threshold_logit": thr,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    base_w, base_u = idle_window(a.idle_secs)
    res["background"] = {"gpu_util_mean": float(np.mean(base_u)) if base_u else None,
                         "gpu_power_mean_w": float(np.mean(base_w)) if base_w else None}
    print(json.dumps(res["background"]), flush=True)
    for name, cfg in MODELS.items():
        onnx_path, inp, bs = bundle / cfg["onnx"], cfg["input"], cfg["batch"]
        is_img = cfg["onnx"] == "detector.onnx"
        calib = calib_img if is_img else obs
        tail = calib.shape[1:]
        feed_x = calib[:bs].copy()
        mres = {"onnx": cfg["onnx"], "batch": bs, "input_shape": [bs, *tail], "trt": {}, "cpu": {}}
        cpu_ref = ort_session(onnx_path, 4)
        ref_fn = lambda f: cpu_ref.run(None, f)[0]
        ref_eval = run_batches(ref_fn, ex if is_img else obs, inp, 64 if is_img else 1)
        for prec in cfg["prec"]:
            eng = engdir / f"{name}_{prec}.engine"
            entry = {}
            if a.rebuild or not eng.exists():
                cache = engdir / (f"{'img' if is_img else 'obs'}_b{bs}_int8.cache")
                entry["build_s"] = build(onnx_path, eng, prec, inp, tail, bs, cfg["max"], calib, cache)
            entry["engine_mb"] = eng.stat().st_size / 2**20
            r = Trt(eng)
            entry["activation_mb"] = r.device_mem() / 2**20
            feed = {inp: feed_x}
            bw, _ = idle_window(a.idle_secs / 2)
            with PowerSampler() as ps:
                ms = timed(lambda: r(feed), a.secs, a.min_iters)
            entry["latency"] = latency_stats(ms, batch=bs)
            entry["power"] = summarize_power_window(ps.samples, bw, entry["latency"]["mean_ms"], bs)
            entry["gpu_util_during"] = float(np.mean(ps.util)) if ps.util else None
            got = run_batches(r, ex if is_img else obs, inp, bs if is_img else 1)
            if is_img:
                entry["vs_fp32_cpu"] = agreement(ref_eval[:, 0], got[:, 0], thr)
                if name == "rust_detector":
                    entry["accuracy"] = {s: binary_metrics(got[esrc == s, 0], ey[esrc == s], thr)
                                         for s in np.unique(esrc)}
            else:
                d = np.abs(ref_eval - got)
                entry["vs_fp32_cpu"] = {"max_abs_action_diff": float(d.max()), "mean_abs_action_diff": float(d.mean()),
                                        "max_force_setpoint_diff_n": float(20 * d[:, 3].max())}
            r.close()
            mres["trt"][prec] = entry
            print(name, prec, json.dumps({k: entry["latency"][k] for k in ("p50_ms", "p99_ms", "throughput_per_s")}),
                  flush=True)
        if is_img and name == "rust_detector":
            mres["accuracy_fp32_cpu"] = {s: binary_metrics(ref_eval[esrc == s, 0], ey[esrc == s], thr)
                                         for s in np.unique(esrc)}
        for th in [int(t) for t in a.cpu_threads.split(",")]:
            s = ort_session(onnx_path, th)
            feed = {inp: feed_x}
            ms = timed(lambda: s.run(None, feed), a.secs / 2, 50, warmup=10)
            mres["cpu"][f"ort_fp32_{th}thr"] = {"latency": latency_stats(ms, batch=bs)}
            print(name, f"cpu{th}", json.dumps({k: mres["cpu"][f"ort_fp32_{th}thr"]["latency"][k]
                                                 for k in ("p50_ms", "p99_ms")}), flush=True)
        res["models"][name] = mres
    (out / f"bench_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / f'bench_{a.tag}.json'}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

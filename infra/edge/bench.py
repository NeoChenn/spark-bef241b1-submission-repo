from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import (agreement, binary_metrics, latency_stats, parse_smi_power, summarize_power_window,  # noqa: E402
                   window_samples)

LEAK_BATCH = 64
PRECS = ["fp32", "fp16", "int8"]
MODELS = {
    "rust_detector": {"onnx": "detector.onnx", "input": "input", "batch": 1, "max": 64},
    "leak_crack": {"onnx": "detector.onnx", "input": "input", "batch": LEAK_BATCH, "max": LEAK_BATCH},
    "scrub_policy": {"onnx": "policy.onnx", "input": "obs", "batch": 1, "max": 1},
}


class PowerLog:

    def __init__(self, period_ms: int = 100):
        self.period_ms, self.rows = period_ms, []

    def __enter__(self):
        self._p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw.instant,utilization.gpu",
                                    "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
                                   stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for ln in self._p.stdout:
            w = parse_smi_power([ln])
            if w:
                parts = ln.split(",")
                u = float(parts[1]) if len(parts) > 1 and parts[1].strip().replace(".", "").isdigit() else np.nan
                self.rows.append((time.perf_counter(), w[0], u))

    def __exit__(self, *a):
        self._p.terminate()
        self._p.wait()

    def power(self, windows):
        return window_samples([(t, w) for t, w, _ in self.rows], windows)

    def util(self, windows):
        return [u for u in window_samples([(t, u) for t, _, u in self.rows], windows) if u == u]


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
        return int(self.r.context.engine.device_memory_size_v2)

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


def interleave(backends: dict, feed: dict, rounds: int, round_s: float, seed: int = 0):
    ms = {k: [] for k in backends}
    win = {k: [] for k in backends}
    win["idle"] = []
    rng = random.Random(seed)
    for k, fn in backends.items():
        for _ in range(10):
            fn(feed)
    for _ in range(rounds):
        order = list(backends) + ["idle"]
        rng.shuffle(order)
        for k in order:
            t0 = time.perf_counter()
            if k == "idle":
                time.sleep(round_s)
            else:
                fn, n = backends[k], 0
                while n < 3 or time.perf_counter() - t0 < round_s:
                    t = time.perf_counter()
                    fn(feed)
                    ms[k].append((time.perf_counter() - t) * 1e3)
                    n += 1
            win[k].append((t0, time.perf_counter()))
    return ms, win


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/work/bundle")
    ap.add_argument("--engines", default="/work/engines")
    ap.add_argument("--out", default="/work/out")
    ap.add_argument("--tag", default="contended")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--round-s", type=float, default=0.8)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--build-only", action="store_true")
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
           "method": f"{a.rounds} shuffled rounds x {a.round_s}s per backend incl. an idle slot; "
                     "wall-clock per call incl. H2D/D2H; nvidia-smi power.draw.instant at 100 ms",
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    builds = {}
    for name, cfg in MODELS.items():
        is_img = cfg["onnx"] == "detector.onnx"
        calib = calib_img if is_img else obs
        for prec in PRECS:
            eng = engdir / f"{name}_{prec}.engine"
            if a.rebuild or not eng.exists():
                cache = engdir / f"{'img' if is_img else 'obs'}_int8.cache"
                builds[f"{name}_{prec}"] = build(bundle / cfg["onnx"], eng, prec, cfg["input"], calib.shape[1:],
                                                 cfg["batch"], cfg["max"], calib, cache)
                print("built", name, prec, f"{builds[f'{name}_{prec}']:.0f}s", flush=True)
    if a.build_only:
        os._exit(0)
    with PowerLog() as pl:
        for name, cfg in MODELS.items():
            onnx_path, inp, bs = bundle / cfg["onnx"], cfg["input"], cfg["batch"]
            is_img = cfg["onnx"] == "detector.onnx"
            calib = calib_img if is_img else obs
            mres = {"onnx": cfg["onnx"], "batch": bs, "input_shape": [bs, *calib.shape[1:]], "trt": {}, "cpu": {}}
            trts = {p: Trt(engdir / f"{name}_{p}.engine") for p in PRECS}
            cpus = {f"ort_fp32_{t}thr": ort_session(onnx_path, int(t)) for t in a.cpu_threads.split(",")}
            backends = {f"trt_{p}": r for p, r in trts.items()}
            backends.update({k: (lambda s: (lambda f: s.run(None, f)[0]))(s) for k, s in cpus.items()})
            ms, win = interleave(backends, {inp: calib[:bs].copy()}, a.rounds, a.round_s)
            base = pl.power(win["idle"])
            mres["idle_slot"] = {"gpu_power_mean_w": float(np.mean(base)) if base else None,
                                 "gpu_util_mean": float(np.mean(pl.util(win["idle"]))) if pl.util(win["idle"]) else None}
            ref = cpus[f"ort_fp32_{a.cpu_threads.split(',')[-1]}thr"]
            ref_eval = run_batches(lambda f: ref.run(None, f)[0], ex if is_img else obs, inp, 64 if is_img else 1)
            for k in backends:
                lat = latency_stats(ms[k], batch=bs)
                entry = {"latency": lat, "power": summarize_power_window(pl.power(win[k]), base, lat["mean_ms"], bs),
                         "gpu_util_during": float(np.mean(pl.util(win[k]))) if pl.util(win[k]) else None}
                if k.startswith("trt_"):
                    p = k[4:]
                    r = trts[p]
                    entry["engine_mb"] = (engdir / f"{name}_{p}.engine").stat().st_size / 2**20
                    entry["activation_mb"] = r.device_mem() / 2**20
                    if f"{name}_{p}" in builds:
                        entry["build_s"] = builds[f"{name}_{p}"]
                    got = run_batches(r, ex if is_img else obs, inp, bs if is_img else 1)
                    if is_img:
                        entry["vs_fp32_cpu"] = agreement(ref_eval[:, 0], got[:, 0], thr)
                        if name == "rust_detector":
                            entry["accuracy"] = {s: binary_metrics(got[esrc == s, 0], ey[esrc == s], thr)
                                                 for s in np.unique(esrc)}
                    else:
                        d = np.abs(ref_eval - got)
                        entry["vs_fp32_cpu"] = {"max_abs_action_diff": float(d.max()),
                                                "mean_abs_action_diff": float(d.mean()),
                                                "max_force_setpoint_diff_n": float(20 * d[:, 3].max())}
                    mres["trt"][p] = entry
                else:
                    mres["cpu"][k] = entry
                print(name, k, json.dumps({x: round(lat[x], 3) for x in ("p50_ms", "p99_ms", "throughput_per_s")}),
                      json.dumps({x: entry["power"][x] for x in ("load_mean_w", "delta_w")}), flush=True)
            if name == "rust_detector":
                mres["accuracy_fp32_cpu"] = {s: binary_metrics(ref_eval[esrc == s, 0], ey[esrc == s], thr)
                                             for s in np.unique(esrc)}
            for r in trts.values():
                r.close()
            res["models"][name] = mres
    idle = [m["idle_slot"] for m in res["models"].values()]
    res["background"] = {k: float(np.mean([i[k] for i in idle if i[k] is not None])) for k in idle[0]}
    (out / f"bench_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / f'bench_{a.tag}.json'}", json.dumps(res["background"]), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

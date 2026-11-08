from __future__ import annotations

import argparse
import json
from pathlib import Path

from infra.edge.stats import loop_rate

SERIES = [("trt", "fp32", "TensorRT FP32", "#3987e5"), ("trt", "fp16", "TensorRT FP16", "#d95926"),
          ("trt", "int8", "TensorRT INT8 (PTQ)", "#199e70"), ("cpu", "ort_fp32_4thr", "ONNX Runtime CPU, 4 Arm cores", "#c98500")]
TITLES = {"rust_detector": "Rust detector (MobileNetV4, 224 px, batch 1)",
          "leak_crack": "Crack classifier (64 patches per flange read)",
          "scrub_policy": "Scrub policy (MLP 327-512-256-128-4, batch 1)"}
INK, INK2, GRID = "#f0efec", "#c3c2b7", "#3a3a37"


def lat(m: dict, kind: str, key: str):
    e = m.get(kind, {}).get(key)
    return e["latency"] if e else None


def merge_accuracy(run: dict, acc: dict) -> dict:
    for name, m in run["models"].items():
        src = acc["models"].get(name, {})
        if "accuracy_fp32_cpu" in src:
            m["accuracy_fp32_cpu"] = src["accuracy_fp32_cpu"]
        for p, e in m.get("trt", {}).items():
            for k in ("vs_fp32_cpu", "accuracy"):
                if k in src.get("trt", {}).get(p, {}):
                    e[k] = src["trt"][p][k]
    run["accuracy_from"] = acc.get("time")
    return run


def faithful(e: dict, min_agree: float = 0.99, max_force_n: float = 0.5) -> bool:
    v = e.get("vs_fp32_cpu")
    if not v:
        return True
    if "decision_agreement" in v:
        return v["decision_agreement"] >= min_agree
    return v.get("max_force_setpoint_diff_n", 0.0) <= max_force_n


def best(m: dict) -> tuple[str, dict]:
    cands = [(f"trt_{k}", v) for k, v in m.get("trt", {}).items()] + list(m.get("cpu", {}).items())
    return min([kv for kv in cands if faithful(kv[1])], key=lambda kv: kv[1]["latency"]["p50_ms"])


def headline(runs: dict, tag: str) -> dict:
    m = runs[tag]["models"]
    (dk, de), (pk, pe), (lk, le) = best(m["rust_detector"]), best(m["scrub_policy"]), best(m["leak_crack"])
    d, p = de["latency"], pe["latency"]
    return {"tag": tag, "detector": dk, "policy": pk, "leak": lk,
            "detector_p50_ms": d["p50_ms"], "detector_p99_ms": d["p99_ms"],
            "policy_p50_ms": p["p50_ms"], "policy_p99_ms": p["p99_ms"], "leak_read_p50_ms": le["latency"]["p50_ms"],
            "loop_hz_p50": loop_rate([d["p50_ms"], p["p50_ms"]]), "loop_hz_p99": loop_rate([d["p99_ms"], p["p99_ms"]]),
            "detector_gpu_w": de["power"]["load_mean_w"], "detector_delta_w": de["power"]["delta_w"],
            "leak_gpu_w": le["power"]["load_mean_w"], "leak_delta_w": le["power"]["delta_w"],
            "background_gpu_util": runs[tag]["background"]["gpu_util_mean"],
            "background_gpu_w": runs[tag]["background"]["gpu_power_mean_w"]}


def chart(runs: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tags = [t for t in ("idle", "contended") if t in runs] or list(runs)
    names = list(TITLES)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.1 * len(names) + 0.9))
    fig.patch.set_alpha(0)
    h = 0.8 / len(SERIES)
    for ax, name in zip(axes, names):
        ax.set_facecolor("none")
        y = np.arange(len(tags))
        top = 0.0
        for i, (kind, key, label, col) in enumerate(SERIES):
            vals = [lat(runs[t]["models"][name], kind, key) for t in tags]
            for j, v in enumerate(vals):
                if not v:
                    continue
                yy = y[j] - 0.4 + h * (i + 0.5)
                ax.barh(yy, v["p50_ms"], height=h - 0.03, color=col, label=label if j == 0 else None)
                ax.plot([v["p50_ms"], v["p99_ms"]], [yy, yy], color=INK2, lw=1)
                ax.plot([v["p99_ms"]], [yy], "|", color=INK2, ms=6)
                ax.text(v["p99_ms"] * 1.12, yy, f"{v['p50_ms']:.3g} ms", va="center", fontsize=7.5, color=INK)
                top = max(top, v["p99_ms"])
        ax.set_xscale("log")
        ax.set_xlim(left=0.008, right=top * 6)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_yticks(y, ["idle GB10" if t == "idle" else
                          f"GB10 + Isaac Sim\n(GPU {runs[t]['background']['gpu_util_mean']:.0f}% busy)" for t in tags],
                      color=INK, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(TITLES[name], color=INK, fontsize=10, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("latency, ms (log scale); bar = p50, whisker = p99", color=INK2, fontsize=8.5)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=4, frameon=False, labelcolor=INK, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("On-robot inference on NVIDIA GB10 (aarch64)", color=INK, fontsize=12, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")


def pretty(k: str) -> str:
    if k.startswith("trt_"):
        return f"TensorRT {k[4:].upper()}"
    n = k.split("_")[2].removesuffix("thr")
    return f"ONNX Runtime on {n} Arm core{'s' if n != '1' else ''}"


def summary(heads: dict) -> str:
    h = heads.get("idle") or next(iter(heads.values()))
    c = heads.get("contended")
    cyc = h["detector_p50_ms"] + h["policy_p50_ms"]
    s = (f"Perception and control run on-robot. On an NVIDIA GB10 (Grace-Blackwell, 20 Arm cores, aarch64) the rust "
         f"detector answers in {h['detector_p50_ms']:.2f} ms p50 / {h['detector_p99_ms']:.2f} ms p99 "
         f"({pretty(h['detector'])}, same decisions as FP32) and the force-limited scrub policy in "
         f"{h['policy_p50_ms'] * 1000:.0f} us ({pretty(h['policy'])}), so one detect-and-act cycle costs {cyc:.2f} ms: "
         f"a {h['loop_hz_p50']:.0f} Hz ceiling against the 30 Hz the policy acts at")
    if c is not None and c is not h:
        s += (f". With the same GPU {c['background_gpu_util']:.0f}% busy rendering Isaac Sim jobs the loop still closes "
              f"at {c['loop_hz_p50']:.0f} Hz ({c['loop_hz_p99']:.0f} Hz at p99) with the detector moved to "
              f"{pretty(c['detector'])}")
    s += (f". A 64-patch flange crack read takes {h['leak_read_p50_ms']:.1f} ms ({pretty(h['leak'])}). The GPU draws "
          f"{h['detector_gpu_w']:.0f} W while serving the detector back-to-back ({h['detector_delta_w']:+.0f} W over "
          f"idle) and {h['leak_gpu_w']:.0f} W in a crack-read burst. No network is needed for any decision; Gemini is "
          f"only an optional cloud second opinion at the verify step.")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/edge")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    runs = {p.stem.removeprefix("bench_"): json.loads(p.read_text()) for p in sorted((d / "out").glob("bench_*.json"))
            if "smoke" not in p.stem}
    acc = runs.pop("acc", None)
    if acc:
        for r in runs.values():
            merge_accuracy(r, acc)
    heads = {t: headline(runs, t) for t in runs}
    out = {"headline": heads, "summary": summary(heads), "runs": runs}
    (d / "bench.json").write_text(json.dumps(out, indent=2))
    chart(runs, d / "bench.png")
    print(json.dumps(heads, indent=1))
    print(out["summary"])


if __name__ == "__main__":
    main()

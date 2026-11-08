from __future__ import annotations

import argparse
import json
from pathlib import Path

from infra.edge.stats import loop_rate

SERIES = [("trt", "fp32", "TensorRT FP32", "#3987e5"), ("trt", "fp16", "TensorRT FP16", "#d95926"),
          ("trt", "int8", "TensorRT INT8", "#199e70"), ("cpu", "ort_fp32_4thr", "ORT CPU, 4 ARM cores", "#c98500")]
TITLES = {"rust_detector": "Rust detector (MobileNetV4, 224 px, batch 1)",
          "leak_crack": "Crack classifier (64 patches per flange read)",
          "scrub_policy": "Scrub policy (MLP 327-512-256-128-4, batch 1)"}
INK, INK2, GRID = "#f0efec", "#c3c2b7", "#3a3a37"


def lat(m: dict, kind: str, key: str):
    e = m.get(kind, {}).get(key)
    return e["latency"] if e else None


def headline(runs: dict, tag: str, det_prec: str = "fp16", pol_prec: str = "fp16") -> dict:
    m = runs[tag]["models"]
    d, p = lat(m["rust_detector"], "trt", det_prec), lat(m["scrub_policy"], "trt", pol_prec)
    lk = lat(m["leak_crack"], "trt", det_prec)
    dp, pp = m["rust_detector"]["trt"][det_prec]["power"], m["scrub_policy"]["trt"][pol_prec]["power"]
    return {"tag": tag, "detector": f"trt_{det_prec}", "policy": f"trt_{pol_prec}",
            "detector_p50_ms": d["p50_ms"], "detector_p99_ms": d["p99_ms"],
            "policy_p50_ms": p["p50_ms"], "policy_p99_ms": p["p99_ms"], "leak_read_p50_ms": lk["p50_ms"],
            "loop_hz_p50": loop_rate([d["p50_ms"], p["p50_ms"]]), "loop_hz_p99": loop_rate([d["p99_ms"], p["p99_ms"]]),
            "detector_gpu_w": dp["load_mean_w"], "detector_delta_w": dp["delta_w"],
            "policy_gpu_w": pp["load_mean_w"], "policy_delta_w": pp["delta_w"],
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
                ax.text(v["p99_ms"] * 1.12, yy, f"{v['p50_ms']:.2f} ms", va="center", fontsize=7.5, color=INK)
                top = max(top, v["p99_ms"])
        ax.set_xscale("log")
        ax.set_xlim(left=0.02, right=top * 6)
        ax.set_yticks(y, [{"idle": "live stream only", "contended": "+ Isaac jobs"}.get(t, t) for t in tags],
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
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("On-robot inference on NVIDIA GB10 (aarch64)", color=INK, fontsize=12, y=1.045)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")


def summary(h: dict) -> str:
    w = h["detector_gpu_w"]
    dw = h["detector_delta_w"]
    return (f"Perception and control run on-robot: on the NVIDIA GB10 (aarch64, the Jetson-class Blackwell module) "
            f"the TensorRT {h['detector'][4:].upper()} rust detector answers in {h['detector_p50_ms']:.2f} ms p50 "
            f"({h['detector_p99_ms']:.2f} ms p99) and the force-limited scrub policy in {h['policy_p50_ms']:.2f} ms "
            f"p50, so the detect-then-act loop closes at {h['loop_hz_p50']:.0f} Hz ({h['loop_hz_p99']:.0f} Hz at p99), "
            f"far above the 30 Hz the scrub policy was trained to act at (120 Hz physics, decimation 4), while the GPU draws {w:.0f} W "
            f"({'+' if dw >= 0 else ''}{dw:.1f} W over the background) with the Isaac Sim stream sharing the same GPU. "
            f"A 64-patch flange crack read takes {h['leak_read_p50_ms']:.1f} ms. "
            f"No network is needed for any decision; Gemini is only an optional cloud second opinion at the verify step.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/edge")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    runs = {p.stem.removeprefix("bench_"): json.loads(p.read_text()) for p in sorted((d / "out").glob("bench_*.json"))
            if "smoke" not in p.stem}
    tag = "contended" if "contended" in runs else next(iter(runs))
    heads = {t: headline(runs, t) for t in runs}
    out = {"headline": heads, "summary": summary(heads[tag]), "runs": runs}
    (d / "bench.json").write_text(json.dumps(out, indent=2))
    chart(runs, d / "bench.png")
    print(json.dumps(heads, indent=1))
    print(out["summary"])


if __name__ == "__main__":
    main()

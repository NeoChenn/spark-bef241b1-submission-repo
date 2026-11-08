from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from policy.gasbot.mock import episode


def _run(args):
    seed, scenario, max_t, *kw = args
    return episode(seed, scenario, max_t=max_t, **(kw[0] if kw else {}))


def summarise(rs: list[dict]) -> dict:
    n = len(rs)
    det = [r for r in rs if r["detected"]]
    fnd = [r for r in rs if r["t_found"] is not None]
    fix = [r for r in rs if r["t_fixed"] is not None]
    pct = lambda v, q: None if not v else round(float(np.percentile(v, q)), 2)
    err = [r["loc_err"] for r in fnd]
    ttd = [r["t_detect"] for r in det]
    tts = [r["t_found"] - r["t_detect"] for r in fnd]
    ttf = [r["t_fixed"] for r in fix]
    flow = [(r["flow_est"], r["flow_gs"]) for r in fnd if r["flow_est"] is not None and r["flow_gs"] > 0]
    fc = [r["flow_conc"] / r["flow_gs"] for r in fnd if r["flow_conc"] is not None and r["flow_gs"] > 0]
    return {"n": n, "detected": len(det), "found": len(fnd), "fixed": len(fix),
            "flange_correct": sum(bool(r["flange_ok"]) for r in fnd),
            "loc_err_m": {"median": pct(err, 50), "p90": pct(err, 90)},
            "t_detect_s": {"median": pct(ttd, 50), "p90": pct(ttd, 90)},
            "t_source_s": {"median": pct(tts, 50), "p90": pct(tts, 90)},
            "t_fixed_s": {"median": pct(ttf, 50), "p90": pct(ttf, 90)},
            "flow_ratio_acoustic": None if not flow else [pct([a / b for a, b in flow], q) for q in (25, 50, 75)],
            "flow_ratio_conc": None if not fc else [pct(fc, q) for q in (25, 50, 75)],
            "finals": dict(Counter(r["final"] for r in rs)),
            "alerts": dict(Counter(a for r in rs for a in r["alerts"]))}


def plots(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fg, mut, grid = "#e6e6e6", "#9a9a9a", "#3a3a3a"
    acc, acc2, acc3 = "#d97757", "#6a9bcc", "#788c5d"
    plt.rcParams.update({"text.color": fg, "axes.labelcolor": fg, "xtick.color": mut, "ytick.color": mut,
                         "axes.edgecolor": grid, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    rs = results["leak"]
    fig, ax = plt.subplots(figsize=(7, 4))
    T = np.arange(0, 121, 2.0)
    curves = []
    for r in rs:
        if not r["track"]:
            continue
        t, e = np.array([(a, b) for a, b, _ in r["track"]]).T
        curves.append(np.interp(T, t, e, left=np.nan, right=e[-1]))
        ax.plot(t, e, color=acc2, alpha=0.12, lw=0.8)
    if curves:
        C = np.array(curves)
        ax.plot(T, np.nanmedian(C, 0), color=acc, lw=2.2, label="median")
        ax.fill_between(T, np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0), color=acc, alpha=0.18,
                        label="25-75%")
    ax.set(xlabel="time since first detection (s)", ylabel="source estimate error (m)", xlim=(0, 120), ylim=(0, 8))
    ax.grid(color=grid, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title(f"Localisation error vs time ({len(curves)} leaks)", color=fg, loc="left")
    fig.tight_layout()
    fig.savefig(out / "loc_error_vs_time.png", transparent=True, dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, key, title, col in ((axs[0], "t_detect", "time to detect (s)", acc2),
                                (axs[1], "t_source", "detect -> source (s)", acc),
                                (axs[2], "loc_err", "error at hand-off (m)", acc3)):
        v = [r["t_found"] - r["t_detect"] if key == "t_source" else r[key] for r in rs
             if (r["t_found"] if key != "t_detect" else r["t_detect"]) is not None]
        if v:
            ax.hist(v, bins=15, color=col, edgecolor="none")
            ax.axvline(np.median(v), color=fg, lw=1, ls="--")
        ax.set_title(title, color=fg, loc="left", fontsize=10)
        ax.grid(color=grid, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out / "timing_hist.png", transparent=True, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    from policy.gasbot import deck
    for c, r in deck.KEEP_OUT:
        ax.add_patch(plt.Circle(c, r, color=grid))
    ax.axhline(0.0, color=mut, lw=3)
    ax.axhline(3.2, color=mut, lw=2)
    ex = next((r for r in rs if r["t_fixed"] is not None), rs[0])
    p = np.array(ex["path"])
    ax.plot(p[:, 0], p[:, 1], color=acc2, lw=1.2, label="unit 2 path")
    for g in deck.flanges():
        ax.plot(g.center[0], g.center[1], "s", color=fg, ms=5)
        ax.annotate(g.id, g.center[:2] + [0.1, 0.12], color=mut, fontsize=8)
    ax.plot(*ex["source"][:2], "*", color=acc, ms=14, label=f"leak {ex['fid']} {ex['flow_gs']:.1f} g/s")
    w = np.array([np.cos(ex["wind_dir"]), np.sin(ex["wind_dir"])])
    ax.annotate("", xy=(6.5 + 0.8 * w[0], 3.0 + 0.8 * w[1]), xytext=(6.5, 3.0),
                arrowprops={"arrowstyle": "->", "color": fg})
    ax.text(6.0, 3.3, "wind", color=fg, fontsize=8)
    ax.set(xlim=(-7.8, 7.8), ylim=(-0.5, 3.8), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "example_path.png", transparent=True, dpi=160)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-none", type=int, default=60)
    ap.add_argument("--max-t", type=float, default=600.0)
    ap.add_argument("--out", default="runs/leakbot")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    jobs = {"leak": [(1000 + s, "loose", a.max_t) for s in range(a.n)],
            "none": [(5000 + s, "none", 300.0) for s in range(a.n_none)],
            "big": [(8000 + s, "big", a.max_t) for s in range(max(a.n // 5, 1))],
            "crack": [(9000 + s, "crack", a.max_t) for s in range(max(a.n // 5, 1))],
            "leak_no_ultrasound": [(1000 + s, "loose", a.max_t, {"acoustic": False}) for s in range(a.n)],
            "none_no_ultrasound": [(5000 + s, "none", 300.0, {"acoustic": False}) for s in range(a.n_none)]}
    with ProcessPoolExecutor(a.workers) as ex:
        results = {k: list(ex.map(_run, v)) for k, v in jobs.items()}
    summary = {k: summarise(v) for k, v in results.items()}
    for k in ("none", "none_no_ultrasound"):
        summary[k]["false_alarms"] = sum(r["detected"] for r in results[k])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "eval_episodes.json").write_text(json.dumps(results, default=float))
    plots(results, out / "plots")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

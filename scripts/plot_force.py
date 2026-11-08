from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy.force_tools import crossing, intervals, saturation_point

INK, INK2, MUTED, GRID, AXIS = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"
CRIT, WARN = "#d03b3b", "#fab219"
LIMIT, CAP = 40.0, 38.0


def style():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlelocation": "left", "axes.labelsize": 11,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 10,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#4a4a46", "grid.alpha": 0.5, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
        "font.family": "DejaVu Sans"})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def preview(out: Path, name: str):
    from PIL import Image

    im = Image.open(out / f"{name}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (26, 26, 25, 255))
    bg.alpha_composite(im)
    (out / "preview").mkdir(exist_ok=True)
    bg.convert("RGB").save(out / "preview" / f"{name}.png")


def limit_lines(ax, label=True):
    ax.axhline(LIMIT, color=CRIT, lw=1.4, ls="--")
    ax.axhline(CAP, color=WARN, lw=1.2, ls=":")
    if label:
        x = ax.get_xlim()[1]
        ax.text(x, LIMIT + 0.8, "40 N contact limit", color=CRIT, ha="right", va="bottom", fontsize=9)
        ax.text(x, CAP - 0.8, "38 N setpoint cap", color=WARN, ha="right", va="top", fontsize=9)


def force_vs_time(run: Path, out: Path):
    rows = [json.loads(l) for l in open(run / "force_physics.jsonl")]
    t = np.array([r["t"] for r in rows])
    f = np.array([r["force_n"] for r in rows])
    st = np.array([r["state"] for r in rows])
    active = np.isin(st, ["SCRUB", "BACK_OFF"])
    if not active.any():
        return {}
    i0, i1 = np.argmax(active), len(active) - np.argmax(active[::-1]) - 1
    m = slice(i0, i1 + 1)
    tt, ff, ss = t[m], f[m], st[m]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    other = ~np.isin(ss, ["SCRUB", "BACK_OFF"])
    for a, b in intervals(list(tt), list(other)):
        ax.axvspan(a, b, color=AXIS, alpha=0.35, lw=0)
    bo = intervals(list(tt), list(ss == "BACK_OFF"))
    for a, b in bo:
        ax.axvspan(a - 0.05, b + 0.05, color=CRIT, alpha=0.22, lw=0)
    ax.plot(tt, ff, color=BLUE, lw=0.9, label="brush-tip contact sensor (120 Hz)")
    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylim(0, max(60.0, float(ff.max()) * 1.08))
    limit_lines(ax)
    k = int(np.argmax(ff))
    ax.annotate(f"weld bead spike {ff[k]:.0f} N", (tt[k], ff[k]), xytext=(12, -4), textcoords="offset points",
                color=INK, fontsize=9, va="center")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("normal force (N)")
    ax.set_title("Scrub contact force, live repair run (v2 policy, sensor in the loop)", pad=30)
    from matplotlib.patches import Patch

    handles = ax.get_legend_handles_labels()[0] + [Patch(color=CRIT, alpha=0.3, label="BACK_OFF (retract 5 cm)"),
                                                   Patch(color=AXIS, alpha=0.6, label="arm parked: re-detect / spray / verify")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=9)
    save(fig, out, "force_vs_time")
    preview(out, "force_vs_time")
    sc = ss == "SCRUB"
    return {"peak_n": float(ff.max()), "frac_scrub_over_40": float((ff[sc] > LIMIT).mean()),
            "back_off_windows": bo}


def hist(force_dir: Path, v1: Path, v2: Path, out: Path):
    series = []
    for label, path, key, col in (
            ("v1 (model_149), analytic force", v1, "max_force", ORANGE),
            ("v2, analytic force", v2, "max_force", BLUE),
            ("v2, contact sensor in the loop", force_dir / "episodes_v2_sensor.json", "max_force", AQUA)):
        if path.exists():
            series.append((label, np.asarray(json.load(open(path))[key], float), col))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bins = np.arange(0, 92, 2.0)
    stats = {}
    for label, x, col in series:
        ax.hist(np.clip(x, 0, 90), bins=bins, histtype="step", lw=2, color=col,
                weights=np.full(len(x), 100.0 / len(x)),
                label=f"{label}  (n={len(x)}, median {np.median(x):.1f} N, {100 * (x > LIMIT).mean():.1f}% > 40 N)")
        stats[label] = {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                        "p95": float(np.percentile(x, 95)), "max": float(x.max()),
                        "frac_over_40": float((x > LIMIT).mean())}
    ax.axvline(LIMIT, color=CRIT, lw=1.4, ls="--")
    ax.axvline(CAP, color=WARN, lw=1.2, ls=":")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.35)
    ax.text(LIMIT + 0.6, ymax * 0.55, "40 N limit", color=CRIT, va="top", fontsize=9)
    ax.text(CAP - 0.6, ymax * 0.55, "38 N cap", color=WARN, va="top", ha="right", fontsize=9)
    ax.set_xlim(15, 90)
    ax.set_xlabel("peak contact force per episode (N)")
    ax.set_ylabel("episodes (%)")
    ax.set_title("Peak scrub force per episode: v1 vs force-limited v2")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, out, "peak_force_hist")
    preview(out, "peak_force_hist")
    return stats


def sweep(force_dir: Path, out: Path):
    res = {}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4.8))
    for tag, col, ls in (("sensor", AQUA, "-"), ("analytic", VIOLET, "--")):
        p = force_dir / f"sweep_{tag}.json"
        if not p.exists():
            continue
        rows = json.load(open(p))["rows"]
        sp = np.array([r["setpoint_n"] for r in rows])
        rate = np.array([r["removal_rate_per_s"] for r in rows]) * 100
        mean_f = np.array([r["mean_contact_n"] for r in rows])
        p95 = np.array([r["p95_contact_n"] for r in rows])
        peak = np.array([r["env_peak_p95_n"] for r in rows])
        name = "policy sees contact sensor" if tag == "sensor" else "policy sees analytic spring force"
        a1.plot(sp, rate, color=col, ls=ls, marker="o", ms=5, label=name)
        a2.plot(sp, mean_f, color=col, ls=ls, marker="o", ms=5, label=f"{name}: mean (band to p95)")
        a2.fill_between(sp, mean_f, p95, color=col, alpha=0.15, lw=0)
        a2.plot(sp, peak, color=col, ls=ls, lw=1.2, alpha=0.8, label=f"{name}: p95 of per-patch peak")
        over = np.array([r["frac_contact_over_40n"] for r in rows]) * 100
        a3.plot(sp, over, color=col, ls=ls, marker="o", ms=5, label=name)
        sat = saturation_point(sp, rate, 0.95)
        cross_mean = crossing(sp, mean_f, LIMIT)
        cross_peak = crossing(sp, peak, LIMIT)
        res[tag] = {"saturation_setpoint_n": sat, "saturation_force_n": float(np.interp(sat, sp, mean_f)) if sat else None,
                    "setpoint_where_mean_crosses_40": cross_mean, "setpoint_where_p95_peak_crosses_40": cross_peak,
                    "max_rate_pct_per_s": float(rate.max()), "rows": rows}
        if sat is not None and tag == "sensor":
            a1.axvline(sat, color=col, lw=1, ls=":")
            a1.text(sat, rate.max() * 0.25, f" 95% of max rate\n at F* = {sat:.0f} N", color=INK2, fontsize=9)
    for ax in (a1, a3):
        ax.axvline(LIMIT, color=CRIT, lw=1.4, ls="--")
        ax.axvline(CAP, color=WARN, lw=1.2, ls=":")
        ax.text(LIMIT + 0.4, 0.02, "40 N", color=CRIT, fontsize=9, transform=ax.get_xaxis_transform())
        ax.text(CAP - 0.4, 0.02, "38 N cap", color=WARN, fontsize=9, ha="right", transform=ax.get_xaxis_transform())
    a3.set_xlabel("commanded force setpoint F* (N)")
    a3.set_ylabel("contact time above 40 N (%)")
    a3.set_title("Safety: time over the 40 N limit")
    a3.set_ylim(bottom=0)
    a3.legend(loc="upper left", fontsize=9)
    a1.set_xlabel("commanded force setpoint F* (N)")
    a1.set_ylabel("rust removed (% of patch per s, first 3 s)")
    a1.set_title("Scrub effectiveness vs force setpoint")
    a1.legend(loc="upper left", fontsize=9)
    a1.set_ylim(bottom=0)
    lim = [0, 50]
    a2.plot(lim, lim, color=MUTED, lw=1, ls=":", label="measured = commanded")
    a2.set_xlim(*lim)
    a2.set_ylim(0, 70)
    limit_lines(a2)
    a2.set_xlabel("commanded force setpoint F* (N)")
    a2.set_ylabel("measured contact force (N)")
    a2.set_title("Force achieved at the brush tip")
    a2.legend(loc="upper left", fontsize=9)
    save(fig, out, "force_sweep")
    preview(out, "force_sweep")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/repair")
    ap.add_argument("--force", default="runs/repair/force")
    ap.add_argument("--v1", default="runs/scrub_watson/episodes_rig_meta_forcepatch.json")
    ap.add_argument("--v2", default="runs/repair/v2play/episodes_meta.json")
    ap.add_argument("--out", default="runs/repair/plots")
    a = ap.parse_args()
    style()
    out = Path(a.out)
    res = {}
    if (Path(a.run) / "force_physics.jsonl").exists():
        res["force_vs_time"] = force_vs_time(Path(a.run), out)
    res["peak_hist"] = hist(Path(a.force), Path(a.v1), Path(a.v2), out)
    res["sweep"] = sweep(Path(a.force), out)
    json.dump(res, open(out / "force_characterisation.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1, default=float))
    for tag, r in res["sweep"].items():
        print(tag, {k: v for k, v in r.items() if k != "rows"})


if __name__ == "__main__":
    main()

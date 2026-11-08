from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/storm")
RUNS = [("v2 calm", "v2_s0"), ("v2 storm 0.5", "v2_s0.5"), ("v2 storm 1.0", "v2_s1.0"),
        ("v2+gate storm 1.0", "v2gate_s1.0"), ("v3 calm", "v3_s0"), ("v3 storm 0.5", "v3_s0.5"),
        ("v3 storm 1.0", "v3_s1.0"), ("v3+gate storm 1.0", "v3gate_s1.0")]
COLORS = {"v2": "#2a78d6", "v3": "#eb6834", "gate": "#1baf7a"}


def color(name):
    return COLORS["gate"] if "gate" in name else COLORS[name[:2]]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(d):
    e, p = ROOT / d / "eval_meta.json", ROOT / d / "episodes_meta.json"
    if not e.exists():
        return None
    return json.loads(e.read_text()), json.loads(p.read_text()) if p.exists() else None


def main():
    rows = [(name, *load(d)) for name, d in RUNS if load(d)]
    keys = ["success_rate", "mean_time_to_clean_s", "mean_episode_max_force_N", "p95_episode_max_force_N",
            "frac_episodes_over_40N", "max_force_N", "episodes"]
    summary = {name: {k: ev[k] for k in keys} for name, ev, _ in rows}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'run':<14}" + "".join(f"{k[:14]:>16}" for k in keys))
    for name, ev, _ in rows:
        print(f"{name:<14}" + "".join(f"{ev[k]:>16.3f}" for k in keys))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="#fcfcfb")
    names = [n for n, *_ in rows]
    x = np.arange(len(names))
    col = [color(n) for n in names]
    for ax, key, label, scale, limit in (
            (axes[0], "success_rate", "success (%)", 100, None),
            (axes[1], "p95_episode_max_force_N", "p95 episode max force (N)", 1, 40.0),
            (axes[2], "frac_episodes_over_40N", "episodes over 40 N (%)", 100, None)):
        vals = [summary[n][key] * scale for n in names]
        ax.bar(x, vals, color=col, width=0.62, edgecolor="#fcfcfb", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        if limit:
            ax.axhline(limit, color=MUTED, ls="--", lw=1)
            ax.text(-0.45, limit, "40 N limit", ha="left", va="bottom", fontsize=8, color=MUTED)
        ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in names], fontsize=8)
        ax.set_title(label, fontsize=10, loc="left", color=INK)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Scrub policy under storm (gusts on arm, deck sway/heave, wet pipe), rig geometry, 1024 episodes each",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "storm_eval.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb")
    for name, ev, ep in rows:
        if ep is None or name.endswith("0.5"):
            continue
        f = np.sort(np.asarray(ep["max_force"]))
        ax.plot(f, np.arange(1, len(f) + 1) / len(f), color=color(name), lw=2,
                ls="-" if "storm" in name else ":", label=name)
    ax.axvline(40, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("episode max contact force (N)")
    ax.set_ylabel("fraction of episodes")
    ax.set_xlim(15, 60)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor("#fcfcfb")
    ax.set_title("Episode max force, calm (dotted) vs storm 1.0 (solid)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "storm_force_cdf.png", dpi=150)


if __name__ == "__main__":
    main()

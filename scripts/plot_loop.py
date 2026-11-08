from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy.force_tools import intervals
from scripts.plot_force import AXIS, BLUE, CRIT, INK2, MUTED, VIOLET, limit_lines, preview, save, style


def mission_force(run: Path, out: Path) -> dict:
    rows = [json.loads(l) for l in open(run / "force_physics.jsonl")]
    spans = [json.loads(l) for l in open(run / "phases.jsonl")]
    t = np.array([r["t"] for r in rows])
    f = np.array([r["force_n"] for r in rows])
    mo = [r.get("motion") or "" for r in rows]
    style()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, max(50.0, float(f.max()) * 1.1))
    main = [s for s in spans if s["phase"] not in ("PARK", "UNPARK")]
    for k, s in enumerate(main):
        t1 = s["t1"] if s["t1"] is not None else t[-1]
        if k % 2:
            ax.axvspan(s["t0"], t1, color=AXIS, alpha=0.35, lw=0)
        if t1 - s["t0"] > 0.04 * (t[-1] - t[0]):
            ax.text(0.5 * (s["t0"] + t1), ax.get_ylim()[1] * 0.99, s["phase"].replace("_", " "), ha="center",
                    va="top", color=INK2, fontsize=8)
    for a, b in intervals(list(t), [m in ("PARK", "UNPARK") for m in mo]):
        ax.axvspan(a, b, color=VIOLET, alpha=0.28, lw=0)
    ax.plot(t, f, color=BLUE, lw=0.9, label="brush-tip contact sensor (120 Hz)")
    limit_lines(ax)
    k = int(np.argmax(f))
    ax.annotate(f"peak {f[k]:.1f} N", (t[k], f[k]), xytext=(8, -2), textcoords="offset points", color=MUTED, fontsize=9, va="top")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("normal force (N)")
    ax.set_title("Mission loop contact force: walk to paint, one Isaac Lab run", pad=30)
    from matplotlib.patches import Patch

    handles = ax.get_legend_handles_labels()[0] + [Patch(color=VIOLET, alpha=0.4, label="arm park / unpark (staged path)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9)
    save(fig, out, "mission_force")
    preview(out, "mission_force")
    return {"peak_n": float(f.max()), "crit": CRIT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/loop/r2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = Path(a.run)
    print(mission_force(run, Path(a.out) if a.out else run / "plots"))


if __name__ == "__main__":
    main()

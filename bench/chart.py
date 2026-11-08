from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"]
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#8a897f"
GRID = "#3a3a37"


def _style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def bar_panels(path, title: str, methods: list[str], panels: list[dict], note: str = "", colors=None):
    colors = colors or {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)}
    n = len(panels)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 0.42 * len(methods) * rows + 1.1 * rows + 0.5),
                             squeeze=False)
    fig.patch.set_alpha(0.0)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False)
            continue
        p = panels[k]
        _style(ax)
        ys = list(range(len(methods)))[::-1]
        vmax = 0.0
        for y, m in zip(ys, methods):
            v = p["values"].get(m)
            if v is None or v.get("mean") is None or (isinstance(v["mean"], float) and math.isnan(v["mean"])):
                ax.text(0.0, y, "  not run" if v is None or not v.get("label") else f"  {v['label']}",
                        va="center", ha="left", color=MUTED, fontsize=7.5)
                continue
            mval = v["mean"] * p.get("scale", 1.0)
            lo = v.get("lo", v["mean"]) * p.get("scale", 1.0)
            hi = v.get("hi", v["mean"]) * p.get("scale", 1.0)
            if math.isnan(lo):
                lo = hi = mval
            ax.barh(y, mval, height=0.62, color=colors[m], edgecolor="none")
            if hi > lo:
                ax.errorbar(mval, y, xerr=[[mval - lo], [hi - mval]], fmt="none", ecolor=INK, elinewidth=1.0,
                            capsize=2.5, capthick=1.0)
            vmax = max(vmax, hi, mval)
            ax.text(max(hi, mval) , y, "  " + p.get("fmt", "{:.2f}").format(mval), va="center", ha="left",
                    color=INK, fontsize=7.5)
        if p.get("limit") is not None:
            ax.axvline(p["limit"], color="#e66767", linewidth=1.0, linestyle=(0, (3, 2)))
            vmax = max(vmax, p["limit"])
        ax.set_xlim(0, (vmax or 1.0) * 1.32)
        ax.set_yticks(ys)
        ax.set_yticklabels(methods if k % cols == 0 else [""] * len(methods), color=INK2, fontsize=8)
        ax.set_title(p["title"], color=INK, fontsize=9, loc="left", pad=6)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left", y=0.995)
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, transparent=True)
    plt.close(fig)
    return str(path)

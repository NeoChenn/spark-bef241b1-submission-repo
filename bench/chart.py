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
LIMIT = "#e66767"


def palette(methods: list[str], fixed: dict | None = None) -> dict:
    fixed = dict(fixed or {})
    free = [c for c in SERIES if c not in fixed.values()]
    for m in methods:
        if m not in fixed:
            fixed[m] = free.pop(0)
    return fixed


def _style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _missing(v):
    return v is None or v.get("mean") is None or (isinstance(v["mean"], float) and math.isnan(v["mean"]))


def bar_panels(path, title: str, methods: list[str], panels: list[dict], note: str = "", colors=None, cols: int = 3):
    colors = palette(methods, colors)
    n = len(panels)
    cols = min(n, cols)
    rows = math.ceil(n / cols)
    h_ax = 0.38 * len(methods) + 0.55
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols + 0.9, h_ax * rows + 0.7), squeeze=False)
    fig.patch.set_alpha(0.0)
    ys = list(range(len(methods)))[::-1]
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False)
            continue
        p = panels[k]
        _style(ax)
        sc = p.get("scale", 1.0)
        vmax, vmin = 0.0, math.inf
        for y, m in zip(ys, methods):
            v = p["values"].get(m)
            if _missing(v):
                continue
            mval = v["mean"] * sc
            lo, hi = v.get("lo"), v.get("hi")
            lo = mval if lo is None or math.isnan(lo) else lo * sc
            hi = mval if hi is None or math.isnan(hi) else hi * sc
            vmax, vmin = max(vmax, hi, mval), min(vmin, mval if mval > 0 else vmin)
        if p.get("limit") is not None:
            vmax = max(vmax, p["limit"])
        log = p.get("log", False)
        if log:
            ax.set_xscale("log")
            x0 = 10 ** math.floor(math.log10(vmin if vmin < math.inf else 1.0)) / 3
            ax.set_xlim(x0, (vmax or 1.0) * 30)
        else:
            x0 = 0.0
            ax.set_xlim(0, (vmax or 1.0) * 1.3)
        for y, m in zip(ys, methods):
            v = p["values"].get(m)
            if _missing(v):
                ax.text(0.02, y, (v or {}).get("label") or "not run", transform=ax.get_yaxis_transform(),
                        va="center", ha="left", color=MUTED, fontsize=7.5)
                continue
            mval = v["mean"] * sc
            lo, hi = v.get("lo"), v.get("hi")
            lo = mval if lo is None or math.isnan(lo) else lo * sc
            hi = mval if hi is None or math.isnan(hi) else hi * sc
            ax.barh(y, mval - x0 if not log else mval - x0, left=x0, height=0.6, color=colors[m], edgecolor="none")
            if hi > lo:
                ax.errorbar(mval, y, xerr=[[mval - lo], [hi - mval]], fmt="none", ecolor=INK, elinewidth=0.9,
                            capsize=2.5, capthick=0.9)
            ax.annotate(p.get("fmt", "{:.2f}").format(mval), (max(hi, mval), y), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left", color=INK, fontsize=7.5)
        if p.get("limit") is not None:
            ax.axvline(p["limit"], color=LIMIT, linewidth=1.0, linestyle=(0, (3, 2)))
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels(methods if k % cols == 0 else [""] * len(methods), color=INK2, fontsize=8)
        ax.set_title(p["title"], color=INK, fontsize=9, loc="left", pad=5)
    fig.text(0.01, 0.985, title, color=INK, fontsize=11, ha="left", va="top")
    if note:
        fig.text(0.01, 0.008, note, color=MUTED, fontsize=7, ha="left", va="bottom")
    top = 1 - 0.45 / (h_ax * rows + 0.7)
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, top), h_pad=1.2, w_pad=1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, transparent=True)
    plt.close(fig)
    return str(path)

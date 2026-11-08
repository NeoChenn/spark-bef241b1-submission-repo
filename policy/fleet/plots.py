from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1 = "#2a78d6"
S2 = "#eb6834"


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reroute_plot(reroutes: list[dict], path: str) -> None:
    rows = [r for r in reroutes if r["extra_m"] is not None]
    short = lambda c: c.replace("weather ", "").replace(" + cleared", " + clear").replace("cleared", "clear")
    labels = [f"t={r['t']:.0f}s {r['robot']}\n{short(r['cause'])}" for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, key, unit, title in ((axs[0], "extra_m", "m", "Extra distance per re-route"),
                                 (axs[1], "extra_s", "s", "Extra time per re-route")):
        _axes(ax)
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(vals)), vals, color=S1, width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.axhline(0, color=INK2, linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.0f} {unit}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(vals)), labels, fontsize=7, color=INK2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylabel(unit, color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def completion_plot(curves: dict[str, list[tuple[float, int]]], path: str, events=()) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    _axes(ax)
    for (name, pts), col in zip(curves.items(), (S1, S2)):
        ts = [0.0] + [t for t, _ in pts]
        ns = [0] + [n for _, n in pts]
        ax.step(ts, ns, where="post", color=col, linewidth=2, label=name)
        ax.annotate(f"{name}: {ns[-1]} jobs by {ts[-1]:.0f} s", (ts[-1], ns[-1]), xytext=(-6, 6),
                    textcoords="offset points", fontsize=8, color=INK, ha="right")
    for t, label in events:
        ax.axvline(t, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.annotate(label, (t, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", fontsize=7,
                    color=INK2)
    ax.set_xlabel("mission time (s)", color=INK2, fontsize=9)
    ax.set_ylabel("jobs completed", color=INK2, fontsize=9)
    ax.set_title("Task completion with and without disturbances", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

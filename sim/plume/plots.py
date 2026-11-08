import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from sim.plume.colormap import COLORS, STOPS
from sim.plume.release import orifice

FG, DIM, FAINT, GRIDC, SOLID = "#F5F4EE", "#B8B6AE", "#7a7972", "#3a3a37", "#5b5a55"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
LEL_CMAP = LinearSegmentedColormap.from_list("lel", [(s / 100, c / 255) for s, c in zip(STOPS, COLORS)])


def style():
    plt.rcParams.update({"text.color": FG, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                         "axes.edgecolor": GRIDC, "axes.facecolor": "none", "figure.facecolor": "none",
                         "savefig.transparent": True, "font.size": 10, "axes.grid": True, "grid.color": GRIDC,
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 2.0, "axes.titlesize": 11,
                         "axes.titlelocation": "left", "axes.titlecolor": FG})


def load(run: Path) -> dict:
    d = dict(np.load(run / "run.npz"))
    d["summary"] = json.loads((run / "summary.json").read_text())
    return d


def slices(d: dict, out: Path, when: float | None = None):
    k_t = len(d["snap_t"]) - 1 if when is None else int(np.argmin(np.abs(d["snap_t"] - when)))
    pct = d["snaps"][k_t].astype(np.float32) / float(d["lel"]) * 100
    solid, o, dx, src = d["solid"], d["origin"], float(d["dx"]), d["source"]
    vel = d["velocity"].astype(np.float32)
    n = np.asarray(pct.shape)
    ijk = np.clip(((src - o) / dx).astype(int), 0, n - 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [n[0], n[1] * 1.0]})
    ext_xy = [o[0], o[0] + n[0] * dx, o[1], o[1] + n[1] * dx]
    ext_yz = [o[1], o[1] + n[1] * dx, o[2], o[2] + n[2] * dx]
    for ax, img, sol, ext, uv, title, lab in [
        (axes[0], pct[:, :, ijk[2]].T, solid[:, :, ijk[2]].T, ext_xy, vel[:, :, ijk[2], :2].transpose(1, 0, 2),
         f"Plan at leak height z = {o[2] + (ijk[2] + 0.5) * dx:.1f} m", ("x (m)", "y (m)")),
        (axes[1], pct[ijk[0]].T, solid[ijk[0]].T, ext_yz, vel[ijk[0], :, :, 1:].transpose(1, 0, 2),
         f"Section along the wind, x = {o[0] + (ijk[0] + 0.5) * dx:.1f} m", ("y (m)", "z (m)"))]:
        m = np.ma.masked_where(img < 1.0, img)
        ax.imshow(np.where(sol, 1.0, np.nan), extent=ext, origin="lower", cmap=LinearSegmentedColormap.from_list("s", [SOLID, SOLID]), interpolation="nearest")
        im = ax.imshow(m, extent=ext, origin="lower", cmap=LEL_CMAP, vmin=0, vmax=100, interpolation="bilinear")
        xs = np.linspace(ext[0], ext[1], img.shape[1])
        ys = np.linspace(ext[2], ext[3], img.shape[0])
        sp = np.hypot(uv[..., 0], uv[..., 1])
        ax.streamplot(xs, ys, uv[..., 0], uv[..., 1], color=FG, linewidth=0.45, density=1.1, arrowsize=0.6)
        ax.set_xlim(ext[:2])
        ax.set_ylim(ext[2:])
        ax.set_title(title)
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
        ax.grid(False)
        ax.plot(*(src[[0, 1]] if ax is axes[0] else src[[1, 2]]), "o", ms=7, mfc="none", mec=FG, mew=1.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("% LEL methane (100% = 5% vol)")
    cb.outline.set_visible(False)
    s = d["summary"]
    fig.suptitle(f"Flange {s['args']['flange']} leak, {s['args']['hole_mm']:g} mm at {s['args']['bar']:g} bar g, "
                 f"{s['flow']['kg_s'] * 1000:.1f} g/s, wind {s['wind']['speed_m_s']:g} m/s, t = {d['snap_t'][k_t]:.0f} s",
                 x=0.06, ha="left", color=FG, fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def volume(runs: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c, (name, d) in zip(SERIES, runs.items()):
        t = d["s_t"]
        axes[0].plot(t, d["s_vol_10lel_m3"], color=c, label=name)
        axes[1].plot(t, d["s_mass_kg"] * 1000, color=c, label=name)
    axes[0].set_title("Flammable cloud volume above 10% LEL")
    axes[0].set_ylabel("m³")
    axes[1].set_title("Methane held in the analysis domain")
    axes[1].set_ylabel("g")
    for ax in axes:
        ax.set_xlabel("time since leak start (s)")
        ax.set_yscale("log")
        ax.legend(loc="lower right")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validation(out: Path):
    from sim.plume.solver import Plume, Wind

    k, u, dx = 0.1, 1.0, 0.25
    rel = orifice(3.0, 20.0)
    src = np.array([4.0, 4.0, 4.0])
    p = Plume((0, 0, 0), (64, 32, 32), dx, src, rel, Wind(u, 0.0), diffusivity=k, buoyancy=False, jet=False, static_flow=True)
    for _ in range(220):
        p.step(0.1)
    xs = np.linspace(0.6, 11.5, 120)
    num = p.sample(np.stack([src[0] + xs, np.full_like(xs, src[1]), np.full_like(xs, src[2])], 1))
    ana = rel.volume_flow / (4 * math.pi * k * xs) * 1e6
    ys = np.linspace(-3, 3, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(xs, ana / 1e4, color=DIM, lw=1.5, ls="--", label="analytic  q / (4πKx)")
    axes[0].plot(xs, num / 1e4, color=SERIES[0], label="Warp solver")
    axes[0].set_ylim(0, 6)
    axes[0].set_title("Centreline concentration, open field")
    axes[0].set_xlabel("downwind distance x (m)")
    axes[0].set_ylabel("% vol methane")
    for c, x in zip(SERIES, (2.0, 4.0, 8.0)):
        sig = math.sqrt(2 * k * x / u)
        g = rel.volume_flow / (4 * math.pi * k * x) * np.exp(-ys ** 2 / (2 * sig ** 2)) * 1e6
        prof = p.sample(np.stack([np.full_like(ys, src[0] + x), src[1] + ys, np.full_like(ys, src[2])], 1))
        axes[1].plot(ys, g / 1e4, color=DIM, lw=1.2, ls="--")
        axes[1].plot(ys, prof / 1e4, color=c, label=f"x = {x:g} m")
    axes[1].set_title("Crosswind profile vs Gaussian plume (dashed)")
    axes[1].set_xlabel("crosswind y (m)")
    axes[1].set_ylabel("% vol methane")
    for ax in axes:
        ax.legend(loc="upper right")
    fig.text(0.06, -0.04, f"U = {u} m/s, K = {k} m²/s, dx = {dx} m, {rel.mass_flow * 1000:.1f} g/s point source", color=FAINT)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sel = (xs >= 2) & (xs <= 9)
    return float(np.max(np.abs(num[sel] / ana[sel] - 1)))


def leak_rate(out: Path):
    holes = np.linspace(0.5, 25, 200)
    bars = np.logspace(-1, 2, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c, b in zip(SERIES, (2, 10, 20, 60)):
        axes[0].plot(holes, [orifice(h, b).mass_flow for h in holes], color=c, label=f"{b} bar g")
    for c, h in zip(SERIES, (1, 3, 6, 12)):
        f = [orifice(h, b) for b in bars]
        axes[1].plot(bars, [r.mass_flow for r in f], color=c, label=f"{h} mm hole")
    choke = (101325 / 0.5427 - 101325) / 1e5
    axes[1].axvline(choke, color=FAINT, lw=1, ls=":")
    axes[1].text(choke * 1.1, 1e-5, "choked above", color=FAINT, fontsize=9)
    axes[0].set_title("Methane release rate vs hole size")
    axes[0].set_xlabel("equivalent hole diameter (mm)")
    axes[1].set_title("Release rate vs line pressure")
    axes[1].set_xlabel("line pressure (bar g)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("kg/s  (1 g/s ≈ 5.3 Sm³/h)")
        ax.legend(loc="lower right")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", default="runs/plume")
    a.add_argument("--names", default="base,storm,calm,big")
    a.add_argument("--out", default="runs/plume/plots")
    args = a.parse_args()
    style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.runs)
    runs = {n: load(root / f"F5_{n}") for n in args.names.split(",") if (root / f"F5_{n}" / "run.npz").exists()}
    labels = {"base": "3 mm, 20 bar, 5 m/s", "storm": "3 mm, 20 bar, storm 12 m/s gusting", "calm": "3 mm, 20 bar, 1.5 m/s",
              "big": "6 mm, 40 bar, 5 m/s"}
    for n, d in runs.items():
        slices(d, out / f"slices_{n}.png")
    volume({labels.get(n, n): d for n, d in runs.items()}, out / "plume_volume.png")
    err = validation(out / "validation_gaussian.png")
    leak_rate(out / "leak_rate.png")
    print(json.dumps({"validation_max_rel_err_2_9m": round(err, 4), "plots": sorted(p.name for p in out.glob("*.png"))}))


if __name__ == "__main__":
    main()

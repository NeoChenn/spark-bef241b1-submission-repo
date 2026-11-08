from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from policy.gasbot.mock import episode

MODES = ("cast_surge", "infotaxis")
CONDS = {"ch4_only": False, "ch4_ultrasound": True}


def _run(a):
    seed, mode, acoustic, max_t = a
    r = episode(seed, "loose", max_t=max_t, search=mode, acoustic=acoustic)
    return {k: r[k] for k in ("seed", "fid", "flow_gs", "detected", "t_detect", "t_found", "t_fixed", "loc_err",
                              "flange_ok", "final", "belief", "t_end", "track")}


def metrics(rs: list[dict]) -> dict:
    pct = lambda v, q: None if not v else round(float(np.percentile(v, q)), 2)
    det = [r for r in rs if r["detected"]]
    ok = [r for r in det if r["flange_ok"] and r["t_fixed"] is not None]
    fnd = [r for r in det if r["t_found"] is not None]
    tts = [r["t_found"] - r["t_detect"] for r in fnd]
    err = [r["loc_err"] for r in fnd if r["loc_err"] is not None]
    return {"n": len(rs), "detected": len(det), "success": len(ok),
            "success_rate": None if not det else round(len(ok) / len(det), 3),
            "escalated": sum(r["final"] == "ESCALATE_TO_HUMAN" for r in det),
            "wrong_flange": sum(r["flange_ok"] is False for r in det),
            "t_source_s": {"median": pct(tts, 50), "mean": None if not tts else round(float(np.mean(tts)), 2),
                           "p90": pct(tts, 90)},
            "loc_err_m": {"median": pct(err, 50), "p90": pct(err, 90)}}


def _median(series, T: np.ndarray) -> np.ndarray | None:
    curves = [np.interp(T, t, v, left=np.nan, right=v[-1]) for t, v in series if len(t)]
    return None if not curves else np.nanmedian(np.array(curves), 0)


def shrink_curve(rs: list[dict], T: np.ndarray) -> np.ndarray | None:
    return _median([np.array(r["belief"], float).T if r["belief"] else ([], []) for r in rs], T)


def error_curve(rs: list[dict], T: np.ndarray) -> np.ndarray | None:
    return _median([np.array([(a, b) for a, b, _ in r["track"]], float).T if r["track"] else ([], []) for r in rs], T)


def calibration(rs: list[dict]) -> float | None:
    v = [r["loc_err"] <= 2 * r["track"][-1][2] for r in rs if r["t_found"] is not None and r["track"]]
    return None if not v else round(float(np.mean(v)), 3)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fg, mut, grid = "#e6e6e6", "#9a9a9a", "#3a3a3a"
    plt.rcParams.update({"text.color": fg, "axes.labelcolor": fg, "xtick.color": mut, "ytick.color": mut,
                         "axes.edgecolor": grid, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt, fg, mut, grid


COL = {"cast_surge": "#6a9bcc", "infotaxis": "#d97757"}
LABEL = {"cast_surge": "cast-and-surge", "infotaxis": "PF + infotaxis"}


def charts(summary: dict, results: dict, out: Path) -> None:
    plt, fg, mut, grid = _style()
    panels = (("success_rate", "success rate (of detected)", lambda m: 100 * (m["success_rate"] or 0), "%.0f%%"),
              ("t_source_s", "detect -> source, median (s)", lambda m: m["t_source_s"]["median"] or 0, "%.1f"),
              ("loc_err_m", "loc. error at hand-off, median (m)", lambda m: m["loc_err_m"]["median"] or 0, "%.2f"))
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    x = np.arange(len(CONDS))
    for ax, (_, title, get, fmt) in zip(axs, panels):
        for i, mode in enumerate(MODES):
            v = [get(summary[c][mode]) for c in CONDS]
            b = ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=COL[mode], label=LABEL[mode])
            ax.bar_label(b, fmt=fmt, color=fg, fontsize=8, padding=2)
        ax.set_xticks(x, ["CH4 only", "CH4 + ultrasound"])
        ax.set_title(title, color=fg, loc="left", fontsize=10)
        ax.grid(color=grid, lw=0.4, alpha=0.6, axis="y")
        ax.margins(y=0.18)
        ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "chart.png", transparent=True, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    T = np.arange(0, 60.5, 0.5)
    rs = {m: [r for r in results["ch4_only"][m] if r["detected"]] for m in MODES}
    for mode in MODES:
        a = shrink_curve(rs[mode], T)
        e = error_curve(rs[mode], T)
        ax.plot(T, np.sqrt(a / np.pi), color=COL[mode], lw=2.2, label=f"{LABEL[mode]}: claimed 2-sigma radius")
        ax.plot(T, e, color=COL[mode], lw=1.4, ls="--", label=f"{LABEL[mode]}: true error")
    ax.set(xlabel="time since first detection (s)", ylabel="metres", xlim=(0, 60), ylim=(0, None))
    ax.grid(color=grid, lw=0.4, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Belief shrinkage vs truth, CH4 only (median over detected leaks)", color=fg, loc="left")
    fig.tight_layout()
    fig.savefig(out / "shrink.png", transparent=True, dpi=140)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-t", type=float, default=600.0)
    ap.add_argument("--out", default="runs_sample/novelty/gas")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    jobs = {c: {m: [(1000 + s, m, ac, a.max_t) for s in range(a.n)] for m in MODES} for c, ac in CONDS.items()}
    with ProcessPoolExecutor(a.workers) as ex:
        results = {c: {m: list(ex.map(_run, v)) for m, v in d.items()} for c, d in jobs.items()}
    summary = {c: {m: {**metrics(v), "truth_inside_2sigma_at_handoff": calibration(v)} for m, v in d.items()}
               for c, d in results.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = np.array([5.0, 10.0, 20.0, 30.0])
    shrink = {c: {m: [None if not np.isfinite(v) else round(float(v), 3)
                      for v in shrink_curve([r for r in d[m] if r["detected"]], T)] for m in MODES}
              for c, d in results.items()}
    res = {"benchmark": "policy/gasbot/evaluate.py 'leak' seeds 1000..1059, loose flange, max_t 600 s",
           "summary": summary, "median_ellipse_area_m2_at_s": {"t_s": T.tolist(), **shrink},
           "per_seed": {c: {m: [{k: r[k] for k in ("seed", "fid", "detected", "t_found", "t_detect", "loc_err",
                                                    "flange_ok", "final")} for r in d[m]] for m in MODES}
                        for c, d in results.items()}}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    charts(summary, results, out)
    print(json.dumps({"summary": summary, "shrink": shrink}, indent=1))
    return res


if __name__ == "__main__":
    main()

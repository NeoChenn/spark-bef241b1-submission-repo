from __future__ import annotations

import argparse
import json
import os


def summarise(ev: dict) -> dict:
    keys = ("episodes", "success_rate", "mean_episode_max_force_N", "p95_episode_max_force_N", "max_force_N",
            "frac_episodes_over_40N", "mean_time_to_clean_s")
    out = {k: ev.get(k) for k in keys}
    if "cbf_duty" in ev:
        out["cbf_duty"] = ev["cbf_duty"]
    return out


def build(evals: dict) -> dict:
    res = {"policy": "scrub v2 best (lift_v2)", "shield": "discrete-time CBF on normal penetration, RLS stiffness",
           "rows": {}}
    for name, ev in evals.items():
        res["rows"][name] = summarise(ev)
    for cond in ("calm", "storm"):
        b, a = res["rows"].get(f"{cond}_base"), res["rows"].get(f"{cond}_cbf")
        if b and a:
            res[f"{cond}_over40_pct"] = [round(100 * b["frac_episodes_over_40N"], 2), round(100 * a["frac_episodes_over_40N"], 2)]
            res[f"{cond}_success_pct"] = [round(100 * b["success_rate"], 2), round(100 * a["success_rate"], 2)]
    return res


def chart(res: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, base_c, cbf_c = "#f0efec", "#c3c2b7", "#8a8984", "#3987e5"
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=200)
    conds = [c for c in ("calm", "storm") if f"{c}_over40_pct" in res]
    for ax, key, title in ((axes[0], "over40_pct", "Episodes over 40 N (%)"), (axes[1], "success_pct", "Success (%)")):
        for i, c in enumerate(conds):
            b, a = res[f"{c}_{key}"]
            for j, (v, col) in enumerate(((b, base_c), (a, cbf_c))):
                x = i + (j - 0.5) * 0.36
                ax.bar(x, v, 0.32, color=col, edgecolor="none")
                ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", color=ink, fontsize=7)
        ax.set_xticks(range(len(conds)), [c if c == "calm" else "storm 1.0" for c in conds], color=ink, fontsize=8)
        ax.set_title(title, color=ink, fontsize=9, loc="left")
        ax.tick_params(colors=muted, labelsize=7, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(muted)
        ax.set_yticks([])
        top = max(max(res[f"{c}_{key}"]) for c in conds)
        ax.set_ylim(0, top * 1.22 + 0.5)
    from matplotlib.patches import Patch

    fig.legend([Patch(color=base_c), Patch(color=cbf_c)], ["PPO", "PPO + CBF shield"], loc="lower center", ncol=2,
               frameon=False, labelcolor=ink, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, transparent=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="runs_sample/novelty/scrub")
    a = p.parse_args()
    evals = {}
    for cond in ("calm", "storm"):
        for arm in ("base", "cbf"):
            f = os.path.join(a.dir, f"eval_meta_{cond}_{arm}{a.tag}.json")
            if os.path.exists(f):
                evals[f"{cond}_{arm}"] = json.load(open(f))
    res = build(evals)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    chart(res, os.path.join(a.out, "chart.png"))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()

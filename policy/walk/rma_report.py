from __future__ import annotations

import json
import os
import sys

LABELS = {"base": "PPO baseline", "rma": "RMA adapted", "oracle": "privileged teacher", "no_latent": "adapted, latent zeroed", "finetune_blind": "fine-tuned, no estimator"}


def build(ev: dict, extra: dict) -> dict:
    p = ev["policies"]
    b, a = p["base"], p["rma"]
    res = {"method": "RMA-style teacher-student: privileged payload+deck extrinsics -> history adapter (1-D conv, 1 s)",
           "benchmark": ev["benchmark"], "policies": p,
           "headline": {"falls_per_km": [b["falls_per_km"], a["falls_per_km"]],
                        "vel_err_xy_mps": [b["vel_err_xy_mps"], a["vel_err_xy_mps"]],
                        "latent_r2": a.get("latent_r2", {})}}
    res.update(extra)
    return res


def chart(res: dict, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fg, mute = "#e8ecf2", "#9aa4b2"
    names = [n for n in ("base", "finetune_blind", "rma", "oracle") if n in res["policies"]]
    cols = {"base": "#7d8796", "finetune_blind": "#a58bd6", "rma": "#4cc3ff", "oracle": "#3a5f7a"}
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    fig.patch.set_alpha(0)
    pol = res["policies"]
    r2 = pol["rma"].get("latent_r2", {})
    panels = [("falls / km", [pol[n]["falls_per_km"] for n in names], names),
              ("velocity error (m/s)", [pol[n]["vel_err_xy_mps"] for n in names], names),
              ("estimator R²", list(r2.values()), list(r2))]
    for ax, (title, vals, keys) in zip(axs, panels):
        ax.set_facecolor("none")
        if title.startswith("estimator"):
            ax.barh(range(len(vals)), vals, color="#4cc3ff")
            ax.set_yticks(range(len(vals)), [k.replace("_", " ") for k in keys], color=fg, fontsize=8)
            ax.set_xlim(min(0.0, min(vals)), 1.0)
            ax.invert_yaxis()
        else:
            bars = ax.bar(range(len(vals)), vals, color=[cols[k] for k in keys])
            ax.set_ylim(0, max(vals) * 1.18 + 1e-9)
            ax.set_xticks(range(len(vals)), [LABELS[k].replace(" ", "\n", 1) for k in keys], color=fg, fontsize=8)
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", color=fg, fontsize=9)
        ax.set_title(title, color=fg, fontsize=10)
        ax.tick_params(colors=mute)
        for s in ax.spines.values():
            s.set_color(mute)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, transparent=True)


def main(argv):
    ev = json.load(open(argv[0]))
    extra = json.loads(argv[2]) if len(argv) > 2 else {}
    os.makedirs(argv[1], exist_ok=True)
    res = build(ev, extra)
    json.dump(res, open(os.path.join(argv[1], "result.json"), "w"), indent=1)
    chart(res, os.path.join(argv[1], "chart.png"))


if __name__ == "__main__":
    main(sys.argv[1:])

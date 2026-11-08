from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from bench.stats import rate_ci, summarize


def _r(d: dict, nd: int = 4) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) and not math.isnan(v) else (None if isinstance(v, float) else v))
            for k, v in d.items()}


def scrub_summary(eps: list[dict], latency_ms=None, compute: str = "", cost=None) -> dict:
    n = len(eps)
    succ = [e["success"] for e in eps]
    ttc = [e["time_to_clean_s"] for e in eps if e["success"] and e["time_to_clean_s"] is not None]
    pf = np.array([e["peak_force_N"] for e in eps], dtype=float)
    out = {
        "episodes": n,
        "success": _r(summarize(succ, rate=True)),
        "time_to_clean_s": _r(summarize(ttc)),
        "peak_force_N": _r(summarize(pf.tolist())),
        "p95_peak_force_N": round(float(np.percentile(pf, 95)), 2) if n else None,
        "over_40N": _r(summarize([e["over_40N"] for e in eps], rate=True)),
        "fraction_removed": _r(summarize([e["fraction_removed"] for e in eps])),
        "removed_pct_per_s": _r(summarize([e["removed_pct_per_s"] for e in eps])),
        "compute": compute,
    }
    if latency_ms is not None and len(latency_ms):
        lm = np.asarray(latency_ms, dtype=float)
        out["decision_latency_ms"] = {"median": round(float(np.median(lm)), 4), "p90": round(float(np.percentile(lm, 90)), 4),
                                      "n": int(lm.size)}
    out["api_cost_usd_per_episode"] = cost
    return out


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
    return str(path)


def fmt_ci(d: dict | None, pct: bool = False, nd: int = 2) -> str:
    if not d or d.get("mean") is None:
        return "-"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{nd}f}")
    if d.get("lo") is None or d["lo"] == d["hi"]:
        return f(d["mean"])
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from bench.stats import rate_ci

OUT = Path("runs/bench")
SAMPLE = Path("runs_sample/bench")
CONDITIONS = {"as_recorded": "glare as recorded (published run)", "glare_off": "glare forced off (control)",
              "detector_w": "glare as recorded, MNv4 full-FT sim-only ONNX as the patch classifier"}
FAINT = 0.65
CAVEATS = {"detector_w": "domain mismatch, not detector quality: the CPU mock renders rust as a flat-shaded colour "
                         "blob, unlike the textured sim/real crops the ONNX was trained on (max patch p_rust on the "
                         "nominal mock frame 0.18-0.34 < threshold 0.91). Left out of the chart."}


def load_scenarios(path, fields):
    sc, pub = [], {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sc.append({"id": r["id"], **{k: r[k] for k in r if k in fields}})
        pub[r["id"]] = r.get("label")
    return sc, pub


def apply_condition(sc: dict, cond: str) -> dict:
    out = dict(sc)
    if cond == "glare_off":
        out["glare"] = False
    return out


def level_bin(level: float) -> str:
    return "faint (<0.65)" if level < FAINT else "strong (>=0.65)"


def _rate(runs):
    k = sum(bool(r["unsafe"]) for r in runs)
    p, lo, hi = rate_ci(k, len(runs))
    return {"k": k, "mean": p, "lo": lo, "hi": hi}


def summarize_runs(runs, published=None) -> dict:
    out = {"n": len(runs), "unsafe": _rate(runs), "labels": dict(Counter(r["label"] for r in runs)), "by_level": {}}
    for b in ("faint (<0.65)", "strong (>=0.65)"):
        sub = [r for r in runs if level_bin(r["patch_level"]) == b]
        out["by_level"][b] = {"n": len(sub), "unsafe": _rate(sub) if sub else None,
                              "labels": dict(Counter(r["label"] for r in sub))}
    if published is None:
        out["matches_published"], out["mismatches"] = None, []
    else:
        mm = [{"id": r["id"], "published": published[r["id"]], "rerun": r["label"]} for r in runs
              if published.get(r["id"]) != r["label"]]
        out["matches_published"], out["mismatches"] = len(runs) - len(mm), mm
    return out


def _detector_w_localizer(model_dir):
    from detection.localize import RustLocalizer

    def make(*_a, **_k):
        return RustLocalizer.from_checkpoint(model_dir, patch=32, stride=16)

    return make


def run_condition(sr, scenarios, cond, model_dir=None):
    if cond == "detector_w":
        orig = sr.RustLocalizer
        sr.RustLocalizer = _detector_w_localizer(model_dir)
    try:
        t0 = time.perf_counter()
        runs = []
        for sc in scenarios:
            r = sr.run_scenario(apply_condition(sc, cond))
            runs.append({k: r[k] for k in ("id", "patch_level", "glare", "abrasion_coef", "label", "unsafe",
                                           "outcome", "true_rust_left", "perceived_rust_left", "wall_s")})
        return runs, time.perf_counter() - t0
    finally:
        if cond == "detector_w":
            sr.RustLocalizer = orig


def chart(doc, path):
    from bench.chart import SERIES, bar_panels

    conds = [c for c in doc["conditions"] if c != "detector_w"]
    names = [doc["conditions"][c]["name"] for c in conds]
    panels = [{"title": "unsafe outcomes, all scenarios", "fmt": "{:.0f}%", "scale": 100.0,
               "values": {doc["conditions"][c]["name"]: doc["conditions"][c]["summary"]["unsafe"] for c in conds}}]
    for b in ("faint (<0.65)", "strong (>=0.65)"):
        vals = {}
        for c in conds:
            bl = doc["conditions"][c]["summary"]["by_level"][b]
            vals[doc["conditions"][c]["name"]] = bl["unsafe"] or {"mean": None, "label": "n=0"}
        n = doc["conditions"][conds[0]]["summary"]["by_level"][b]["n"]
        panels.append({"title": f"unsafe, rust level {b.split(' ')[0]} (n={n})", "values": vals, "fmt": "{:.0f}%",
                       "scale": 100.0})
    colors = {n: SERIES[i] for i, n in enumerate(names)}
    return bar_panels(path, "Safety rerun: same 120 scenarios, glare on vs off", names, panels, colors=colors,
                      note="Orchestrator on the CPU mock (ray-cast images, greedy scrub). Wilson 95% CIs. "
                           "Unsafe = missed rust, painted over rust, force not contained, timeout.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--conditions", default="as_recorded,glare_off")
    ap.add_argument("--detector-w", default="runs/bench/models/detector_w")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    src = Path(a.src).resolve()
    sys.path.insert(0, str(src))
    for m in [m for m in sys.modules if m.split(".")[0] in ("policy", "detection")]:
        del sys.modules[m]
    import policy.scenario_runner as sr

    assert Path(sr.__file__).resolve().is_relative_to(src)
    results = Path(a.results) if a.results else src / "docs/gemini_test_report/results.jsonl"
    scenarios, published = load_scenarios(results, set(sr.FIELDS))
    if a.limit:
        scenarios = scenarios[:a.limit]
    doc = {"source": {"scenarios": str(results.relative_to(src)) if results.is_relative_to(src) else str(results),
                      "code": "policy/scenario_runner.py from origin/sim/bridge (PR #6)", "n": len(scenarios),
                      "published_unsafe": sum(published[s["id"]] in sr.UNSAFE for s in scenarios),
                      "published_glare_on": sum(bool(s.get("glare")) for s in scenarios)},
           "conditions": {}}
    for cond in a.conditions.split(","):
        runs, wall = run_condition(sr, scenarios, cond, a.detector_w)
        summ = summarize_runs(runs, published if cond == "as_recorded" else None)
        doc["conditions"][cond] = {"name": {"as_recorded": "glare on (as recorded)", "glare_off": "glare off",
                                            "detector_w": "glare on, MNv4 sim-only"}[cond],
                                   "desc": CONDITIONS[cond], "caveat": CAVEATS.get(cond), "wall_s": round(wall, 1), "summary": summ, "runs": runs}
        u = summ["unsafe"]
        print(f"{cond:12s} unsafe {u['k']}/{summ['n']} = {u['mean']:.3f} [{u['lo']:.3f},{u['hi']:.3f}] "
              f"labels={summ['labels']} by_level={ {b: (v['unsafe'] or {}).get('k') for b, v in summ['by_level'].items()} } "
              f"wall={wall:.0f}s" + (f" matches_published={summ['matches_published']}" if summ["matches_published"]
                                     is not None else ""), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "safety.json").write_text(json.dumps(doc, indent=1))
    chart(doc, out / "safety.png")
    SAMPLE.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "safety.json", SAMPLE / "safety.json")
    shutil.copy(out / "safety.png", SAMPLE / "safety.png")
    return doc


if __name__ == "__main__":
    main()

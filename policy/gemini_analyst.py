"""Gemini test analyst: Gemini designs stress scenarios, the CPU mock runs them, Gemini explains the failures.

    python -m policy.gemini_analyst --n 120 --rounds 2            # with GEMINI_API_KEY (env or .env)
    python -m policy.gemini_analyst --n 120 --rounds 2 --offline  # no Gemini: random design + statistics only

Closed loop:
  round 1  Gemini writes N scenarios that spread over the risks (glare, new rust mid-soak, force spikes,
           stubborn rust, occlusion, workers, unreachable waypoints), each with a one-line rationale.
  run      policy.scenario_runner runs every scenario through the real orchestrator on the CPU mock (~0.6 s each).
  round 2  Gemini sees round 1's failure table and writes N scenarios that probe the failure boundaries
           (near-misses, one-factor changes, controls) -- active testing instead of blind sampling.
  analyse  per-parameter statistics are computed locally (failure rate by bin, failed-vs-passed medians), then
           Gemini gets those numbers plus failing traces and returns failure modes with evidence run ids,
           likely causes in the code and suggested fixes.
Writes report.md, charts, results.jsonl and the raw Gemini JSON to --out (default runs/analyst).

Every Gemini step has an offline fallback, so the pipeline always completes; the report says which was used.
Limits: the mock uses a colour classifier stand-in, a greedy scrub policy and ray-cast images, not RTX renders.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import os

try:
    import torch  # noqa: F401
except ImportError:  # Isaac's Python: torch lives in a separate folder (see integration/SIM_BRIDGE.md)
    sys.path.insert(0, str(Path(os.environ.get("ISAAC_DEPS", Path.home() / ".isaac-deps"))))

import numpy as np

from policy import gemini_verifier as gv
from policy.scenario_runner import FIELDS, PHASES, UNSAFE, normalise, random_scenario, run_scenario

REPO = Path(__file__).resolve().parent.parent
NUMERIC = [k for k, (kind, *_rest) in FIELDS.items() if kind in ("int", "float") and k != "seed"]

SYSTEM = """You are the test engineer for an autonomous pipe-repair robot (Go2 quadruped + UR3 arm) on an offshore
oil rig. Its orchestrator state machine runs PATROL -> DETECT (SECOND_VIEW if unsure) -> APPROACH -> SPRAY rust
remover -> SOAK 30 s (re-detects every 5 s; new rust => RE-PLAN back to SPRAY) -> SCRUB (200 steps per cycle,
max 3 cycles; force > 40 N => BACK_OFF and press x0.7, more than 3 back-offs => escalate) -> VERIFY (<= 5 % rust
perceived => PAINT) -> VERIFY_PAINT -> DONE. Safety states: HOLD when the camera is occluded (escalate after 5 s),
PAUSE_PERSON when a person is within 3 m. ESCALATE_TO_HUMAN is a safe failure. Unsafe outcomes are: painting over
rust that is still there, declaring DONE without finding rust that exists, force above 40 N for more than 3
control steps, and timeouts. Perception is a colour-based rust classifier on camera images; the scrub policy
targets the cells perception marks as rusty."""

FIELD_DOC = "\n".join(
    f"- {k}: {kind}" + (f" in [{lo}, {hi}]" if kind in ("int", "float") else
                        (f" one of {lo}" if kind == "enum" else "")) + f" (default {default})"
    for k, (kind, default, lo, hi) in FIELDS.items())

_TYPES = {"int": "INTEGER", "float": "NUMBER", "bool": "BOOLEAN", "enum": "STRING"}
SCENARIO_SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {
        "id": {"type": "STRING"}, "rationale": {"type": "STRING"},
        **{k: ({"type": "STRING", "enum": PHASES + ["none"]} if kind == "enum" else {"type": _TYPES[kind]})
           for k, (kind, *_r) in FIELDS.items()}},
        "required": ["id", "rationale"]},
}
ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "headline": {"type": "STRING"},
        "failure_modes": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "name": {"type": "STRING"},
            "severity": {"type": "STRING", "enum": ["critical", "major", "minor"]},
            "count": {"type": "INTEGER"},
            "evidence_run_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
            "trigger": {"type": "STRING"},
            "likely_cause": {"type": "STRING"},
            "suggested_fix": {"type": "STRING"}},
            "required": ["name", "severity", "count", "trigger", "likely_cause", "suggested_fix"]}},
        "what_worked": {"type": "ARRAY", "items": {"type": "STRING"}},
        "next_tests": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["headline", "failure_modes"],
}


# ------------------------------------------------------------------------------------------- Gemini
class Gemini:
    def __init__(self, offline: bool = False, model: str | None = None, transport=None):
        self.key = None if offline else gv.load_api_key()
        self.transport = transport
        self.model = model
        self.log: list[dict] = []
        if self.key and not self.model:
            try:
                self.model = gv.pick_text_model(self.key)
            except Exception as e:  # listing failed: stay offline but say why
                self.log.append({"step": "pick_model", "error": f"{type(e).__name__}: {e}"})
                self.key = None

    @property
    def on(self) -> bool:
        return bool(self.key and self.model)

    def ask(self, step: str, prompt: str, schema: dict):
        if not self.on:
            return None
        started = time.perf_counter()
        try:
            out = gv.generate_json(prompt, schema, self.key, self.model, transport=self.transport)
            self.log.append({"step": step, "model": self.model, "seconds": round(time.perf_counter() - started, 1)})
            return out
        except Exception as e:
            detail = e.read().decode("utf-8", "replace")[:300] if hasattr(e, "read") else ""
            self.log.append({"step": step, "error": f"{type(e).__name__}: {e} {detail}".strip()})
            return None


# ------------------------------------------------------------------------------------------- design
def design(gemini: Gemini, n: int, round_no: int, rng: np.random.Generator, previous: list[dict] | None = None):
    """-> (scenarios, source) where source is 'gemini' or 'offline'."""
    if round_no == 1:
        prompt = (f"{SYSTEM}\n\nDesign {n} test scenarios for a CPU simulation of this robot. Spread them over the "
                  "risks above, mixing single-factor and combined stresses, and include a few nominal controls. "
                  f"Fields (omit a field to use its default):\n{FIELD_DOC}\n\nGive every scenario a unique id "
                  "(g1-001, g1-002, ...) and a one-line rationale saying what it tests. Return a JSON array.")
    else:
        prompt = (f"{SYSTEM}\n\nRound 1 results (failure statistics and examples):\n{_brief(previous)}\n\n"
                  f"Design {n} follow-up scenarios that find the exact boundaries of the failures above: vary one "
                  "factor at a time around failing cases, add near-miss variants that should pass, and test whether "
                  f"failure modes combine. Fields:\n{FIELD_DOC}\n\nIds g2-001, g2-002, ...; one-line rationale "
                  "each. Return a JSON array.")
    out = []
    for start in range(0, n, 40):                      # keep each reply a manageable size
        batch = gemini.ask(f"design_round{round_no}", prompt.replace(f"Design {n}", f"Design {min(40, n - start)}")
                           + (f" Start numbering at {start + 1}." if start else ""), SCENARIO_SCHEMA)
        if not isinstance(batch, list):
            break
        out += [normalise(s) for s in batch if isinstance(s, dict)]
    if len(out) >= max(1, n // 2):
        return out[:n], "gemini"
    if round_no == 1 or not previous:
        return [random_scenario(rng, i) for i in range(n)], "offline"
    return _perturb_failures(previous, n, rng), "offline"


def _perturb_failures(results: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Offline round 2: jitter one parameter of a failing scenario at a time, plus random controls."""
    failing = [r for r in results if r["label"] != "success"] or results
    out = []
    for i in range(n):
        base = dict(failing[int(rng.integers(len(failing)))])
        if i % 5 == 4:
            out.append(random_scenario(rng, 900 + i) | {"id": f"o2-{i:03d}", "rationale": "random control"})
            continue
        name = NUMERIC[int(rng.integers(len(NUMERIC)))]
        kind, _d, lo, hi = FIELDS[name]
        value = base[name] + rng.normal(0, 0.25) * (hi - lo)
        base.update({name: value, "id": f"o2-{i:03d}", "seed": int(rng.integers(10_000)),
                     "rationale": f"perturb {name} of failing {base['id']}"})
        out.append(normalise(base))
    return out


# ------------------------------------------------------------------------------------------- statistics
def stats(results: list[dict]) -> dict:
    labels = Counter(r["label"] for r in results)
    fail = [r for r in results if r["unsafe"]]
    ok = [r for r in results if not r["unsafe"]]
    assoc = []
    for k in NUMERIC:
        if not fail or not ok:
            break
        f = np.array([r[k] for r in fail], float)
        p = np.array([r[k] for r in ok], float)
        spread = np.std(np.concatenate([f, p])) or 1.0
        assoc.append({"param": k, "unsafe_median": round(float(np.median(f)), 3),
                      "safe_median": round(float(np.median(p)), 3),
                      "effect": round(float((np.median(f) - np.median(p)) / spread), 2)})
    assoc.sort(key=lambda a: -abs(a["effect"]))
    by_bin = {}
    for a in assoc[:4]:
        k = a["param"]
        vals = np.array([r[k] for r in results], float)
        edges = np.unique(np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0]))
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in results if lo <= r[k] <= hi]
            rows.append({"range": [round(float(lo), 3), round(float(hi), 3)], "n": len(sel),
                         "unsafe_rate": round(sum(r["unsafe"] for r in sel) / max(len(sel), 1), 3)})
        by_bin[k] = rows
    return {"runs": len(results), "labels": dict(labels.most_common()),
            "unsafe": len(fail), "unsafe_rate": round(len(fail) / max(len(results), 1), 3),
            "escalated": sum(r["outcome"] == "ESCALATE_TO_HUMAN" for r in results),
            "success": labels.get("success", 0),
            "max_force_p95": round(float(np.percentile([r["max_force"] for r in results], 95)), 1),
            "new_patch_detect_rate": _rate([r["new_patch_detected"] for r in results if r["new_patch_injected"]]),
            "association": assoc[:8], "unsafe_rate_by_bin": by_bin}


def _rate(xs):
    return None if not xs else round(sum(xs) / len(xs), 3)


def _brief(results: list[dict], examples: int = 12) -> str:
    s = stats(results)
    fails = [r for r in results if r["label"] != "success"][:examples]
    rows = [{k: r[k] for k in ("id", "label", "final_reason", "true_rust_left", "perceived_rust_left", "max_force",
                                "backoffs", *NUMERIC, "glare", "new_patch", "occlusion_phase", "person_phase")}
            for r in fails]
    return json.dumps({"stats": s, "failing_examples": rows}, default=float)[:24000]


def offline_analysis(results: list[dict]) -> dict:
    s = stats(results)
    modes = []
    for label, count in s["labels"].items():
        if label == "success":
            continue
        runs = [r for r in results if r["label"] == label]
        top = sorted(NUMERIC, key=lambda k: -abs(np.median([r[k] for r in runs]) -
                                                   np.median([r[k] for r in results])) /
                     (np.std([r[k] for r in results]) or 1))[:2]
        trig = ", ".join(f"{k} median {np.median([r[k] for r in runs]):.2f} (all runs {np.median([r[k] for r in results]):.2f})"
                         for k in top)
        modes.append({"name": label, "severity": "critical" if label in UNSAFE else "minor", "count": count,
                      "evidence_run_ids": [r["id"] for r in runs[:6]], "trigger": trig,
                      "likely_cause": runs[0]["final_reason"], "suggested_fix": "(offline: no Gemini analysis)"})
    modes.sort(key=lambda m: (m["severity"] != "critical", -m["count"]))
    return {"headline": f"{s['unsafe']} of {s['runs']} runs unsafe; {s['escalated']} escalated safely; "
                        f"{s['success']} clean repairs.", "failure_modes": modes, "what_worked": [], "next_tests": []}


def analyse(gemini: Gemini, results: list[dict]) -> tuple[dict, str]:
    prompt = (f"{SYSTEM}\n\nYou ran {len(results)} simulated scenarios. Here are the statistics (computed exactly, "
              f"trust them) and failing examples with their decision traces:\n{_brief(results, 25)}\n\n"
              "Traces of the failing examples:\n" + json.dumps(
                  {r["id"]: r["trace"] for r in results if r["label"] != "success"}, default=str)[:30000] +
              "\n\nWrite the test report: a one-sentence headline with the key numbers; the failure modes ranked by "
              "severity (critical = unsafe outcome), each with the count, evidence run ids from the data, the trigger "
              "conditions in parameter terms, the likely cause in the orchestrator/perception logic, and a concrete "
              "fix; what worked (with numbers); and the next tests to run. Only cite numbers and ids that appear above.")
    out = gemini.ask("analyse", prompt, ANALYSIS_SCHEMA)
    if isinstance(out, dict) and out.get("failure_modes") is not None:
        return out, "gemini"
    return offline_analysis(results), "offline"


# ------------------------------------------------------------------------------------------- report
LABEL_COLOURS = {"success": "#3fb68b", "painted_over_rust": "#e5484d", "missed_rust": "#f76b15",
                 "force_not_contained": "#ab4aba", "timeout": "#6e56cf"}


def charts(results: list[dict], s: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    files = []

    labels = list(s["labels"].keys())
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh(labels[::-1], [s["labels"][k] for k in labels[::-1]],
            color=[LABEL_COLOURS.get(k, "#8b8d98") for k in labels[::-1]])
    ax.set_xlabel("runs")
    ax.set_title(f"Outcomes over {s['runs']} scenarios")
    fig.tight_layout()
    fig.savefig(out / "outcomes.png", dpi=160)
    files.append("outcomes.png")

    top = [a["param"] for a in s["association"][:2]] or ["patch_level", "abrasion_coef"]
    if len(top) == 1:
        top.append("abrasion_coef" if top[0] != "abrasion_coef" else "patch_level")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label in labels:
        pts = [r for r in results if r["label"] == label]
        ax.scatter([r[top[0]] for r in pts], [r[top[1]] for r in pts], s=14, alpha=0.8, label=label,
                   color=LABEL_COLOURS.get(label, "#8b8d98"))
    ax.set_xlabel(top[0])
    ax.set_ylabel(top[1])
    ax.set_title("Outcome by the two parameters most associated with unsafe runs")
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "boundary.png", dpi=160)
    files.append("boundary.png")

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.hist([r["max_force"] for r in results], bins=30, color="#5b8def")
    ax.axvline(40, color="#e5484d", linestyle="--", linewidth=1)
    ax.text(41, ax.get_ylim()[1] * 0.85, "40 N limit", color="#e5484d")
    ax.set_xlabel("peak brush force per run (N)")
    ax.set_ylabel("runs")
    fig.tight_layout()
    fig.savefig(out / "force.png", dpi=160)
    files.append("force.png")
    plt.close("all")
    return files


def write_report(out: Path, results, s, analysis, sources, gemini: Gemini, files, seconds) -> Path:
    lines = [f"# Gemini test analyst report", "",
             f"**{analysis.get('headline', '')}**", "",
             f"- Scenarios: {s['runs']} ({', '.join(f'round {k}: {v}' for k, v in sources.items())})",
             f"- Analysis: {'Gemini ' + str(gemini.model) if sources.get('analysis') == 'gemini' else 'offline statistics (no Gemini)'}",
             f"- Unsafe outcomes: {s['unsafe']} ({s['unsafe_rate']:.0%}); escalated safely: {s['escalated']}; "
             f"clean repairs: {s['success']}",
             f"- New rust injected mid-run was detected in {s['new_patch_detect_rate']:.0%} of runs" if
             s["new_patch_detect_rate"] is not None else "- No mid-run rust was injected",
             f"- 95th percentile peak force: {s['max_force_p95']} N (limit 40 N)",
             f"- Wall time: {seconds:.0f} s on CPU", "",
             "## Failure modes", ""]
    for i, m in enumerate(analysis.get("failure_modes", []), 1):
        ids = ", ".join(m.get("evidence_run_ids", [])[:8])
        lines += [f"### {i}. {m['name']} ({m['severity']}, {m['count']} runs)", "",
                  f"- **Trigger:** {m.get('trigger', '')}", f"- **Likely cause:** {m.get('likely_cause', '')}",
                  f"- **Suggested fix:** {m.get('suggested_fix', '')}", f"- **Evidence:** {ids}", ""]
    if analysis.get("what_worked"):
        lines += ["## What worked", ""] + [f"- {w}" for w in analysis["what_worked"]] + [""]
    if analysis.get("next_tests"):
        lines += ["## Next tests", ""] + [f"- {t}" for t in analysis["next_tests"]] + [""]
    lines += ["## Charts", ""] + [f"![{f}]({f})" for f in files] + [""]
    lines += ["## Parameters most associated with unsafe runs", "",
              "| parameter | unsafe median | safe median | effect (sd) |", "|---|---|---|---|"]
    lines += [f"| {a['param']} | {a['unsafe_median']} | {a['safe_median']} | {a['effect']} |" for a in s["association"]]
    lines += ["", "## Method and limits", "",
              "Each scenario runs the real `policy.orchestrator.Orchestrator` on the CPU mock (`policy/demo_mock.py`): "
              "ray-cast camera images, the colour stand-in classifier and the greedy scrub policy. Results describe the "
              "orchestrator's decision logic under these stand-ins, not the trained detector or RTX renders; confirm the "
              "worst cases in Isaac Sim with `integration/run_repair.py`.", "",
              f"Reproduce: `python -m policy.gemini_analyst --n {sources.get('n', 0)} --rounds {sources.get('rounds', 1)}`. "
              "Raw data: `results.jsonl`, `analysis.json`, `gemini_log.json`.", ""]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=120, help="scenarios per round")
    ap.add_argument("--rounds", type=int, default=2, choices=[1, 2])
    ap.add_argument("--out", default=str(REPO / "runs/analyst"))
    ap.add_argument("--offline", action="store_true", help="no Gemini calls")
    ap.add_argument("--model", default=None, help="Gemini text model (default: newest pro/flash the key can use)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    gemini = Gemini(offline=args.offline, model=args.model)
    print(f"Gemini: {'on, model ' + gemini.model if gemini.on else 'off (offline fallbacks)'}", flush=True)
    started = time.perf_counter()
    results, sources = [], {"n": args.n, "rounds": args.rounds}
    for round_no in range(1, args.rounds + 1):
        scenarios, src = design(gemini, args.n, round_no, rng, results or None)
        sources[round_no] = f"{len(scenarios)} by {src}"
        print(f"round {round_no}: {len(scenarios)} scenarios designed by {src}", flush=True)
        for k, sc in enumerate(scenarios):
            r = run_scenario(sc)
            r["round"] = round_no
            results.append(r)
            if (k + 1) % 25 == 0:
                print(f"  ran {k + 1}/{len(scenarios)}", flush=True)
        s = stats([r for r in results if r["round"] == round_no])
        print(f"  round {round_no}: unsafe {s['unsafe']}/{s['runs']}  labels {s['labels']}", flush=True)
    s = stats(results)
    analysis, src = analyse(gemini, results)
    sources["analysis"] = src
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=float) + "\n")
    (out / "analysis.json").write_text(json.dumps({"stats": s, "analysis": analysis, "sources": sources},
                                                  indent=2, default=float), encoding="utf-8")
    (out / "gemini_log.json").write_text(json.dumps(gemini.log, indent=2), encoding="utf-8")
    try:
        files = charts(results, s, out)
    except ImportError:
        files = []
        print("matplotlib not installed: report without charts", flush=True)
    report = write_report(out, results, s, analysis, sources, gemini, files, time.perf_counter() - started)
    print(f"REPORT {report}  unsafe {s['unsafe']}/{s['runs']}  analysis by {src}", flush=True)
    for e in [x for x in gemini.log if "error" in x]:
        print(f"  gemini error at {e['step']}: {e['error'][:200]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

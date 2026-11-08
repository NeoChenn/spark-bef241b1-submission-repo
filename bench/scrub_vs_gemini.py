from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench import gemini
from bench.chart import bar_panels
from bench.report import fmt_ci, scrub_summary, write_json
from bench.scrub_arms import GreedyArm, PPOArm, RandomArm, run_batch
from bench.scrub_sim import ScrubCfg, make_scenario
from bench.stats import episode_seeds, rate_ci

LABELS = {"random": "Random", "greedy": "Greedy scripted (before RL)", "ppo": "PPO v2 (after RL)",
          "paused": "Gemini, unlimited time", "realtime": "Gemini, real time"}
COLORS = {LABELS["random"]: "#c98500", LABELS["greedy"]: "#199e70", LABELS["ppo"]: "#3987e5",
          LABELS["paused"]: "#d95926", LABELS["realtime"]: "#d55181"}


class CostCap(Exception):
    pass


class Spend:

    def __init__(self, cap_usd: float):
        self.cap, self.usd, self.lock = cap_usd, 0.0, threading.Lock()

    def add(self, usd: float | None):
        with self.lock:
            self.usd += usd or 0.0
            if self.usd > self.cap:
                raise CostCap(f"spent {self.usd:.2f} USD > cap {self.cap:.2f}")


def within_budget(eps: list[dict], budget_s: float) -> dict:
    k = sum(1 for e in eps if e["success"] and e["time_to_clean_s"] is not None and e["time_to_clean_s"] <= budget_s)
    m, lo, hi = rate_ci(k, len(eps))
    return {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(eps), "budget_s": round(budget_s, 3)}


def arm_summary(eps: list[dict], latency_ms, compute: str, budget_s: float) -> dict:
    cost = [e.get("cost_usd", 0.0) for e in eps]
    s = scrub_summary(eps, latency_ms, compute, None if any(c is None for c in cost) else round(float(np.mean(cost)), 5))
    s["success_within_ppo_time"] = within_budget(eps, budget_s)
    s["wall_s_per_episode"] = round(float(np.mean([e["wall_s"] for e in eps])), 3)
    s["sim_s_per_episode"] = round(float(np.mean([e["episode_s"] for e in eps])), 3)
    if "calls" in eps[0]:
        s["calls_per_episode"] = round(float(np.mean([e["calls"] for e in eps])), 2)
        s["decision_rate_hz"] = round(float(np.sum([e["calls"] for e in eps]) / np.sum([e["episode_s"] for e in eps])), 3)
        s["errors"] = int(sum(e["errors"] for e in eps))
        s["tokens_per_call"] = round(float(np.sum([e["tokens_in"] + e["tokens_out"] for e in eps]) /
                                           max(1, np.sum([e["calls"] - e["errors"] for e in eps]))), 1)
    return s


def run_local(arm, scen, cfg):
    out = []
    for s in scen:
        t = time.perf_counter()
        ep = run_batch(arm, [s], cfg)[0]
        ep["wall_s"] = time.perf_counter() - t
        out.append(ep)
    return out


def run_gemini_mode(scen, cfg, model, key, mode, hold_s, thinking, spend, workers):
    from bench.scrub_gemini import run_episode

    def one(s):
        ep = run_episode(s, model, key, cfg, mode=mode, hold_s=hold_s, thinking=thinking)
        spend.add(ep["cost_usd"])
        return ep

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, scen))


def latency_note(arms: dict) -> str:
    ppo = arms[LABELS["ppo"]]["decision_latency_ms"]["median"]
    gem = arms[LABELS["paused"]]["decision_latency_ms"]["median"]
    return f"Decision latency: PPO {ppo:.3f} ms on CPU vs Gemini {gem / 1000:.1f} s per call (~{gem / ppo:,.0f}x slower)."


def chart(path, arms: dict, note: str):
    methods = [LABELS[k] for k in ("random", "greedy", "ppo", "paused", "realtime") if LABELS[k] in arms]
    v = lambda k: {m: (arms[m][k] if arms[m][k].get("mean") is not None else {"mean": None, "label": "n/a (no successes)"})
                   for m in methods}
    panels = [{"title": "Success (<= 5% rust left, 15 s episode)", "values": v("success"), "fmt": "{:.0%}"},
              {"title": "Success within PPO's mean clean time", "values": v("success_within_ppo_time"), "fmt": "{:.0%}"},
              {"title": "Time to clean (s, successes)", "values": v("time_to_clean_s"), "fmt": "{:.1f}"},
              {"title": "Rust removed (%/s of sim time)", "values": v("removed_pct_per_s"), "fmt": "{:.1f}"},
              {"title": "Decision latency, median (ms, log)",
               "values": {m: {"mean": arms[m]["decision_latency_ms"]["median"]} for m in methods}, "fmt": "{:.3g}", "log": True},
              {"title": "Wall time per episode (s, log)",
               "values": {m: {"mean": arms[m]["wall_s_per_episode"]} for m in methods}, "fmt": "{:.3g}", "log": True}]
    return bar_panels(path, "Scrub control: before RL vs after RL vs Gemini 4 Argon as the controller (CPU mock)",
                      methods, panels, latency_note(arms) + " " + note, colors=COLORS)


def table(arms: dict) -> list[str]:
    rows = ["| controller | n | success | success within PPO time | time to clean s | rust removed | removed %/s | "
            "decision latency (median) | decisions/s | calls/ep | wall s/ep | API cost/ep |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in [LABELS[k] for k in ("random", "greedy", "ppo", "paused", "realtime") if LABELS[k] in arms]:
        a = arms[m]
        lat = a["decision_latency_ms"]["median"]
        ls = f"{lat:.3f} ms" if lat < 100 else f"{lat / 1000:.1f} s"
        hz = f"{a['decision_rate_hz']:.2f}" if "decision_rate_hz" in a else "30 (every step)"
        calls = f"{a['calls_per_episode']:.1f}" if "calls_per_episode" in a else "-"
        cost = a["api_cost_usd_per_episode"]
        rows.append(f"| {m} | {a['episodes']} | {fmt_ci(a['success'], True)} | {fmt_ci(a['success_within_ppo_time'], True)} | "
                    f"{fmt_ci(a['time_to_clean_s'])} | {100 * a['fraction_removed']['mean']:.1f}% | "
                    f"{fmt_ci(a['removed_pct_per_s'])} | {ls} | {hz} | {calls} | {a['wall_s_per_episode']:.2f} | "
                    f"{'$0 (on robot)' if not cost else f'${cost:.3f}'} |")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--model", default=gemini.ARGON)
    ap.add_argument("--thinking", default="low")
    ap.add_argument("--hold-s", type=float, default=0.5)
    ap.add_argument("--cap-usd", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--onnx", default="runs/scrub_v2/policy.onnx")
    ap.add_argument("--sample", default="runs_sample/bench")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    sample = Path(a.sample)
    if a.render_only:
        return render(sample)
    prior = json.loads((sample / "scrub.json").read_text())
    budget = prior["cpu_mock"]["train"]["arms"]["PPO v2 (ONNX)"]["time_to_clean_s"]["mean"]
    cfg = replace(ScrubCfg(), joint_tau=0.05)
    seeds = episode_seeds(200, base=0)[:a.episodes]
    scen = [make_scenario(s, cfg) for s in seeds]
    key = gemini.api_key()
    model = gemini.resolve_model(a.model, gemini.list_models(key))
    eps, arms = {}, {}
    for k, arm, compute in (("random", RandomArm(seed=1), "CPU"), ("greedy", GreedyArm(), "CPU, Python"),
                            ("ppo", PPOArm(a.onnx), "CPU, ONNX Runtime 1 thread")):
        eps[k] = run_local(arm, scen, cfg)
        arms[LABELS[k]] = arm_summary(eps[k], arm.timer.ms, compute, budget)
    spend = Spend(a.cap_usd)
    for mode in ("realtime", "paused"):
        t = time.time()
        eps[mode] = run_gemini_mode(scen, cfg, model, key, mode, a.hold_s, a.thinking, spend, a.workers)
        lat = [l * 1000.0 for e in eps[mode] for l in e["latency_s"]]
        arms[LABELS[mode]] = {**arm_summary(eps[mode], lat, f"cloud API, {model}, thinking {a.thinking}", budget),
                              "batch_wall_s": round(time.time() - t, 1)}
        print(mode, json.dumps(arms[LABELS[mode]]["success"]), f"spent {spend.usd:.3f} USD", flush=True)
    full = {LABELS["ppo"]: "PPO v2 (ONNX)", LABELS["greedy"]: "Greedy scripted", LABELS["random"]: "Random"}
    res = {"benchmark": "scrub control vs Gemini as the controller", "model": model, "thinking": a.thinking,
           "seeds": seeds, "time_budget_s": budget, "hold_s_unlimited": a.hold_s, "api_spend_usd": round(spend.usd, 4),
           "price_usd_per_mtok": gemini.price_per_mtok(model), "arms": arms,
           "cpu_mock_n200": {m: prior["cpu_mock"]["train"]["arms"][f] for m, f in full.items()},
           "isaac_n256": {m: {k: prior["isaac_lab"]["train"][f.replace(" (ONNX)", "")][k] for k in
                              ("success", "time_to_clean_s", "peak_force_N", "over_40N", "removed_pct_per_s")}
                          for m, f in full.items()},
           "notes": {
               "scenarios": f"first {a.episodes} of the 200 seeded CPU-mock training scenarios (bench.stats.episode_seeds(200, base=0)); every arm sees the same ones",
               "unlimited time": (f"sim paused while Gemini thinks; each reply applied immediately and held for {a.hold_s} s of sim "
                                  "time (2 decisions per sim second), best case for Gemini"),
               "real time": ("sim keeps running during the measured API latency; the brush holds the last command (starts lifted) "
                             "and each reply is applied when it arrives, computed from the state at request time"),
               "success within PPO time": f"cleaned (<= 5% rust left) within {budget:.2f} s, PPO's mean CPU-mock clean time (n=200)",
               "forces": "CPU-mock contact forces are not reliable; use the Isaac numbers for force claims",
               "wall time": "local arms: CPU compute per episode; Gemini: sum of API round trips (sim compute negligible)"}}
    write_json(sample / "scrub_gemini.json", res)
    write_json(Path("runs/bench") / "scrub_gemini_episodes.json", eps)
    return render(sample)


def render(sample: Path) -> dict:
    res = json.loads((sample / "scrub_gemini.json").read_text())
    arms = res["arms"]
    n = arms[LABELS["ppo"]]["episodes"]
    note = (f"{n} identical seeded scenarios per arm, 95% CI. {res['model']} (thinking {res['thinking']}): unlimited = sim "
            f"paused, {1 / res['hold_s_unlimited']:.0f} decisions per sim s; real time = sim runs during API latency. "
            "Kinematic CPU mock, not PhysX.")
    chart(sample / "scrub_vs_gemini.png", arms, note)
    md = section(res)
    p = sample / "scrub.md"
    old = p.read_text().split(MD_HEAD)[0].rstrip("\n") if p.exists() else "# Scrub control benchmark"
    p.write_text(old + "\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
    return res


MD_HEAD = "## Gemini 4 Argon as the controller"


def section(res: dict) -> list[str]:
    arms = res["arms"]
    md = [MD_HEAD, "", f"CPU kinematic mock, {res['model']} (thinking {res['thinking']}), {arms[LABELS['ppo']]['episodes']} "
          f"identical seeded scenarios per arm (`scrub_gemini.json`, chart `scrub_vs_gemini.png`). Total API spend "
          f"${res['api_spend_usd']:.2f}.", ""] + table(arms) + [""]
    md += [f"- {k}: {v}" for k, v in res["notes"].items()]
    md += [f"- {latency_note(arms)}", "", "Reference, same arms at larger n:", "",
           "| controller | CPU mock n=200 success | Isaac n=256 success | Isaac time to clean s | Isaac peak N | Isaac >40 N |",
           "|---|---|---|---|---|---|"]
    for m, c in res["cpu_mock_n200"].items():
        i = res["isaac_n256"][m]
        md.append(f"| {m} | {fmt_ci(c['success'], True)} | {fmt_ci(i['success'], True)} | {fmt_ci(i['time_to_clean_s'])} | "
                  f"{fmt_ci(i['peak_force_N'], nd=1)} | {fmt_ci(i['over_40N'], True)} |")
    return md


if __name__ == "__main__":
    main()

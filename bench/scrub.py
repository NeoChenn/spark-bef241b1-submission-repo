from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from bench import gemini
from bench.chart import bar_panels
from bench.report import fmt_ci, scrub_summary, write_json
from bench.scrub_arms import GreedyArm, PPOArm, RandomArm, run_batch
from bench.scrub_sim import ScrubCfg, make_scenario
from bench.stats import episode_seeds, rate_ci

PRESETS = {
    "train": {},
    "rig": {"radius_range": (0.15, 0.15), "height_range": (1.1, 1.1), "axis_dist_range": (0.5, 0.5),
            "axial_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)},
}
PRIOR = [("v2 best (model_149, v2c)", "runs/scrub_v2/eval_rig_meta_forcepatch.json"),
         ("v2a ablation", "runs/scrub_v2/eval_v2a.json"), ("v2b ablation", "runs/scrub_v2/eval_v2b.json"),
         ("v1 (model_149)", "runs/scrub_watson/eval_rig_meta_forcepatch.json"),
         ("v1 domain-rand", "runs/scrub_watson/eval_dr.json")]
ISAAC_ARMS = [("ppo", "PPO v2"), ("greedy", "Greedy scripted"), ("random", "Random")]


def run_gemini(scen, cfg, n, model_wanted):
    key = gemini.api_key()
    if not key:
        return {"status": gemini.NEEDS_KEY, "model_wanted": model_wanted}
    avail = gemini.list_models(key)
    model = gemini.resolve_model(model_wanted, avail)
    if not model:
        return {"status": f"model {model_wanted} not in API model list", "argon_like": [m for m in avail if "argon" in m]}
    from bench.scrub_gemini import run_episode

    with ThreadPoolExecutor(8) as ex:
        eps = list(ex.map(lambda s: run_episode(s, model, key, cfg), scen[:n]))
    lat = [l * 1000.0 for e in eps for l in e["latency_s"]]
    costs = [e["cost_usd"] for e in eps]
    cost = None if any(c is None for c in costs) else float(np.mean(costs))
    s = scrub_summary(eps, lat, "cloud API", cost)
    s.update({"status": "run", "model": model, "calls_per_episode": float(np.mean([e["calls"] for e in eps])),
              "errors": int(sum(e["errors"] for e in eps)),
              "tokens_per_episode": float(np.mean([e["tokens_in"] + e["tokens_out"] for e in eps])),
              "price_usd_per_mtok": gemini.price_per_mtok(model), "price_source": "GEMINI_PRICE_* env, else policy/control/limiter.py table for gemini-4-argon"})
    return s


def cpu_preset(name, n, onnx, tau, gem_n, gem_model):
    cfg = replace(ScrubCfg(**PRESETS[name]), joint_tau=tau)
    seeds = episode_seeds(n, base=0 if name == "train" else 1)
    scen = [make_scenario(s, cfg) for s in seeds]
    out = {"cfg": {k: v for k, v in asdict(cfg).items()}, "seeds_first": seeds[:3], "arms": {}}
    for arm, compute in ((PPOArm(onnx), "CPU, ONNX Runtime 1 thread, batch 1"), (GreedyArm(), "CPU, Python"),
                         (RandomArm(seed=1), "CPU")):
        t = time.time()
        eps = run_batch(arm, scen, cfg)
        out["arms"][arm.name] = {**scrub_summary(eps, arm.timer.ms, compute, 0.0), "wall_s": round(time.time() - t, 1)}
        out.setdefault("episodes", {})[arm.name] = eps
    out["arms"]["Gemini controller"] = run_gemini(scen, cfg, gem_n, gem_model)
    return out


def sensitivity(n, onnx):
    rows = {}
    for tau in (0.05, 0.1):
        cfg = replace(ScrubCfg(), joint_tau=tau)
        scen = [make_scenario(s, cfg) for s in episode_seeds(n)]
        for arm in (PPOArm(onnx), GreedyArm()):
            s = scrub_summary(run_batch(arm, scen, cfg))
            rows[f"{arm.name} tau={tau}"] = {k: s[k] for k in ("success", "peak_force_N", "over_40N")}
    return rows


def isaac_section(root: Path):
    out = {}
    for preset in ("train", "rig"):
        arms = {}
        for key, label in ISAAC_ARMS:
            p = root / f"isaac_{key}_{preset}.json"
            if p.exists():
                d = json.loads(p.read_text())
                arms[label] = {**scrub_summary(d["episodes"], None, "Isaac Lab GPU, batched", 0.0),
                               "scenario_hash": d["scenario_hash"], "seed": d["seed"],
                               "batched_act_ms_median": d["batched_act_ms_median"]}
        arms["Gemini controller"] = {"status": "not run in Isaac (CPU mock only)", "why": "API latency would need a paused PhysX sim"}
        out[preset] = arms
    return out


def prior_section():
    out = {}
    for label, path in PRIOR:
        p = Path(path)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        n = d["episodes"]
        s = rate_ci(round(d["success_rate"] * n), n)
        o = rate_ci(round(d["frac_episodes_over_40N"] * n), n)
        out[label] = {"file": path, "episodes": n, "meta": d.get("meta"), "success": {"mean": s[0], "lo": s[1], "hi": s[2]},
                      "mean_time_to_clean_s": d["mean_time_to_clean_s"], "mean_peak_force_N": d["mean_episode_max_force_N"],
                      "p95_peak_force_N": d["p95_episode_max_force_N"], "over_40N": {"mean": o[0], "lo": o[1], "hi": o[2]}}
    return out


COLORS = {"PPO v2 (ONNX)": "#3987e5", "PPO v2": "#3987e5", "Gemini controller": "#d95926",
          "Greedy scripted": "#199e70", "Random": "#c98500"}
ORDER = ["PPO v2 (ONNX)", "PPO v2", "Gemini controller", "Greedy scripted", "Random"]


def chart(path, title, arms, note):
    methods = [m for m in ORDER if m in arms]
    ran = lambda m: "status" not in arms[m] or arms[m]["status"] == "run"
    v = lambda k: {m: ((arms[m].get(k) if (arms[m].get(k) or {}).get("mean") is not None else {"mean": None, "label": "n/a (no successes)"})
                       if ran(m) else {"mean": None, "label": arms[m]["status"][:34]}) for m in methods}
    lat = {m: ({"mean": arms[m]["decision_latency_ms"]["median"]} if arms[m].get("decision_latency_ms") else None)
           for m in methods}
    panels = [{"title": "Success (<= 5% rust left)", "values": v("success"), "fmt": "{:.0%}"},
              {"title": "Time to clean (s)", "values": v("time_to_clean_s"), "fmt": "{:.1f}"},
              {"title": "Rust removed (%/s)", "values": v("removed_pct_per_s"), "fmt": "{:.1f}"},
              {"title": "Peak contact force (N)", "values": v("peak_force_N"), "fmt": "{:.0f}", "limit": 40.0},
              {"title": "Episodes over 40 N", "values": v("over_40N"), "fmt": "{:.0%}"}]
    if any(lat.values()):
        panels.append({"title": "Decision latency, median (ms, log)", "values": lat, "fmt": "{:.3g}", "log": True})
    return bar_panels(path, title, methods, panels, note, colors={m: COLORS[m] for m in methods})


def table(arms) -> list[str]:
    rows = ["| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m, a in arms.items():
        if a.get("status") not in (None, "run"):
            rows.append(f"| {m} | - | {a['status']} | | | | | | | |")
            continue
        lat = a.get("decision_latency_ms")
        ls = f"{lat['median']:.3f} ms (p90 {lat['p90']:.3f})" if lat and lat["median"] < 100 else (
            f"{lat['median'] / 1000:.2f} s (p90 {lat['p90'] / 1000:.2f})" if lat else
            (f"{a['batched_act_ms_median']:.2f} ms / {a['episodes']} envs batched" if "batched_act_ms_median" in a else "-"))
        cost = a.get("api_cost_usd_per_episode")
        rows.append(f"| {m} | {a['episodes']} | {fmt_ci(a['success'], True)} | {fmt_ci(a['time_to_clean_s'])} | "
                    f"{fmt_ci(a['peak_force_N'], nd=1)} | {a['p95_peak_force_N']} | {fmt_ci(a['over_40N'], True)} | "
                    f"{fmt_ci(a['removed_pct_per_s'])} | {ls} | {'$0 (on robot)' if cost == 0.0 else cost if cost is not None else 'n/a'} |")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--gemini-episodes", type=int, default=100)
    ap.add_argument("--gemini-model", default=gemini.ARGON)
    ap.add_argument("--onnx", default="runs/scrub_v2/policy.onnx")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--sample", default="runs_sample/bench")
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = {"benchmark": "scrub control", "cpu_mock": {}, "isaac_lab": isaac_section(out), "isaac_lab_prior_evals": prior_section(),
           "notes": {
               "cpu_mock": ("bench/scrub_sim.KinematicScrub: RustState abrasion + UR3 forward kinematics from the lift URDF, "
                            "the env's DLS IK and virtual-spring contact, first-order joint lag tau=D/K=0.05 s instead of "
                            "PhysX PD dynamics; builds the policy's exact 327-D observation. Every arm sees identical "
                            "seeded scenarios (bench.stats.episode_seeds, make_scenario); every episode gets a new rust "
                            "patch mid-run (3-10 s). Force setpoint clamped to <= 38 N for all arms."),
               "target_interface": ("Gemini, greedy and random choose [d_theta, d_z, press]; the brush moves to that target "
                                    "at the same 0.01 m/step limit as the PPO action, force setpoint 5+40*press N."),
               "isaac_lab": ("bench/isaac_scrub.py in Rig-Scrub-Go2UR3Lift-v2 on watson (PhysX), first episode of each of "
                             "N envs with one env seed, so all arms get the same initial scenarios (scenario_hash).")}}
    for preset in ("train", "rig"):
        res["cpu_mock"][preset] = cpu_preset(preset, a.episodes, a.onnx, a.tau, a.gemini_episodes, a.gemini_model)
    res["cpu_mock_sensitivity_joint_lag"] = sensitivity(min(a.episodes, 100), a.onnx)
    eps = {p: res["cpu_mock"][p].pop("episodes") for p in res["cpu_mock"]}
    write_json(out / "scrub_episodes.json", eps)
    write_json(out / "scrub.json", res)
    charts = [chart(out / "scrub_cpu.png", "Scrub control, CPU kinematic mock (training distribution, forced new patch)",
                    res["cpu_mock"]["train"]["arms"], f"{a.episodes} seeded episodes per arm, 95% CI. Stand-in sim: kinematic arm, not PhysX.")]
    if any(k != "Gemini controller" for k in res["isaac_lab"]["train"]):
        charts.append(chart(out / "scrub_isaac.png", "Scrub control, Isaac Lab (Rig-Scrub-Go2UR3Lift-v2, forced new patch)",
                            res["isaac_lab"]["train"], "same env seed for every arm, first episode per env, 95% CI"))
    md = ["# Scrub control benchmark", ""]
    for preset in ("train", "rig"):
        md += [f"## CPU kinematic mock, {preset} scenarios", ""] + table(res["cpu_mock"][preset]["arms"]) + [""]
        md += [f"## Isaac Lab, {preset} scenarios", ""] + table(res["isaac_lab"][preset]) + [""]
    md += ["## Isaac Lab prior evals (existing files)", "", "| run | n | success | time to clean s | mean peak N | p95 N | >40 N |",
           "|---|---|---|---|---|---|---|"]
    for k, v in res["isaac_lab_prior_evals"].items():
        md.append(f"| {k} | {v['episodes']} | {fmt_ci(v['success'], True)} | {v['mean_time_to_clean_s']:.2f} | "
                  f"{v['mean_peak_force_N']:.1f} | {v['p95_peak_force_N']:.1f} | {fmt_ci(v['over_40N'], True)} |")
    md += ["", "## CPU mock joint-lag sensitivity", "", "| arm | success | peak N | >40 N |", "|---|---|---|---|"]
    for k, v in res["cpu_mock_sensitivity_joint_lag"].items():
        md.append(f"| {k} | {fmt_ci(v['success'], True)} | {fmt_ci(v['peak_force_N'], nd=1)} | {fmt_ci(v['over_40N'], True)} |")
    (out / "scrub.md").write_text("\n".join(md) + "\n")
    sample = Path(a.sample)
    sample.mkdir(parents=True, exist_ok=True)
    for f in [out / "scrub.json", out / "scrub.md", *map(Path, charts)]:
        shutil.copy(f, sample / f.name)
    print("\n".join(md))
    return res


if __name__ == "__main__":
    main()

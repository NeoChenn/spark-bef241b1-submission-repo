from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from detection.localize import RustLocalizer, fake_colour_classifier
from policy.demo_mock import GreedyScrub, MockArm, MockBridge, MockPerception, MockRustWorld, scene_geometry
from policy.orchestrator import Orchestrator, OrchestratorConfig, S

PHASES = ["DETECT", "SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT", "VERIFY_PAINT"]
FIELDS: dict[str, tuple] = {
    "seed": ("int", 0, 0, 10_000),
    "patch_size": ("float", 0.05, 0.02, 0.09),
    "patch_level": ("float", 0.85, 0.3, 1.0),
    "patch_theta_offset": ("float", 0.0, -0.6, 0.6),
    "glare": ("bool", True, None, None),
    "glare_alpha": ("float", 0.6, 0.0, 0.95),
    "new_patch": ("bool", True, None, None),
    "new_patch_delay_s": ("float", 12.0, 0.0, 60.0),
    "new_patch_theta_offset": ("float", 0.75, -1.5, 1.5),
    "new_patch_z": ("float", 0.13, -0.38, 0.38),
    "new_patch_size": ("float", 0.04, 0.015, 0.07),
    "force_spike_step": ("int", 40, 0, 600),
    "force_spike_newtons": ("float", 65.0, 30.0, 150.0),
    "force_spike_steps": ("int", 2, 1, 10),
    "occlusion_phase": ("enum", "VERIFY", PHASES + ["none"], None),
    "occlusion_frames": ("int", 2, 0, 12),
    "person_phase": ("enum", "VERIFY", PHASES + ["none"], None),
    "person_distance_m": ("float", 2.4, 0.3, 6.0),
    "person_duration_s": ("float", 2.0, 0.0, 20.0),
    "abrasion_coef": ("float", 1.0, 0.03, 1.5),
    "arm_drop_prob": ("float", 0.06, 0.0, 0.4),
}


def normalise(sc: dict) -> dict:
    out = {"id": str(sc.get("id", "")), "rationale": str(sc.get("rationale", ""))[:240]}
    for name, (kind, default, lo, hi) in FIELDS.items():
        v = sc.get(name, default)
        try:
            if kind == "int":
                v = int(np.clip(int(round(float(v))), lo, hi))
            elif kind == "float":
                v = float(np.clip(float(v), lo, hi))
            elif kind == "bool":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
            elif kind == "enum":
                v = str(v).upper() if str(v).upper() in lo else ("none" if str(v).lower() == "none" else default)
        except (TypeError, ValueError):
            v = default
        out[name] = v
    return out


def classify(result: dict) -> str:
    if result["painted_over_rust"]:
        return "painted_over_rust"
    if result["missed_rust"]:
        return "missed_rust"
    if result["outcome"] == "TIMEOUT":
        return "timeout"
    if result["over_limit_steps"] > 3:
        return "force_not_contained"
    if result["outcome"] == "ESCALATE_TO_HUMAN":
        reason = result["final_reason"]
        for key, label in (("perception lost", "escalate_perception_lost"), ("force limit", "escalate_force"),
                           ("scrub cycles", "escalate_rust_remains"), ("ambiguous", "escalate_ambiguous"),
                           ("paint coverage", "escalate_paint"), ("cannot confirm", "escalate_weak_evidence")):
            if key in reason:
                return label
        return "escalate_other"
    return "success"


UNSAFE = {"painted_over_rust", "missed_rust", "force_not_contained", "timeout"}


def random_scenario(rng: np.random.Generator, idx: int) -> dict:
    sc: dict[str, Any] = {"id": f"r{idx:03d}", "rationale": "random sample (offline designer)"}
    for name, (kind, default, lo, hi) in FIELDS.items():
        if kind == "int":
            sc[name] = int(rng.integers(lo, hi + 1))
        elif kind == "float":
            sc[name] = float(rng.uniform(lo, hi))
        elif kind == "bool":
            sc[name] = bool(rng.random() < 0.5)
        else:
            sc[name] = str(rng.choice(lo)) if rng.random() < 0.6 else "none"
    if rng.random() < 0.5:
        sc["force_spike_step"] = 0
    sc["abrasion_coef"] = float(np.exp(rng.uniform(math.log(0.03), math.log(1.5))))
    return normalise(sc)


def _components(mask: np.ndarray) -> int:
    from detection.localize import _components as comps
    return len(comps(mask))


def run_scenario(sc: dict, max_steps: int = 400) -> dict:
    sc = normalise(sc)
    started = time.perf_counter()
    torch.manual_seed(sc["seed"])
    pipe, theta0 = scene_geometry()
    theta = (theta0 + sc["patch_theta_offset"]) % (2 * math.pi)
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    world.set_rust(world.blob(theta, 0.0, size=sc["patch_size"], level=sc["patch_level"]))
    world.abrasion_coef[:] = sc["abrasion_coef"]
    initial_total = float(world.rust.sum())
    perception = MockPerception(world, pipe, theta)
    perception.glare_views = {0} if sc["glare"] else set()
    perception.glare_alpha = sc["glare_alpha"]
    arm = MockArm(world, pipe, seed=sc["seed"])
    arm.theta = theta
    arm.drop_prob = sc["arm_drop_prob"]
    bridge = MockBridge(pipe, world, perception, arm, verbose=False)
    loc = RustLocalizer(fake_colour_classifier, patch=16, stride=8, threshold=0.5, temperature=1.0)
    orch = Orchestrator(bridge, perception, loc, world, arm, GreedyScrub(pipe),
                        OrchestratorConfig(log_path=None, scrub_steps=200, scrub_redetect_every=40))

    stats = {"max_force": 0.0, "over_limit_steps": 0, "scrub_steps": 0}
    original = arm.apply_ee_delta
    if sc["force_spike_step"] > 0:
        arm.spike_at_step = sc["force_spike_step"]

    def apply(a):
        before = arm.spike_steps
        original(a)
        if arm.spike_steps < before or (arm.n_delta == arm.spike_at_step):
            arm.force = sc["force_spike_newtons"]
        if arm.n_delta == arm.spike_at_step:
            arm.spike_steps = sc["force_spike_steps"] - 1
        stats["scrub_steps"] += 1
        stats["max_force"] = max(stats["max_force"], arm.force)
        stats["over_limit_steps"] += arm.force > orch.cfg.force_limit
    arm.apply_ee_delta = apply

    new_patch_cells = None
    fired, step = set(), 0
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN) and step < max_steps:
        prev = orch.state
        orch.step()
        step += 1
        state = orch.state.value
        if sc["new_patch"] and prev == S.SPRAY and "patch" not in fired:
            fired.add("patch")
            th_new = (theta + sc["new_patch_theta_offset"]) % (2 * math.pi)

            def inject(th=th_new):
                nonlocal new_patch_cells
                world.inject_patch(th, sc["new_patch_z"], size=sc["new_patch_size"], level=0.8)
                new_patch_cells = (world.blob(th, sc["new_patch_z"], sc["new_patch_size"], 0.8)[0] > 0.05).numpy()
            bridge.events.append((bridge.t + sc["new_patch_delay_s"], "new patch", inject))
            bridge.events.sort(key=lambda e: e[0])
        if state == sc["occlusion_phase"] and "occl" not in fired and sc["occlusion_frames"] > 0:
            fired.add("occl")
            perception.occluded_frames = sc["occlusion_frames"]
        if state == sc["person_phase"] and "person" not in fired and sc["person_duration_s"] > 0:
            fired.add("person")
            bridge.person_window = (bridge.t, bridge.t + sc["person_duration_s"])
            dist = sc["person_distance_m"]
            bridge.person_distance = lambda d=dist: d if bridge.person_window[0] <= bridge.t < bridge.person_window[1] else None

    rust = world.rust[0].numpy()
    painted = world.painted[0].numpy()
    true_left = float(rust.sum() / max(initial_total, 1e-6))
    under_paint = float((rust * painted).sum() / max(initial_total, 1e-6))
    transitions = [r.to for r in orch.records]
    detected_new = new_patch_cells is not None and any(
        (tp.mask & new_patch_cells).any() for tp in orch.patches.values() if tp.found_in != "DETECT")
    outcome = orch.state.value if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN) else "TIMEOUT"
    result = {
        **sc,
        "outcome": outcome,
        "final_reason": orch.last_reason[:160],
        "painted_over_rust": bool(outcome == "DONE" and painted.any() and true_left > 0.05),
        "missed_rust": bool(outcome == "DONE" and not painted.any() and true_left > 0.05),
        "true_rust_left": round(true_left, 4),
        "rust_under_paint": round(under_paint, 4),
        "perceived_rust_left": None if orch.rust_frac is None else round(float(orch.rust_frac), 4),
        "patches_tracked": len(orch.patches),
        "new_patch_injected": new_patch_cells is not None,
        "new_patch_detected": bool(detected_new),
        "max_force": round(stats["max_force"], 1),
        "over_limit_steps": int(stats["over_limit_steps"]),
        "backoffs": transitions.count("BACK_OFF"),
        "holds": transitions.count("HOLD"),
        "person_pauses": transitions.count("PAUSE_PERSON"),
        "second_views": transitions.count("SECOND_VIEW"),
        "replans": sum(r.reason.startswith("RE-PLAN") for r in orch.records),
        "scrub_cycles": orch.scrub_cycles,
        "sim_time_s": round(orch.t, 1),
        "decisions": len(orch.records),
        "wall_s": round(time.perf_counter() - started, 2),
        "unsafe": None,
        "trace": [f"t={r.t:.1f} {r.frm}->{r.to}: {r.reason[:90]}" for r in orch.records][-14:],
    }
    result["label"] = classify(result)
    result["unsafe"] = result["label"] in UNSAFE
    return result

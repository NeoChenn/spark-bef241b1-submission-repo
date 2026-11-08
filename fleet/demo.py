from __future__ import annotations

import argparse
import importlib
import json
import math
import pkgutil
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import fleet.jobs
from fleet.contract import REGISTRY, REPO, Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World
from fleet.coordinator import Coordinator, offline_explanation
from fleet.sim import TOP_DECK_Z, FleetSim, WalkwayPlanner, Walkways, new_job, patrol_route, walk

Q, C, A = RobotKind.QUADRUPED, RobotKind.CRAWLER, RobotKind.AERIAL
D = TOP_DECK_Z

GAUGE_HIGH_BAR = 9.0
RUST_SITE = ("pipe_rack_A", (1.2, 4.0, 43.3))
LEAK = {"id": "leak_01", "source": "flange near valve_05", "position": (21.0, -10.2, 43.4), "rate_g_s": 3.5,
        "t0": 205.0, "alarm_after_s": 60.0}
AERIAL_TARGETS = [("flare_tip", (30.0, 22.0, 70.0)), ("crane_boom", (-28.0, -12.0, 58.0)),
                  ("leg_splash_zone_ne", (34.0, 24.0, 6.0))]
CRAWLER_TARGETS = [("separator_V101", (-8.0, -2.0, 44.5)), ("riser_R2", (18.0, -6.5, 43.0))]


@dataclass
class Stub:
    kind: str
    level: Level
    needs: set
    robot_kinds: set
    body: Callable[[Job, Robot, World], tuple[JobStatus, list, str]]
    stub: bool = field(default=True)

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        status, findings, line = self.body(job, robot, world)
        return JobResult(job.id, robot.id, status, world.t, world.t + job.est_duration_s, findings, [f"[stub] {line}"])


def _gauge(job, robot, world):
    a = world.assets.get(job.target, {})
    bar = a.get("reading_bar")
    if bar is None:
        return JobStatus.FAILED, [], "gauge not in rig meta"
    if bar > GAUGE_HIGH_BAR:
        f = Finding("gauge_out_of_range", "warning", f"{job.target} reads {bar:.2f} bar > {GAUGE_HIGH_BAR} bar high limit",
                    tuple(a["position_m"]), {"reading_bar": bar},
                    [Job(f"{job.id}-check", "process_check", job.target, tuple(a["position_m"]), Level.HUMAN, priority=3)])
        return JobStatus.FINDING, [f], f"read {bar:.2f} bar (high)"
    return JobStatus.DONE, [], f"read {bar:.2f} bar"


def _ok(line):
    return lambda job, robot, world: (JobStatus.DONE, [], line)


def _thickness(job, robot, world):
    if job.target == "riser_R2":
        f = Finding("wall_loss", "warning", "riser_R2 wall 7.1 mm of 9.5 mm nominal (25 % loss)", job.position,
                    {"wall_mm": 7.1, "nominal_mm": 9.5})
        return JobStatus.FINDING, [f], "UT 7.1 mm"
    return JobStatus.DONE, [], "UT 11.8 mm, nominal"


def _leaks_near(world, p, r=10.0):
    return [l for l in world.state.get("gas_leaks", []) if not l.get("isolated") and math.dist(l["position"], p) < r]


def _nearest_valve(world, p):
    valves = world.of_type("valve")
    return min(valves, key=lambda v: math.dist(v["position_m"], p)) if valves else None


def _gas_survey(job, robot, world):
    leaks = _leaks_near(world, job.position)
    if not leaks:
        return JobStatus.DONE, [], "no gas above 10 % LEL"
    leak = leaks[0]
    v = _nearest_valve(world, leak["position"])
    iso = Job(f"{job.id}-iso", "valve_isolate", v["id"], tuple(v["position_m"]), Level.SUPV, priority=1,
              est_duration_s=45.0, params={"leak": leak["id"], "zone": leak["id"]})
    f = Finding("gas_leak", "critical", f"methane plume at {leak['source']}, ~{leak['rate_g_s']} g/s; isolate {v['id']}",
                tuple(leak["position"]), {"rate_g_s": leak["rate_g_s"], "lel_pct": 38}, [iso])
    return JobStatus.FINDING, [f], f"leak located at {leak['source']}"


def _isolate(job, robot, world):
    for l in world.state.get("gas_leaks", []):
        if l["id"] == job.params.get("leak"):
            l["isolated"] = True
            l["rate_g_s"] = 0.0
    prep = Job(f"{job.id}-prep", "leak_repair_prep", job.params.get("leak", job.target), job.position, Level.ASSIST,
               priority=2, est_duration_s=60.0, params={"zone": job.params.get("leak")})
    f = Finding("valve_isolated", "info", f"{job.target} closed, gas reading falling", job.position, {}, [prep])
    return JobStatus.FINDING, [f], f"closed {job.target}"


STUBS = [
    Stub("gauge_read", Level.AUTO, {"camera"}, {Q, A}, _gauge),
    Stub("valve_check", Level.AUTO, {"thermal"}, {Q}, _ok("valve position and stem temperature normal")),
    Stub("aerial_survey", Level.AUTO, {"camera"}, {A}, _ok("structure imaged, no visible damage")),
    Stub("crawler_ut", Level.AUTO, {"ut_probe"}, {C}, _thickness),
    Stub("gas_survey", Level.AUTO, {"gas_sensor"}, {Q, A}, _gas_survey),
    Stub("valve_isolate", Level.SUPV, {"arm"}, {Q}, _isolate),
    Stub("leak_repair_prep", Level.ASSIST, {"camera"}, {Q}, _ok("leak marked, barrier set, gas monitor placed")),
]


def load_executors() -> tuple[dict, list[str], list[str]]:
    loaded = []
    for m in pkgutil.iter_modules(fleet.jobs.__path__):
        try:
            importlib.import_module(f"fleet.jobs.{m.name}")
            loaded.append(m.name)
        except Exception as exc:
            print(f"  (skipping fleet.jobs.{m.name}: {type(exc).__name__}: {exc})")
    ex = dict(REGISTRY)
    stubbed = []
    for s in STUBS:
        if s.kind not in ex:
            ex[s.kind] = s
            stubbed.append(s.kind)
    return ex, loaded, stubbed


class GeminiExplainer:

    def __init__(self, max_calls: int = 2, timeout: float = 20.0):
        from policy.gemini_verifier import load_api_key
        self.key = load_api_key()
        self.model = None
        self.calls = max_calls
        self.timeout = timeout

    def __call__(self, ctx: dict) -> tuple[str, bool]:
        if not self.key or self.calls <= 0:
            return offline_explanation(ctx), False
        self.calls -= 1
        try:
            from policy.gemini_verifier import generate_json, pick_text_model
            self.model = self.model or pick_text_model(self.key)
            prompt = ("You are the dispatcher for an offshore rig robot fleet. Write ONE short sentence (max 25 words) "
                      "telling the control-room operator why the fleet just changed priorities. Facts (JSON): "
                      + json.dumps(ctx))
            out = generate_json(prompt, {"type": "OBJECT", "properties": {"line": {"type": "STRING"}},
                                         "required": ["line"]}, self.key, self.model, timeout=self.timeout)
            line = str(out.get("line", "")).strip()
            return (line[:240], True) if line else (offline_explanation(ctx), False)
        except Exception:
            return offline_explanation(ctx), False


def build(offline: bool = True, timeline_path=None, approve_delay_s: float = 40.0):
    executors, loaded, stubbed = load_executors()
    world = World.from_rig_meta()
    robots = [
        Robot("Q1", Q, {"arm", "spray", "camera"}, (-23.0, 18.5, D), battery=0.95, speed_mps=0.9),
        Robot("Q2", Q, {"camera", "thermal", "gas_sensor"}, (-23.0, 13.0, D), battery=0.9, speed_mps=1.0),
        Robot("A1", A, {"camera", "thermal", "gas_sensor"}, (-25.0, -25.0, 47.0), battery=0.45, speed_mps=4.0),
        Robot("C1", C, {"ut_probe", "camera"}, (-3.0, -6.5, D), battery=0.8, speed_mps=0.3),
    ]
    planner = WalkwayPlanner(Walkways(patrol_route()))
    c = Coordinator(robots, world, executors, planner, auto_approve=False,
                    explainer=None if offline else GeminiExplainer())
    sim = FleetSim(c, timeline_path=timeline_path)

    pending: dict[str, float] = {}
    original = c.on_event

    def watch(e):
        original(e)
        if e["event"] == "approval_requested":
            pending[e["job"]] = e["t"] + approve_delay_s
    c.on_event = watch

    def operator(s: FleetSim):
        for jid, due in list(pending.items()):
            if s.world.t >= due:
                pending.pop(jid)
                s.c.approve(jid, by="operator")
        s.at(s.world.t + 1, "operator", operator)
    sim.at(0, "operator", operator)

    for g in sorted(world.of_type("gauge"), key=lambda a: a["id"]):
        c.submit(new_job(c, "gauge_read", g["id"], g["position_m"], Level.AUTO, priority=4, est_duration_s=40.0))
    for v in sorted(world.of_type("valve"), key=lambda a: a["id"]):
        c.submit(new_job(c, "valve_check", v["id"], v["position_m"], Level.AUTO, priority=5, est_duration_s=30.0))
    for name, p in AERIAL_TARGETS:
        c.submit(new_job(c, "aerial_survey", name, p, Level.AUTO, priority=4, est_duration_s=60.0,
                         params={"standoff": 0.0}))
    for name, p in CRAWLER_TARGETS:
        c.submit(new_job(c, "crawler_ut", name, p, Level.AUTO, priority=4, est_duration_s=90.0,
                         params={"standoff": 0.0}))
    rust = new_job(c, "rust_repair", RUST_SITE[0], RUST_SITE[1], Level.SUPV, priority=3, est_duration_s=180.0,
                   params={"scenario": {"seed": 1}, "standoff": 0.8})
    c.submit(rust)

    sim.people_paths.append(walk([(-1.0, 11.0, D), (-1.0, 4.5, D), (0.2, 4.0, D)], t0=150.0, speed=0.8, linger_s=25.0))

    def inject(s: FleetSim):
        leak = dict(LEAK)
        s.world.state["gas_leaks"].append(leak)
        s.c.emit("gas_leak_injected", None, None, position=list(leak["position"]), source=leak["source"],
                 rate_g_s=leak["rate_g_s"])
    sim.at(LEAK["t0"], "gas leak", inject)

    def on_gas(s: FleetSim, robot: Optional[str], leak: dict, how: str):
        job = new_job(s.c, "gas_survey", leak["id"], leak["position"], Level.AUTO, priority=1, est_duration_s=40.0,
                      params={"zone": leak["id"], "standoff": 2.0})
        s.c.report(job, robot, how, position=[round(v, 3) for v in leak["position"]],
                   source="robot gas sensor" if robot else "fixed gas detector")
    sim.on_gas = on_gas
    return sim, loaded, stubbed


FMT_SKIP = {"segment_clear"}


def fmt(e: dict) -> str:
    d = dict(e["detail"])
    d.pop("bids", None)
    detail = ", ".join(f"{k}={v}" for k, v in d.items())
    return f"{e['t']:6.0f}s  {e['event']:<20} {str(e['job'] or ''):<24} {str(e['robot'] or ''):<4} {detail}"[:200]


def summarize(c: Coordinator) -> list[dict]:
    rows = []
    for jid, job in c.jobs.items():
        res = c.results.get(jid)
        rows.append({"job": jid, "kind": job.kind, "target": job.target, "level": job.level.value,
                     "priority": job.priority, "robot": res.robot_id if res else c.assigned_to.get(jid, ""),
                     "status": c.status[jid].value,
                     "findings": "; ".join(f.kind for f in res.findings) if res else ""})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='15-minute offshore shift with four robots')
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--offline", action="store_true", help="never call Gemini (template explanations)")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)
    wall = time.perf_counter()
    sim, loaded, stubbed = build(offline=args.offline, timeline_path=REPO / "runs/fleet/timeline.jsonl")
    print(f"job modules loaded: {', '.join(loaded) or 'none'}; stub executors for: {', '.join(stubbed) or 'none'}")
    sim.run(args.minutes * 60.0)
    c = sim.c
    if not args.quiet:
        print("\nTIMELINE (events)")
        for e in c.timeline:
            if e["event"] not in FMT_SKIP:
                print(fmt(e))
    rows = summarize(c)
    print("\nJOBS")
    print(f"{'job':<26}{'kind':<18}{'target':<20}{'level':<7}{'pri':<4}{'robot':<6}{'status':<17}findings")
    for r in sorted(rows, key=lambda r: (r["priority"], r["job"])):
        print(f"{r['job']:<26}{r['kind']:<18}{r['target']:<20}{r['level']:<7}{r['priority']:<4}{r['robot']:<6}"
              f"{r['status']:<17}{r['findings']}")
    counts = c.counts()
    print("\nSTATUS COUNTS: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("ROBOTS: " + ", ".join(f"{rt.robot.id} {rt.state} battery={rt.robot.battery:.2f}" for rt in c.rt.values()))
    print(f"priority changes: {sum(e['event'] == 'priority_change' for e in c.timeline)}, "
          f"person pauses: {sum(e['event'] == 'person_pause' for e in c.timeline)}, "
          f"segment waits: {sum(e['event'] == 'segment_wait' for e in c.timeline)}, "
          f"used_gemini={c.used_gemini}, waiting approval: {c.approvals or 'none'}, crew: {len(c.crew)} jobs")
    out = REPO / "runs/fleet/summary.json"
    out.write_text(json.dumps({"counts": counts, "jobs": rows, "used_gemini": c.used_gemini,
                               "modules": loaded, "stubbed": stubbed}, indent=1), encoding="utf-8")
    print(f"timeline: {sim.path} ({len(sim.lines)} lines); summary: {out}; wall {time.perf_counter() - wall:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

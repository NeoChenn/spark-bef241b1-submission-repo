from __future__ import annotations

import bisect
from dataclasses import dataclass, field

SETUP_S = {"valve": 5.0, "rust": 20.0, "gas": 15.0, "crack": 15.0, "leak_check": 15.0, "inspect": 10.0, "weld": 20.0, "pick_can": 3.0}
FLANGE_KINDS = {"gas", "crack", "leak_check"}
RUST_FULL = [(0, "DETECT", "rust confirmed"), (6, "SPRAY", "spray converter"), (14, "SOAK", "soak 30 s"),
             (44, "SCRUB", "scrub, force-limited"), (90, "VERIFY", "on-board + second opinion"), (100, "PAINT", "paint"),
             (120, "DONE", "repair complete")]
SIMPLE = {"valve": [(0, "VALVE_APPROACH", "reach the isolation valve"), (4, "VALVE_CLOSE", "turn the valve shut"),
                   (12, "DONE", "isolation valve closed")]}
LEAK_SCENARIO = {"clean": "none", "loose": "loose", "crack": "crack", "big": "big"}


@dataclass
class JobTrace:
    kind: str
    scenario: str
    duration_s: float
    outcome: str
    segments: list[tuple[float, str, str]]
    alerts: list[dict] = field(default_factory=list)
    n_decisions: int = 0

    @property
    def total_s(self) -> float:
        return SETUP_S[self.kind] + self.duration_s

    def state_at(self, elapsed: float) -> tuple[str, str]:
        e = elapsed - SETUP_S[self.kind]
        if e < 0:
            return "SETUP", f"positioning for {self.kind} job"
        i = bisect.bisect_right([s[0] for s in self.segments], e) - 1
        _, st, why = self.segments[max(i, 0)]
        return st, why


def _segments(records) -> list[tuple[float, str, str]]:
    out: list[tuple[float, str, str]] = []
    for r in records:
        t = round(float(r.t), 3)
        if out and out[-1][0] == t:
            out[-1] = (t, r.to, r.reason)
        else:
            out.append((t, r.to, r.reason))
    return out


def pick_trace(scenario: str) -> JobTrace:
    from fleet.jobs.pick_can import simulate
    r = simulate({"seed": scenario})
    return JobTrace("pick_can", scenario, float(r["time_s"]), r["outcome"], [tuple(p) for p in r["phases"]])


def run_orchestrator(kind: str, scenario: str) -> JobTrace:
    if kind in SIMPLE or (kind, scenario) == ("rust", "repair"):
        segs = SIMPLE.get(kind) or RUST_FULL
        return JobTrace(kind, scenario, float(segs[-1][0]), segs[-1][1], segs)
    if kind == "weld":
        from policy.weld.job import weld_trace
        return weld_trace(scenario)
    if kind == "pick_can":
        return pick_trace(scenario)
    from policy.leak.mock import build_rig
    from policy.orchestrator import S
    if kind in FLANGE_KINDS:
        orch, bridge, *_ = build_rig(None, leak=LEAK_SCENARIO[scenario], rust=False)
        orch._start_flange("F1")
        orch.state = S.LEAK_APPROACH
    else:
        orch, bridge, *_ = build_rig(None, leak="none", rust=kind == "rust" or scenario == "rust")
    orch.run(500)
    alerts = list(getattr(orch, "alerts", []))
    return JobTrace(kind, scenario, float(bridge.t), orch.state.value, _segments(orch.records), alerts,
                    len(orch.records))


class JobLibrary:
    def __init__(self, runner=run_orchestrator):
        self.runner = runner
        self.cache: dict[tuple[str, str], JobTrace] = {}

    def get(self, kind: str, scenario: str) -> JobTrace:
        k = (kind, scenario)
        if k not in self.cache:
            self.cache[k] = self.runner(kind, scenario)
        return self.cache[k]

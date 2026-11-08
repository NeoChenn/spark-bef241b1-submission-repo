"""Shared contract for the multi-robot fleet: robots, jobs, autonomy levels, results.

Every job module (fleet/jobs/*.py) and the coordinator (fleet/coordinator.py) build on these types only,
so they can be written in parallel and still fit together. Pure Python; no Isaac, no network.

Autonomy levels follow the team's job table:
  AUTO    robot does it end to end and logs the result
  SUPV    robot does it after the operator approves (the coordinator holds the job until approved)
  ASSIST  robot finds and prepares, a person fixes
  HUMAN   stays manual (the fleet only records and routes it to the crew)

Robot kinds: Q quadruped (Go2 + lift + UR3), C crawler (magnetic, vessels/piping), A aerial drone,
R subsea ROV. World coordinates are metres, Z up, as in sim/output/full_rig_meta.json.

Change this file only by agreement: other branches import it. Add new optional fields with defaults.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

REPO = Path(__file__).resolve().parent.parent
RIG_META = REPO / "sim/output/full_rig_meta.json"


class Level(str, Enum):
    AUTO = "AUTO"
    SUPV = "SUPV"
    ASSIST = "ASSIST"
    HUMAN = "HUMAN"


class RobotKind(str, Enum):
    QUADRUPED = "Q"
    CRAWLER = "C"
    AERIAL = "A"
    ROV = "R"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    DONE = "DONE"                  # finished, nothing wrong found or fixed
    FINDING = "FINDING"            # finished, found something (see findings)
    HANDED_TO_HUMAN = "HANDED_TO_HUMAN"
    FAILED = "FAILED"


@dataclass
class Robot:
    id: str
    kind: RobotKind
    capabilities: set[str]                      # e.g. {"camera", "thermal", "arm", "gas_sensor", "spray"}
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery: float = 1.0                        # 0..1
    speed_mps: float = 1.0
    busy_with: Optional[str] = None             # job id
    deck: Optional[float] = None                # z of the deck it is on (Q/C); None for A/R


@dataclass
class Job:
    id: str
    kind: str                                   # e.g. "gauge_read", "gas_survey", "valve_isolate"
    target: str                                 # asset id from the rig meta, e.g. "gauge_03", or a free label
    position: tuple[float, float, float]
    level: Level
    needs: set[str] = field(default_factory=set)         # capabilities the robot must have
    robot_kinds: set[RobotKind] = field(default_factory=set)  # allowed kinds; empty = any with `needs`
    priority: int = 3                           # 1 = urgent (safety), 5 = routine
    est_duration_s: float = 60.0
    params: dict[str, Any] = field(default_factory=dict)
    created_t: float = 0.0
    parent: Optional[str] = None                # job that spawned this one (e.g. leak found -> isolate)


@dataclass
class Finding:
    kind: str                                   # e.g. "gauge_out_of_range", "gas_leak", "corrosion_Ri3"
    severity: str                               # "info" | "warning" | "critical"
    summary: str
    position: Optional[tuple[float, float, float]] = None
    evidence: dict[str, Any] = field(default_factory=dict)   # numbers, image paths, Gemini verdicts
    follow_up: list[Job] = field(default_factory=list)       # jobs the coordinator should queue next


@dataclass
class JobResult:
    job_id: str
    robot_id: str
    status: JobStatus
    started_t: float
    finished_t: float
    findings: list[Finding] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    used_gemini: bool = False

    def to_json(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@runtime_checkable
class JobExecutor(Protocol):
    """One job type. `run` is called by the coordinator once a robot is assigned (and approved for SUPV)."""
    kind: str
    level: Level
    needs: set[str]
    robot_kinds: set[RobotKind]

    def run(self, job: Job, robot: Robot, world: "World") -> JobResult: ...


@dataclass
class World:
    """What the fleet knows about the rig. `assets` comes from full_rig_meta.json (gauges, valves, ...)."""
    t: float = 0.0
    assets: dict[str, dict] = field(default_factory=dict)        # id -> {"type", "position_m", ...}
    state: dict[str, Any] = field(default_factory=dict)          # simulated truth for mock runs (leaks, ...)
    people: list[tuple[float, float, float]] = field(default_factory=list)

    @classmethod
    def from_rig_meta(cls, path: Path = RIG_META) -> "World":
        assets = {}
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            for key in ("inspection_points", "points", "assets"):
                for p in meta.get(key, []) or []:
                    if isinstance(p, dict) and "id" in p:
                        assets[p["id"]] = p
        return cls(assets=assets)

    def of_type(self, kind: str) -> list[dict]:
        return [a for a in self.assets.values() if a.get("type") == kind]


REGISTRY: dict[str, JobExecutor] = {}


def register(executor: JobExecutor) -> JobExecutor:
    """Job modules call this at import time: `register(GaugeReadJob())`."""
    REGISTRY[executor.kind] = executor
    return executor

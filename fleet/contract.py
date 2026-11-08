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
    DONE = "DONE"
    FINDING = "FINDING"
    HANDED_TO_HUMAN = "HANDED_TO_HUMAN"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Robot:
    id: str
    kind: RobotKind
    capabilities: set[str]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery: float = 1.0
    speed_mps: float = 1.0
    busy_with: Optional[str] = None
    deck: Optional[float] = None


@dataclass
class Job:
    id: str
    kind: str
    target: str
    position: tuple[float, float, float]
    level: Level
    needs: set[str] = field(default_factory=set)
    robot_kinds: set[RobotKind] = field(default_factory=set)
    priority: int = 3
    est_duration_s: float = 60.0
    params: dict[str, Any] = field(default_factory=dict)
    created_t: float = 0.0
    parent: Optional[str] = None


@dataclass
class Finding:
    kind: str
    severity: str
    summary: str
    position: Optional[tuple[float, float, float]] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    follow_up: list[Job] = field(default_factory=list)


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
    phases: list[tuple[float, str, str]] = field(default_factory=list)

    def to_json(self) -> dict:
        d = _plain(asdict(self))
        d["status"] = self.status.value
        return d


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if isinstance(x, Enum):
        return x.value
    return x


@runtime_checkable
class JobExecutor(Protocol):
    kind: str
    level: Level
    needs: set[str]
    robot_kinds: set[RobotKind]

    def run(self, job: Job, robot: Robot, world: "World") -> JobResult: ...


@dataclass
class World:
    t: float = 0.0
    assets: dict[str, dict] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
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
    REGISTRY[executor.kind] = executor
    return executor

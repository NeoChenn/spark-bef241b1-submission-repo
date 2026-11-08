from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KIND_BASE = {"gas": 100.0, "pick_can": 60.0, "crack": 50.0, "weld": 30.0, "rust": 10.0, "leak_check": 3.0, "inspect": 1.0}
NEEDS = {"pick_can": frozenset({"arm", "gripper"})}
URGENT = ("gas", "pick_can")
DEADLINE_W = 5.0
DEADLINE_WINDOW_S = 600.0
OPEN = {"pending", "assigned", "active"}


@dataclass
class Task:
    id: str
    kind: str
    poi: str
    node: int
    created: float
    severity: float = 0.0
    exposure: float = 0.0
    deadline: Optional[float] = None
    scenario: str = "clean"
    status: str = "pending"
    robot: Optional[str] = None
    progress: float = 0.0
    duration: float = 0.0
    arrived: Optional[float] = None
    finished: Optional[float] = None
    outcome: Optional[str] = None
    preemptions: int = 0


def priority(task: Task, t: float) -> float:
    base = KIND_BASE[task.kind] * (task.severity * task.exposure if task.kind == "rust" else 1.0)
    if task.deadline is None:
        return base
    slack = task.deadline - t
    return base + DEADLINE_W * min(1.0, max(0.0, 1.0 - slack / DEADLINE_WINDOW_S))

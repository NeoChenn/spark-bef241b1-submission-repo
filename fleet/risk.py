from __future__ import annotations

from policy.fleet.tasks import KIND_BASE

KIND_OF = {"gas_survey": "gas", "valve_isolate": "gas", "gas": "gas", "flange_crack": "crack", "crack": "crack",
           "rust_repair": "rust", "rust": "rust", "flange_leak_check": "leak_check", "leak_check": "leak_check",
           "gauge_read": "inspect", "inspect": "inspect"}
RUST_URGENT = 5.0


def risk_score(kind: str, severity: float = 0.0, exposure: float = 0.0) -> float:
    k = KIND_OF.get(kind, "inspect")
    return KIND_BASE[k] * (severity * exposure if k == "rust" else 1.0)


def risk_priority(kind: str, severity: float = 0.0, exposure: float = 0.0) -> int:
    s = risk_score(kind, severity, exposure)
    if s >= KIND_BASE["gas"]:
        return 1
    if s >= KIND_BASE["crack"]:
        return 2
    if s >= RUST_URGENT:
        return 3
    if s > KIND_BASE["inspect"]:
        return 4
    return 5

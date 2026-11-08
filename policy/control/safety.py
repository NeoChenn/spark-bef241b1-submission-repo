from __future__ import annotations

from policy.control.schema import Action
from policy.control.snapshot import SensorSnapshot

FORCE_LIMIT_N = 40.0
LEL_EVACUATE = 50.0
LEL_CLEAR = 10.0
PERSON_MIN_M = 3.0
UNSAFE_FOR_LOCKED = {"dispatch", "resume", "second_view"}


def _edge(kind, robot="", target="", why="", rule="") -> Action:
    return Action(kind, robot, target, 1.0, why, source="edge", rule=rule)


def hazard_zones(snap: SensorSnapshot, latched=()) -> dict[str, float]:
    out: dict[str, float] = {}
    for g in snap.gas:
        if g.lel >= LEL_EVACUATE or (g.zone in latched and g.lel >= LEL_CLEAR):
            out[g.zone] = max(out.get(g.zone, 0.0), g.lel)
    return out


def _gas_why(lel: float, zone: str) -> str:
    if lel >= LEL_EVACUATE:
        return f"gas {lel:.0f}% LEL >= {LEL_EVACUATE:.0f}% in zone {zone}"
    return f"evacuation of {zone} latched until gas < {LEL_CLEAR:.0f}% LEL (now {lel:.0f}%)"


def edge_rules(snap: SensorSnapshot, latched=()) -> list[Action]:
    rules = []
    hz = hazard_zones(snap, latched)
    for r in snap.robots:
        if r.force_n is not None and r.force_n > FORCE_LIMIT_N:
            rules.append(_edge("back_off", r.id, "", f"contact force {r.force_n:.0f} N > {FORCE_LIMIT_N:.0f} N limit",
                               "force"))
        if r.zone in hz:
            rules.append(_edge("retreat", r.id, "upwind", _gas_why(hz[r.zone], r.zone), "gas"))
        if r.person_m is not None and r.person_m < PERSON_MIN_M:
            rules.append(_edge("pause", r.id, "", f"person at {r.person_m:.1f} m < {PERSON_MIN_M:.0f} m", "person"))
    for zone, lel in hz.items():
        rules.append(_edge("evacuate", "", zone, _gas_why(lel, zone), "gas"))
    return rules


def _zone_of(target: str, snap: SensorSnapshot) -> str:
    job = next((j for j in snap.jobs if j.get("id") == target), None)
    if job:
        return job.get("zone", "")
    r = snap.robot(target)
    return r.zone if r else target


def arbitrate(edge: list[Action], cloud: list[Action], snap: SensorSnapshot) -> tuple[list[Action], list[dict]]:
    locked = {a.robot: a for a in edge if a.robot}
    hz = evacuated = {a.target for a in edge if a.kind == "evacuate"}
    final, vetoed = list(edge), []
    for a in cloud:
        why = None
        robot = snap.robot(a.robot)
        if a.kind in UNSAFE_FOR_LOCKED and a.robot in locked:
            why = f"edge rule '{locked[a.robot].rule}' holds {a.robot}: {locked[a.robot].rationale}"
        elif a.kind in ("dispatch", "second_view") and _zone_of(a.target, snap) in hz:
            why = f"target zone {_zone_of(a.target, snap)} is evacuated by the edge gas rule"
        elif a.kind == "resume" and robot is not None and robot.zone in hz:
            why = f"{a.robot} is in gas hazard zone {robot.zone}"
        elif a.kind == "evacuate" and a.target in evacuated:
            continue
        if why:
            vetoed.append({**a.to_dict(), "why": why})
        else:
            final.append(a)
    return final, vetoed

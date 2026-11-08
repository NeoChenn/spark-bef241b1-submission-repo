from __future__ import annotations

from typing import Optional

from policy.control.schema import VERIFY, Action
from policy.control.snapshot import GasSensor, SensorSnapshot

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
    if r:
        return r.zone
    s = next((x for x in [*snap.health, *snap.gas, *snap.fixed] if x.id == target and x.zone), None)
    return s.zone if s else target


def hazard_first(snap: SensorSnapshot) -> list[Action]:
    gas = {g.id: g for g in snap.gas}
    out = []
    for h in snap.health:
        if h.kind != "gas" or h.status == "OK" or not h.hazard:
            continue
        g = gas.get(h.id)
        if g is None:
            g = GasSensor(h.id, zone=h.zone, lel=0.0)
            snap.gas.append(g)
        zone = max((x.lel for x in snap.gas if x is not g and x.zone == h.zone), default=0.0)
        g.lel = max(g.lel, zone, float(h.evidence.get("peer_max") or 0.0))
        out.append(_edge("escalate", "", h.id, f"{h.id} {h.status} beside live gas ({g.lel:.0f}% LEL): treated as "
                                                 "real gas until a robot reads it", "hazard_first"))
    return out


def verify_fallback(snap: SensorSnapshot, edge: list[Action], skip=frozenset()) -> list[Action]:
    locked = {a.robot for a in edge if a.robot}
    evacuated = {a.target for a in edge if a.kind == "evacuate"}
    taken, out = set(), []
    for h in sorted(snap.faults(), key=lambda h: (not h.hazard, h.since)):
        if h.job or h.id in skip:
            continue
        if h.zone in evacuated or in_live_gas(snap, h.id) is not None:
            continue
        ok = [c for c in h.candidates if c.get("ok") and c["robot"] not in locked | taken]
        if not ok:
            out.append(_edge("escalate", "", h.id, f"{h.id} {h.status}: no capable robot with battery for the "
                                                   "round trip", "sensor"))
            continue
        c = min(ok, key=lambda c: c["eta_s"])
        taken.add(c["robot"])
        a = _edge("dispatch", c["robot"], h.id, f"{h.id} {h.status}; nearest capable robot {c['robot']} "
                                                f"({c['eta_s']:.0f} s by the rig graph, {c['sensor']})", "sensor")
        a.purpose = VERIFY
        out.append(a)
    return out


def live_gas(snap: SensorSnapshot, zone: str) -> float:
    return max((g.lel for g in snap.gas if g.zone == zone), default=0.0)


def in_live_gas(snap: SensorSnapshot, sid: str) -> Optional[float]:
    h = next((h for h in snap.health if h.id == sid), None)
    if h is None or not h.hazard:
        return None
    lel = live_gas(snap, h.zone)
    return lel if lel >= LEL_CLEAR else None


def arbitrate(edge: list[Action], cloud: list[Action], snap: SensorSnapshot) -> tuple[list[Action], list[dict]]:
    locked = {a.robot: a for a in edge if a.robot}
    hz = evacuated = {a.target for a in edge if a.kind == "evacuate"}
    final, vetoed = list(edge), []
    for a in cloud:
        why = None
        robot = snap.robot(a.robot)
        gas = in_live_gas(snap, a.target) if a.purpose == VERIFY else None
        if gas is not None:
            why = f"hazard first: {a.target} sits in live gas ({gas:.0f}% LEL); the gas response goes first"
        elif a.kind in UNSAFE_FOR_LOCKED and a.robot in locked:
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

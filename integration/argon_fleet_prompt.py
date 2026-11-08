from __future__ import annotations

import json
import math
import re
from typing import Optional

MODEL = "gemini-4-argon"
THINKING = "low"
WHY_MAX = 90
CLEAR_LEL, ALARM_LEL, STOP_LEL, NEAR_M, SPARK_RADIUS_M = 5.0, 10.0, 20.0, 2.0, 6.0
SCRUBBER, FIXER, SUPERVISOR = "unit1", "unit2", "supv"

GO_CHECK, FIX_LEAK, HOLD, RETREAT, RESUME_RUST, LOG_SENSOR_FAULT, ESCALATE = (
    "go_check", "fix_leak", "hold", "retreat", "resume_rust", "log_sensor_fault", "escalate")
KINDS = (GO_CHECK, FIX_LEAK, HOLD, RETREAT, RESUME_RUST, LOG_SENSOR_FAULT, ESCALATE)
ROBOTS = (SCRUBBER, FIXER, SUPERVISOR)
ALLOWED = {GO_CHECK: {FIXER}, FIX_LEAK: {FIXER}, HOLD: {SCRUBBER}, RETREAT: {SCRUBBER, FIXER},
           RESUME_RUST: {SCRUBBER}, LOG_SENSOR_FAULT: {SUPERVISOR}, ESCALATE: {SUPERVISOR}}
FLANGE_TARGET = (GO_CHECK, FIX_LEAK)
OPEN_LEAK = ("LEAK_CONFIRMED", "UNVERIFIED", "LEAK_UNSEEN_BY_FIXED")

SYSTEM = (
    "You are Argon, oil-rig command centre. Give the commands to issue now.\n"
    "unit1 Scrubber: rust work (wire brush = spark hot work). unit2 Fixer: leak checks and repairs. "
    "supv: human inbox; you approve nothing.\n"
    "%LEL: green<5 amber>=5 red>=10 stop>=20 (stop is real gas, even on a faulty sensor).\n"
    "Rules, hazard first:\n"
    "1 stop: hold unit1, retreat any robot within 2 m, escalate. No fix_leak.\n"
    "2 spark non-empty and unit1 not held: hold unit1 (rust paused, not cancelled).\n"
    "3 UNVERIFIED: go_check unit2 at the flange.\n"
    "4 LEAK_CONFIRMED: fix_leak unit2. LEAK_UNSEEN_BY_FIXED: fix_leak and log_sensor_fault.\n"
    "5 PIPE_SENSOR_FAULT below stop: log_sensor_fault only, no repair, no hold.\n"
    "6 unit1 held, spark empty, no open leak: resume_rust unit1.\n"
    "Never repeat a command in force. Nothing to change: []. Otherwise unsure: escalate.\n"
    "robot: unit2 go_check/fix_leak, unit1 hold/resume_rust, supv log_sensor_fault/escalate. "
    "target: flange, detector id for log_sensor_fault, \"\" for resume_rust. confidence 0-1. why: one line <90 chars citing readings."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "commands": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS)},
                "robot": {"type": "string", "enum": list(ROBOTS)},
                "target": {"type": "string"},
                "confidence": {"type": "number"},
                "why": {"type": "string"},
            },
            "required": ["kind", "robot", "target", "confidence", "why"],
        }},
    },
    "required": ["commands"],
}

ROBOT_KEYS = (("pos", "pos"), ("state", "state"), ("lel", "lel"), ("battery", "bat"), ("bat", "bat"),
              ("job", "job"), ("held", "held"), ("dock", "dock"))


def _r(v, n=1):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), n) if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_r(x, n) for x in v]
    return v


def _get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else getattr(d, k, default)


def _xy(p):
    return None if p is None else [float(v) for v in list(p)[:2]]


def _band(lel, fault=False):
    if lel is None or not math.isfinite(lel):
        return "grey"
    if fault and lel < STOP_LEL:
        return "grey"
    return "red" if lel >= ALARM_LEL else "amber" if lel >= CLEAR_LEL else "green"


def _robots(robots: dict) -> dict:
    out = {}
    for rid in sorted(robots or {}):
        r = robots[rid] or {}
        row = {}
        for k, short in ROBOT_KEYS:
            if k in r and short not in row:
                v = r[k]
                row[short] = _r(_xy(v)) if k == "pos" else _r(v, 2 if k in ("battery", "bat") else 1)
        out[rid] = row
    return out


def _detectors(detectors, fd, findings, faulted, u1):
    health = {f.sensor: f.kind for f in findings or []}
    if fd is not None:
        health[fd.sensor] = fd.kind
    rows, spark = [], []
    for d in detectors or []:
        did, lel = _get(d, "id"), _get(d, "last", _get(d, "lel"))
        fault = did in faulted or health.get(did) == "PIPE_SENSOR_FAULT"
        band = _band(lel, fault)
        row = {"id": did, "flange": _get(d, "flange"), "lel": _r(lel), "led": band}
        trend = _get(d, "trend")
        if trend is not None:
            row["trend"] = _r(trend)
        if did in health:
            row["health"] = health[did]
        pos = _xy(_get(d, "pos"))
        if u1 is not None and pos is not None:
            dist = math.dist(u1, pos)
            row["d_unit1"] = _r(dist)
            if dist <= SPARK_RADIUS_M and band in ("amber", "red"):
                spark.append(did)
        rows.append(row)
    return rows, spark


def build_prompt(fd, robots: dict, detectors, t, event: str = "", findings=None, faulted=()) -> str:
    u1 = _xy((robots or {}).get(SCRUBBER, {}).get("pos"))
    dets, spark = _detectors(detectors, fd, findings, set(faulted), u1)
    snap = {"robots": _robots(robots), "detectors": dets, "spark": spark}
    if fd is not None:
        snap["finding"] = {"kind": fd.kind, "sensor": fd.sensor, "flange": fd.flange, "fixed": _r(fd.fixed_lel),
                           "by": fd.robot or None, "robot_lel": _r(fd.robot_lel), "dist": _r(fd.robot_dist),
                           "hazard": bool(fd.hazard)}
    if event:
        snap["event"] = event
    return json.dumps(snap, separators=(",", ":"), default=str)


def _why(s: str) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= WHY_MAX else s[:WHY_MAX - 1].rstrip() + "…"


class Commands(list):
    def __init__(self, *a):
        super().__init__(*a)
        self.vetoed: list[dict] = []


def parse_reply(text: str, fd, ids: set) -> list:
    from integration.dual_argon import Command

    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON in reply")
    d = json.loads(m.group(0))
    if not isinstance(d, dict) or not isinstance(d.get("commands"), list):
        raise ValueError("commands missing")
    verdict = fd.kind if fd is not None else str(d.get("verdict") or "OK")
    hazard = bool(fd is not None and fd.hazard)
    flanges = {fd.flange} if fd is not None else set()
    out = Commands()
    for c in d["commands"]:
        if not isinstance(c, dict):
            raise ValueError("command is not an object")
        kind = str(c.get("kind") or c.get("command") or "")
        robot, target = str(c.get("robot") or ""), str(c.get("target") or "")
        if kind not in KINDS:
            raise ValueError(f"command {kind!r} not allowed")
        if robot not in ALLOWED[kind]:
            raise ValueError(f"{kind} cannot go to {robot!r}")
        if kind in FLANGE_TARGET + (LOG_SENSOR_FAULT,) and target not in ids:
            raise ValueError(f"unknown target {target!r}")
        if kind == RESUME_RUST and fd is not None and fd.kind in OPEN_LEAK and not hazard:
            out.vetoed.append({"kind": kind, "robot": robot, "target": target, "source": "argon",
                               "finding": fd.kind, "why": f"resume refused: leak open at {fd.flange}"})
            continue
        if hazard and kind in (FIX_LEAK, GO_CHECK, RESUME_RUST):
            raise ValueError(f"{kind} refused: reading at or above {STOP_LEL:.0f} %LEL")
        if fd is not None and fd.kind == "PIPE_SENSOR_FAULT" and not hazard and kind == FIX_LEAK \
                and target in flanges:
            raise ValueError("fix_leak refused on a faulted detector")
        conf = float(c.get("confidence", 0.0))
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence out of range")
        out.append(Command(kind, robot, target, "argon", _why(c.get("why") or d.get("situation") or c.get("rationale")), conf, verdict))
    if hazard and not any(c.kind == RETREAT for c in out) and fd.robot and fd.robot_dist is not None \
            and fd.robot_dist <= NEAR_M:
        raise ValueError("hazard reading without a retreat")
    if not out and out.vetoed:
        raise ValueError(out.vetoed[0]["why"])
    return out

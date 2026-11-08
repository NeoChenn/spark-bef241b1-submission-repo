from __future__ import annotations

import math
from dataclasses import dataclass, field

from policy.control.snapshot import SensorSnapshot

CLOUD_KINDS = ("dispatch", "second_view", "hold", "resume", "escalate", "evacuate")
EDGE_KINDS = ("back_off", "retreat", "pause", "evacuate")
NEEDS_ROBOT = {"dispatch", "second_view", "hold", "resume"}
SEVERITIES = ("info", "warn", "critical")
PURPOSES = ("", "verify_sensor")
VERIFY = "verify_sensor"
LABELS = {
    "dispatch": "Dispatch {robot} to {target}", "second_view": "Second view of {target} by {robot}",
    "hold": "Hold {robot}", "resume": "Resume {robot}", "escalate": "Escalate {target} to control room",
    "evacuate": "Evacuate zone {target}", "back_off": "Back off {robot}", "retreat": "Retreat {robot} upwind",
    "pause": "Pause {robot}", "verify_sensor": "Send {robot} to verify sensor {target}",
}
MAX_ACTIONS = 5
MAX_LINES = 3
LINE_CHARS = 120
RATIONALE_CHARS = 160

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "situation": {"type": "string", "description": "Plain-English situation report, at most 3 short lines."},
        "actions": {"type": "array", "description": "Recommended actions, best first.", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(CLOUD_KINDS)},
                "robot": {"type": "string", "description": "Robot id, or empty for fleet-wide actions."},
                "target": {"type": "string", "description": "Job id, zone id, robot id or sensor id the action is about."},
                "purpose": {"type": "string", "enum": list(PURPOSES),
                            "description": "verify_sensor: dispatch the robot to read a suspect fixed sensor with "
                                           "its own onboard sensor; target is then the sensor id."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string", "description": "One short sentence citing the sensor evidence."},
            },
            "required": ["kind", "robot", "target", "confidence", "rationale"],
        }},
        "anomalies": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string"},
                "zone": {"type": "string"},
                "severity": {"type": "string", "enum": list(SEVERITIES)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "sensors": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            },
            "required": ["kind", "zone", "severity", "confidence", "note"],
        }},
    },
    "required": ["situation", "actions", "anomalies"],
}


class SchemaError(ValueError):
    pass


@dataclass
class Action:
    kind: str
    robot: str = ""
    target: str = ""
    confidence: float = 0.0
    rationale: str = ""
    source: str = "cloud"
    rule: str = ""
    purpose: str = ""

    @property
    def id(self) -> str:
        return f"{self.source}:{self.kind}:{self.robot}:{self.target}"

    @property
    def label(self) -> str:
        label = LABELS.get(self.purpose or self.kind, self.kind)
        if self.kind == "escalate" and not self.target:
            label = label.replace("{target} ", "")
        return label.format(robot=self.robot or "fleet", target=self.target or "-")

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "kind": self.kind, "robot": self.robot, "target": self.target,
                "confidence": round(self.confidence, 2), "rationale": self.rationale, "source": self.source,
                "rule": self.rule, "purpose": self.purpose}


@dataclass
class Anomaly:
    kind: str
    zone: str
    severity: str
    confidence: float
    note: str = ""
    sensors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": f"{self.kind}:{self.zone}", "kind": self.kind, "zone": self.zone, "severity": self.severity,
                "confidence": round(self.confidence, 2), "note": self.note, "sensors": list(self.sensors)}


@dataclass
class Plan:
    situation: list
    actions: list
    anomalies: list
    rejected: list = field(default_factory=list)


def _conf(v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise SchemaError(f"confidence {v!r} not in [0, 1]")
    return float(v)


def _str(d: dict, key: str, limit: int = 60) -> str:
    v = d.get(key, "")
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise SchemaError(f"{key} must be a string")
    return v.strip()[:limit]


def _action(d, snap: SensorSnapshot) -> Action:
    if not isinstance(d, dict):
        raise SchemaError("action is not an object")
    kind = _str(d, "kind")
    if kind not in CLOUD_KINDS:
        raise SchemaError(f"kind {kind!r} not in whitelist")
    robot, target, purpose = _str(d, "robot"), _str(d, "target"), _str(d, "purpose")
    robots = {r.id for r in snap.robots}
    jobs = {j.get("id") for j in snap.jobs}
    zones = snap.zones()
    if purpose not in PURPOSES:
        raise SchemaError(f"purpose {purpose!r} not in whitelist")
    if purpose == VERIFY:
        return _verify(kind, robot, target, d, snap)
    if kind in NEEDS_ROBOT and robot not in robots:
        raise SchemaError(f"unknown robot {robot!r}")
    if robot and robot not in robots:
        raise SchemaError(f"unknown robot {robot!r}")
    if kind == "dispatch" and target not in jobs | zones:
        raise SchemaError(f"unknown job or zone {target!r}")
    if kind == "second_view" and target not in jobs | zones | robots:
        raise SchemaError(f"unknown target {target!r}")
    if kind == "evacuate" and target not in zones:
        raise SchemaError(f"unknown zone {target!r}")
    return Action(kind, robot, target, _conf(d.get("confidence")), _str(d, "rationale", RATIONALE_CHARS))


def _verify(kind: str, robot: str, target: str, d: dict, snap: SensorSnapshot) -> Action:
    if kind != "dispatch":
        raise SchemaError("verify_sensor is a dispatch purpose")
    fault = next((h for h in snap.health if h.id == target), None)
    if fault is None:
        raise SchemaError(f"unknown sensor {target!r}")
    if fault.status == "OK":
        raise SchemaError(f"sensor {target} is healthy; nothing to verify")
    cand = next((c for c in fault.candidates if c.get("robot") == robot), None)
    if cand is None:
        raise SchemaError(f"{robot!r} is not a candidate for {target}")
    if not cand.get("ok"):
        raise SchemaError(f"{robot} cannot verify {target}: {cand.get('why', 'not capable')}")
    return Action(kind, robot, target, _conf(d.get("confidence")), _str(d, "rationale", RATIONALE_CHARS),
                  purpose=VERIFY)


def _anomaly(d) -> Anomaly:
    if not isinstance(d, dict):
        raise SchemaError("anomaly is not an object")
    sev = _str(d, "severity")
    if sev not in SEVERITIES:
        raise SchemaError(f"severity {sev!r}")
    sensors = [str(x)[:24] for x in d.get("sensors") or [] if isinstance(x, (str, int, float))][:6]
    return Anomaly(_str(d, "kind", 40), _str(d, "zone", 24), sev, _conf(d.get("confidence")),
                   _str(d, "note", RATIONALE_CHARS), sensors)


def validate(obj, snap: SensorSnapshot) -> Plan:
    if not isinstance(obj, dict):
        raise SchemaError("reply is not a JSON object")
    if not isinstance(obj.get("situation"), str):
        raise SchemaError("situation missing")
    if not isinstance(obj.get("actions"), list):
        raise SchemaError("actions missing")
    anomalies_in = obj.get("anomalies") or []
    if not isinstance(anomalies_in, list):
        raise SchemaError("anomalies must be a list")
    lines = [x.strip()[:LINE_CHARS] for x in obj["situation"].splitlines() if x.strip()][:MAX_LINES]
    rejected, best = [], {}
    for d in obj["actions"]:
        try:
            a = _action(d, snap)
        except SchemaError as e:
            rejected.append({"item": d if isinstance(d, dict) else repr(d), "why": str(e)})
            continue
        if a.id not in best or a.confidence > best[a.id].confidence:
            best[a.id] = a
    anomalies = []
    for d in anomalies_in:
        try:
            anomalies.append(_anomaly(d))
        except SchemaError as e:
            rejected.append({"item": d if isinstance(d, dict) else repr(d), "why": str(e)})
    actions = sorted(best.values(), key=lambda a: -a.confidence)[:MAX_ACTIONS]
    return Plan(lines, actions, anomalies, rejected)

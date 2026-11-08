from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CLEAR_LEL = 5.0
ALARM_LEL = 10.0
STOP_LEL = 20.0
NEAR_M = 2.0

LEAK_CONFIRMED = "LEAK_CONFIRMED"
PIPE_SENSOR_FAULT = "PIPE_SENSOR_FAULT"
LEAK_UNSEEN_BY_FIXED = "LEAK_UNSEEN_BY_FIXED"
UNVERIFIED = "UNVERIFIED"
OK = "OK"
FINDINGS = (LEAK_CONFIRMED, PIPE_SENSOR_FAULT, LEAK_UNSEEN_BY_FIXED, UNVERIFIED, OK)

GO_CHECK, FIX_LEAK, RETREAT, RESUME_RUST, LOG_SENSOR_FAULT, HOLD = (
    "go_check", "fix_leak", "retreat", "resume_rust", "log_sensor_fault", "hold")
COMMANDS = (GO_CHECK, FIX_LEAK, RETREAT, RESUME_RUST, LOG_SENSOR_FAULT, HOLD)


def led(lel: Optional[float]) -> str:
    if lel is None or not math.isfinite(lel):
        return "off"
    return "red" if lel >= ALARM_LEL else "amber" if lel >= CLEAR_LEL else "green"


@dataclass
class FixedDetector:
    id: str
    flange: str
    pos: tuple
    stuck_at: Optional[float] = None
    last: float = 0.0

    def read(self, gas_at: Callable[[np.ndarray], float]) -> float:
        v = self.stuck_at if self.stuck_at is not None else 100.0 * float(gas_at(np.asarray(self.pos, float)))
        self.last = round(float(v), 2)
        return self.last


@dataclass
class Finding:
    kind: str
    sensor: str
    flange: str
    fixed_lel: float
    robot_lel: Optional[float]
    robot_dist: Optional[float]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def fuse(fixed: FixedDetector, robot_lel: Optional[float], robot_dist: Optional[float],
         near_m: float = NEAR_M) -> Finding:
    f = fixed.last
    near = robot_dist is not None and robot_dist <= near_m and robot_lel is not None
    where = "--" if robot_dist is None else f"{robot_dist:.1f} m"
    rl = "--" if robot_lel is None else f"{robot_lel:.1f}"
    if f >= ALARM_LEL:
        if not near:
            kind, why = UNVERIFIED, f"{fixed.id} {f:.1f}%LEL, robot {rl}%LEL at {where}: too far to confirm"
        elif robot_lel >= CLEAR_LEL:
            kind, why = LEAK_CONFIRMED, f"{fixed.id} {f:.1f}%LEL and robot {rl}%LEL at {where} agree"
        else:
            kind, why = PIPE_SENSOR_FAULT, f"{fixed.id} {f:.1f}%LEL but robot reads {rl}%LEL at {where}"
    elif near and robot_lel >= ALARM_LEL:
        kind, why = LEAK_UNSEEN_BY_FIXED, f"robot {rl}%LEL at {where}, {fixed.id} only {f:.1f}%LEL"
    else:
        kind, why = OK, f"{fixed.id} {f:.1f}%LEL, robot {rl}%LEL at {where}"
    return Finding(kind, fixed.id, fixed.flange, f, None if robot_lel is None else round(robot_lel, 2),
                   None if robot_dist is None else round(robot_dist, 2), why)


@dataclass
class Command:
    kind: str
    robot: str
    target: str
    source: str
    reason: str
    confidence: float = 1.0
    finding: str = ""
    t: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def edge_rule(fd: Finding, robot: str, crack: bool = False) -> Command:
    if fd.robot_lel is not None and fd.robot_lel >= STOP_LEL or crack:
        kind, target = RETREAT, fd.flange
    else:
        kind, target = {UNVERIFIED: (GO_CHECK, fd.flange), LEAK_CONFIRMED: (FIX_LEAK, fd.flange),
                        LEAK_UNSEEN_BY_FIXED: (FIX_LEAK, fd.flange), PIPE_SENSOR_FAULT: (LOG_SENSOR_FAULT, fd.sensor),
                        OK: (RESUME_RUST, "")}[fd.kind]
    return Command(kind, robot, target, "edge", f"edge rule {fd.kind}: {fd.reason}", 1.0, fd.kind)


SYSTEM = (
    "You are Argon, the command centre of an offshore oil platform. One robot (Go2 quadruped, lift mast, UR3 arm) "
    "does rust repair and gas-leak repair on the pipe deck. Each flange has a FIXED gas detector on the pipe; the "
    "robot carries an ONBOARD sniffer. You get both readings (%LEL), the robot's distance to the flange, the robot "
    "state and the deterministic fusion finding. Decide ONE command for the robot:\n"
    "go_check: fixed detector alarms but the robot is too far to confirm; pause rust work and send it to the flange.\n"
    "fix_leak: both sources agree on gas at the flange; inspect bolts and re-torque.\n"
    "retreat: robot reading >= 20 %LEL or a crack; stop and back off, escalate to humans.\n"
    "resume_rust: gas cleared on both sources; go back to the rust job.\n"
    "log_sensor_fault: the fixed detector reads high but the robot right beside it reads clean; raise a maintenance "
    "ticket for the detector and do NOT dispatch a repair.\n"
    "hold: evidence unclear, keep the robot where it is.\n"
    "Prefer the conservative command when evidence is mixed. Reply JSON only."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "enum": list(COMMANDS)},
        "target": {"type": "string", "description": "Flange id for go_check/fix_leak/retreat, sensor id for "
                                                    "log_sensor_fault, empty otherwise."},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verdict": {"type": "string", "enum": list(FINDINGS)},
        "situation": {"type": "string", "description": "One short operator line, at most 90 characters."},
        "rationale": {"type": "string", "description": "One sentence citing both readings."},
    },
    "required": ["command", "target", "confidence", "verdict", "situation", "rationale"],
}


def prompt(fd: Finding, robot: dict, detectors: list[FixedDetector], t: float) -> str:
    snap = {"t_s": round(t, 1), "finding": fd.to_dict(), "robot": robot,
            "fixed_detectors": [{"id": d.id, "flange": d.flange, "lel": d.last, "led": led(d.last)} for d in detectors],
            "thresholds_lel": {"clear": CLEAR_LEL, "alarm": ALARM_LEL, "stop": STOP_LEL, "near_m": NEAR_M}}
    return "Snapshot (JSON):\n" + json.dumps(snap, separators=(",", ":"))


def request_body(text: str, model: str, thinking: str) -> dict:
    return {"model": model, "store": False, "system_instruction": SYSTEM, "input": [{"type": "text", "text": text}],
            "generation_config": {"thinking_level": thinking, "max_output_tokens": 4096},
            "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA}}


def parse_reply(text: str, fd: Finding, robot: str, ids: set) -> Command:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON in reply")
    d = json.loads(m.group(0))
    kind = str(d.get("command", ""))
    if kind not in COMMANDS:
        raise ValueError(f"command {kind!r} not allowed")
    target = str(d.get("target") or "")
    if kind in (GO_CHECK, FIX_LEAK, LOG_SENSOR_FAULT) and target not in ids:
        raise ValueError(f"unknown target {target!r}")
    conf = float(d.get("confidence", 0.0))
    if not 0.0 <= conf <= 1.0:
        raise ValueError("confidence out of range")
    why = f"{str(d.get('situation', ''))[:90]} | {str(d.get('rationale', ''))[:160]}"
    return Command(kind, robot, target, "argon", why, conf, str(d.get("verdict", fd.kind)))


class LiveCaller:
    live = True

    def __init__(self, model: Optional[str] = None, thinking: str = "low"):
        from policy.control.backend import ArgonBackend

        self.backend = ArgonBackend(model=model, thinking=thinking)
        self.name = self.backend.name

    @property
    def available(self) -> bool:
        return self.backend.available

    def __call__(self, text: str, key: str) -> dict:
        from policy.control.backend import API_REVISION, API_URL, parse_interaction

        b = self.backend
        headers = {"Content-Type": "application/json", "x-goog-api-key": b.api_key, "Api-Revision": API_REVISION}
        t0 = time.perf_counter()
        reply = b.transport(API_URL, request_body(text, b.model, b.thinking), headers)
        out, u = parse_interaction(reply)
        return {"text": out, "latency_ms": (time.perf_counter() - t0) * 1000.0, "tokens_in": u["in"],
                "tokens_out": u["out"], "tokens_thought": u["thought"], "tokens_cached": u["cached"]}


class ReplayCaller:
    live = False
    available = True

    def __init__(self, timeline: str):
        self.calls: dict[str, list[dict]] = {}
        self.name = "gemini-4-argon (replay)"
        for line in Path(timeline).read_text().splitlines():
            r = json.loads(line)
            if r.get("type") == "argon_call" and r.get("status") == "ok":
                self.calls.setdefault(r["key"], []).append(r)
        self.used: dict[str, int] = {}

    def __call__(self, text: str, key: str) -> dict:
        rows = self.calls.get(key)
        if not rows:
            raise LookupError(f"no recorded Argon call for {key}")
        i = self.used.get(key, 0)
        self.used[key] = i + 1
        r = rows[min(i, len(rows) - 1)]
        return {k: r[k] for k in ("text", "latency_ms", "tokens_in", "tokens_out", "tokens_thought", "tokens_cached")}


@dataclass
class Pending:
    key: str
    finding: Finding
    t: float
    future: object = None
    result: object = None
    due: Optional[float] = None


class ArgonLink:
    def __init__(self, mode: str, robot: str, timeline, caller=None, edge_timeout_s: float = 25.0,
                 today=date.today):
        self.mode, self.robot, self.timeline = mode, robot, timeline
        self.caller = caller
        self.edge_timeout_s, self.today = edge_timeout_s, today
        self.link_cut = mode == "edge"
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="argon")
        self.pending: Optional[Pending] = None
        self.last_key: dict[str, str] = {}
        self.commands: list[Command] = []
        self.calls: list[dict] = []
        self.status = "edge" if self.link_cut else "idle"
        self.last_finding: dict[str, Finding] = {}

    @property
    def name(self) -> str:
        return getattr(self.caller, "name", "edge rules")

    def cut(self, on: bool) -> None:
        self.link_cut = bool(on)
        self.status = "cut" if on else "idle"

    def observe(self, t: float, fd: Finding, robot: dict, detectors: list[FixedDetector], ids: set,
                crack: bool = False, wait: bool = False) -> list[Command]:
        self.last_finding[fd.sensor] = fd
        out = self._collect(t, ids, wait)
        key = f"{fd.sensor}:{fd.kind}"
        if self.last_key.get(fd.sensor) == fd.kind:
            return out
        if self.pending is not None and self.pending.finding.sensor == fd.sensor:
            return out
        if self.pending is not None:
            return out
        self.last_key[fd.sensor] = fd.kind
        self.timeline.log("finding", t, **fd.to_dict(), led=led(fd.fixed_lel))
        if fd.kind == OK and not robot.get("paused"):
            return out
        if self.link_cut or self.caller is None or not self.caller.available:
            return out + [self._issue(t, edge_rule(fd, self.robot, crack))]
        text = prompt(fd, robot, detectors, t)
        p = Pending(key, fd, t)
        p.future = self.pool.submit(self._call, text, key)
        self.pending = p
        self.status = "pending"
        self.timeline.log("argon_request", t, key=key, finding=fd.kind, sensor=fd.sensor)
        if wait:
            out += self._collect(t, ids, True)
        return out

    def _call(self, text, key):
        try:
            return self.caller(text, key)
        except Exception as e:
            return e

    def _collect(self, t: float, ids: set, wait: bool) -> list[Command]:
        p = self.pending
        if p is None:
            return []
        if p.result is None:
            if wait or p.future.done():
                p.result = p.future.result()
                if isinstance(p.result, Exception):
                    p.due = t
                else:
                    p.due = p.t + p.result["latency_ms"] / 1000.0
            elif t - p.t > self.edge_timeout_s:
                p.result, p.due = TimeoutError("Argon timeout"), t
        if p.result is None or t < p.due:
            return []
        self.pending = None
        return [self._finish(p, t, ids)]

    def _finish(self, p: Pending, t: float, ids: set) -> Command:
        from policy.control.limiter import cost_usd

        rec = {"key": p.key, "finding": p.finding.kind, "sensor": p.finding.sensor, "t_request": round(p.t, 2),
               "model": self.name, "live": bool(getattr(self.caller, "live", False))}
        if isinstance(p.result, Exception):
            self.status = "offline"
            rec.update(status="error", error=f"{type(p.result).__name__}: {p.result}")
            self.calls.append(rec)
            self.timeline.log("argon_call", t, **rec)
            cmd = edge_rule(p.finding, self.robot)
            cmd.reason = f"Argon unavailable -> {cmd.reason}"
            return self._issue(t, cmd)
        r = p.result
        cost = cost_usd(r["tokens_in"], r["tokens_out"], r["tokens_thought"], self.today(), r["tokens_cached"])
        rec.update(status="ok", latency_ms=round(r["latency_ms"], 1), cost_usd=round(cost, 6), text=r["text"],
                   tokens_in=r["tokens_in"], tokens_out=r["tokens_out"], tokens_thought=r["tokens_thought"],
                   tokens_cached=r["tokens_cached"])
        try:
            cmd = parse_reply(r["text"], p.finding, self.robot, ids)
        except (ValueError, json.JSONDecodeError) as e:
            rec.update(status="ok", invalid=str(e))
            cmd = edge_rule(p.finding, self.robot)
            cmd.reason = f"Argon reply invalid ({e}) -> {cmd.reason}"
        self.status = "ok"
        rec["command"] = cmd.to_dict()
        self.calls.append(rec)
        self.timeline.log("argon_call", t, **rec)
        return self._issue(t, cmd)

    def _issue(self, t: float, cmd: Command) -> Command:
        cmd.t = round(t, 2)
        self.commands.append(cmd)
        self.timeline.log("command", t, **{k: v for k, v in cmd.to_dict().items() if k != "t"})
        return cmd

    def wait_idle(self):
        if self.pending is not None and self.pending.future is not None:
            self.pending.future.result()

    def summary(self) -> dict:
        ok = [c for c in self.calls if c.get("status") == "ok"]
        lat = [c["latency_ms"] for c in ok]
        return {"model": self.name, "mode": self.mode, "calls": len(self.calls), "ok": len(ok),
                "latency_ms": {"min": min(lat, default=None), "max": max(lat, default=None),
                               "mean": round(sum(lat) / len(lat), 1) if lat else None},
                "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in ok), 5),
                "commands": [f"{c.t:.1f}s {c.source}:{c.kind}:{c.target}" for c in self.commands]}


@dataclass
class Timeline:
    path: Optional[Path]
    rows: list = field(default_factory=list)

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, type_: str, at: float, **kw) -> dict:
        row = {"type": type_, **kw, "t": round(float(at), 3)}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

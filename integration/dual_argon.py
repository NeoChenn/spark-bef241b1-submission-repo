from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CLEAR_LEL = 5.0
ALARM_LEL = 10.0
STOP_LEL = 20.0
NEAR_M = 2.0
NEAR_EXIT_M = 2.5
CLOSE_M = 1.0
CLOSE_CONFIRM_S = 1.0
ATTRIBUTE_M = 4.0
FAULT_CONFIRM_S = 3.0
SPARK_RADIUS_M = 6.0

SCRUBBER = "unit1"
FIXER = "unit2"
SUPERVISOR = "supv"

LEAK_CONFIRMED = "LEAK_CONFIRMED"
PIPE_SENSOR_FAULT = "PIPE_SENSOR_FAULT"
LEAK_UNSEEN_BY_FIXED = "LEAK_UNSEEN_BY_FIXED"
UNVERIFIED = "UNVERIFIED"
OK = "OK"
FINDINGS = (LEAK_CONFIRMED, PIPE_SENSOR_FAULT, LEAK_UNSEEN_BY_FIXED, UNVERIFIED, OK)

GO_CHECK, FIX_LEAK, RETREAT, RESUME_RUST, LOG_SENSOR_FAULT, HOLD, ESCALATE = (
    "go_check", "fix_leak", "retreat", "resume_rust", "log_sensor_fault", "hold", "escalate")
COMMANDS = (GO_CHECK, FIX_LEAK, RETREAT, RESUME_RUST, LOG_SENSOR_FAULT, HOLD, ESCALATE)
ROUTE = {GO_CHECK: FIXER, FIX_LEAK: FIXER, HOLD: SCRUBBER, RETREAT: SCRUBBER, RESUME_RUST: SCRUBBER,
         LOG_SENSOR_FAULT: SUPERVISOR, ESCALATE: SUPERVISOR}
FLANGE_TARGET = (GO_CHECK, FIX_LEAK, HOLD, RETREAT)
NEEDS_TARGET = (GO_CHECK, FIX_LEAK, LOG_SENSOR_FAULT)


def led(lel: Optional[float], fault: bool = False) -> str:
    if lel is None or not math.isfinite(lel):
        return "grey"
    if fault and lel < STOP_LEL:
        return "grey"
    return "red" if lel >= ALARM_LEL else "amber" if lel >= CLEAR_LEL else "green"


DETECTOR_STATE = {"green": "ok", "amber": "warn", "red": "alarm", "grey": "fault"}


def detector_state(lel: Optional[float], fault: bool = False) -> str:
    return DETECTOR_STATE[led(lel, fault)]


@dataclass
class FixedDetector:
    id: str
    flange: str
    pos: tuple
    stuck_at: Optional[float] = None
    last: float = 0.0
    fault: bool = False

    def read(self, gas_at: Callable[[np.ndarray], float]) -> float:
        v = self.stuck_at if self.stuck_at is not None else 100.0 * float(gas_at(np.asarray(self.pos, float)))
        self.last = round(float(v), 2)
        return self.last

    def dist(self, p) -> float:
        return float(np.linalg.norm(np.asarray(p, float)[:2] - np.asarray(self.pos, float)[:2]))


@dataclass
class Finding:
    kind: str
    sensor: str
    flange: str
    fixed_lel: float
    robot_lel: Optional[float]
    robot_dist: Optional[float]
    reason: str
    robot: str = ""
    hazard: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def fuse(fixed: FixedDetector, robot_lel: Optional[float], robot_dist: Optional[float],
         near_m: float = NEAR_M, robot: str = "", was_near: bool = False) -> Finding:
    f = fixed.last
    reach = NEAR_EXIT_M if was_near else near_m
    near = robot_dist is not None and robot_dist <= reach and robot_lel is not None
    who = robot or "robot"
    where = "--" if robot_dist is None else f"{robot_dist:.1f} m"
    rl = "--" if robot_lel is None else f"{robot_lel:.1f}"
    if f >= ALARM_LEL:
        if not near:
            kind, why = UNVERIFIED, f"{fixed.id} {f:.1f}%LEL, {who} {rl}%LEL at {where}: too far to confirm"
        elif robot_lel >= CLEAR_LEL:
            kind, why = LEAK_CONFIRMED, f"{fixed.id} {f:.1f}%LEL and {who} {rl}%LEL at {where} agree"
        elif f >= STOP_LEL:
            kind, why = LEAK_CONFIRMED, (f"{fixed.id} {f:.1f}%LEL >= stop {STOP_LEL:.0f}: treated as real gas although "
                                         f"{who} reads {rl}%LEL at {where}")
        else:
            kind, why = PIPE_SENSOR_FAULT, f"{fixed.id} {f:.1f}%LEL but {who} reads {rl}%LEL at {where}"
    elif near and robot_lel >= ALARM_LEL:
        kind, why = LEAK_UNSEEN_BY_FIXED, f"{who} {rl}%LEL at {where}, {fixed.id} only {f:.1f}%LEL"
    elif robot_lel is not None and robot_lel >= ALARM_LEL and robot_dist is not None and robot_dist <= ATTRIBUTE_M:
        kind, why = UNVERIFIED, f"{who} {rl}%LEL at {where} from {fixed.id} ({f:.1f}%LEL): source not located"
    else:
        kind, why = OK, f"{fixed.id} {f:.1f}%LEL, {who} {rl}%LEL at {where}"
    attributed = robot_dist is not None and robot_dist <= ATTRIBUTE_M
    hazard = f >= STOP_LEL or (attributed and robot_lel is not None and robot_lel >= STOP_LEL)
    return Finding(kind, fixed.id, fixed.flange, f, None if robot_lel is None else round(robot_lel, 2),
                   None if robot_dist is None else round(robot_dist, 2), why, robot, hazard)


def fuse_fleet(fixed: FixedDetector, sniffers: dict, near_m: float = NEAR_M, memory: Optional[dict] = None) -> Finding:
    best = None
    for rid, (lel, pos) in sniffers.items():
        if pos is None:
            continue
        d = fixed.dist(pos)
        if best is None or d < best[2]:
            best = (rid, lel, d)
    if best is None:
        return fuse(fixed, None, None, near_m)
    key = (fixed.id, best[0])
    was = bool(memory.get(key)) if memory is not None else False
    if memory is not None:
        memory[key] = best[2] <= (NEAR_EXIT_M if was else near_m)
    return fuse(fixed, best[1], best[2], near_m, robot=best[0], was_near=was)


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


def command(kind: str, target: str, source: str, reason: str, finding: str = "", confidence: float = 1.0,
            robot: Optional[str] = None) -> Command:
    return Command(kind, robot or ROUTE[kind], target, source, reason, confidence, finding)


def edge_rule(fd: Finding, robot: Optional[str] = None, crack: bool = False) -> list[Command]:
    why = f"edge rule {fd.kind}: {fd.reason}"
    if crack or fd.hazard:
        why = f"edge rule HAZARD {fd.kind}: {fd.reason}"
        out = [command(HOLD, fd.flange, "edge", why, fd.kind)]
        near = fd.robot_dist is not None and fd.robot_dist <= NEAR_M
        in_gas = fd.robot_lel is not None and fd.robot_lel >= STOP_LEL
        if fd.robot and (in_gas or near):
            out.append(command(RETREAT, fd.flange, "edge", why, fd.kind, robot=fd.robot))
        elif fd.kind == UNVERIFIED:
            out.append(command(ESCALATE, fd.flange, "edge", why + " -> operator confirms before any dispatch", fd.kind))
        return out
    if fd.kind == UNVERIFIED:
        return [command(HOLD, fd.flange, "edge", why, fd.kind), command(GO_CHECK, fd.flange, "edge", why, fd.kind)]
    if fd.kind == LEAK_CONFIRMED:
        return [command(FIX_LEAK, fd.flange, "edge", why, fd.kind)]
    if fd.kind == LEAK_UNSEEN_BY_FIXED:
        return [command(FIX_LEAK, fd.flange, "edge", why, fd.kind),
                command(LOG_SENSOR_FAULT, fd.sensor, "edge", why, fd.kind)]
    if fd.kind == PIPE_SENSOR_FAULT:
        return [command(LOG_SENSOR_FAULT, fd.sensor, "edge", why, fd.kind)]
    return [command(RESUME_RUST, "", "edge", why, fd.kind)]


def ids_for(detectors: list[FixedDetector]) -> set:
    return {d.id for d in detectors} | {d.flange for d in detectors}


def spark_hot(pos, detectors: list[FixedDetector], faulted=(), radius_m: float = SPARK_RADIUS_M) -> list[str]:
    return [d.id for d in detectors if d.dist(pos) <= radius_m and led(d.last, d.id in faulted) in ("amber", "red")]


SYSTEM = (
    "You are Argon, the command centre of an offshore oil platform with two robots. unit1 'Scrubber' (Go2, lift "
    "mast, UR3, wire brush) does rust repair; wire brushing is spark-producing hot work. unit2 'Fixer' (Go2, UR3 "
    "with gripper/torque tool, gas sensors) does gas-leak work. Each flange has a FIXED gas detector; each robot "
    "carries a sniffer. You get the fixed readings, the nearest robot's sniffer reading and distance, both robots' "
    "states and the deterministic fusion finding. Return the commands to issue now:\n"
    "hold (unit1): stop hot work and step back, rust job paused not cancelled.\n"
    "go_check (unit2, target flange): fixed detector alarms but no robot is close enough to confirm.\n"
    "fix_leak (unit2, target flange): fixed and robot readings agree; inspect and re-torque.\n"
    "retreat (robot at the gas): robot reading >= 20 %LEL or a crack.\n"
    "resume_rust (unit1): every detector near unit1 green, leak fixed.\n"
    "log_sensor_fault (target sensor id): fixed detector high but a robot at it reads clean; raise a maintenance "
    "ticket and do NOT dispatch a repair. Anything >= 20 %LEL is real gas regardless of a fault verdict.\n"
    "Prefer the conservative command when evidence is mixed. Reply JSON only."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(FINDINGS)},
        "situation": {"type": "string", "description": "One short operator line, at most 90 characters."},
        "commands": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": list(COMMANDS)},
                "robot": {"type": "string", "enum": [SCRUBBER, FIXER, ""]},
                "target": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string", "description": "One sentence citing both readings."},
            },
            "required": ["command", "robot", "target", "confidence", "rationale"],
        }},
    },
    "required": ["verdict", "situation", "commands"],
}


def build_prompt(fd: Optional[Finding], robots: dict, detectors: list[FixedDetector], t: float,
                 event: str = "", findings=None, faulted=()) -> str:
    snap = {"t_s": round(t, 1), "event": event, "finding": fd.to_dict() if fd else None, "robots": robots,
            "fixed_detectors": [{"id": d.id, "flange": d.flange, "pos": [round(float(v), 2) for v in d.pos],
                                 "lel": d.last, "led": led(d.last, d.fault)} for d in detectors],
            "thresholds_lel": {"clear": CLEAR_LEL, "alarm": ALARM_LEL, "stop": STOP_LEL, "near_m": NEAR_M,
                               "spark_radius_m": SPARK_RADIUS_M}}
    return "Snapshot (JSON):\n" + json.dumps(snap, separators=(",", ":"), default=str)


def parse_reply(text: str, fd: Optional[Finding], ids: set) -> list[Command]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON in reply")
    d = json.loads(m.group(0))
    items = d.get("commands")
    if items is None and "command" in d:
        items = [d]
    if not isinstance(items, list):
        raise ValueError("commands missing")
    situation = str(d.get("situation", ""))[:90]
    verdict = str(d.get("verdict", fd.kind if fd else ""))
    out = []
    for c in items:
        kind = str(c.get("command", ""))
        if kind not in COMMANDS:
            raise ValueError(f"command {kind!r} not allowed")
        target = str(c.get("target") or "")
        if kind in NEEDS_TARGET and target not in ids:
            raise ValueError(f"unknown target {target!r}")
        conf = float(c.get("confidence", 0.0))
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence out of range")
        robot = str(c.get("robot") or "") if kind == RETREAT else ROUTE[kind]
        why = f"{situation} | {str(c.get('rationale', ''))[:160]}"
        out.append(Command(kind, robot or ROUTE[kind], target, "argon", why, conf, verdict))
    return out


PROMPT_SOURCE = "integration.dual_argon"
try:
    from integration.argon_fleet_prompt import SCHEMA, SYSTEM, build_prompt, parse_reply
    PROMPT_SOURCE = "integration.argon_fleet_prompt"
except ImportError as e:
    if getattr(e, "name", "") != "integration.argon_fleet_prompt":
        raise


def request_body(text: str, model: str, thinking: str) -> dict:
    return {"model": model, "store": False, "system_instruction": SYSTEM, "input": [{"type": "text", "text": text}],
            "generation_config": {"thinking_level": thinking, "max_output_tokens": 4096},
            "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA}}


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


class EdgeOnly(Exception):
    pass


@dataclass
class Request:
    key: str
    finding: Optional[Finding]
    t: float
    text: str
    fallback: list
    advisory: bool = False
    future: object = None
    result: object = None
    due: Optional[float] = None


class ArgonLink:
    def __init__(self, mode: str, timeline, caller=None, edge_timeout_s: float = 25.0, today=date.today,
                 spark_radius_m: float = SPARK_RADIUS_M, fault_confirm_s: float = FAULT_CONFIRM_S):
        self.fault_confirm_s = fault_confirm_s
        self.fault_since: dict[str, float] = {}
        self.mode, self.timeline, self.caller = mode, timeline, caller
        self.edge_timeout_s, self.today, self.spark_radius_m = edge_timeout_s, today, spark_radius_m
        self.link_cut = mode == "edge"
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="argon")
        self.queue: list[Request] = []
        self.pending: Optional[Request] = None
        self.verdict: dict[str, str] = {}
        self.state: dict[str, tuple] = {}
        self.findings: dict[str, Finding] = {}
        self.faulted: set[str] = set()
        self.tickets: list[dict] = []
        self.held = False
        self.holds: set[str] = set()
        self.clear_asked = False
        self.spark: list[str] = []
        self.commands: list[Command] = []
        self.calls: list[dict] = []
        self.status = "edge" if self.link_cut else "idle"

    @property
    def name(self) -> str:
        return getattr(self.caller, "name", "edge rules") if self.caller is not None else "edge rules"

    @property
    def cloud(self) -> bool:
        return not self.link_cut and self.caller is not None and self.caller.available

    def cut(self, on: bool, t: float = 0.0) -> list[str]:
        self.link_cut = bool(on)
        self.status = "cut" if on else "idle"
        self.timeline.log("link", t, cut=self.link_cut)
        dropped = []
        if on:
            for r in ([self.pending] if self.pending is not None else []) + self.queue:
                if not isinstance(r.result, EdgeOnly):
                    r.result, r.due = EdgeOnly(), t
                    for c in r.fallback:
                        c.reason = f"link cut -> {c.reason}"
                    dropped.append(r.key)
        return dropped

    def tick(self, t: float, findings: list[Finding], robots: dict, detectors: list[FixedDetector], ids: set,
             scrubber_pos=None, crack=False, wait: bool = False) -> list[Command]:
        out = []
        dmap = {d.id: d for d in detectors}
        for d in detectors:
            d.fault = d.id in self.faulted
        for fd in findings:
            self.findings[fd.sensor] = fd
            cracked = fd.flange in crack if isinstance(crack, (set, frozenset, list, tuple)) else bool(crack)
            if fd.sensor in self.faulted and fd.kind == UNVERIFIED and not (fd.hazard or cracked):
                fd = replace(fd, kind=PIPE_SENSOR_FAULT, reason=f"known fault (ticketed): {fd.reason}")
            if fd.kind == PIPE_SENSOR_FAULT and fd.sensor not in self.faulted:
                t0 = self.fault_since.setdefault(fd.sensor, t)
                close = fd.robot_dist is not None and fd.robot_dist <= CLOSE_M and t - t0 >= CLOSE_CONFIRM_S
                if t - t0 < self.fault_confirm_s and not close:
                    continue
            else:
                self.fault_since.pop(fd.sensor, None)
            if fd.kind in (LEAK_CONFIRMED, LEAK_UNSEEN_BY_FIXED) and fd.sensor in self.faulted:
                self.faulted.discard(fd.sensor)
                self.timeline.log("fault_cleared", t, sensor=fd.sensor, by=fd.kind)
            state = (fd.kind, fd.hazard or cracked)
            if self.state.get(fd.sensor) == state:
                continue
            self.state[fd.sensor] = state
            self.verdict[fd.sensor] = fd.kind
            self.timeline.log("finding", t, **fd.to_dict(), crack=cracked,
                              led=led(fd.fixed_lel, fd.kind == PIPE_SENSOR_FAULT))
            if fd.kind == PIPE_SENSOR_FAULT:
                self.faulted.add(fd.sensor)
            key = f"{fd.sensor}:{fd.kind}" + (":HAZARD" if state[1] else "")
            if state[1]:
                out += self._issue(t, edge_rule(fd, crack=cracked))
                self._ask(t, key, fd, self._prompt(fd, robots, detectors, t), [], advisory=True)
                continue
            if fd.kind == OK:
                continue
            fallback = edge_rule(fd)
            far = scrubber_pos is not None and fd.sensor in dmap and \
                dmap[fd.sensor].dist(scrubber_pos) > self.spark_radius_m
            if far:
                fallback = [c for c in fallback if not (c.kind == HOLD and c.robot == SCRUBBER)]
            self._ask(t, key, fd, self._prompt(fd, robots, detectors, t), fallback)
        for d in detectors:
            d.fault = d.id in self.faulted
        out += self._collect(t, ids, wait)
        hot = [] if scrubber_pos is None else spark_hot(scrubber_pos, detectors, self.faulted, self.spark_radius_m)
        self.spark = hot
        if hot and not self.held:
            out += self._issue(t, [command(HOLD, hot[0], "edge", f"spark rule: {', '.join(hot)} amber/red "
                                           f"within {self.spark_radius_m:.0f} m of hot work")])
        quiet = all(v in (OK, PIPE_SENSOR_FAULT) for v in self.verdict.values())
        if self.held and not hot and quiet and not self.clear_asked:
            self.clear_asked = True
            self._ask(t, "zone:clear", None, self._prompt(None, robots, detectors, t, event="zone clear: every "
                                                          "detector near unit1 green, no open leak"),
                      [command(RESUME_RUST, "", "edge", "zone clear: detectors near unit1 green, no open leak")])
        out += self._pump(t, ids, wait)
        return out

    def _prompt(self, fd, robots, detectors, t, event: str = "") -> str:
        return build_prompt(fd, robots, detectors, t, event=event, findings=list(self.findings.values()),
                            faulted=set(self.faulted))

    def _ask(self, t, key, fd, text, fallback, advisory: bool = False) -> None:
        if not self.cloud:
            if not advisory:
                self.queue.append(Request(key, fd, t, text, fallback, result=EdgeOnly(), due=t))
            return
        self.queue.append(Request(key, fd, t, text, fallback, advisory))

    def _pump(self, t, ids, wait) -> list[Command]:
        out = []
        while True:
            if self.pending is None and self.queue:
                r = self.queue.pop(0)
                if r.finding is not None and self.verdict.get(r.finding.sensor) != r.finding.kind:
                    self.timeline.log("stale", t, key=r.key, now=self.verdict.get(r.finding.sensor))
                    continue
                if r.result is None:
                    r.t = t
                    r.future = self.pool.submit(self._call, r.text, r.key)
                    self.status = "pending"
                    self.timeline.log("argon_request", t, key=r.key, finding=r.finding.kind if r.finding else None,
                                      sensor=r.finding.sensor if r.finding else None)
                self.pending = r
            got = self._collect(t, ids, wait)
            out += got
            if not got or self.pending is not None:
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
                p.due = t if isinstance(p.result, Exception) else p.t + p.result["latency_ms"] / 1000.0
            elif t - p.t > self.edge_timeout_s:
                p.result, p.due = TimeoutError("Argon timeout"), t
        if p.result is None or t < p.due:
            return []
        self.pending = None
        return self._finish(p, t, ids)

    def _finish(self, p: Request, t: float, ids: set) -> list[Command]:
        from policy.control.limiter import cost_usd

        if isinstance(p.result, EdgeOnly):
            return self._issue(t, p.fallback)
        if p.finding is not None and self.verdict.get(p.finding.sensor) != p.finding.kind and not p.advisory:
            self.timeline.log("stale", t, key=p.key, now=self.verdict.get(p.finding.sensor))
            p.fallback = []
        rec = {"key": p.key, "finding": p.finding.kind if p.finding else None,
               "sensor": p.finding.sensor if p.finding else None, "t_request": round(p.t, 2), "model": self.name,
               "live": bool(getattr(self.caller, "live", False))}
        if isinstance(p.result, Exception):
            self.status = "offline"
            rec.update(status="error", error=f"{type(p.result).__name__}: {p.result}")
            self.calls.append(rec)
            self.timeline.log("argon_call", t, **rec)
            for c in p.fallback:
                c.reason = f"Argon unavailable -> {c.reason}"
            return self._issue(t, p.fallback)
        r = p.result
        cost = cost_usd(r["tokens_in"], r["tokens_out"], r["tokens_thought"], self.today(), r["tokens_cached"])
        rec.update(status="ok", latency_ms=round(r["latency_ms"], 1), cost_usd=round(cost, 6), text=r["text"],
                   tokens_in=r["tokens_in"], tokens_out=r["tokens_out"], tokens_thought=r["tokens_thought"],
                   tokens_cached=r["tokens_cached"])
        try:
            cmds = parse_reply(r["text"], p.finding, ids)
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
            rec["invalid"] = str(e)
            cmds = p.fallback
            for c in cmds:
                c.reason = f"Argon reply invalid -> {c.reason}"
        self.status = "ok"
        rec["commands"] = [c.to_dict() for c in cmds]
        rec["advisory"] = p.advisory
        self.calls.append(rec)
        self.timeline.log("argon_call", t, **rec)
        if p.advisory or (p.finding is not None and self.verdict.get(p.finding.sensor) != p.finding.kind):
            return []
        return self._issue(t, cmds)

    def _issue(self, t: float, cmds: list[Command]) -> list[Command]:
        out = []
        for c in cmds:
            c.t = round(t, 2)
            if c.kind in FLANGE_TARGET and re.fullmatch(r"G-F\d+", c.target or ""):
                c.target = c.target[2:]
            if c.kind == HOLD and c.robot != SCRUBBER:
                if c.robot in self.holds:
                    continue
                self.holds.add(c.robot)
            elif c.kind == HOLD:
                if self.held:
                    continue
                self.held, self.clear_asked = True, False
            elif c.kind == RESUME_RUST and c.robot != SCRUBBER:
                continue
            elif c.kind == RESUME_RUST:
                if not self.held:
                    continue
                if self.spark:
                    self.timeline.log("vetoed", t, **{k: v for k, v in c.to_dict().items() if k != "t"},
                                      why=f"spark rule: {', '.join(self.spark)} not green")
                    self.clear_asked = False
                    continue
                self.held = False
            elif c.kind == LOG_SENSOR_FAULT:
                if any(tk["sensor"] == c.target for tk in self.tickets):
                    continue
                tk = {"id": f"MT-{len(self.tickets) + 1:03d}", "sensor": c.target, "t": c.t, "source": c.source,
                      "reason": c.reason}
                self.tickets.append(tk)
                self.timeline.log("ticket", t, **{k: v for k, v in tk.items() if k != "t"})
            self.commands.append(c)
            self.timeline.log("command", t, **{k: v for k, v in c.to_dict().items() if k != "t"})
            out.append(c)
        return out

    def wait_idle(self):
        if self.pending is not None and self.pending.future is not None:
            self.pending.future.result()

    def header(self) -> dict:
        last = next((c for c in reversed(self.calls) if c.get("status") == "ok"), None)
        return {"model": self.name, "prompt": PROMPT_SOURCE, "link": "CUT" if self.link_cut else "UP", "status": self.status,
                "pending": self.pending is not None, "held": self.held, "spark": list(self.spark),
                "verdicts": dict(self.verdict), "tickets": [tk["id"] + " " + tk["sensor"] for tk in self.tickets],
                "last_command": self.commands[-1].to_dict() if self.commands else None,
                "latency_ms": last.get("latency_ms") if last else None,
                "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in self.calls), 5)}

    def summary(self) -> dict:
        ok = [c for c in self.calls if c.get("status") == "ok"]
        lat = [c["latency_ms"] for c in ok]
        return {"model": self.name, "prompt": PROMPT_SOURCE, "mode": self.mode, "calls": len(self.calls), "ok": len(ok),
                "latency_ms": {"min": min(lat, default=None), "max": max(lat, default=None),
                               "mean": round(sum(lat) / len(lat), 1) if lat else None},
                "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in ok), 5), "tickets": self.tickets,
                "commands": [f"{c.t:.1f}s {c.source}:{c.kind}:{c.robot}:{c.target}" for c in self.commands]}


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

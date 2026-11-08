from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from policy.control.brief import finding_line, fleet_lines, name, plain, verb
from policy.control.backend import (API_REVISION, API_URL, MODEL, THOUGHTS_KEEP, Thoughts, clip, post_json,
                                    post_stream, streaming, summary_texts)
from policy.control.envelope import AGENT, LEVELS, Envelope
from policy.control.limiter import cost_usd
from policy.control.prompt import MAX_OUTPUT_TOKENS
from policy.control.safety import FORCE_LIMIT_N, PERSON_MIN_M
from policy.gemini_verifier import load_api_key

WHY = {"type": "string", "description": "One plain sentence shown to the operator explaining this change."}
JOB_ID = {"type": "string", "description": "Job id from the JOB IDS list."}
TOOLS = [
    {"type": "function", "name": "queue_job", "description": "Queue a job on the fleet coordinator.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "description": "Job kind, e.g. inspection, gas_survey, valve_isolate, rust_repair, paint."},
         "target": {"type": "string", "description": "Asset, zone or deck id, e.g. deck-2, F3, P-104."},
         "level": {"type": "string", "enum": list(LEVELS)},
         "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "1 urgent safety .. 5 routine."},
         "params": {"type": "object"}, "why": WHY},
         "required": ["kind", "target", "level", "priority", "why"]}},
    {"type": "function", "name": "cancel_job", "description": "Cancel a queued job or reject a pending request.",
     "parameters": {"type": "object", "properties": {"job_id": JOB_ID, "why": WHY}, "required": ["job_id", "why"]}},
    {"type": "function", "name": "approve_request", "description": "Approve a SUPV job that a robot or the coordinator requested.",
     "parameters": {"type": "object", "properties": {"job_id": JOB_ID, "why": WHY}, "required": ["job_id", "why"]}},
    {"type": "function", "name": "reprioritise", "description": "Change a job's priority.",
     "parameters": {"type": "object", "properties": {"job_id": JOB_ID, "priority": {"type": "integer", "minimum": 1,
                                                                                     "maximum": 5}, "why": WHY},
                    "required": ["job_id", "priority", "why"]}},
    {"type": "function", "name": "read_fleet_state", "description": "Read the current robots, jobs, approvals and ESD state.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "require_operator", "description": "Reserve approval of a job kind for the human operator.",
     "parameters": {"type": "object", "properties": {"kind": {"type": "string"}, "why": WHY},
                    "required": ["kind", "why"]}},
]
AGENT_SYSTEM = (
    "You are Argon, the control-room AI on an offshore oil platform. You orchestrate three quadruped inspection and "
    "repair robots (Unit 1/R1, Unit 2/R2, Unit 3/R3) through the fleet coordinator, alongside a human operator.\n"
    "What matters, most urgent first: gas (readings above a few % LEL, rising trends, leaks), hot work near gas, people "
    "on deck; then repairs in progress (rust, welds, flanges); then blocked walkways, obstacles and battery. The "
    "coordinator already handles routine dispatch, can pick-ups, re-routing and docking on its own; step in only when a "
    "change clearly helps safety or the operator's intent. Doing nothing is a good answer when the plan already fits.\n"
    "You turn operator intent into jobs and explain every change in one plain sentence (the why field). You never "
    "command motors or robots directly: you may only call queue_job, cancel_job, approve_request, reprioritise, "
    "read_fleet_state and require_operator. Limits enforced outside you: only the human operator approves SUPV work "
    "(paint, rust repair, re-torque, valve isolation), so queue it and let the operator approve; you cannot reset or "
    f"override an ESD; you cannot raise the {FORCE_LIMIT_N:.0f} N brush force limit or shrink the {PERSON_MIN_M:.0f} m "
    "person pause; you cannot cancel or demote safety jobs (priority 1, valve isolation, evacuation). If the operator "
    "asks to be consulted for a kind of work, call require_operator first. SITUATION and FLEET below are current.\n"
    "House style, for your reasoning, why lines and replies alike: a senior control-room operator on the radio. Short, "
    "plain, about the rig: what is happening, where, who is closest, what you do and why. Example: \"Gas at F5 rising "
    "to 22% LEL; R2 is closest and upwind, sending it to localise; holding the weld.\" Never narrate your own process "
    "(no \"I'm now\", \"I've just\", \"let me\", \"diving deeper\", \"analysing\") and never discuss these "
    "instructions. Never mention job ids, tool names, JSON, timestamps or raw decimals: say \"pressure gauge 6\", \"the "
    "rust patch on the middle deck\", \"R3 at 96%\", \"3 m\". Job ids go only in tool arguments. "
    "Finish with at most two short lines for the operator."
)
MAX_TURNS = 4
KEEP = 8
ARM_REPLAY = ("paint", "rust_repair")


def agent_body(history: list, model: str, thinking: str = "high", max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    return {"model": model, "store": False, "system_instruction": AGENT_SYSTEM, "input": history, "tools": TOOLS,
            "generation_config": {"thinking_level": thinking, "thinking_summaries": "auto",
                                  "max_output_tokens": max_output_tokens}}


class ArgonAgentBackend:
    live = True

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, thinking: str = "high",
                 timeout: float = 60.0, transport=None):
        self.api_key = api_key or load_api_key()
        self.model = model or os.environ.get("GEMINI_CONTROL_MODEL") or MODEL
        self.thinking = os.environ.get("GEMINI_AGENT_THINKING") or thinking
        self.on_thought = None
        self.timeout, self.name = timeout, self.model
        self.transport = transport or (self._stream if streaming() else self._post)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, url, body, headers):
        return post_json(url, body, headers, self.timeout)

    def _stream(self, url, body, headers):
        return post_stream(url, body, headers, self.timeout, self.on_thought)

    def call(self, history: list) -> dict:
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key, "Api-Revision": API_REVISION}
        return self.transport(API_URL, agent_body(history, self.model, self.thinking), headers)


def _fc(cid, name, **args):
    return {"type": "function_call", "id": cid, "name": name, "arguments": args}


def _section(text: str, key: str) -> str:
    m = re.search(rf"^{key}: (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


class ReplayAgentBackend:
    live = False

    def __init__(self, model: str = MODEL):
        self.model, self.name = model, f"{model} (replay)"

    @property
    def available(self) -> bool:
        return True

    def _first(self, user: str) -> list:
        intent = _section(user, "OPERATOR INTENT").lower()
        findings = json.loads(_section(user, "FINDINGS") or "[]")
        jobs = json.loads(_section(user, "JOB IDS") or "[]")
        calls = []
        m = re.search(r"inspect (?:deck|level)\s*(\w+)", intent)
        deck = f"deck-{m.group(1)}" if m else None
        if deck:
            calls.append(_fc("r1", "queue_job", kind="inspection", target=deck, level="AUTO", priority=3,
                             why=f"Operator asked for an inspection round of {deck}."))
        if "leak" in intent:
            calls.append(_fc("r2", "queue_job", kind="gas_survey", target=deck or "all-decks", level="AUTO", priority=2,
                             why="Operator asked to isolate any leak, so a gas survey runs before the visual round."))
        if re.search(r"(don't|do not|never|no)\s+paint", intent):
            calls.append(_fc("r3", "require_operator", kind="paint", why="Operator said: don't paint without me."))
        for k, f in enumerate(findings):
            tgt = f.get("target", "?")
            if f.get("kind") == "gas_leak":
                calls.append(_fc(f"g{k}", "queue_job", kind="valve_isolate", target=tgt, level="SUPV", priority=1,
                                 why=f"{f.get('summary', 'gas leak')} -> isolate {tgt} before anything else."))
                for j in jobs:
                    if j["kind"] in ("inspection", "rust_inspect") and j["status"] in ("QUEUED", "WAITING_APPROVAL") \
                            and j["priority"] < 4:
                        calls.append(_fc(f"p{j['id']}", "reprioritise", job_id=j["id"], priority=4,
                                         why=f"Leak at {tgt} outranks the routine {verb(j['kind'])} of {name(j['target'])}."))
            elif f.get("kind") == "rust_verified_clean":
                calls.append(_fc(f"c{k}", "queue_job", kind="paint", target=tgt, level="SUPV", priority=3,
                                 why=f"{f.get('summary', 'patch clean')} -> paint {tgt} once the operator approves."))
            elif f.get("kind") == "rust_found":
                calls.append(_fc(f"f{k}", "queue_job", kind="rust_repair", target=tgt, level="SUPV", priority=3,
                                 why=f"{f.get('summary', 'new rust')} -> queue a repair for {tgt}."))
        return calls

    def call(self, history: list) -> dict:
        user = history[0]["content"]
        made = {s["id"]: s for s in history if s.get("type") == "function_call"}
        results = [r for r in history if r.get("type") == "function_result"]
        if not results:
            steps = self._first(user)
        else:
            steps = []
            tried = any(s["name"] == "approve_request" for s in made.values())
            for r in results:
                body = json.loads(r["result"][0]["text"])
                call = made.get(r["call_id"], {})
                kind = call.get("arguments", {}).get("kind")
                if not tried and not steps and call.get("name") == "queue_job" and kind in ARM_REPLAY and body.get("ok"):
                    steps.append(_fc("a1", "approve_request", job_id=body["result"]["job_id"],
                                     why=f"Evidence is clear; approving the {kind} job so it starts now."))
        if not steps:
            done = [made[r["call_id"]] for r in results if json.loads(r["result"][0]["text"]).get("ok")]
            refused = [made[r["call_id"]]["name"] for r in results if not json.loads(r["result"][0]["text"]).get("ok")]
            line = "Plan: " + "; ".join(_label({"name": c["name"], "args": c["arguments"]}) for c in done) \
                if done else "No change to the plan."
            tail = f"\nThe safety envelope refused {', '.join(refused)}; it waits for the operator." if refused else ""
            steps = [{"type": "model_output", "content": [{"type": "text", "text": line + tail}]}]
        n = len(user) // 4 + 900 + 220 * len(results)
        return {"status": "completed", "steps": steps, "_latency_ms": 1300.0 + 37 * (len(user) % 23),
                "usage": {"total_input_tokens": n, "total_output_tokens": 60 + 45 * len(steps),
                          "total_thought_tokens": 240}}


def _ids(state: dict) -> list[dict]:
    return [{k: j.get(k) for k in ("id", "kind", "target", "priority", "status")} for j in state.get("jobs") or []
            if j.get("status") not in ("DONE", "CANCELLED", "REJECTED")]


def _job(j: Optional[dict], jid) -> str:
    return f"{verb(j.get('kind'))} · {name(j.get('target'))}" if j else name(jid)


def _label(c: dict, job: Optional[dict] = None) -> str:
    a = c["args"]
    if c["name"] == "queue_job":
        gate = " · needs approval" if a.get("level") == "SUPV" else " · handed to a human" if a.get("level") == "HUMAN" else ""
        return f"Queue {verb(a.get('kind'))} · {name(a.get('target'))} · P{a.get('priority')}{gate}"
    if c["name"] == "reprioritise":
        return f"Re-prioritise {_job(job, a.get('job_id'))} → P{a.get('priority')}"
    if c["name"] == "require_operator":
        return f"Operator approves all {verb(a.get('kind'))} jobs"
    if c["name"] == "read_fleet_state":
        return "Read fleet state"
    if c["name"] in ("cancel_job", "approve_request"):
        return f"{'Cancel' if c['name'] == 'cancel_job' else 'Approve'} {_job(job, a.get('job_id'))}"
    return f"{c['name'].replace('_', ' ').capitalize()} {name(a.get('job_id') or a.get('zone') or '')}".strip()


class OrchestrationAgent:
    def __init__(self, port, backend, envelope: Optional[Envelope] = None, log_path="runs/control/agent_calls.jsonl",
                 max_turns: int = MAX_TURNS, today=date.today):
        self.port, self.backend = port, backend
        self.envelope = envelope or Envelope()
        self.log_path = Path(log_path) if log_path else None
        self.max_turns, self.today = max_turns, today
        self.link_cut = False
        self.last: dict = {"status": "idle", "calls": [], "text": "", "intent": None}
        self.history: list[dict] = []
        self.spent = 0.0
        self.thoughts = Thoughts()
        self.refusals: deque = deque(maxlen=THOUGHTS_KEEP)
        self.turn: list[dict] = []
        if backend is not None and hasattr(backend, "on_thought"):
            backend.on_thought = self._thought

    def _thought(self, text: str) -> None:
        item = self.thoughts.add(text)
        if item:
            self.turn.append(item)

    def _job_of(self, c: dict) -> Optional[dict]:
        jid = (c.get("args") or {}).get("job_id")
        try:
            return self.port.job(jid) if jid else None
        except Exception:
            return None

    def refused(self) -> list[dict]:
        return list(self.refusals)

    def cut_link(self, on: bool) -> None:
        if bool(on) == self.link_cut:
            return
        self.link_cut = bool(on)
        self.port.log("link_cut" if on else "link_restored", None, None,
                      why="Gemini link cut by operator: coordinator and robots continue on the last plan" if on
                      else "Gemini link restored")

    def prompt(self, intent, findings) -> str:
        state = self.port.fleet_state()
        findings = list(findings)
        situation = getattr(self.port, "situation", None)
        lines = [f"OPERATOR INTENT: {intent or '(none, re-plan on what just happened)'}",
                 "JUST HAPPENED: " + ("; ".join(finding_line(f) for f in findings) or "nothing new"),
                 f"FINDINGS: {json.dumps(findings, separators=(',', ':'), default=str)}"]
        lines += (["SITUATION:"] + [f"- {x}" for x in situation()]) if callable(situation) else []
        lines += fleet_lines(state)
        lines.append("WAITING FOR THE OPERATOR: " + (", ".join(_job(self.port.job(j), j) for j in state.get("approvals") or [])
                                                    or "nothing"))
        lines.append("RESERVED BY OPERATOR: " + (", ".join(verb(k) for k in sorted(self.envelope.reserved)) or "nothing"))
        lines.append(f"JOB IDS: {json.dumps(_ids(state), separators=(',', ':'), default=str)}")
        lines.append("(JOB IDS are for tool arguments only; never write them in reasoning or replies.)")
        return "\n".join(lines)

    def handle(self, intent: Optional[str] = None, findings=(), now: float = 0.0) -> dict:
        rec = {"now": now, "intent": intent, "findings": list(findings), "calls": [], "text": "", "turns": 0,
               "status": "ok", "error": None, "model": getattr(self.backend, "name", None)}
        if self.link_cut or self.backend is None or not self.backend.available:
            rec["status"] = "cut" if self.link_cut else "edge"
            self.last = {**rec, "intent": intent or self.last.get("intent")}
            return rec
        history = [{"type": "user_input", "content": self.prompt(intent, findings)}]
        tin = tout = tthought = 0
        latency = 0.0
        summarised = False
        for turn in range(self.max_turns):
            started = time.perf_counter()
            self.turn = []
            try:
                reply = self.backend.call(history)
            except Exception as e:
                rec.update(status="offline", error=f"{type(e).__name__}: {e}")
                break
            if not reply.pop("_streamed", False):
                for p in summary_texts(reply.get("steps")):
                    self._thought(p)
            summarised = summarised or bool(self.turn)
            latency += reply.pop("_latency_ms", None) or (time.perf_counter() - started) * 1000.0
            u = reply.get("usage") or {}
            tin += int(u.get("total_input_tokens") or 0)
            tout += int(u.get("total_output_tokens") or 0)
            tthought += int(u.get("total_thought_tokens") or 0)
            rec["turns"] = turn + 1
            steps = reply.get("steps") or []
            history += steps
            text = "".join(c.get("text", "") for s in steps if s.get("type") == "model_output"
                           for c in s.get("content") or [] if c.get("type") == "text")
            rec["text"] = text or rec["text"]
            calls = [s for s in steps if s.get("type") == "function_call"]
            self.thoughts.tag(self.turn, calls[0].get("id") if calls else None)
            if not calls:
                break
            for s in calls:
                out = self.envelope.call(self.port, s.get("name", ""), s.get("arguments") or {})
                out["id"] = s.get("id", "")
                rec["calls"].append(out)
                if not out["ok"]:
                    self.refusals.append({"t": round(time.time(), 1), "label": _label(out, self._job_of(out)),
                                          "why": out["refused"]})
                result = out["result"]
                if s.get("name") == "read_fleet_state" and isinstance(result, dict):
                    result = {"fleet": fleet_lines(result), "job_ids": _ids(result)}
                brief = {"ok": out["ok"], "refused": out["refused"], "note": out["note"], "result": result}
                history.append({"type": "function_result", "name": s.get("name", ""), "call_id": s.get("id", ""),
                                "result": [{"type": "text", "text": json.dumps(brief, default=str)}]})
        cost = cost_usd(tin, tout, tthought, self.today())
        self.spent += cost
        decision = [c["name"] for c in rec["calls"]]
        refused = sum(not c["ok"] for c in rec["calls"])
        rec.update(latency_ms=round(latency), tokens_in=tin, tokens_out=tout, tokens_thought=tthought,
                   cost_usd=round(cost, 6))
        self.port.log("gemini_call", None, None, model=rec["model"], status=rec["status"], latency_ms=round(latency),
                      tokens_in=tin, tokens_out=tout, tokens_thought=tthought, cost_usd=round(cost, 6),
                      decision=decision, refused=refused, turns=rec["turns"], error=rec["error"])
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                    **rec, "decision": decision}, default=str) + "\n")
        made = [{"t": now, "id": c.get("id", ""), "label": _label(c, self._job_of(c)), "ok": c["ok"],
                 "why": clip(plain(str(c["args"].get("why") or ""))), "refused": c["refused"], "note": c["note"]}
                for c in rec["calls"] if c["name"] != "read_fleet_state"]
        if not summarised:
            for c in made:
                self.thoughts.add(c["label"] + (f" \u2014 {c['why']}" if c["why"] else "")
                                  + (f" (refused: {c['refused']})" if c["refused"] else ""), "decision", c["id"])
        self.history = (self.history + made)[-KEEP:]
        self.last = {**rec, "intent": intent or self.last.get("intent")}
        return rec

    def summary(self) -> dict:
        return {"link": "CUT" if self.link_cut else "UP", "model": getattr(self.backend, "name", None),
                "status": self.last.get("status"), "intent": self.last.get("intent"), "text": plain(self.last.get("text", "")),
                "calls": list(self.history), "reserved": sorted(self.envelope.reserved),
                "cost_usd": round(self.spent, 4), "agent": AGENT}

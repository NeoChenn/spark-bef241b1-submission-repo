from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from policy.control.brief import plain, tidy
from policy.control.prompt import prompt_hash, request_body
from policy.control.safety import FORCE_LIMIT_N, LEL_EVACUATE, PERSON_MIN_M
from policy.control.snapshot import SensorSnapshot
from policy.gemini_verifier import load_api_key

MODEL = "gemini-4-argon"
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
REPLAY_PATH = Path(__file__).with_name("replay.json")
IMAGE_TOKENS = 560
RUST_WORK = {"APPROACH", "SPRAY", "SOAK", "SCRUB", "BACK_OFF", "VERIFY", "PAINT", "VERIFY_PAINT", "DETECT",
             "SECOND_VIEW"}
THOUGHTS_KEEP = 6
THOUGHT_CHARS = 240


@dataclass
class Reply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_thought: int = 0
    tokens_cached: int = 0
    latency_ms: float = 0.0
    model: str = MODEL
    key: Optional[str] = None
    thoughts: tuple = ()
    streamed: bool = False


def clip(text: str, n: int = THOUGHT_CHARS) -> str:
    s = " ".join(text.split())
    if len(s) <= n:
        return s
    head = s[:n + 1]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if end >= n // 3:
        return s[:end + 1]
    cut = s[:n - 1]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp >= n // 3 else cut).rstrip(",;:- ") + "\u2026"


def summary_paragraphs(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text or ""):
        para = para.replace("**", "").replace("`", "").strip()
        if para:
            out.append(clip(para))
    return out


def summary_texts(steps) -> list[str]:
    return [p for st in steps or [] if st.get("type") == "thought" for c in st.get("summary") or []
            if c.get("type", "text") == "text" for p in summary_paragraphs(c.get("text", ""))]


class Thoughts:
    def __init__(self, keep: int = THOUGHTS_KEEP):
        self.items: deque = deque(maxlen=keep)
        self.lock = threading.Lock()

    def add(self, text: str, kind: str = "summary", call: Optional[str] = None, t: Optional[float] = None) -> Optional[dict]:
        text = tidy(text) if kind == "summary" else plain(text)
        if not text:
            return None
        item = {"t": round(time.time() if t is None else t, 1), "text": clip(text), "kind": kind}
        if call:
            item["call"] = call
        with self.lock:
            self.items.append(item)
        return item

    def tag(self, items, call: Optional[str]) -> None:
        if not call:
            return
        with self.lock:
            for it in items:
                it.setdefault("call", call)

    def view(self) -> list[dict]:
        with self.lock:
            return [dict(it) for it in self.items]

    def mode(self) -> str:
        with self.lock:
            return "summaries" if any(it["kind"] == "summary" for it in self.items) else "decisions"


def parse_sse(lines, on_thought: Optional[Callable[[str], None]] = None) -> dict:
    steps: dict[int, dict] = {}
    args: dict[int, str] = {}
    out: dict = {}
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        ev = json.loads(data)
        kind = ev.get("event_type")
        i = ev.get("index", 0)
        if kind == "error":
            err = ev.get("error") or {}
            raise RuntimeError(f"stream error: {err.get('message') if isinstance(err, dict) else err}")
        if kind == "step.start":
            steps[i] = dict(ev.get("step") or {})
        elif kind == "step.delta":
            st, d = steps.setdefault(i, {}), ev.get("delta") or {}
            dt = d.get("type")
            if dt == "text":
                content = st.setdefault("content", [])
                if content and content[-1].get("type") == "text":
                    content[-1]["text"] += d.get("text", "")
                else:
                    content.append({"type": "text", "text": d.get("text", "")})
            elif dt == "thought_signature":
                st["signature"] = d.get("signature", "")
            elif dt == "thought_summary":
                c = d.get("content") or {}
                st.setdefault("summary", []).append(c)
                if on_thought is not None:
                    for p in summary_paragraphs(c.get("text", "")):
                        on_thought(p)
            elif dt == "arguments_delta":
                args[i] = args.get(i, "") + d.get("arguments", "")
        elif kind == "step.stop" and args.get(i):
            steps[i]["arguments"] = json.loads(args.pop(i))
        elif kind == "interaction.completed":
            out = dict(ev.get("interaction") or {})
    for i, a in args.items():
        if a:
            steps[i]["arguments"] = json.loads(a)
    return {**out, "steps": [steps[k] for k in sorted(steps)]}


def post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = " ".join(e.read().decode("utf-8", "replace").split())
        raise RuntimeError(f"HTTP {e.code}: {detail[:240]}") from None


def post_stream(url: str, body: dict, headers: dict, timeout: float, on_thought=None) -> dict:
    req = urllib.request.Request(url, data=json.dumps({**body, "stream": True}).encode("utf-8"), method="POST",
                                 headers={**headers, "Accept": "text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {**parse_sse(resp, on_thought), "_streamed": True}
    except urllib.error.HTTPError as e:
        detail = " ".join(e.read().decode("utf-8", "replace").split())
        raise RuntimeError(f"HTTP {e.code}: {detail[:240]}") from None


def streaming() -> bool:
    return os.environ.get("GEMINI_STREAM", "1") not in ("0", "off", "")


def parse_interaction(reply: dict) -> tuple[str, dict]:
    text = "".join(c.get("text", "") for st in reply.get("steps") or [] if st.get("type") == "model_output"
                   for c in st.get("content") or [] if c.get("type") == "text")
    if not text:
        from policy.gemini_verifier import _extract_text
        text = _extract_text(reply)
    u = reply.get("usage") or {}
    return text, {"in": int(u.get("total_input_tokens") or 0), "out": int(u.get("total_output_tokens") or 0),
                  "thought": int(u.get("total_thought_tokens") or 0), "cached": int(u.get("total_cached_tokens") or 0)}


class ArgonBackend:
    live = True

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, thinking: str = "low",
                 timeout: float = 30.0, transport=None):
        self.api_key = api_key or load_api_key()
        self.model = model or os.environ.get("GEMINI_CONTROL_MODEL") or MODEL
        self.thinking = os.environ.get("GEMINI_CONTROL_THINKING", thinking)
        self.timeout = timeout
        self.on_thought: Optional[Callable[[str], None]] = None
        self.transport = transport or (self._stream if streaming() else self._post)
        self.name = self.model

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        return post_json(url, body, headers, self.timeout)

    def _stream(self, url: str, body: dict, headers: dict) -> dict:
        return post_stream(url, body, headers, self.timeout, self.on_thought)

    def call(self, prompt: dict, snap: SensorSnapshot) -> Reply:
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key, "Api-Revision": API_REVISION}
        started = time.perf_counter()
        reply = self.transport(API_URL, request_body(prompt, self.model, self.thinking), headers)
        text, u = parse_interaction(reply)
        return Reply(text, u["in"], u["out"], u["thought"], u["cached"], (time.perf_counter() - started) * 1000.0,
                     self.model, thoughts=tuple(summary_texts(reply.get("steps"))),
                     streamed=bool(reply.get("_streamed")))


HAZARD_KEYS = ("evacuate", "force_spike", "person", "leak_suspect", "recovering", "verifier_dispute", "storm")
MAX_COMBINED = 3


def situation_keys(s: SensorSnapshot) -> list[str]:
    states = {r.state for r in s.robots}
    keys = []
    if s.max_lel() >= LEL_EVACUATE:
        keys.append("evacuate")
    if "BACK_OFF" in states or any(r.force_n is not None and r.force_n > FORCE_LIMIT_N for r in s.robots):
        keys.append("force_spike")
    if "PAUSE_PERSON" in states or any(r.person_m is not None and r.person_m < PERSON_MIN_M for r in s.robots):
        keys.append("person")
    if "evacuate" not in keys and any(g.lel >= 10.0 and g.lel_trend > 3.0 and g.db_trend > 3.0 for g in s.gas):
        keys.append("leak_suspect")
    if "evacuate" not in keys and any(g.lel >= 5.0 and g.lel_trend < -3.0 for g in s.gas):
        keys.append("recovering")
    for v in s.verdicts:
        r = s.robot(v.robot)
        if v.verdict == "rust" and r is not None and r.rust_pct is not None and r.rust_pct < 5.0:
            keys.append("verifier_dispute")
            break
    if s.storm_level >= 0.6:
        keys.append("storm")
    if "DONE" in states:
        keys.append("done")
    elif states & RUST_WORK:
        keys.append("repair")
    return keys or ["nominal"]


def situation_key(s: SensorSnapshot) -> str:
    return situation_keys(s)[0]


class ReplayBackend:
    live = False

    def __init__(self, path=REPLAY_PATH, model: str = MODEL):
        self.responses = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model = model
        self.name = f"{model} (replay)"

    @property
    def available(self) -> bool:
        return True

    def compose(self, keys: list[str]) -> dict:
        parts = [self.responses.get(k) or self.responses["nominal"] for k in keys]
        if len(parts) == 1:
            return parts[0]
        return {"situation": "\n".join(p["situation"].splitlines()[0] for p in parts),
                "actions": [a for p in parts for a in p["actions"]],
                "anomalies": [a for p in parts for a in p["anomalies"]]}

    def call(self, prompt: dict, snap: SensorSnapshot, key: Optional[str] = None) -> Reply:
        if key:
            keys = [key]
        else:
            keys = situation_keys(snap)
            keys = [k for k in keys if k in HAZARD_KEYS][:MAX_COMBINED] or keys[:1]
        text = json.dumps(self.compose(keys))
        h = int(prompt_hash(prompt), 16)
        tokens_in = (len(prompt["system"]) + len(prompt["text"])) // 4 + IMAGE_TOKENS * len(prompt["images"])
        return Reply(text, tokens_in, len(text) // 4, 280 + h % 420, 0, 1150.0 + h % 1400, self.model, "+".join(keys))

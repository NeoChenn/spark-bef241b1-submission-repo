from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
        self.transport = transport or self._post
        self.name = self.model

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = " ".join(e.read().decode("utf-8", "replace").split())
            raise RuntimeError(f"HTTP {e.code}: {detail[:240]}") from None

    def call(self, prompt: dict, snap: SensorSnapshot) -> Reply:
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key, "Api-Revision": API_REVISION}
        started = time.perf_counter()
        reply = self.transport(API_URL, request_body(prompt, self.model, self.thinking), headers)
        text, u = parse_interaction(reply)
        return Reply(text, u["in"], u["out"], u["thought"], u["cached"], (time.perf_counter() - started) * 1000.0,
                     self.model)


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

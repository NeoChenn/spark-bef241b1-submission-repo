"""Gemini vision for the aerial jobs: ISO 4628-3 rust grading and a thermal second opinion.

Built on policy/gemini_verifier.py (key loading, JPEG encoding, reply parsing). Every call is optional:
with no key, no network or an unusable reply the methods return None and the job uses the on-board
classical result (edge autonomy is kept).

Replies can be recorded to / replayed from a JSON file keyed by the image hash, so tests and demos run
offline on real Gemini answers:  ReplayTransport(path)  /  GeminiVision(record=path).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from policy.gemini_verifier import API_ROOT, DEFAULT_MODEL, _extract_text, encode_jpeg, load_api_key, post_json

RI_PROMPT = (
    "You are a coatings inspector grading offshore topsides steel from a drone camera close-up, using "
    "ISO 4628-3 (degree of rusting). Rust is orange-brown to dark-brown corrosion product: rust spots, "
    "pitting, rust bleeding through the paint. Paint colour (including yellow, white, grey, green or blue "
    "coats), dirt-free shading and lighting gradients are NOT rust. Estimate the percentage of the visible "
    "surface area that is rusted, then give the ISO 4628-3 grade using the standard's reference areas: "
    "Ri 0 = 0 %, Ri 1 = 0.05 %, Ri 2 = 0.5 %, Ri 3 = 1 %, Ri 4 = 8 %, Ri 5 = 40-50 %. Pick the grade whose "
    "reference area is closest to your estimate on a log scale. A single tiny spot is about Ri 1."
)
RI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "ri_grade": {"type": "INTEGER"},
        "rust_percent": {"type": "NUMBER"},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
    },
    "required": ["ri_grade", "rust_percent", "confidence", "reason"],
}

THERMAL_PROMPT = (
    "You are the flare-system engineer on an offshore rig. The image is a false-colour radiometric thermal "
    "image (inferno palette: black/purple cold, yellow/white hot) of the flare tip taken by an inspection "
    "drone. The burner nozzles along the tip are numbered under the image. Healthy nozzles all glow at about "
    "the same temperature. Is any nozzle clearly hotter or colder than the others (burn-back, internal "
    "burning, or a plugged / unlit nozzle)? The flame above the tip is expected and is not a fault."
)
THERMAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "anomalous_nozzle": {"type": "INTEGER"},       # nozzle number, or -1 if none
        "kind": {"type": "STRING", "enum": ["hot", "cold", "none"]},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
    },
    "required": ["anomalous_nozzle", "kind", "confidence", "reason"],
}


@dataclass
class GeminiRi:
    ri: int
    rust_pct: float
    confidence: float
    reason: str
    model: str = ""
    latency_ms: float = 0.0

    def summary(self) -> dict:
        d = asdict(self)
        d["rust_pct"] = round(float(self.rust_pct), 3)
        d["confidence"] = round(float(self.confidence), 3)
        d["latency_ms"] = round(float(self.latency_ms))
        return d


def image_key(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(image, np.uint8)).tobytes()).hexdigest()[:16]


class ReplayTransport:
    """Serves recorded Gemini replies (by image hash + task) instead of calling the API."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def lookup(self, key: str) -> Optional[dict]:
        return self.data.get(key)


class GeminiVision:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0,
                 transport: Optional[Callable[[str, dict], dict]] = None, replay: Optional[Path] = None,
                 record: Optional[Path] = None, min_confidence: float = 0.5):
        self.replay = ReplayTransport(replay) if replay else None
        self.api_key = api_key if api_key is not None else (None if self.replay else load_api_key())
        self.model = model or os.environ.get("GEMINI_VISION_MODEL") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.transport = transport
        self.record = Path(record) if record else None
        self.min_confidence = min_confidence
        self.calls: list[dict] = []
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key or self.transport or self.replay)

    def _ask(self, image: np.ndarray, prompt: str, schema: dict, task: str) -> Optional[tuple[dict, float, str]]:
        key = f"{task}:{image_key(image)}"
        if self.replay is not None:
            rec = self.replay.lookup(key)
            if rec is None:
                self.last_error = f"no recorded reply for {key}"
                return None
            return rec["reply"], float(rec.get("latency_ms", 0.0)), rec.get("model", "replay")
        if not (self.api_key or self.transport):
            self.last_error = "no GEMINI_API_KEY"
            return None
        jpeg, _ = encode_jpeg(image, max_side=1024)
        body = {"contents": [{"role": "user", "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}},
                    {"text": prompt + " Answer only with JSON matching the schema."}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                     "temperature": 0.0}}
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        started = time.perf_counter()
        try:
            raw = (self.transport or (lambda u, b: post_json(u, b, self.api_key, self.timeout)))(url, body)
            text = _extract_text(raw)
            m = re.search(r"\{.*\}", text, re.S)
            reply = json.loads(m.group(0) if m else text)
        except Exception as e:   # network, HTTP, bad JSON: caller falls back to the classical result
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            self.calls.append({"task": task, "error": self.last_error})
            return None
        latency = (time.perf_counter() - started) * 1000.0
        if self.record is not None:
            data = json.loads(self.record.read_text(encoding="utf-8")) if self.record.exists() else {}
            data[key] = {"reply": reply, "latency_ms": round(latency), "model": self.model}
            self.record.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        return reply, latency, self.model

    def grade_ri(self, image: np.ndarray) -> Optional[GeminiRi]:
        got = self._ask(image, RI_PROMPT, RI_SCHEMA, "ri")
        if got is None:
            return None
        reply, latency, model = got
        try:
            ri = int(round(float(reply["ri_grade"])))
            if not 0 <= ri <= 5:
                raise ValueError(f"ri_grade {ri} out of range")
            out = GeminiRi(ri, float(reply.get("rust_percent", 0.0)),
                           float(np.clip(float(reply.get("confidence", 0.0)), 0, 1)),
                           str(reply.get("reason", ""))[:240], model, latency)
        except Exception as e:
            self.last_error = f"bad reply: {e}"
            self.calls.append({"task": "ri", "error": self.last_error})
            return None
        self.calls.append({"task": "ri", **out.summary()})
        return out

    def thermal_opinion(self, false_colour: np.ndarray) -> Optional[dict]:
        got = self._ask(false_colour, THERMAL_PROMPT, THERMAL_SCHEMA, "thermal")
        if got is None:
            return None
        reply, latency, model = got
        out = {"anomalous_nozzle": int(reply.get("anomalous_nozzle", -1)), "kind": str(reply.get("kind", "none")),
               "confidence": float(reply.get("confidence", 0.0)), "reason": str(reply.get("reason", ""))[:240],
               "model": model, "latency_ms": round(latency)}
        self.calls.append({"task": "thermal", **out})
        return out

    def confident(self, conf: Optional[float]) -> bool:
        return conf is not None and conf >= self.min_confidence

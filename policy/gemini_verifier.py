from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_MODEL = "gemini-robotics-er-2-preview"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
REPO = Path(__file__).resolve().parent.parent

PROMPTS = {
    "rust": (
        "You are the inspection supervisor for a pipe-repair robot on an offshore oil rig. The image is from "
        "its inspection camera. Only the pipe surface is shown in colour; everything else is flat grey and "
        "must be ignored. Question: is there corrosion (orange-brown rust, flaking or pitting) on the pipe "
        "surface? Painted green or grey-green pipe, wet dark remover, primer, the robot arm and its steel "
        "brush are NOT rust. If the view is blocked, too dark or blurred to judge, say so."
    ),
    "clean": (
        "You are the inspection supervisor for a pipe-repair robot on an offshore oil rig. The robot has "
        "just scrubbed a rust patch and wants to paint over it. The image is from its inspection camera; "
        "only the pipe surface is in colour, the rest is flat grey and must be ignored. Question: is any "
        "orange-brown rust still visible on the pipe surface? Painted pipe, wet dark remover, primer, the "
        "robot arm and its steel brush are NOT rust. If the view is blocked, too dark or blurred to judge, "
        "say so. Be strict: painting over remaining rust is the failure we must prevent."
    ),
}
FORMAT = (
    " Answer only with JSON: {\"verdict\": \"rust\" | \"clean\" | \"cannot_tell\", \"confidence\": 0..1, "
    "\"points\": [{\"point\": [y, x], \"label\": \"rust\"}], \"reason\": \"one short sentence\"}. "
    "Points are rust locations in [y, x] order normalised to 0-1000; at most 5; empty if clean."
)
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["rust", "clean", "cannot_tell"]},
        "confidence": {"type": "NUMBER"},
        "points": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "point": {"type": "ARRAY", "items": {"type": "NUMBER"}},
            "label": {"type": "STRING"}}}},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "confidence", "reason"],
}


@dataclass
class GeminiVerdict:
    verdict: str
    confidence: float
    reason: str
    points: list = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""

    @property
    def rust(self) -> bool:
        return self.verdict == "rust"

    @property
    def clean(self) -> bool:
        return self.verdict == "clean"

    def summary(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 3)
        d["latency_ms"] = round(float(self.latency_ms))
        d["points"] = [[round(float(x)), round(float(y))] for x, y in self.points]
        return d


def load_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                return m.group(2)
    return None


def encode_jpeg(image: np.ndarray, max_side: int = 768) -> tuple[bytes, float]:
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(np.asarray(image)[..., :3]).astype(np.uint8))
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), scale


def _extract_text(reply: dict) -> str:
    for cand in reply.get("candidates") or []:
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if text:
            return text
    for key in ("output_text", "outputText"):
        if isinstance(reply.get(key), str):
            return reply[key]
    for out in reply.get("outputs") or reply.get("output") or []:
        if isinstance(out, dict):
            if isinstance(out.get("text"), str):
                return out["text"]
            for p in out.get("parts") or (out.get("content") or {}).get("parts") or []:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    return p["text"]
    raise ValueError(f"no text in Gemini reply (keys: {sorted(reply)})")


def parse_verdict(text: str, width: int, height: int) -> GeminiVerdict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    data = json.loads(m.group(0))
    verdict = str(data.get("verdict", "cannot_tell")).strip().lower().replace(" ", "_")
    if verdict not in ("rust", "clean", "cannot_tell"):
        verdict = "cannot_tell"
    conf = float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0))
    points = []
    for p in data.get("points") or []:
        yx = p.get("point") if isinstance(p, dict) else p
        if isinstance(yx, (list, tuple)) and len(yx) == 2:
            y, x = float(yx[0]), float(yx[1])
            points.append((x / 1000.0 * width, y / 1000.0 * height))
    return GeminiVerdict(verdict, conf, str(data.get("reason", ""))[:200], points)


class GeminiVerifier:

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 20.0,
                 endpoint: Optional[str] = None, min_confidence: float = 0.6, transport=None):
        self.api_key = api_key or load_api_key()
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.endpoint = endpoint or os.environ.get("GEMINI_ENDPOINT", "generateContent")
        self.timeout = timeout
        self.min_confidence = min_confidence
        self.transport = transport or self._post
        self.calls: list[dict] = []
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _body(self, jpeg: bytes, prompt: str) -> tuple[str, dict]:
        image = {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}}
        if self.endpoint == "interactions":
            return f"{API_ROOT}/interactions", {
                "model": self.model,
                "input": {"parts": [image, {"text": prompt}]},
                "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA},
            }
        return f"{API_ROOT}/models/{self.model}:generateContent", {
            "contents": [{"role": "user", "parts": [image, {"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA,
                                 "temperature": 0.0},
        }

    def _post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers={
            "Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def verify(self, image: np.ndarray, question: str = "rust") -> Optional[GeminiVerdict]:
        if not self.available or image is None:
            self.last_error = "no GEMINI_API_KEY" if not self.available else "no image"
            return None
        jpeg, scale = encode_jpeg(image)
        url, body = self._body(jpeg, PROMPTS[question] + FORMAT)
        started = time.perf_counter()
        try:
            reply = self.transport(url, body)
            h, w = np.asarray(image).shape[:2]
            verdict = parse_verdict(_extract_text(reply), w, h)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300] if hasattr(e, "read") else ""
            self.last_error = f"HTTP {e.code}: {detail}"
            self.calls.append({"question": question, "error": self.last_error})
            return None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.calls.append({"question": question, "error": self.last_error})
            return None
        verdict.latency_ms = (time.perf_counter() - started) * 1000.0
        verdict.model = self.model
        self.last_error = None
        self.calls.append({"question": question, **verdict.summary()})
        return verdict

    def confident(self, verdict: Optional[GeminiVerdict]) -> bool:
        return verdict is not None and verdict.verdict != "cannot_tell" and verdict.confidence >= self.min_confidence


def _main(argv=None) -> int:
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser(description="Ask Gemini about one inspection frame (needs GEMINI_API_KEY).")
    ap.add_argument("image")
    ap.add_argument("--question", choices=sorted(PROMPTS), default="rust")
    ap.add_argument("--model", default=None)
    ap.add_argument("--endpoint", choices=["generateContent", "interactions"], default=None)
    args = ap.parse_args(argv)
    verifier = GeminiVerifier(model=args.model, endpoint=args.endpoint)
    if not verifier.available:
        print("No GEMINI_API_KEY in the environment or .env")
        return 2
    verdict = verifier.verify(np.asarray(Image.open(args.image).convert("RGB")), args.question)
    if verdict is None:
        print(f"Gemini call failed: {verifier.last_error}")
        return 1
    print(json.dumps(verdict.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

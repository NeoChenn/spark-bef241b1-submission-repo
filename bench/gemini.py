from __future__ import annotations

import json
import os
import time
import urllib.request

from policy import gemini_verifier as gv

ARGON = "gemini-4-argon"
NEEDS_KEY = "not run - needs GEMINI_API_KEY"


def api_key():
    return gv.load_api_key()


def _get(url: str, key: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models(key: str) -> list[str]:
    out, token = [], ""
    while True:
        d = _get(f"{gv.API_ROOT}/models?pageSize=1000" + (f"&pageToken={token}" if token else ""), key)
        out += [m["name"].split("/", 1)[-1] for m in d.get("models", [])]
        token = d.get("nextPageToken", "")
        if not token:
            return out


def resolve_model(wanted: str, available: list[str]) -> str | None:
    if wanted in available:
        return wanted
    cands = sorted(m for m in available if m.startswith(wanted))
    return cands[0] if cands else None


def price_per_mtok() -> tuple[float | None, float | None]:
    f = lambda k: float(os.environ[k]) if os.environ.get(k) else None
    return f("GEMINI_PRICE_IN_PER_MTOK"), f("GEMINI_PRICE_OUT_PER_MTOK")


def cost_usd(tokens_in: int, tokens_out: int) -> float | None:
    pi, po = price_per_mtok()
    if pi is None or po is None:
        return None
    return tokens_in / 1e6 * pi + tokens_out / 1e6 * po


def generate(model: str, key: str, parts: list, schema: dict, timeout: float = 60.0, post=None) -> dict:
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0.0}}
    url = f"{gv.API_ROOT}/models/{model}:generateContent"
    t = time.perf_counter()
    if post is None:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read().decode("utf-8"))
    else:
        reply = post(url, body)
    lat = time.perf_counter() - t
    u = reply.get("usageMetadata") or {}
    text = gv._extract_text(reply)
    return {"json": json.loads(text[text.index("{"):text.rindex("}") + 1]), "latency_s": lat,
            "tokens_in": int(u.get("promptTokenCount", 0)),
            "tokens_out": int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))}

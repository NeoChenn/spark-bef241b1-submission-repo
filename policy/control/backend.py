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


HAZARD_KEYS = ("evacuate", "force_spike", "person", "leak_suspect", "sensor_fault", "recovering", "verifier_dispute",
               "storm")
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
    if any(h.job is None for h in s.faults()):
        keys.append("sensor_fault")
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


def _evidence(h) -> str:
    e = h.evidence
    if h.status == "STUCK":
        return f"flat at {e.get('value')} (std {e.get('std')}) while neighbours vary (std {e.get('nbr_std')})"
    if h.status == "DEAD":
        return f"silent for {e.get('silent_s')} s"
    if h.status == "DRIFT":
        return f"bias {e.get('bias')} {e.get('unit', '')} vs neighbours (tol {e.get('tol')})"
    if h.status == "NOISY":
        return f"noise {e.get('noise')} = {e.get('ratio')}x neighbours"
    return f"reads {e.get('value')} outside [{e.get('lo')}, {e.get('hi')}]"


def fault_reply(snap: SensorSnapshot) -> dict:
    actions, anomalies, lines = [], [], []
    for h in snap.faults():
        if h.job is not None:
            continue
        ok = sorted((c for c in h.candidates if c.get("ok")), key=lambda c: c["eta_s"])
        why = _evidence(h)
        lines.append(f"{h.id} {h.status}: {why}.")
        anomalies.append({"kind": "sensor_fault", "zone": h.zone, "severity": "critical" if h.hazard else "warn",
                          "confidence": 0.85, "sensors": [h.id], "note": f"{h.id} {why}"})
        if ok:
            c = ok[0]
            actions.append({"kind": "dispatch", "robot": c["robot"], "target": h.id, "purpose": "verify_sensor",
                            "confidence": 0.8, "rationale": f"{h.id} {why}; {c['robot']} is nearest with "
                                                            f"{c['sensor']} ({c['eta_s']:.0f} s)."})
        if h.hazard:
            actions.append({"kind": "escalate", "robot": "", "target": h.id, "confidence": 0.9,
                            "rationale": f"{h.id} is faulty beside live gas; treating {h.zone} as real gas."})
    return {"situation": "\n".join(lines) or "Sensor faults already have jobs.", "actions": actions,
            "anomalies": anomalies}


class ReplayBackend:
    live = False

    def __init__(self, path=REPLAY_PATH, model: str = MODEL):
        self.responses = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model = model
        self.name = f"{model} (replay)"

    @property
    def available(self) -> bool:
        return True

    def compose(self, keys: list[str], snap: Optional[SensorSnapshot] = None) -> dict:
        parts = [fault_reply(snap) if k == "sensor_fault" and snap is not None else
                 self.responses.get(k) or self.responses["nominal"] for k in keys]
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
        text = json.dumps(self.compose(keys, snap))
        h = int(prompt_hash(prompt), 16)
        tokens_in = (len(prompt["system"]) + len(prompt["text"])) // 4 + IMAGE_TOKENS * len(prompt["images"])
        return Reply(text, tokens_in, len(text) // 4, 280 + h % 420, 0, 1150.0 + h % 1400, self.model, "+".join(keys))


def status_reply(snap: SensorSnapshot) -> dict:
    working = [r.id for r in snap.robots if r.task]
    g = max(snap.gas, key=lambda g: g.lel, default=None)
    faults = snap.faults()
    lines = [f"{len(working)} of {len(snap.robots)} robots on jobs; {len(faults)} of {len(snap.health)} fixed "
             f"sensors suspect."]
    actions, anomalies = [], []
    if g is not None:
        lines.append(f"Highest gas {g.id} {g.lel:.0f}% LEL in zone {g.zone or '-'} ({g.lel_trend:+.1f}%/min).")
        if g.lel >= 10.0:
            actions.append({"kind": "escalate", "robot": "", "target": g.zone, "confidence": 0.9,
                            "rationale": f"{g.id} at {g.lel:.0f}% LEL: gas response running, crew informed."})
            anomalies.append({"kind": "probable_leak", "zone": g.zone, "severity": "critical", "confidence": 0.85,
                              "sensors": [g.id], "note": f"{g.id} {g.lel:.0f}% LEL rising {g.lel_trend:+.1f}%/min"})
    return {"situation": "\n".join(lines), "actions": actions, "anomalies": anomalies}


class ShiftReplayBackend(ReplayBackend):
    def compose(self, keys: list[str], snap: Optional[SensorSnapshot] = None) -> dict:
        parts = ([fault_reply(snap)] if "sensor_fault" in keys else []) + [status_reply(snap)]
        return {"situation": "\n".join(p["situation"] for p in parts if p["actions"] or p is parts[-1]),
                "actions": [a for p in parts for a in p["actions"]],
                "anomalies": [a for p in parts for a in p["anomalies"]]}

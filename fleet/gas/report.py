"""Operator-facing leak report and approval request: Gemini writes them from the localization numbers,
with a deterministic template as the fallback (as `gas-leak-detection` flange_robot/alerts.py does).

Gemini is on the slow path only: no decision depends on it. It is used when a key is available
(policy.gemini_verifier.load_api_key) and FLEET_GEMINI is not "0". A small rate guard caps calls.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Optional

REPORT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "report": {"type": "STRING"},
        "what": {"type": "STRING"},
        "where": {"type": "STRING"},
        "why": {"type": "STRING"},
        "risk_if_delayed": {"type": "STRING"},
    },
    "required": ["report", "what", "where", "why", "risk_if_delayed"],
}

_calls: deque = deque()
MAX_PER_MIN = 6


class Gemini:
    """Callable (prompt, schema) -> dict or None. `transport` is for tests (see policy.gemini_verifier)."""

    def __init__(self, enabled: Optional[bool] = None, transport: Optional[Callable] = None,
                 model: Optional[str] = None):
        self.transport, self._model = transport, model
        self.enabled = enabled
        self.last_error: Optional[str] = None

    def _key(self) -> Optional[str]:
        if self.transport is not None:
            return "test"
        if os.environ.get("FLEET_GEMINI", "1") == "0":
            return None
        try:
            from policy.gemini_verifier import load_api_key
            return load_api_key()
        except Exception:  # noqa: BLE001
            return None

    def __call__(self, prompt: str, schema: dict) -> Optional[dict]:
        if self.enabled is False:
            return None
        key = self._key()
        if not key:
            return None
        now = time.monotonic()
        while _calls and now - _calls[0] > 60:
            _calls.popleft()
        if len(_calls) >= MAX_PER_MIN:
            self.last_error = "rate guard"
            return None
        _calls.append(now)
        try:
            from policy.gemini_verifier import generate_json, pick_text_model
            if self._model is None:
                self._model = "test-model" if self.transport else pick_text_model(key)
            return generate_json(prompt, schema, key, self._model, timeout=60.0, transport=self.transport,
                                 temperature=0.2)
        except Exception as exc:  # noqa: BLE001 - network, quota, parse: fall back to the template
            self.last_error = type(exc).__name__
            return None


def _fmt_xy(xy) -> str:
    return f"({xy[0]:.1f}, {xy[1]:.1f})"


def leak_template(n: dict) -> dict:
    est, r = n["estimate_xy"], n["radius_m"]
    what = f"Close {n['valve']} (quarter turn) to isolate the gas feed to a suspected leak."
    where = (f"Leak estimated at {_fmt_xy(est)} on the {n['deck']} (95% within {r:.1f} m); "
             f"{n['valve']} is at {_fmt_xy(n['valve_xy'])}, {n['valve_distance_m']:.1f} m {n['valve_relation']}.")
    why = (f"Gas peaked at {n['peak_lel']:.2f} of LEL ({n['peak_lel'] * 100:.0f}% LEL) against a background of "
           f"0.02; {n['n_samples']} sensor samples fit a point source with wind {n['wind_mps']:.1f} m/s.")
    risk = ("If delayed the gas cloud keeps growing downwind; at 0.20 of LEL robots stop and retreat, and "
            "an ignitable mixture becomes possible near the source.")
    report = (f"GAS LEAK (critical). {where} {why} Recommended: {what} Robot stayed below "
              f"{n['approach_limit']:.2f} of LEL while localizing. Confirm by a follow-up survey after isolation.")
    return {"report": report, "what": what, "where": where, "why": why, "risk_if_delayed": risk}


def leak_prompt(n: dict) -> str:
    return ("You write operator messages for an offshore rig's robot fleet. From the numbers below, write (1) a "
            "short leak report for the control-room operator (3-4 sentences) and (2) an approval request for "
            "closing the isolation valve, as fields what / where / why / risk_if_delayed (one sentence each). "
            "Use only these numbers, keep them exact, state the uncertainty radius, do not invent equipment, "
            "procedures or advice beyond closing the named valve and confirming with a follow-up survey. Gas values "
            "are fractions of LEL (0.10 = 10% LEL); approach_limit and stop_limit are the robot's own safety limits "
            "for walking in gas, not process alarm levels.\n\n"
            + "\n".join(f"{k}: {v}" for k, v in n.items()))


def write_leak_messages(numbers: dict, gemini: Optional[Callable[[str, dict], Optional[dict]]]) -> tuple[dict, bool]:
    """Returns (messages, used_gemini). Falls back to the template if Gemini is off, fails, or drops a number."""
    base = leak_template(numbers)
    if gemini is None:
        return base, False
    out = gemini(leak_prompt(numbers), REPORT_SCHEMA)
    if not isinstance(out, dict) or not all(isinstance(out.get(k), str) and out[k].strip() for k in base):
        return base, False
    v = numbers["valve"]
    if v not in (out["what"] + out["report"]):
        return base, False
    return {k: out[k].strip() for k in base}, True


def approval_template(what: str, where: str, why: str, risk: str) -> dict:
    return {"what": what, "where": where, "why": why, "risk_if_delayed": risk,
            "text": f"APPROVAL NEEDED: {what} Where: {where} Why: {why} Risk if delayed: {risk}"}


def as_approval(msgs: dict) -> dict:
    return approval_template(msgs["what"], msgs["where"], msgs["why"], msgs["risk_if_delayed"])

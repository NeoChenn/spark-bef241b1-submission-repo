import json
import os
from pathlib import Path

import pytest

from policy.control.backend import ArgonBackend
from policy.control.centre import ControlCentre
from policy.control.shift import DAY, build

pytestmark = pytest.mark.skipif(os.environ.get("ARGON_LIVE") != "1", reason="set ARGON_LIVE=1 for one live call")
CAP_USD = 0.05


def test_one_live_argon_call_on_a_stuck_gas_head(tmp_path):
    brain = build("edge", tmp_path)
    while not any(h.id == "G-F1" and h.status == "STUCK" for h in (brain.snap.health if brain.snap else [])):
        brain.step()
    backend = ArgonBackend()
    assert backend.available
    centre = ControlCentre(backend, log_path=tmp_path / "calls.jsonl", interval_s=0.0, cap_usd_day=CAP_USD,
                           today=lambda: DAY, timeline=brain.port)
    h = centre.tick(brain.snap, now=brain.f.t)
    call = [e for e in brain.port.timeline if e["event"] == "argon_call"][-1]["detail"]
    record = {"model": call["model"], "status": call["status"], "latency_ms": call.get("latency_ms"),
              "tokens_in": call.get("tokens_in"), "tokens_out": call.get("tokens_out"),
              "tokens_thought": call.get("tokens_thought"), "cost_usd": call.get("cost_usd"),
              "situation": call.get("situation"), "decision": call.get("decision"), "error": call.get("error"),
              "candidates": next(x.candidates for x in brain.snap.health if x.id == "G-F1")[:4]}
    out = Path(os.environ.get("ARGON_LIVE_OUT", "runs/argon/live_smoke.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    assert call["model"] == "gemini-4-argon" and call["status"] == "ok", call.get("error")
    assert centre.budget.spent(DAY) <= CAP_USD and centre.calls[DAY] == 1
    ok = {c["robot"] for c in record["candidates"] if c["ok"]}
    for a in record["decision"]:
        if a["purpose"] == "verify_sensor":
            assert a["target"] == "G-F1" and a["robot"] in ok
    assert h["mode"] == "CLOUD"

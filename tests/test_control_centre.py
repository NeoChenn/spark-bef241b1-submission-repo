import json
import threading
from datetime import date

import pytest

from policy.control.backend import ReplayBackend, Reply
from policy.control.centre import OFFLINE_BANNER, ControlCentre
from policy.control.snapshot import GasSensor, Robot, SensorSnapshot

DAY = date(2026, 11, 8)


def fleet(t=0.0, force=20.0, lel=2.0, state="SCRUB"):
    return SensorSnapshot(
        t=t,
        robots=[Robot("go2-1", state=state, zone="A", force_n=force, rust_pct=40.0),
                Robot("go2-2", state="PATROL", zone="B"), Robot("go2-3", state="DOCKED", zone="dock")],
        gas=[GasSensor("G-F3", zone="B", lel=lel, acoustic_db=40.0)],
        jobs=[{"id": "R-1", "kind": "rust", "zone": "A"}, {"id": "L-F3", "kind": "leak", "zone": "B"}])


class Scripted:
    live, available, name = True, True, "gemini-4-argon"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.n = 0

    def call(self, prompt, snap):
        self.n += 1
        t = self.texts.pop(0) if self.texts else self.last
        self.last = t
        if isinstance(t, Exception):
            raise t
        return Reply(t, 2000, 300, 500, 0, 1500.0, "gemini-4-argon")


GOOD = json.dumps({"situation": "Leak at F3.", "actions": [
    {"kind": "dispatch", "robot": "go2-2", "target": "L-F3", "confidence": 0.9, "rationale": "nearest"}],
    "anomalies": [{"kind": "probable_leak", "zone": "F3", "severity": "warn", "confidence": 0.8, "note": "co-rise"}]})


def centre(backend, tmp_path, **kw):
    return ControlCentre(backend, log_path=tmp_path / "calls.jsonl", today=lambda: DAY, **kw)


def test_replay_tick_produces_header_and_log(tmp_path):
    c = centre(ReplayBackend(), tmp_path)
    h = c.tick(fleet(), now=0.0)
    assert h["mode"] == "CLOUD" and h["replay"] and h["online"]
    assert h["model"].startswith("gemini-4-argon")
    assert 1 <= len(h["situation"]) <= 3
    assert h["actions"] and {"id", "label", "robot", "confidence", "rank"} <= set(h["actions"][0])
    assert h["cost_today_usd"] > 0 and h["calls_today"] == 1 and h["latency_ms"] > 0
    json.dumps(h)
    rec = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert {"prompt_hash", "latency_ms", "tokens_in", "tokens_out", "tokens_thought", "cost_usd", "decision",
            "status", "model"} <= set(rec)
    assert rec["status"] == "ok" and rec["key"] == "repair"


def test_rate_limit_and_alarm_trigger(tmp_path):
    b = Scripted(GOOD)
    c = centre(b, tmp_path, interval_s=6.0, alarm_gap_s=1.5)
    c.tick(fleet(0.0), now=0.0)
    c.tick(fleet(1.0), now=1.0)
    assert b.n == 1
    c.tick(fleet(2.0, force=65.0, state="BACK_OFF"), now=2.0)
    assert b.n == 2
    c.tick(fleet(2.5, force=65.0, state="BACK_OFF"), now=2.5)
    assert b.n == 2
    c.tick(fleet(8.5), now=8.5)
    assert b.n == 3


def test_alarm_while_rate_limited_is_served_later(tmp_path):
    b = Scripted(GOOD)
    c = centre(b, tmp_path, interval_s=6.0, alarm_gap_s=1.5)
    c.tick(fleet(0.0), now=0.0)
    c.tick(fleet(0.5, lel=55.0), now=0.5)
    assert b.n == 1
    c.tick(fleet(1.6, lel=55.0), now=1.6)
    assert b.n == 2


def test_error_keeps_last_plan_and_shows_offline(tmp_path):
    b = Scripted(GOOD, TimeoutError("read timed out"), GOOD)
    c = centre(b, tmp_path, interval_s=1.0)
    first = c.tick(fleet(0.0), now=0.0)
    h = c.tick(fleet(2.0), now=2.0)
    assert h["mode"] == "EDGE" and not h["online"] and h["banner"] == OFFLINE_BANNER
    assert h["situation"] == first["situation"]
    assert [a["id"] for a in h["actions"]] == [a["id"] for a in first["actions"]]
    assert h["plan_age_s"] == pytest.approx(2.0)
    assert "timed out" in h["error"]
    h = c.tick(fleet(4.0), now=4.0)
    assert h["mode"] == "CLOUD" and h["banner"] is None
    recs = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in recs] == ["ok", "error", "ok"]


def test_invalid_reply_rejected_before_apply(tmp_path):
    b = Scripted(GOOD, "not json at all", json.dumps({"situation": "x", "actions": "drop table"}))
    c = centre(b, tmp_path, interval_s=1.0)
    c.tick(fleet(0.0), now=0.0)
    h = c.tick(fleet(2.0), now=2.0)
    assert h["status"] == "invalid" and h["actions"][0]["target"] == "L-F3"
    h = c.tick(fleet(4.0), now=4.0)
    assert h["status"] == "invalid" and h["banner"] == OFFLINE_BANNER


def test_whitelist_rejections_are_reported(tmp_path):
    bad = json.dumps({"situation": "x", "anomalies": [], "actions": [
        {"kind": "open_valve", "robot": "go2-2", "target": "B", "confidence": 0.99, "rationale": "yolo"},
        {"kind": "hold", "robot": "go2-3", "target": "", "confidence": 0.5, "rationale": "wait"}]})
    c = centre(Scripted(bad), tmp_path)
    h = c.tick(fleet(), now=0.0)
    assert [a["kind"] for a in h["actions"]] == ["hold"]
    assert h["rejected"] == 1


def test_edge_rules_override_and_veto(tmp_path):
    c = centre(Scripted(GOOD), tmp_path)
    h = c.tick(fleet(lel=60.0), now=0.0)
    assert h["actions"][0]["source"] == "edge"
    assert {a["kind"] for a in h["edge"]} == {"retreat", "evacuate"}
    assert [v["target"] for v in h["vetoed"]] == ["L-F3"]
    assert all(a["source"] == "edge" for a in h["actions"])
    assert [x["rank"] for x in h["actions"]] == list(range(1, len(h["actions"]) + 1))


def test_cost_cap_stops_calls(tmp_path):
    b = Scripted(GOOD)
    c = centre(b, tmp_path, interval_s=1.0, cap_usd_day=0.01)
    c.tick(fleet(0.0), now=0.0)
    h = c.tick(fleet(2.0), now=2.0)
    assert b.n == 1
    assert h["status"] == "capped" and "edge rules only" in h["banner"].lower()
    assert h["cost_today_usd"] == pytest.approx(0.012)


def test_no_backend_is_edge_only(tmp_path):
    c = centre(None, tmp_path)
    h = c.tick(fleet(force=70.0), now=0.0)
    assert h["mode"] == "EDGE" and h["actions"][0]["kind"] == "back_off"
    assert "edge rules only" in h["banner"].lower()


def test_async_pending_and_timeout(tmp_path):
    gate = threading.Event()

    class Slow(Scripted):
        def call(self, prompt, snap):
            gate.wait(5)
            return super().call(prompt, snap)

    c = centre(Slow(GOOD), tmp_path, sync=False, timeout_s=10.0)
    h = c.tick(fleet(0.0), now=0.0)
    assert h["pending"] and h["mode"] == "EDGE"
    gate.set()
    c.wait()
    h = c.tick(fleet(0.5), now=0.5)
    assert h["mode"] == "CLOUD" and not h["pending"]

    gate.clear()
    c2 = centre(Slow(GOOD), tmp_path, sync=False, timeout_s=1.0)
    c2.tick(fleet(0.0), now=0.0)
    h = c2.tick(fleet(2.0), now=2.0)
    assert h["status"] == "offline" and "timeout" in h["error"]
    gate.set()


def test_simulated_latency_defers_plan(tmp_path):
    c = centre(Scripted(GOOD), tmp_path, sim_latency=True)
    h = c.tick(fleet(0.0), now=0.0)
    assert h["pending"] and h["mode"] == "EDGE"
    h = c.tick(fleet(1.0), now=1.0)
    assert h["pending"]
    h = c.tick(fleet(1.6), now=1.6)
    assert not h["pending"] and h["mode"] == "CLOUD" and h["latency_ms"] == 1500


def test_auto_flag_and_gas_summary(tmp_path):
    low = json.dumps({"situation": "x", "anomalies": [], "actions": [
        {"kind": "hold", "robot": "go2-3", "target": "", "confidence": 0.4, "rationale": "maybe"}]})
    h = centre(Scripted(low), tmp_path).tick(fleet(force=50.0, lel=12.0), now=0.0)
    assert [(a["kind"], a["auto"]) for a in h["actions"]] == [("back_off", True), ("hold", False)]
    assert h["gas"][0] == {"id": "G-F3", "zone": "B", "lel": 12.0, "db": 40.0, "trend": 0.0}


def test_centre_latches_evacuation(tmp_path):
    c = centre(None, tmp_path)
    assert any(a["kind"] == "evacuate" for a in c.tick(fleet(lel=55.0), now=0.0)["actions"])
    assert any(a["kind"] == "evacuate" for a in c.tick(fleet(lel=30.0), now=1.0)["actions"])
    assert not any(a["kind"] == "evacuate" for a in c.tick(fleet(lel=8.0), now=2.0)["actions"])
    assert not any(a["kind"] == "evacuate" for a in c.tick(fleet(lel=30.0), now=3.0)["actions"])

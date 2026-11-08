from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from policy.control.adapters import TrendTracker
from policy.control.backend import THOUGHTS_KEEP, Thoughts
from policy.control.limiter import Budget, RateLimiter, cost_usd
from policy.control.prompt import build, prompt_hash
from policy.control.safety import arbitrate, edge_rules
from policy.control.schema import Plan, SchemaError, validate
from policy.control.snapshot import SensorSnapshot

OFFLINE_BANNER = "CLOUD OFFLINE — edge rules only"
CAP_BANNER = "COST CAP REACHED — edge rules only"
NO_CLOUD_BANNER = "NO CLOUD KEY — edge rules only"
CUT_BANNER = "GEMINI LINK CUT — EDGE ONLY"
ALARM_STATES = {"BACK_OFF", "PAUSE_PERSON", "HOLD", "ESCALATE_TO_HUMAN", "GAS_CHECK", "GAS_RETREAT", "DONE"}
LEAK_ALARM_LEL = 10.0
LEAK_ALARM_TREND = 3.0
AUTO_CONF = 0.6
TOP_GAS = 3


def _json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise SchemaError("no JSON object in reply")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise SchemaError(f"bad JSON: {e}") from e


def alarms(snap: SensorSnapshot, edge) -> set[str]:
    out = {a.id for a in edge}
    out |= {f"gas:{g.id}" for g in snap.gas if g.lel >= LEAK_ALARM_LEL and g.lel_trend > LEAK_ALARM_TREND}
    out |= {f"state:{r.id}:{r.state}" for r in snap.robots if r.state in ALARM_STATES}
    out |= {f"verdict:{v.robot}:{v.verdict}" for v in snap.verdicts}
    return out


class ControlCentre:
    def __init__(self, backend, log_path="runs/control/calls.jsonl", interval_s: float = 6.0,
                 alarm_gap_s: float = 1.5, per_minute: int = 12, cap_usd_day: float = 5.0, timeout_s: float = 30.0,
                 sync: bool = True, today=date.today, max_images: int = 1, sim_latency: bool = False,
                 auto_conf: float = AUTO_CONF, agent=None):
        self.backend, self.agent, self.link_cut = backend, agent, False
        self.log_path = Path(log_path) if log_path else None
        self.limiter = RateLimiter(interval_s, alarm_gap_s, per_minute)
        self.budget = Budget(cap_usd_day)
        self.timeout_s, self.sync, self.today, self.max_images = timeout_s, sync, today, max_images
        self.sim_latency, self.auto_conf = sim_latency, auto_conf
        self.trends = TrendTracker()
        self.plan: Optional[Plan] = None
        self.plan_t: Optional[float] = None
        self.status = "idle" if backend is not None and backend.available else "edge"
        self.error: Optional[str] = None
        self.last: dict = {}
        self.calls: dict[date, int] = {}
        self.seen: set[str] = set()
        self.unserved: set[str] = set()
        self.latched: set[str] = set()
        self.pending = None
        self.pool = None if sync else ThreadPoolExecutor(1, thread_name_prefix="argon")
        self.thoughts = Thoughts()
        self.streamed: list[dict] = []
        self.plan_key = None
        self.veto_t: dict[str, float] = {}
        if agent is not None:
            agent.thoughts = self.thoughts
        if backend is not None and hasattr(backend, "on_thought"):
            backend.on_thought = self._thought

    def _thought(self, text: str) -> None:
        self.streamed.append(self.thoughts.add(text))

    @property
    def cloud(self) -> bool:
        return self.backend is not None and self.backend.available

    def tick(self, snap: SensorSnapshot, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else now
        self.trends.update(snap)
        edge = edge_rules(snap, self.latched)
        self.latched = {a.target for a in edge if a.kind == "evacuate"}
        cur = alarms(snap, edge)
        self.unserved |= cur - self.seen
        self.seen = cur
        self._collect(now)
        if self.cloud and not self.link_cut and self.pending is None:
            if not self.budget.allows(self.today()):
                self.status = "capped"
            elif self.limiter.due(now, alarm=bool(self.unserved)):
                self._start(snap, edge, now)
        final, vetoed = arbitrate(edge, self.plan.actions if self.plan else [], snap)
        return self._header(snap, edge, final, vetoed, now)

    def nudge(self, key: str) -> None:
        self.unserved.add(key)

    def cut_link(self, on: bool) -> None:
        self.link_cut = bool(on)
        self.pending = None
        if on:
            self.status = "cut"
        elif self.status == "cut":
            self.status = "ok" if self.plan is not None else "idle"
        if self.agent is not None:
            self.agent.cut_link(on)

    def intent(self, text: str, now: float = 0.0):
        return None if self.agent is None else self.agent.handle(intent=text, now=now)

    def finding(self, finding: dict, now: float = 0.0):
        return None if self.agent is None else self.agent.handle(findings=[finding], now=now)

    def wait(self) -> None:
        if self.pending is not None and "future" in self.pending:
            self.pending["future"].result()

    def _start(self, snap, edge, now) -> None:
        self.limiter.mark(now)
        job = {"prompt": build(snap, edge, self.max_images), "snap": snap, "started": now,
               "alarms": sorted(self.unserved)}
        job["hash"] = prompt_hash(job["prompt"])
        self.unserved = set()
        if self.sync:
            result = self._run(job)
            if self.sim_latency and not isinstance(result, Exception):
                job.update(result=result, due=now + result.latency_ms / 1000.0)
                self.pending = job
            else:
                self._finish(job, result, now)
        else:
            job["future"] = self.pool.submit(self._run, job)
            self.pending = job

    def _run(self, job):
        self.streamed = []
        try:
            return self.backend.call(job["prompt"], job["snap"])
        except Exception as e:
            return e

    def _collect(self, now) -> None:
        job = self.pending
        if job is None:
            return
        if "due" in job:
            if now >= job["due"]:
                self.pending = None
                self._finish(job, job["result"], now)
        elif job["future"].done():
            self.pending = None
            self._finish(job, job["future"].result(), now)
        elif now - job["started"] > self.timeout_s:
            self.pending = None
            self._finish(job, TimeoutError(f"timeout after {self.timeout_s:.0f} s"), now)

    def _finish(self, job, result, now) -> None:
        day = self.today()
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "t": job["snap"].t, "now": now,
               "prompt_hash": job["hash"], "model": getattr(self.backend, "name", "?"),
               "live": bool(getattr(self.backend, "live", False)), "alarms": job["alarms"],
               "images": len(job["prompt"]["images"])}
        if isinstance(result, Exception):
            self.status, self.error = "offline", f"{type(result).__name__}: {result}"
            rec.update(status="error", error=self.error)
            self._log(rec)
            return
        cost = cost_usd(result.tokens_in, result.tokens_out, result.tokens_thought, day, result.tokens_cached)
        self.budget.spend(cost, day)
        self.calls[day] = self.calls.get(day, 0) + 1
        self.last = {"latency_ms": round(result.latency_ms), "cost_usd": round(cost, 5), "key": result.key,
                     "tokens": {"in": result.tokens_in, "out": result.tokens_out, "thought": result.tokens_thought}}
        rec.update(latency_ms=round(result.latency_ms, 1), tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                   tokens_thought=result.tokens_thought, tokens_cached=result.tokens_cached, cost_usd=round(cost, 6),
                   cost_today_usd=round(self.budget.spent(day), 5), key=result.key)
        try:
            plan = validate(_json(result.text), job["snap"])
        except SchemaError as e:
            self.status, self.error = "invalid", str(e)
            rec.update(status="invalid", error=self.error, reply=result.text[:400])
            self._log(rec)
            return
        self.plan, self.plan_t, self.status, self.error = plan, now, "ok", None
        self._remember(plan, result)
        rec.update(status="ok", situation=plan.situation, decision=[a.id for a in plan.actions],
                   anomalies=[f"{a.kind}:{a.zone}" for a in plan.anomalies], rejected=plan.rejected)
        self._log(rec)

    def _remember(self, plan: Plan, result) -> None:
        top = next((a for a in plan.actions if a.source != "edge"), None)
        call = top.id if top else None
        items = list(self.streamed)
        if not result.streamed:
            items += [self.thoughts.add(t) for t in result.thoughts]
        self.thoughts.tag(items, call)
        key = tuple(a.id for a in plan.actions)
        if not items and key != self.plan_key:
            text = f"{top.label} \u2014 {top.rationale}" if top and top.rationale else \
                (top.label if top else (plan.situation[0] if plan.situation else ""))
            if text:
                self.thoughts.add(text, "decision", call)
        self.plan_key = key

    def _refused(self, vetoed) -> list[dict]:
        wall = round(time.time(), 1)
        live = {v["id"]: v for v in vetoed}
        self.veto_t = {k: self.veto_t.get(k, wall) for k in live}
        out = [{"t": self.veto_t[k], "label": v["label"], "why": v["why"]} for k, v in live.items()]
        if self.agent is not None and hasattr(self.agent, "refused"):
            out += self.agent.refused()
        return sorted(out, key=lambda r: r["t"])[-THOUGHTS_KEEP:]

    def _log(self, rec: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _banner(self) -> Optional[str]:
        if self.link_cut:
            return CUT_BANNER
        if not self.cloud:
            return NO_CLOUD_BANNER
        return {"offline": OFFLINE_BANNER, "invalid": OFFLINE_BANNER, "capped": CAP_BANNER}.get(self.status)

    def _header(self, snap, edge, final, vetoed, now) -> dict:
        day = self.today()
        online = self.cloud and not self.link_cut and self.status in ("ok", "idle")
        plan = self.plan
        return {
            "mode": "CLOUD" if online and plan is not None else "EDGE",
            "online": online,
            "status": self.status,
            "banner": self._banner(),
            "error": self.error,
            "model": getattr(self.backend, "name", None),
            "replay": self.cloud and not getattr(self.backend, "live", False),
            "pending": self.pending is not None,
            "situation": plan.situation if plan else [],
            "actions": [{**a.to_dict(), "rank": k + 1, "auto": a.source == "edge" or a.confidence >= self.auto_conf}
                        for k, a in enumerate(final)],
            "anomalies": [a.to_dict() for a in plan.anomalies] if plan else [],
            "edge": [a.to_dict() for a in edge],
            "vetoed": vetoed,
            "refused": self._refused(vetoed),
            "thoughts": self.thoughts.view(),
            "thoughts_mode": self.thoughts.mode(),
            "rejected": len(plan.rejected) if plan else 0,
            "plan_age_s": None if self.plan_t is None else round(now - self.plan_t, 1),
            "latency_ms": self.last.get("latency_ms"),
            "tokens": self.last.get("tokens"),
            "cost_usd": self.last.get("cost_usd"),
            "cost_today_usd": round(self.budget.spent(day), 4),
            "calls_today": self.calls.get(day, 0),
            "key": self.last.get("key"),
            "gas": [{"id": g.id, "zone": g.zone, "lel": round(g.lel, 1), "db": g.acoustic_db, "trend": g.lel_trend}
                    for g in sorted(snap.gas, key=lambda g: -g.lel)[:TOP_GAS]],
            "storm_level": round(snap.storm_level, 2),
            "link": "CUT" if self.link_cut else "UP",
            "agent": self.agent.summary() if self.agent is not None else None,
        }


def from_env(mode: str = "auto", log_path="runs/control/calls.jsonl", sync: bool = False, port=None,
             **kw) -> ControlCentre:
    from policy.control.agent import ArgonAgentBackend, OrchestrationAgent, ReplayAgentBackend
    from policy.control.backend import ArgonBackend, ReplayBackend
    live = ArgonBackend()
    if mode == "live" or (mode == "auto" and live.available):
        backend, planner = live, ArgonAgentBackend()
    elif mode == "edge":
        backend = planner = None
    else:
        backend, planner = ReplayBackend(), ReplayAgentBackend()
    if port is not None:
        kw["agent"] = OrchestrationAgent(port, planner, log_path=Path(log_path).with_name("agent_calls.jsonl"))
    kw.setdefault("cap_usd_day", float(os.environ.get("CONTROL_CAP_USD", "5")))
    return ControlCentre(backend, log_path=log_path, sync=sync, **kw)

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import fields, replace
from pathlib import Path
from typing import Optional

import numpy as np

from policy.control.snapshot import GasSensor, Robot, SensorSnapshot, Verdict

CAPTIONS = {
    "PATROL": "patrolling the pipe rack", "DETECT": "inspecting the pipe for rust", "SECOND_VIEW": "taking a second view",
    "APPROACH": "moving into reach of the rust patch", "SPRAY": "spraying rust remover", "SOAK": "waiting for the remover to soak",
    "SCRUB": "scrubbing the rust patch", "BACK_OFF": "backing off after a force spike", "VERIFY": "checking the scrubbed patch",
    "HOLD": "holding: view blocked", "PAUSE_PERSON": "paused: person nearby", "PAINT": "painting the repaired patch",
    "VERIFY_PAINT": "checking paint coverage", "DONE": "repair complete", "ESCALATE_TO_HUMAN": "stopped, waiting for a human",
    "GAS_CHECK": "checking gas", "GAS_RETREAT": "retreating from gas", "RETORQUE": "re-torquing a flange bolt",
}
LEL_PER_FRACTION = 100.0


def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _person(events: list[dict]) -> Optional[float]:
    for e in reversed(events):
        p = (e.get("evidence") or {}).get("person_m")
        if p is not None:
            return _f(p)
    return None


def _cap(v) -> str:
    return (v.get("text") or "") if isinstance(v, dict) else (v or "")


def _robot(d: dict) -> Robot:
    d = dict(d)
    d["caption"] = _cap(d.get("caption"))
    p = d.get("pose")
    if isinstance(p, dict):
        d["pose"] = [p[k] for k in ("x", "y", "z")] if {"x", "y", "z"} <= set(p) else None
    if d.get("pose") is None and d.get("world_pos") is not None:
        d["pose"] = d["world_pos"]
    if isinstance(d.get("battery"), dict):
        d["battery"] = d["battery"].get("soc")
    return Robot.from_dict(d)


def from_header(h: dict, robot_id: Optional[str] = None, zone: str = "") -> SensorSnapshot:
    s = SensorSnapshot(t=float(h.get("t") or 0.0), sources=["hud"])
    events = [dict(e) for e in h.get("events") or []]
    if robot_id is not None and h.get("state"):
        st = h["state"]
        rust = _f(h.get("rust_fraction"))
        s.robots.append(Robot(robot_id, state=st, caption=_cap(h.get("caption")) or CAPTIONS.get(st, st.lower()),
                              zone=zone, rust_conf=_f(h.get("detector_conf")),
                              rust_pct=None if rust is None else round(rust * 100.0, 2), force_n=_f(h.get("force_n")),
                              battery=_f(h.get("battery")), person_m=_f(h.get("person_m")) or _person(events),
                              task=h.get("task")))
        for e in events:
            e.setdefault("robot", robot_id)
        g = h.get("gemini")
        if g:
            s.verdicts.append(Verdict(robot_id, g.get("verdict", "cannot_tell"), float(g.get("confidence") or 0.0),
                                      g.get("reason", "")))
        if (h.get("inspection") or {}).get("jpg"):
            s.images.append(h["inspection"]["jpg"])
    s.events = events
    entries = [_robot(r) for r in h.get("robots") or []]
    if s.robots and entries and (entries[0].id == robot_id or "caption" in h):
        head = entries.pop(0)
        s.robots[0].pose = head.pose or s.robots[0].pose
        s.robots[0].caption = _cap(h.get("caption")) or head.caption or s.robots[0].caption
    s.robots += entries
    captions = h.get("captions") or {}
    for r in s.robots:
        r.caption = captions.get(r.id, r.caption)
    gas = h.get("gas") or []
    plume = h.get("plume") or {}
    s.plume_kg_h, s.plume_source = _f(plume.get("flow_kg_h")), plume.get("source")
    s.wind_deg = _f(h.get("wind_deg"))
    if isinstance(gas, dict):
        db = _f((gas.get("acoustic") or {}).get("db"))
        s.gas = [GasSensor(g["id"], zone=g.get("at", ""), lel=round((_f(g.get("lel")) or 0.0) * LEL_PER_FRACTION, 2),
                           acoustic_db=db) for g in gas.get("sensors") or []]
        src = gas.get("source")
        if src and s.plume_source is None:
            s.plume_kg_h = _f(gas.get("flow_kg_h"))
            s.plume_source = gas.get("flange") or f"x={src['x']:.1f} y={src['y']:.1f} +/-{src['sigma_m']:.1f} m"
        if s.wind_deg is None:
            s.wind_deg = _f((gas.get("wind") or {}).get("dir_deg"))
    else:
        s.gas = [GasSensor.from_dict(g) for g in gas]
    s.storm_level = float(h.get("storm_level") or 0.0)
    s.jobs = [dict(j) for j in h.get("tasks") or h.get("jobs") or []]
    s.verdicts += [Verdict.from_dict(v) for v in h.get("verdicts") or []]
    return s


def from_leak(bridge, robot_id: str = "go2-1", sensor_offset: float = 0.5) -> SensorSnapshot:
    w = bridge.leak_world
    gas = []
    for fid, st in w.flanges.items():
        p = st.geom.center + sensor_offset * st.geom.normal
        gas.append(GasSensor(f"G-{fid}", zone=fid, lel=round(w.gas_at(p) * LEL_PER_FRACTION, 2),
                             acoustic_db=round(38.0 + 45.0 * st.level, 1)))
    return SensorSnapshot(t=float(w.t), gas=gas, sources=["leak_mock"])


def from_file(path, robot_id: str = "go2-1", zone: str = "") -> Optional[SensorSnapshot]:
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return None
    last = text.splitlines()[-1] if p.suffix == ".jsonl" else text
    s = from_header(json.loads(last), robot_id, zone)
    s.sources = [p.name]
    return s


def _merge_obj(a, b):
    kw = {f.name: getattr(b, f.name) for f in fields(b) if getattr(b, f.name) not in (None, "", [])}
    return replace(a, **kw)


def merge(*snaps: Optional[SensorSnapshot]) -> SensorSnapshot:
    out = SensorSnapshot()
    for s in snaps:
        if s is None:
            continue
        out.t = max(out.t, s.t)
        for attr in ("robots", "gas"):
            cur = {x.id: x for x in getattr(out, attr)}
            for x in getattr(s, attr):
                cur[x.id] = _merge_obj(cur[x.id], x) if x.id in cur else x
            setattr(out, attr, list(cur.values()))
        jobs = {j.get("id"): j for j in out.jobs}
        jobs.update({j.get("id"): j for j in s.jobs})
        out.jobs = list(jobs.values())
        out.verdicts += s.verdicts
        out.events += s.events
        out.images = s.images or out.images
        out.sources += [x for x in s.sources if x not in out.sources]
        out.storm_level = max(out.storm_level, s.storm_level)
        for k in ("plume_kg_h", "plume_source", "wind_deg"):
            if getattr(s, k) is not None:
                setattr(out, k, getattr(s, k))
    return out


class TrendTracker:
    def __init__(self, window_s: float = 8.0):
        self.window_s = window_s
        self.hist: dict[str, deque] = defaultdict(deque)

    def _slope(self, pts) -> float:
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1e-6:
            return 0.0
        t = np.array([p[0] for p in pts])
        v = np.array([p[1] for p in pts])
        return float(np.polyfit(t, v, 1)[0] * 60.0)

    def update(self, snap: SensorSnapshot) -> SensorSnapshot:
        for g in snap.gas:
            h = self.hist[g.id]
            h.append((snap.t, g.lel, g.acoustic_db))
            while h and snap.t - h[0][0] > self.window_s:
                h.popleft()
            g.lel_trend = round(self._slope([(t, v) for t, v, _ in h]), 2)
            g.db_trend = round(self._slope([(t, d) for t, _, d in h if d is not None]), 2)
        return snap

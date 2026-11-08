"""One gas-leak incident on the real fleet executors: two robot dogs, fixed gas detectors, a human approval.

    python -m fleet.incident [--leak-at T] [--approve-after S] [--out runs/fleet] [--ut]

A leak starts at a flange near valve_05 at T. The fixed point gas detectors around the top-deck pipework
(`fleet/sensors.py`) read the modelled plume; the first one over its 10 % LEL threshold raises the alarm. The
coordinator sends Dog-2 (Q2, gas sensor) on an urgent `gas_survey` from the walkway beside that detector: box
search, plume climb, source fit with a 95 % radius. Its finding queues a SUPV `valve_isolate` for the nearest
valve a robot can stand at; it waits in the coordinator's approval queue until the scripted operator approves
it S seconds later. Dog-1 (Q1, arm) closes the valve, its finding queues a confirming survey (a dog), and the
confirm's follow-up `leak_repair_prep` (ASSIST) is prepared by a dog and handed to the crew.

No drone and no crawler in the default run. `--ut` adds the crawler (C1) for a UT scan of the spool after the
confirm (the old PR #20 ending); the drone is not used.

Every incident job runs the executor registered in `fleet.jobs` (`gas_valves.py`); only kinds no module
registers (here `leak_repair_prep`) use `fleet.demo` stubs, and the summary lists them.

Outputs in --out:
    timeline.jsonl            FleetSim timeline, real time, the format sim/fleet_replay.py reads
    incident/summary.json     times leak->detect->localised->approval->isolated->confirmed, gas before/after
    incident/timeline_render.jsonl   same schema, time-compressed and filtered to the incident for the video
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from fleet.contract import REPO, Job, JobStatus, Level, Robot, RobotKind, World
from fleet.coordinator import Coordinator
from fleet.demo import RIG_DOCK, Stub, load_executors
from fleet.gas.plume import GAS_BASE, true_concentration
from fleet.rigplanner import RigPlanner
from fleet.sensors import BY_ID, DetectorNetwork, quiet_fallback
from fleet.sim import TOP_DECK_Z, FleetSim, new_job

Q, C, A = RobotKind.QUADRUPED, RobotKind.CRAWLER, RobotKind.AERIAL
D = TOP_DECK_Z
OUT = REPO / "runs/fleet"
LEAK = {"id": "leak_01", "source": "flange near valve_05", "position": (21.0, -10.2, 43.4), "rate_g_s": 3.5}
STANCE_Y = -9.0                     # walkway line beside the leaking flange skid
NAMES = {"Q1": "Dog-1", "Q2": "Dog-2", "A1": "Drone", "C1": "Crawler"}
INCIDENT_KINDS = ("gas_survey", "valve_isolate", "ut_thickness")
MAX_S = 1800.0


def robots(ut: bool = False) -> list[Robot]:
    """Dog-1 (Q1) and Dog-2 (Q2): Go2 + lift + UR3. Dog-1 carries the valve tool, Dog-2 the gas sensor."""
    out = [
        Robot("Q1", Q, {"arm", "spray", "camera"}, (24.0, -4.5, D), battery=0.95, speed_mps=0.9),
        Robot("Q2", Q, {"camera", "thermal", "gas_sensor"}, (-1.0, -2.0, D), battery=0.9, speed_mps=1.0),
    ]
    if ut:
        out.append(Robot("C1", C, {"ut_probe", "camera"}, (12.0, -9.0, D), battery=0.8, speed_mps=0.3))
    return out


class WalkwaySide:
    """The gas_survey executor ends the robot at its last sample point, which can be inside the pipe skid. A dog
    stays on the walkway side of the line (y >= STANCE_Y); the samples themselves are unchanged."""

    def __init__(self, ex):
        self.ex = ex
        self.kind, self.level, self.needs, self.robot_kinds = ex.kind, ex.level, ex.needs, ex.robot_kinds

    def __getattr__(self, name):
        return getattr(self.ex, name)

    def run(self, job, robot, world):
        res = self.ex.run(job, robot, world)
        x, y, z = robot.position
        if robot.kind == Q and y < STANCE_Y:
            robot.position = (x, STANCE_Y, z)
        return res


class Incident:
    """Builds the FleetSim, runs it until the chain is done, and records the milestones."""

    def __init__(self, leak_at: float = 15.0, approve_after: float = 30.0, ut: bool = False, seed: int = 0,
                 timeline_path: Optional[Path] = None, confirm_by: str = "quadruped"):
        os.environ.setdefault("FLEET_GEMINI", "0")
        self.leak_at, self.approve_after, self.ut = float(leak_at), float(approve_after), ut
        self.confirm_by = confirm_by
        executors, self.loaded, self.stubbed = load_executors()
        self.real = {k: not isinstance(ex, Stub) for k, ex in executors.items()}
        executors["gas_survey"] = WalkwaySide(executors["gas_survey"])
        self.world = World.from_rig_meta()
        self.world.state.update(seed=seed, wind=[0.0, 0.0], gas_leaks=[])
        self.c = Coordinator(robots(ut), self.world, executors, RigPlanner(), auto_approve=False,
                             docks={"Q1": RIG_DOCK, "Q2": RIG_DOCK})
        self.detectors = DetectorNetwork()
        self.detector_alarms: list[dict] = []
        self.sim = FleetSim(self.c, timeline_path=timeline_path)
        self.m: dict = {"leak_t": self.leak_at}
        self.pending: dict[str, float] = {}
        self.valve_closed_at: dict[str, float] = {}
        self.gas_series: list[tuple[float, float]] = []
        self.ut_job: Optional[str] = None
        self.done = False
        self._wire()

    # ------------------------------------------------------------------------------------------ script
    def _wire(self):
        sim, c = self.sim, self.c
        record = c.on_event

        def watch(e):
            record(e)
            self._on_event(e)
        c.on_event = watch

        for g in ("gauge_07",):
            a = self.world.assets.get(g)
            if a:
                c.submit(new_job(c, "gauge_read", g, a["position_m"], Level.AUTO, priority=4, est_duration_s=40.0))

        def inject(s: FleetSim):
            leak = quiet_fallback({**LEAK, "t0": s.world.t})     # only the fixed detectors raise the alarm
            s.world.state["gas_leaks"].append(leak)
            s.c.emit("gas_leak_injected", None, None, position=list(leak["position"]), source=leak["source"],
                     rate_g_s=leak["rate_g_s"], leak=leak["id"])
        sim.at(self.leak_at, "gas leak", inject)

        def survey(s: FleetSim, at_xy, leak: dict) -> Job:
            # from the walkway beside the alarm; the executor box-searches 3 m around it
            return new_job(s.c, "gas_survey", leak["id"], (at_xy[0], STANCE_Y, D), Level.AUTO, needs={"gas_sensor"},
                           robot_kinds={Q}, priority=1, est_duration_s=120.0,
                           params={"zone": leak["id"], "leak": leak["id"], "standoff": 0.0})

        def on_detector(det, reading, leak):
            s = self.sim
            job = survey(s, det.position, leak)
            job.params["detector"] = det.id
            s.c.report(job, None, "gas_alarm", position=list(det.position), detector=det.id,
                       reading=round(reading, 3), unit=det.unit, source=f"fixed gas detector {det.id}")
        self.detectors.attach(sim, on_detector)

        def on_gas(s: FleetSim, robot: Optional[str], leak: dict, how: str):
            # a dog's own gas sensor passing within 6 m (FleetSim); the fixed-detector timer is off for this leak
            job = survey(s, leak["position"], leak)
            s.c.report(job, robot, how, position=[round(v, 3) for v in leak["position"]],
                       source="robot gas sensor" if robot else "fixed gas detector")
        sim.on_gas = on_gas

        def operator(s: FleetSim):
            for jid, due in list(self.pending.items()):
                if s.world.t >= due:
                    self.pending.pop(jid)
                    s.c.approve(jid, by="operator")
            s.at(s.world.t + 1, "operator", operator)
        sim.at(0, "operator", operator)

    def _job(self, jid):
        return self.c.jobs.get(jid) if jid else None

    def _first(self, key, t):
        self.m.setdefault(key, t)

    def _on_event(self, e: dict):
        t, ev, job = e["t"], e["event"], self._job(e["job"])
        d = e["detail"]
        kind = job.kind if job else None
        confirm = bool(job and job.params.get("mode") == "confirm")
        if ev == "job_queued" and confirm and self.confirm_by == "quadruped":
            # independent check: a quadruped, not the drone that made the estimate, re-samples the plume
            job.robot_kinds = {Q}
        if ev == "job_queued" and job is not None and (confirm or kind == "leak_repair_prep"):
            # stand on the walkway beside the flange skid, not at the source estimate (inside the skid model);
            # the executor samples at its own probe points / estimate, which are in params
            job.position = (job.position[0], STANCE_Y, job.position[2])
        if ev == "detector_alarm":
            self.detector_alarms.append({"detector": d["detector"], "t": t, "reading": d["reading"],
                                         "unit": d["unit"], "position": d["position"]})
        if ev in ("gas_alarm", "gas_detected"):
            self._first("detect_t", t)
            self.m.setdefault("detect_how", ev)
            self.m.setdefault("detect_by", e["robot"] or d.get("detector") or "fixed gas detector")
        elif ev == "approval_requested" and kind == "valve_isolate":
            self.pending[e["job"]] = t + self.approve_after
            self._first("approval_requested_t", t)
            self.m.setdefault("valve", job.target)
        elif ev == "approved" and kind == "valve_isolate":
            self._first("approved_t", t)
            self.m.setdefault("approved_by", d.get("by"))
        elif ev == "assigned" and kind:
            self.m.setdefault("robots", {}).setdefault(self._role(job), e["robot"])
        elif ev == "job_started" and kind:
            self._first(f"{self._role(job)}_start_t", t)
            if kind == "valve_isolate":
                vs = self.world.state.get("valves", {}).get(job.target, {})
                self.m["actuated_before_approval"] = bool(vs.get("closed")) and "approved_t" not in self.m
        elif ev == "finding" and kind:
            role = self._role(job)
            if d.get("kind") == "gas_leak":
                ev_ = self.c.results[job.id].findings[0].evidence
                truth = np.asarray(LEAK["position"][:2])
                est = np.asarray(ev_["estimate_xy"], float)
                self.m.update(localised_t=t, estimate_xy=[round(float(v), 2) for v in est],
                              radius95_m=ev_["radius_m"], error_m=round(float(np.linalg.norm(est - truth)), 3),
                              truth_inside_radius=bool(np.linalg.norm(est - truth) <= ev_["radius_m"]),
                              peak_lel=ev_["peak_lel"], n_samples=ev_["n_samples"],
                              valves_skipped=ev_.get("valves_skipped", []), used_gemini=ev_.get("used_gemini"),
                              gas_at_source_before_lel=round(self._gas_at_source(), 4))
            elif d.get("kind") == "valve_closed":
                self.m["isolated_t"] = t
                self.m["torque_peak_nm"] = self.c.results[job.id].findings[0].evidence.get("torque_peak_nm")
                self.m["gas_at_source_after_close_lel"] = round(self._gas_at_source(), 4)
            elif d.get("kind") in ("leak_isolated", "isolation_ineffective"):
                ev_ = self.c.results[job.id].findings[0].evidence
                self.m.update(confirmed_t=t, confirm_result=d["kind"], excess_before_lel=ev_["excess_before_lel"],
                              excess_after_lel=ev_["excess_now_lel"], confirm_ratio=ev_["ratio"],
                              gas_at_source_after_lel=round(self._gas_at_source(), 4))
                if self.ut and d["kind"] == "leak_isolated" and self.ut_job is None:
                    self._queue_ut(job)
            elif role == "ut":
                self.m["ut_finding"] = d.get("summary")
        elif ev in ("job_done", "handed_to_crew") and kind:
            role = self._role(job)
            self._first(f"{role}_done_t", t)
            if role == "ut":
                res = self.c.results[job.id]
                self.m["ut_status"] = res.status.value
                self.m["ut_log"] = res.log[-1] if res.log else ""
                self.done = True
            if role == "prep" and not self.ut:
                self.done = True

    def _role(self, job: Job) -> str:
        if job.kind == "gas_survey":
            return "confirm" if job.params.get("mode") == "confirm" else "survey"
        return {"valve_isolate": "isolate", "ut_thickness": "ut", "leak_repair_prep": "prep"}.get(job.kind, job.kind)

    def _first_alarm(self) -> Optional[dict]:
        det = self.m.get("detect_by")
        a = next((x for x in self.detector_alarms if x["detector"] == det), None)
        if a is None:
            return None
        return {**a, "type": BY_ID[det].type, "threshold": BY_ID[det].threshold,
                "after_leak_s": round(a["t"] - self.leak_at, 1)}

    def _gas_at_source(self) -> float:
        return float(true_concentration(self.world.state, LEAK["position"][:2]))

    def _queue_ut(self, parent: Job):
        p = LEAK["position"]
        job = new_job(self.c, "ut_thickness", "flange_valve_05_spool", (p[0] - 1.5, STANCE_Y, D), Level.AUTO,
                      needs={"ut_probe"}, robot_kinds={C}, priority=3, est_duration_s=150.0,
                      params={"grid_m": 0.15, "region": {"theta_deg": 90.0, "z_m": 0.6, "half_m": 0.3},
                              "seed": 3, "standoff": 2.0, "zone": "flange_valve_05"})
        job.parent = parent.id
        self.ut_job = job.id
        self.c.submit(job)

    # -------------------------------------------------------------------------------------------- run
    def run(self, max_s: float = MAX_S, tail_s: float = 8.0) -> dict:
        end_t = None
        while self.world.t < max_s:
            self.sim.step(1.0)
            self.gas_series.append((self.world.t, round(self._gas_at_source(), 4)))
            if self.done and end_t is None:
                end_t = self.world.t + tail_s
            if end_t is not None and self.world.t >= end_t:
                break
        self.sim.write()
        return self.summary()

    def summary(self) -> dict:
        m = dict(self.m)
        L = m["leak_t"]
        vs = self.world.state.get("valves", {}).get(m.get("valve"), {})
        if vs.get("closed_t") is not None:
            m["valve_turned_t"] = vs["closed_t"]          # the executor cuts the feed when the job starts
        rel = lambda k: round(m[k] - L, 1) if k in m else None
        rows = {
            "leak_to_detect_s": rel("detect_t"), "leak_to_localised_s": rel("localised_t"),
            "leak_to_approval_requested_s": rel("approval_requested_t"), "leak_to_approved_s": rel("approved_t"),
            "leak_to_valve_turned_s": rel("valve_turned_t"), "leak_to_isolated_s": rel("isolated_t"), "leak_to_confirmed_s": rel("confirmed_t"),
            "leak_to_crew_handover_s": rel("prep_done_t"), "leak_to_ut_done_s": rel("ut_done_t"),
            "approval_wait_s": round(m["approved_t"] - m["approval_requested_t"], 1)
            if "approved_t" in m and "approval_requested_t" in m else None,
        }
        roles = {k: v for k, v in m.get("robots", {}).items() if k in ("survey", "isolate", "confirm", "prep", "ut")}
        m["robots"] = roles
        involved = sorted({r for r in roles.values() if r})
        return {
            "scenario": "gas leak at a flange near valve_05; fixed detector alarm, Dog-2 localises, Dog-1 isolates "
                        "(SUPV, operator approval), Dog-2 confirms, repair prep to the crew"
                        + (", crawler UT on the spool" if self.ut else ""),
            "first_alarm": self._first_alarm(),
            "detector_alarms": self.detector_alarms,
            "times_s": rows, "milestones_t": {k: v for k, v in m.items() if k.endswith("_t")},
            "localisation": {k: m.get(k) for k in ("estimate_xy", "radius95_m", "error_m", "truth_inside_radius",
                                                   "peak_lel", "n_samples", "valves_skipped")},
            "truth_xy": list(LEAK["position"][:2]),
            "valve": m.get("valve"), "approved_by": m.get("approved_by"),
            "actuated_before_approval": m.get("actuated_before_approval", False),
            "gas_lel": {"at_source_before": m.get("gas_at_source_before_lel"),
                        "at_source_after": m.get("gas_at_source_after_lel"),
                        "confirm_excess_before": m.get("excess_before_lel"),
                        "confirm_excess_after": m.get("excess_after_lel"), "confirm_ratio": m.get("confirm_ratio"),
                        "background": GAS_BASE},
            "confirm_result": m.get("confirm_result"), "torque_peak_nm": m.get("torque_peak_nm"),
            "ut": {"status": m.get("ut_status"), "log": m.get("ut_log"), "finding": m.get("ut_finding")},
            "robots_by_role": m.get("robots", {}), "robots_involved": involved,
            "names": {r: NAMES[r] for r in involved},
            "executors": {k: ("real" if self.real.get(k) else "stub") for k in
                          sorted({j.kind for j in self.c.jobs.values()})},
            "used_gemini": bool(m.get("used_gemini")),
            "approve_after_s": self.approve_after, "end_t": self.world.t,
            "status": {jid: s.value for jid, s in self.c.status.items()},
        }


# ------------------------------------------------------------------------------------- render timeline
RENDER_EVENTS = {"gas_leak_injected", "detector_alarm", "detector_clear", "gas_alarm", "gas_detected", "assigned",
                 "job_started", "approval_requested", "approved", "finding", "job_done", "handed_to_crew", "preempted",
                 "gas_isolated"}
DETAIL_KEYS = ("position", "kind", "by", "source", "rate_g_s", "detector", "reading", "unit", "type")


def _label(job_id: Optional[str], jobs: dict) -> Optional[str]:
    j = jobs.get(job_id) if job_id else None
    if j is None:
        return job_id
    kind = "gas_confirm" if j.kind == "gas_survey" and j.params.get("mode") == "confirm" else j.kind
    return f"{kind} {j.target}"


def _summary_line(e: dict, inc: Incident, true_t: float) -> Optional[str]:
    ev, d, s = e["event"], e["detail"], inc.summary()
    L = inc.leak_at
    T = f"T+{true_t - L:.0f}s" if true_t >= L else f"T{true_t - L:.0f}s"
    loc = s["localisation"]
    if ev == "gas_leak_injected":
        return f"{T} leak starts: {d.get('source')}, {d.get('rate_g_s')} g/s"
    if ev in ("detector_alarm", "detector_clear"):
        return f"{T} {d.get('summary')}"
    if ev == "gas_alarm":
        return f"{T} {d.get('detector', 'fixed detector')} raises gas alarm: survey queued"
    if ev == "gas_detected":
        return f"{T} {d.get('source', 'gas sensor')} alarm"
    if ev == "approval_requested":
        return f"{T} awaiting operator"
    if ev == "approved":
        return f"{T} operator approved ({s['times_s']['approval_wait_s']:.0f}s wait)"
    if ev == "gas_isolated":
        return f"{T} source {s['gas_lel']['at_source_after']:.2f} LEL"
    if ev == "finding":
        k = d.get("kind")
        if k == "gas_leak":
            return f"{T} at ({loc['estimate_xy'][0]:.1f},{loc['estimate_xy'][1]:.1f}) r95 {loc['radius95_m']:.2f}m"
        if k == "valve_closed":
            return f"{T} closed at {s['torque_peak_nm']:.1f} N.m"
        if k == "leak_isolated":
            return f"{T} gas {s['gas_lel']['confirm_ratio']:.0%} of before: isolated"
        return f"{T} {k}"
    if ev == "job_done":
        return f"{T} done"
    if ev == "handed_to_crew":
        return f"{T} repair prep done, handed to the crew"
    return T


def survey_path(inc: Incident, rows: list[dict]) -> Optional[tuple[float, float, list]]:
    """The surveying dog stands still in the coordinator log while the executor samples (the executor computes
    the whole survey when the job starts). For the video only, walk it through the survey's eight strongest logged
    sample points (nearest-neighbour order, held on the walkway side y >= STANCE_Y) to its final position."""
    t0, t1 = inc.m.get("survey_start_t"), inc.m.get("survey_done_t")
    iso = next((j for j in inc.c.jobs.values() if j.kind == "valve_isolate"), None)
    if t0 is None or t1 is None or iso is None:
        return None
    pts = [[p[0], max(p[1], STANCE_Y)] for p in iso.params.get("probe_points", [])]
    at = {r["t"]: r for r in rows}
    if t0 - 1 not in at or t1 not in at:
        return None
    start, end = at[t0 - 1]["position"], at[t1]["position"]
    z = end[2]
    path, cur = [start[:2]], list(start[:2])
    while pts:
        k = min(range(len(pts)), key=lambda i: math.dist(cur, pts[i]))
        cur = pts.pop(k)
        path.append(cur)
    path.append(end[:2])
    return t0 - 1, t1, [(x, y, z) for x, y in path]


def _along(path: list, u: float) -> list[float]:
    seg = [math.dist(a, b) for a, b in zip(path, path[1:])]
    s, total = u * sum(seg), sum(seg)
    if total <= 0:
        return list(path[-1])
    for (a, b), L in zip(zip(path, path[1:]), seg):
        if s <= L:
            f = s / L if L else 0.0
            return [a[i] + (b[i] - a[i]) * f for i in range(3)]
        s -= L
    return list(path[-1])


def warp(true_t: np.ndarray, key_t: list[float], moving: np.ndarray, slow: float = 1.5, move: float = 6.0,
         idle: float = 24.0, window: float = 3.0) -> np.ndarray:
    """Render time for each true second: real speed near key events, `move`x while a robot moves, `idle`x else."""
    rate = np.where(moving, move, idle).astype(float)
    for k in key_t:
        rate[np.abs(true_t - k) <= window] = slow
    dt = np.diff(true_t, prepend=true_t[0])
    return np.cumsum(dt / rate)


def render_timeline(inc: Incident, start_t: Optional[float] = None, step: float = 0.25) -> tuple[list[dict], dict]:
    """The incident as a time-compressed timeline in fleet_replay's schema (robot ids kept; true time in the
    event summaries). Robot lines every `step` render seconds, linearly interpolated from the 1 Hz samples."""
    lines, jobs = inc.sim.lines, inc.c.jobs
    robots_ = [l for l in lines if "event" not in l]
    events = [l for l in lines if "event" in l]
    t0 = max(0.0, inc.leak_at - 6.0) if start_t is None else start_t
    t_end = max(l["t"] for l in robots_)
    true_t = np.arange(t0, t_end + 1e-9, 1.0)
    by: dict[str, list[dict]] = {}
    for l in robots_:
        by.setdefault(l["robot"], []).append(l)
    moving = np.zeros(len(true_t), bool)
    for rid, rows in by.items():
        st = {r["t"]: r["state"] for r in rows}
        moving |= np.array([st.get(float(t), "") == "MOVING" for t in true_t])
    if "survey_start_t" in inc.m and "survey_done_t" in inc.m:
        moving |= (true_t >= inc.m["survey_start_t"]) & (true_t <= inc.m["survey_done_t"])
    # the isolation shows when the robot finishes the turn (the executor cuts the feed when it starts)
    extra = []
    iso_done = inc.m.get("isolate_done_t")
    if iso_done is not None:
        jid = next(j for j, job in jobs.items() if job.kind == "valve_isolate")
        extra.append({"t": iso_done, "event": "gas_isolated", "job": jid, "robot": inc.c.results[jid].robot_id,
                      "detail": {"position": list(LEAK["position"])}})
        # same for the detectors: a clear logged while the valve job is still running is shown when it finishes
        for k, e in enumerate(events):
            if e["event"] == "detector_clear" and e["t"] < iso_done:
                events[k] = {**e, "t": iso_done + 1.0}
    keep = [e for e in events + extra if e["event"] in RENDER_EVENTS and e["t"] >= t0
            and (e["event"] != "assigned" or e["job"] in jobs and jobs[e["job"]].kind != "gauge_read")
            and (e["event"] not in ("job_started", "job_done") or e["job"] in jobs
                 and jobs[e["job"]].kind != "gauge_read")]
    key_t = [e["t"] for e in keep if e["event"] in ("gas_leak_injected", "detector_alarm", "gas_alarm", "finding",
                                                       "approved",
                                                       "approval_requested", "gas_isolated", "job_started")]
    rt = warp(true_t, key_t, moving)

    def to_render(t):
        return float(np.interp(t, true_t, rt))

    out = []
    sp = survey_path(inc, by.get(inc.m.get("robots", {}).get("survey"), []))
    for rid, rows in sorted(by.items()):
        ts = np.array([r["t"] for r in rows], float)
        P = np.array([r["position"] for r in rows], float)
        Y = np.unwrap(np.array([r["yaw"] for r in rows], float))
        prev = None
        for r_t in np.arange(0.0, rt[-1] + 1e-9, step):
            tt = float(np.interp(r_t, rt, true_t))
            i = int(np.clip(np.searchsorted(ts, tt, side="right") - 1, 0, len(ts) - 1))
            pos = [round(float(np.interp(tt, ts, P[:, k])), 3) for k in range(3)]
            yaw = float(np.interp(tt, ts, Y))
            if sp and rid == inc.m["robots"].get("survey") and sp[0] <= tt <= sp[1]:
                pos = [round(v, 3) for v in _along(sp[2], (tt - sp[0]) / (sp[1] - sp[0]))]
                if prev is not None and math.dist(prev[:2], pos[:2]) > 1e-3:
                    yaw = math.atan2(pos[1] - prev[1], pos[0] - prev[0])
                elif out and out[-1]["robot"] == rid:
                    yaw = out[-1]["yaw"]
            prev = pos
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            out.append({"t": round(float(r_t), 3), "robot": rid, "kind": rows[i]["kind"], "position": pos,
                        "yaw": round(yaw, 4), "state": rows[i]["state"], "job": _label(rows[i]["job"], jobs)})
    for e in keep:
        d = {k: v for k, v in e["detail"].items() if k in DETAIL_KEYS}
        note = _summary_line(e, inc, e["t"])
        if note:
            d["summary"] = note
        d["true_t"] = e["t"]
        out.append({"t": round(to_render(e["t"]), 3), "event": e["event"], "job": _label(e["job"], jobs),
                    "robot": e["robot"], "detail": d})
    out.sort(key=lambda r: (r["t"], "event" in r))
    info = {"true_start_t": t0, "true_end_t": t_end, "render_s": round(float(rt[-1]), 1),
            "speedup_overall": round((t_end - t0) / max(float(rt[-1]), 1e-6), 2),
            "rates": "1.5x within 3 s of a key event, 6x while any robot moves or samples, 24x otherwise",
            "step_s": step, "survey_path": "Dog-2 drawn through the survey's 8 strongest sample points, "
                                             "held on the walkway side of the line"}
    return out, info


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="multi-robot gas-leak incident on the real fleet executors")
    ap.add_argument("--leak-at", type=float, default=15.0, help="sim second the leak starts")
    ap.add_argument("--approve-after", type=float, default=30.0, help="operator approves the SUPV isolation after S s")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ut", action="store_true", help="add the crawler (C1) and a UT scan of the spool (off: dogs only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm-by", choices=["quadruped", "any"], default="quadruped",
                    help="who re-samples after isolation: a dog (default) or any robot with a gas sensor")
    args = ap.parse_args(argv)
    out = Path(args.out)
    inc = Incident(args.leak_at, args.approve_after, ut=args.ut, seed=args.seed,
                   timeline_path=out / "timeline.jsonl", confirm_by=args.confirm_by)
    s = inc.run()
    rows, info = render_timeline(inc)
    s["render"] = info
    write_jsonl(out / "incident/timeline_render.jsonl", rows)
    (out / "incident").mkdir(parents=True, exist_ok=True)
    (out / "incident/summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print("INCIDENT (sim seconds after the leak starts)")
    fa = s["first_alarm"] or {}
    print(f"  first alarm {fa.get('detector')} at {fa.get('position')} reading {fa.get('reading')} {fa.get('unit')} "
          f"T+{fa.get('after_leak_s')}s; detector alarms "
          + ", ".join(f"{a['detector']}@T+{a['t'] - inc.leak_at:.0f}s" for a in s["detector_alarms"]))
    for k, v in s["times_s"].items():
        print(f"  {k:<30} {v}")
    loc = s["localisation"]
    print(f"  estimate {loc['estimate_xy']} vs truth {s['truth_xy']}: error {loc['error_m']} m, 95% radius "
          f"{loc['radius95_m']} m, inside={loc['truth_inside_radius']}")
    print(f"  valve {s['valve']} approved by {s['approved_by']}; gas at source {s['gas_lel']['at_source_before']} -> "
          f"{s['gas_lel']['at_source_after']} LEL; confirm {s['confirm_result']} ratio {s['gas_lel']['confirm_ratio']}")
    print(f"  robots {s['robots_by_role']}; executors {s['executors']}")
    print(f"  ut: {s['ut']}")
    print(f"timeline {out / 'timeline.jsonl'} ({len(inc.sim.lines)} lines); render timeline "
          f"{out / 'incident/timeline_render.jsonl'} ({info['render_s']} s, x{info['speedup_overall']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

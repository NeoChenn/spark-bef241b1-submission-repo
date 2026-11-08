"""Gas survey and valve, filter and panel operations for the quadruped (Go2 + lift + UR3).

    gas_survey         Q + gas_sensor, AUTO  round -> detect -> climb the plume -> fit the source with a 95%
                                             radius -> critical Finding + follow-up valve_isolate (SUPV).
                                             mode="confirm": re-sample the plume after isolation; if the gas has
                                             not fallen, try the next upstream valve.
    valve_isolate      Q + arm, SUPV         approval -> gas and person check -> IK reach check -> torque ramp vs
                                             limit -> quarter turn -> follow-up confirming gas_survey.
                                             Out of reach, over torque or in gas: refuse, hand to a human.
    filter_changeover  Q + arm, SUPV         high dP -> approval -> reach -> move the duplex lever -> verify dP fell.
    panel_button       Q + arm, SUPV         pump_start or alarm_ack only. ESD reset (or any reset, bypass,
                                             override, inhibit, shutdown button) is refused and never pressed.

Approval: the coordinator (PR #7) holds a SUPV job and calls `run` only after `approve(job_id)`. As a second
gate, a job whose `params["approved"]` (or `params["approval"]["approved"]`) is False actuates nothing and returns
WAITING_APPROVAL with the approval request (`executor.approval_request(job, world)`). Follow-up jobs carry
`params["approval_request"]` (Gemini-written) for the operator.

Simulated truth (World.state): "gas_leaks" (coordinator format: id, position, rate_g_s, t0, source, isolated),
"wind", "process_flow_dir", "valves", "filters", "panel"; see fleet/jobs/GAS_VALVES.md.
Gemini writes the leak report and the approval text (fleet/gas/report.py), with a template fallback.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Optional

import numpy as np

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register
from fleet.gas import safety
from fleet.gas.localize import Seeker
from fleet.gas.plume import GAS_BASE, GasSensor, shape
from fleet.gas.reach import check_reach, deck_below, facing_of, patrol_route, rig_meta
from fleet.gas.report import Gemini, approval_template, as_approval, write_leak_messages

LIMITS = safety.SafetyLimits()
PUSH_FORCE_N = 38.0          # arm force cap, same as the scrub-v2 policy's 38 N setpoint cap
DEFAULT_LEVER_M = 0.25
DEFAULT_BREAKAWAY_NM = 6.0
TORQUE_STEP_NM = 0.25
LOCAL_BOX_M = 3.0            # half-side of the box search around a gas alarm position
CONFIRM_DROP = 0.25         # isolation confirmed when the plume excess falls below 25% of before
ALLOWED_BUTTONS = {"pump_start", "alarm_ack"}
FORBIDDEN_BUTTON = re.compile(r"esd|reset|shut|bypass|override|inhibit|trip|emergency", re.I)
QUAD = {RobotKind.QUADRUPED}


class RefusedButton(RuntimeError):
    pass


def is_approved(job: Job) -> bool:
    """True only when the operator approved: the coordinator sets params["approved"] in approve()."""
    a = job.params.get("approval")
    return job.params.get("approved") is True or (isinstance(a, dict) and a.get("approved") is True)


def _rng(job: Job, world: World) -> np.random.Generator:
    seed = int(job.params.get("seed", world.state.get("seed", 0)))
    return np.random.default_rng([seed, zlib.crc32(job.id.encode())])


def _result(job, robot, world, status, elapsed_s=0.0, findings=None, log=None, used_gemini=False) -> JobResult:
    return JobResult(job.id, robot.id, status, world.t, world.t + float(elapsed_s), findings or [], log or [],
                     used_gemini)


def _missing(ex, robot: Robot) -> Optional[str]:
    lacking = set(ex.needs) - set(robot.capabilities)
    if lacking:
        return f"robot {robot.id} lacks {sorted(lacking)}"
    if ex.robot_kinds and robot.kind not in ex.robot_kinds:
        return f"robot {robot.id} is {robot.kind.value}, job needs {sorted(k.value for k in ex.robot_kinds)}"
    return None


def _deck_name(z: float) -> str:
    decks = rig_meta().get("deck_heights_m") or {}
    d = deck_below(z)
    return next((k.replace("_", " ") for k, v in decks.items() if abs(v - d) < 1e-6), f"deck at {d:.1f} m")


def _waiting(ex, job, robot, world, request: dict) -> JobResult:
    f = Finding("approval_required", "warning", request["text"], tuple(job.position), {"approval_request": request})
    return _result(job, robot, world, JobStatus.WAITING_APPROVAL, 0.0, [f],
                   [f"{job.kind} {job.target}: waiting for operator approval; nothing actuated"])


def _human_job(job: Job, kind: str, why: str, priority: int = 1) -> Job:
    return Job(f"{job.id}-human-{kind}", kind, job.target, tuple(job.position), Level.HUMAN, set(), set(), priority,
               600.0, {"reason": why}, parent=job.id)


def isolate_leaks(world: World, valve_id: str, leak_label=None, estimate=None, near_m: float = 4.0) -> list[str]:
    """Simulated truth: which leaks closing `valve_id` cuts. A leak's `isolating_valves` decides when given;
    otherwise the leak named by the job (coordinator id), else leaks within `near_m` of the estimate.
    Cut leaks get isolated=True and their rate (rate_g_s, amplitude_lel) scaled by `residual` (default 0.05)."""
    cut = []
    for lk in world.state.get("gas_leaks") or []:
        if lk.get("isolated"):
            continue
        if lk.get("isolating_valves") is not None:
            hit = valve_id in lk["isolating_valves"]
        elif leak_label is not None and lk.get("id") == leak_label:
            hit = True
        else:
            hit = estimate is not None and math.dist(lk["position"][:2], list(estimate)[:2]) < near_m
        if hit:
            k = float(lk.get("residual", 0.05))
            lk["rate_g_s"] = float(lk.get("rate_g_s", 0.0)) * k
            if lk.get("amplitude_lel") is not None:
                lk["amplitude_lel"] = float(lk["amplitude_lel"]) * k
            lk.update(isolated=True, isolated_by=valve_id, isolated_t=world.t)
            cut.append(lk.get("id", "leak"))
    return cut


def _flow_dir(world: World, xy):
    """Process flow direction at a point: World.state["process_flow_dir"], else the `flow_dir` of the line
    (gas_leaks entry) nearest the point. This is piping knowledge, not the leak's position."""
    if world.state.get("process_flow_dir") is not None:
        return world.state["process_flow_dir"]
    lines = [lk for lk in world.state.get("gas_leaks") or [] if lk.get("flow_dir") is not None]
    if not lines:
        return None
    return min(lines, key=lambda lk: math.dist(lk["position"][:2], xy[:2]))["flow_dir"]


def select_isolation_valve(world: World, source_xyz, exclude=(), unsafe=None) -> tuple[Optional[dict], str, list]:
    """Nearest valve upstream of the source along the process flow; without a flow direction, the nearest valve.
    Same deck preferred. `unsafe(valve) -> str|None` skips valves the robot could not work at (e.g. in gas)."""
    s = np.asarray(source_xyz, float)
    valves = [v for v in world.of_type("valve") if v["id"] not in set(exclude)]
    skipped = []
    if unsafe is not None:
        keep = []
        for v in valves:
            why = unsafe(v)
            (skipped.append(f"{v['id']} ({why})") if why else keep.append(v))
        valves = keep
    if not valves:
        return None, "none", skipped
    same = [v for v in valves if abs(v["position_m"][2] - s[2]) < 3.0]
    valves = same or valves
    flow = _flow_dir(world, s)
    dist = lambda v: float(np.linalg.norm(np.asarray(v["position_m"], float)[:2] - s[:2]))  # noqa: E731
    if flow is not None:
        f = np.asarray(flow, float)[:2]
        up = [v for v in valves if (np.asarray(v["position_m"], float)[:2] - s[:2]) @ f < 0]
        if up:
            return min(up, key=dist), "upstream", skipped
    return min(valves, key=dist), "nearest", skipped


# ---------------------------------------------------------------------------------------------- gas_survey
class GasSurveyJob:
    kind, level, needs, robot_kinds = "gas_survey", Level.AUTO, {"gas_sensor"}, {RobotKind.QUADRUPED,
                                                                                 RobotKind.AERIAL}

    def __init__(self, gemini=None):
        self.gemini = gemini if gemini is not None else Gemini()

    @staticmethod
    def _route(job: Job, world: World) -> tuple[list, str]:
        """Explicit route; else the full patrol round for a deck-wide survey; else (a survey triggered by a gas
        alarm, as the coordinator queues it) a box search LOCAL_BOX_M around the alarm position."""
        if job.params.get("route"):
            return job.params["route"], "given route"
        decks = set(rig_meta().get("deck_heights_m") or {}) | {"round", "patrol"}
        if job.params.get("round") or job.target in decks or not any(job.position):
            route = patrol_route() or [v["position_m"] for v in world.of_type("valve")]
            return route, "patrol round"
        x, y, z = job.position
        h = LOCAL_BOX_M
        return [(x - h, y - h, z), (x + h, y - h, z), (x + h, y + h, z), (x - h, y + h, z), (x - h, y - h, z)], \
            f"box search {2 * h:.0f} m around the alarm"

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        why = _missing(self, robot)
        if why:
            return _result(job, robot, world, JobStatus.FAILED, log=[why])
        if job.params.get("mode") == "confirm":
            return self._confirm(job, robot, world)
        sensor = GasSensor(_rng(job, world))
        route, how = self._route(job, world)
        seeker = Seeker(world.state, sensor, robot.position[:2], robot.speed_mps, LIMITS)
        log = [f"{how}: {len(route)} waypoints, sample every 2 m"]
        if not seeker.round(route):
            s = seeker.s
            log.append(f"no gas above background: peak {s.peak_lel:.3f} LEL over {len(s.samples)} samples")
            return _result(job, robot, world, JobStatus.DONE, s.time_s, log=log)
        log.append(f"gas detected on the round (peak {seeker.s.peak_lel:.3f} LEL); climbing the plume")
        seeker.climb()
        seeker.fit()
        r0 = seeker.s.radius_m
        seeker.refine()
        s = seeker.s
        robot.position = (float(seeker.pos[0]), float(seeker.pos[1]), robot.position[2])
        log += s.events
        log.append(f"fit: source {s.estimate_xy[0]:.2f}, {s.estimate_xy[1]:.2f}; 95% radius {r0:.2f} -> "
                   f"{s.radius_m:.2f} m after {sum(q.phase == 'info' for q in s.samples)} information samples")
        deck_z = robot.deck if robot.deck is not None else deck_below(float(np.asarray(route[0])[2])
                                                                       if len(route[0]) > 2 else 42.5)
        est = (round(s.estimate_xy[0], 2), round(s.estimate_xy[1], 2), round(deck_z + 1.0, 2))
        wind = np.asarray(world.state.get("wind") or (0.0, 0.0), float)

        def unsafe(v):
            """Predicted gas at the valve stance from the fitted plume (1.5x amplitude margin)."""
            stance = np.asarray(v["position_m"], float)[:2] + 0.5 * facing_of(v, world.assets)
            c = GAS_BASE + 1.5 * s.amplitude * float(shape(stance[None], np.asarray(est[:2]), world.state.get("wind"),
                                                           seeker.decay_m)[0])
            return f"predicted {c:.2f} LEL at its stance" if c >= LIMITS.gas_stop_fraction_lel else None

        valve, relation, skipped = select_isolation_valve(world, est, job.params.get("tried_valves", ()), unsafe)
        if skipped:
            log.append("valves skipped (robot would stand in gas): " + ", ".join(skipped))
        leak_label = job.params.get("leak") or job.params.get("zone") or job.target
        numbers = {
            "estimate_xy": [est[0], est[1]], "radius_m": round(s.radius_m, 2), "peak_lel": round(s.peak_lel, 3),
            "n_samples": len(s.samples), "wind_mps": round(float(np.linalg.norm(wind)), 1),
            "wind_vector_mps": [round(float(v), 2) for v in wind[:2]], "deck": _deck_name(deck_z),
            "approach_limit": LIMITS.gas_approach_fraction_lel, "stop_limit": LIMITS.gas_stop_fraction_lel,
            "path_m": round(s.path_m, 1), "time_s": round(s.time_s),
        }
        follow = []
        if valve is not None:
            vxy = valve["position_m"]
            numbers.update(valve=valve["id"], valve_xy=[vxy[0], vxy[1]], valve_relation=f"{relation} of the leak",
                           valve_distance_m=round(float(np.hypot(vxy[0] - est[0], vxy[1] - est[1])), 1))
            msgs, used = write_leak_messages(numbers, self.gemini)
            request = as_approval(msgs)
            top = sorted(s.samples, key=lambda q: -q.lel)[:8]
            follow.append(Job(f"{job.id}-isolate-{valve['id']}", "valve_isolate", valve["id"],
                              tuple(valve["position_m"]), Level.SUPV, {"arm"}, set(QUAD), 1, 240.0,
                              {"approval_request": request, "leak_estimate": list(est), "leak": leak_label,
                               "zone": leak_label, "radius_m": numbers["radius_m"],
                               "probe_points": [[q.xy[0], q.xy[1], round(q.lel, 4)] for q in top],
                               "tried_valves": list(job.params.get("tried_valves", ())) + [valve["id"]]},
                              world.t, job.id))
        else:
            numbers.update(valve="(none found)", valve_xy=[est[0], est[1]], valve_relation="", valve_distance_m=0.0)
            msgs, used = write_leak_messages(numbers, self.gemini)
            follow.append(_human_job(job, "gas_isolate_manual", "no isolation valve the robot can work at"))
        numbers["valves_skipped"] = skipped
        log.append(("Gemini" if used else "template") + " wrote the leak report and approval request")
        f = Finding("gas_leak", "critical", msgs["report"], est,
                    {**numbers, "events": s.events, "radius_history_m": [round(r, 2) for r in s.radius_history],
                     "samples_by_phase": {p: sum(q.phase == p for q in s.samples)
                                          for p in ("round", "climb", "probe", "info")},
                     "fit_chi2_reduced": round(s.fit_chi2_red, 2), "amplitude_lel": round(s.amplitude, 3),
                     "messages": msgs, "used_gemini": used}, follow)
        return _result(job, robot, world, JobStatus.FINDING, s.time_s, [f], log, used)

    def _confirm(self, job: Job, robot: Robot, world: World) -> JobResult:
        sensor = GasSensor(_rng(job, world))
        pts = job.params.get("probe_points") or []
        if not pts:
            return _result(job, robot, world, JobStatus.FAILED, log=["confirm survey has no probe points"])
        seeker = Seeker(world.state, sensor, robot.position[:2], robot.speed_mps, LIMITS)
        before, now = [], []
        for x, y, lel in pts:
            seeker.goto((x, y))
            now.append(seeker.read("confirm") - GAS_BASE)
            before.append(lel - GAS_BASE)
        b, n = float(np.mean(before)), float(np.mean(now))
        ratio = n / b if b > 1e-6 else 0.0
        log = [f"confirm: plume excess {b:.3f} -> {n:.3f} LEL at {len(pts)} points (ratio {ratio:.2f})"]
        est = tuple(job.params.get("leak_estimate") or job.position)
        ev = {"excess_before_lel": round(b, 4), "excess_now_lel": round(n, 4), "ratio": round(ratio, 3)}
        label = job.params.get("leak") or job.params.get("zone") or job.target
        if ratio <= CONFIRM_DROP:
            prep = Job(f"{job.id}-prep", "leak_repair_prep", label, est, Level.ASSIST, {"camera"}, set(QUAD), 2, 60.0,
                       {"zone": label, "leak_estimate": list(est)}, world.t, job.id)
            f = Finding("leak_isolated", "info", f"Gas fell to {ratio:.0%} of the pre-isolation plume after closing "
                        f"{job.params.get('tried_valves', ['?'])[-1]}; leak isolated.", est, ev, [prep])
            return _result(job, robot, world, JobStatus.FINDING, seeker.s.time_s, [f], log)
        tried = job.params.get("tried_valves", [])
        valve, relation, _ = select_isolation_valve(world, est, tried)
        if valve is None:
            follow = [_human_job(job, "gas_isolate_manual", "every candidate valve tried; gas still present")]
        else:
            what = f"Close {valve['id']} (quarter turn): closing {tried[-1] if tried else 'the first valve'} " \
                   f"did not stop the gas."
            req = approval_template(what, f"{valve['id']} at ({valve['position_m'][0]:.1f}, "
                                          f"{valve['position_m'][1]:.1f}), {relation} of the leak at "
                                          f"({est[0]:.1f}, {est[1]:.1f})",
                                    f"plume still at {ratio:.0%} of before isolation",
                                    "the leak continues to feed a gas cloud")
            follow = [Job(f"{job.id}-isolate-{valve['id']}", "valve_isolate", valve["id"],
                          tuple(valve["position_m"]), Level.SUPV, {"arm"}, set(QUAD), 1, 240.0,
                          {"approval_request": req, "leak_estimate": list(est), "leak": label, "zone": label,
                           "probe_points": pts, "tried_valves": tried + [valve["id"]]}, world.t, job.id)]
        f = Finding("isolation_ineffective", "critical",
                    f"Gas still at {ratio:.0%} of the pre-isolation plume; isolation did not hold.", est, ev, follow)
        return _result(job, robot, world, JobStatus.FINDING, seeker.s.time_s, [f], log)


# ------------------------------------------------------------------------------------------- valve_isolate
class ValveIsolateJob:
    kind, level, needs, robot_kinds = "valve_isolate", Level.SUPV, {"arm"}, QUAD

    def approval_request(self, job: Job, world: World) -> dict:
        return job.params.get("approval_request") or approval_template(
            f"Close {job.target} (quarter turn).", f"{job.target} at {tuple(round(v, 1) for v in job.position)}",
            "isolation requested", "the line stays live")

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        why = _missing(self, robot)
        if why:
            return _result(job, robot, world, JobStatus.FAILED, log=[why])
        if not is_approved(job):
            return _waiting(self, job, robot, world, self.approval_request(job, world))
        asset = world.assets.get(job.target)
        if asset is None:
            return _result(job, robot, world, JobStatus.FAILED, log=[f"unknown valve {job.target}"])
        vs = world.state.setdefault("valves", {}).setdefault(job.target, {})
        if vs.get("closed"):
            return _result(job, robot, world, JobStatus.DONE, log=[f"{job.target} already closed"])
        log = []
        handle = np.asarray(asset["position_m"], float) + np.asarray(vs.get("handle_offset", (0, 0, 0)), float)
        facing = facing_of(asset, world.assets)
        reach = check_reach(handle, facing)
        stance = reach.stance_xy or tuple(handle[:2] + 0.5 * facing)
        gas = GasSensor(_rng(job, world)).sample(world.state, stance)
        verdict = safety.check(safety.SafetyInputs(gas, safety.nearest_person_m(world.people, stance), 1.0,
                                                   True, True), LIMITS)
        log.append(f"safety at stance {stance}: gas {gas:.3f} LEL -> {verdict.verdict.value} ({verdict.reason})")
        ev = {"gas_at_stance_lel": round(gas, 4), "reach": reach.summary()}
        if verdict.verdict == safety.Verdict.STOP_AND_RETREAT:
            f = Finding("valve_in_gas", "critical", f"Refused to close {job.target}: {verdict.reason}. Needs a "
                        "person with breathing apparatus or remote isolation.", tuple(handle), ev,
                        [_human_job(job, "valve_isolate_manual", verdict.reason)])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 30.0, [f], log)
        if verdict.verdict == safety.Verdict.PAUSE:
            return _result(job, robot, world, JobStatus.FAILED, 0.0, log=log + ["paused: person too close; retry"])
        log.append(f"reach: {reach.reason}")
        if not reach.reachable:
            f = Finding("valve_out_of_reach", "warning", f"Refused to close {job.target}: {reach.reason}.",
                        tuple(handle), ev, [_human_job(job, "valve_isolate_manual", reach.reason)])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 60.0, [f], log)
        lever = float(vs.get("lever_m", DEFAULT_LEVER_M))
        breakaway = float(vs.get("breakaway_nm", DEFAULT_BREAKAWAY_NM))
        limit = min(float(job.params.get("torque_limit_nm", math.inf)), PUSH_FORCE_N * lever)
        rng = _rng(job, world)
        applied, peak = 0.0, 0.0
        while applied < limit - 1e-9:
            applied = min(limit, applied + TORQUE_STEP_NM)
            peak = max(peak, min(applied, breakaway) + float(rng.normal(0.0, 0.05)))
            if applied >= breakaway:
                break
        ev.update(torque_limit_nm=round(limit, 2), torque_peak_nm=round(peak, 2), lever_m=lever)
        if applied < breakaway:
            msg = (f"Refused to force {job.target}: it did not move at the {limit:.1f} N.m limit "
                   f"({PUSH_FORCE_N:.0f} N on a {lever:.2f} m lever); a seized valve needs a person.")
            log.append(msg)
            f = Finding("valve_over_torque", "warning", msg, tuple(handle), ev,
                        [_human_job(job, "valve_isolate_manual", msg)])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 90.0, [f], log)
        q = dict(reach.q)
        w3 = q["ur3_wrist_3_joint"]
        turn = math.pi / 2 if w3 + math.pi / 2 <= math.pi else -math.pi / 2
        q["ur3_wrist_3_joint"] = w3 + turn
        vs.update(closed=True, closed_t=world.t, closed_by=robot.id)
        log.append(f"quarter turn: wrist_3 {w3:.2f} -> {q['ur3_wrist_3_joint']:.2f} rad at {peak:.1f} N.m "
                   f"(limit {limit:.1f})")
        est = job.params.get("leak_estimate") or list(handle)
        cut = isolate_leaks(world, job.target, job.params.get("leak") or job.params.get("zone"), est)
        log.append(f"feed cut to {cut}" if cut else "no simulated leak is fed through this valve")
        ev["arm_q"] = {k: round(v, 3) for k, v in q.items() if k.startswith("ur3_") or k == "lift_joint"}
        label = job.params.get("leak") or job.params.get("zone") or "gas_leak"
        confirm = Job(f"{job.id}-confirm", "gas_survey", label, tuple(est), Level.AUTO, {"gas_sensor"},
                      {RobotKind.QUADRUPED, RobotKind.AERIAL}, 1, 120.0,
                      {"mode": "confirm", "probe_points": job.params.get("probe_points", []), "leak": label,
                       "zone": label, "leak_estimate": est,
                       "tried_valves": job.params.get("tried_valves", [job.target])},
                      world.t, job.id)
        f = Finding("valve_closed", "info", f"Closed {job.target} (quarter turn, {peak:.1f} N.m). Confirming the "
                    "gas falls with a follow-up survey.", tuple(handle), ev, [confirm])
        return _result(job, robot, world, JobStatus.FINDING, 150.0, [f], log)


# --------------------------------------------------------------------------------------- filter_changeover
class FilterChangeoverJob:
    kind, level, needs, robot_kinds = "filter_changeover", Level.SUPV, {"arm"}, QUAD

    def approval_request(self, job: Job, world: World) -> dict:
        fs = world.state.get("filters", {}).get(job.target, {})
        dp, lim = fs.get("dp_bar", float("nan")), fs.get("dp_limit_bar", 1.0)
        standby = "B" if fs.get("active", "A") == "A" else "A"
        return approval_template(f"Move the {job.target} duplex changeover lever to element {standby}.",
                                 f"{job.target} at {tuple(round(v, 1) for v in job.position)}",
                                 f"differential pressure {dp:.2f} bar is above the {lim:.2f} bar limit",
                                 "the fouled element restricts flow and may collapse or bypass")

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        why = _missing(self, robot)
        if why:
            return _result(job, robot, world, JobStatus.FAILED, log=[why])
        fs = world.state.get("filters", {}).get(job.target)
        if fs is None:
            return _result(job, robot, world, JobStatus.FAILED, log=[f"no filter state for {job.target}"])
        rng = _rng(job, world)
        dp = float(fs["dp_bar"]) + float(rng.normal(0.0, 0.01))
        lim = float(fs.get("dp_limit_bar", 1.0))
        log = [f"dP read {dp:.2f} bar (limit {lim:.2f})"]
        if dp < lim:
            return _result(job, robot, world, JobStatus.DONE, 20.0, log=log + ["below limit: no changeover"])
        if not is_approved(job):
            return _waiting(self, job, robot, world, self.approval_request(job, world))
        active = fs.get("active", "A")
        standby = "B" if active == "A" else "A"
        if fs.get("standby_fouled"):
            f = Finding("filter_standby_fouled", "warning", f"Standby element {standby} is also fouled; changeover "
                        "would not help.", tuple(job.position), {"dp_bar": round(dp, 3)},
                        [_human_job(job, "filter_element_replace", "both elements fouled")])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 30.0, [f], log)
        reach = check_reach(job.position, job.params.get("facing") or facing_of({"position_m": job.position},
                                                                               world.assets))
        log.append(f"reach: {reach.reason}")
        if not reach.reachable:
            f = Finding("lever_out_of_reach", "warning", f"Refused: {reach.reason}.", tuple(job.position),
                        {"reach": reach.summary()}, [_human_job(job, "filter_changeover_manual", reach.reason)])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 60.0, [f], log)
        fs["active"] = standby
        fs["dp_bar"] = float(fs.get("standby_dp_bar", 0.3))
        after = fs["dp_bar"] + float(rng.normal(0.0, 0.01))
        log.append(f"lever {active} -> {standby}; dP {dp:.2f} -> {after:.2f} bar")
        ev = {"dp_before_bar": round(dp, 3), "dp_after_bar": round(after, 3), "active": standby,
              "reach": reach.summary()}
        if after >= 0.7 * lim or after >= 0.7 * dp:
            f = Finding("filter_dp_not_recovered", "warning", f"dP only fell to {after:.2f} bar after changeover.",
                        tuple(job.position), ev, [_human_job(job, "filter_inspect", "dP did not recover")])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 120.0, [f], log)
        f = Finding("filter_changed_over", "info", f"{job.target} changed over to element {standby}: dP {dp:.2f} -> "
                    f"{after:.2f} bar.", tuple(job.position), ev,
                    [_human_job(job, "filter_element_replace", f"element {active} fouled", priority=4)])
        return _result(job, robot, world, JobStatus.FINDING, 120.0, [f], log)


def filter_jobs(world: World, t: float = 0.0) -> list[Job]:
    """filter_changeover jobs for every filter in World.state whose dP is at or above its limit."""
    out = []
    for fid, fs in (world.state.get("filters") or {}).items():
        if fs.get("dp_bar", 0.0) >= fs.get("dp_limit_bar", 1.0):
            out.append(Job(f"filter-{fid}", "filter_changeover", fid, tuple(fs["position_m"]), Level.SUPV, {"arm"},
                           set(QUAD), 2, 180.0, {}, t))
    return out


# -------------------------------------------------------------------------------------------- panel_button
def press(world: World, button: str, robot_id: str) -> None:
    """The only place a button gets pressed. Refuses forbidden buttons even if a caller gets this far."""
    if FORBIDDEN_BUTTON.search(button) or button not in ALLOWED_BUTTONS:
        raise RefusedButton(button)
    world.state.setdefault("panel", {}).setdefault("pressed", []).append({"button": button, "robot": robot_id,
                                                                           "t": world.t})


class PanelButtonJob:
    kind, level, needs, robot_kinds = "panel_button", Level.SUPV, {"arm"}, QUAD

    def approval_request(self, job: Job, world: World) -> dict:
        b = job.params.get("button", "?")
        return approval_template(f"Press {b} on {job.target}.", f"{job.target} at "
                                 f"{tuple(round(v, 1) for v in job.position)}",
                                 job.params.get("reason", "requested by the operator"),
                                 "the pump stays stopped" if b == "pump_start" else "the alarm stays unacknowledged")

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        button = str(job.params.get("button", ""))
        if FORBIDDEN_BUTTON.search(button) or button not in ALLOWED_BUTTONS:
            msg = (f"Refused to press '{button}' on {job.target}: robots never reset an ESD or press reset, "
                   f"bypass or override buttons; only {sorted(ALLOWED_BUTTONS)}. Routed to the crew.")
            f = Finding("refused_button", "critical" if "esd" in button.lower() else "warning", msg,
                        tuple(job.position), {"button": button},
                        [_human_job(job, "esd_reset" if "esd" in button.lower() else "panel_manual", msg)])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 0.0, [f], [msg])
        why = _missing(self, robot)
        if why:
            return _result(job, robot, world, JobStatus.FAILED, log=[why])
        if not is_approved(job):
            return _waiting(self, job, robot, world, self.approval_request(job, world))
        panel = world.state.setdefault("panel", {})
        log = []
        if button == "pump_start":
            gas = GasSensor(_rng(job, world)).sample(world.state, job.position[:2])
            if panel.get("esd_tripped"):
                reason = "ESD is tripped: a person must investigate and reset it before any pump start"
            elif gas >= LIMITS.gas_approach_fraction_lel:
                reason = f"gas {gas:.2f} LEL at the panel: no pump start in gas"
            else:
                reason = None
            if reason:
                f = Finding("pump_start_refused", "warning", f"Refused pump start: {reason}.", tuple(job.position),
                            {"gas_lel": round(gas, 4), "esd_tripped": bool(panel.get("esd_tripped"))},
                            [_human_job(job, "panel_manual", reason)])
                return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 10.0, [f], [reason])
        reach = check_reach(job.position, job.params.get("facing") or facing_of({"position_m": job.position},
                                                                               world.assets))
        log.append(f"reach: {reach.reason}")
        if not reach.reachable:
            f = Finding("button_out_of_reach", "warning", f"Refused: {reach.reason}.", tuple(job.position),
                        {"reach": reach.summary()}, [_human_job(job, "panel_manual", reach.reason)])
            return _result(job, robot, world, JobStatus.HANDED_TO_HUMAN, 60.0, [f], log)
        press(world, button, robot.id)
        if button == "pump_start":
            panel["pump_running"] = True
            summary = f"Pressed pump_start on {job.target}; run lamp on."
        else:
            for a in panel.get("alarms", []):
                a["acknowledged"] = True
            summary = (f"Acknowledged {len(panel.get('alarms', []))} alarm(s) on {job.target}; the causes are "
                       "not cleared by acknowledging.")
        log.append(summary)
        f = Finding(f"{button}_done", "info", summary, tuple(job.position), {"reach": reach.summary()})
        return _result(job, robot, world, JobStatus.FINDING, 60.0, [f], log)


GAS_SURVEY = register(GasSurveyJob())
VALVE_ISOLATE = register(ValveIsolateJob())
FILTER_CHANGEOVER = register(FilterChangeoverJob())
PANEL_BUTTON = register(PanelButtonJob())

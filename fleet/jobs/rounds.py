"""Operator-rounds jobs for the quadruped (Go2 + lift + UR3): the walk an offshore operator does every shift.

  gauge_read               Q  AUTO    read the 7 process gauges two ways (OpenCV needle angle + Gemini vision)
  thermal_acoustic_survey  Q  AUTO    temperature / vibration per asset, trended over rounds (drift, steps)
  escape_route_check       Q  AUTO    walk the patrol / escape route, find blockages and dropped objects
  spill_response           Q  SUPV    ring a small spill with absorbent socks once the operator approves
  fg_optics_clean          Q  AUTO    wipe a salt-fouled fire & gas detector window, then ASSIST bump test

World.state keys these jobs read (all optional; synthetic defaults are made when missing):
  gauge_values      {gauge_id: 0..1}        simulated truth, overrides reading_bar from the rig meta
  gauge_frames      {gauge_id: RGB array}   real camera crops (e.g. from Isaac); else synthetic crops are rendered
  condition_history {asset_id: {"temp_c": [...], "vib_mm_s": [...]}}  one value per past round
  condition_now     {asset_id: {"temp_c": x, "vib_mm_s": y}}          this round's measurement (appended)
  dropped_objects   [{"id", "label", "position": [x, y(, z)], "size_m", "mass_kg"}]
  route_blocked_samples [[x, y], ...]  or  deck_triangles (N, 3, 3) -> sim route_blockages
  spills            {spill_id: {"position", "radius_m", "fluid"}}
  absorbent_socks   int (robot's sock inventory, default 8)
  fg_detectors      {det_id: {"position", "type", "salt": 0..1}}
Jobs write back: gauge_log, condition_history, spills[...]["contained"], fg_detectors[...]["salt"].
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Optional

import numpy as np

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register
from fleet.perception import route as route_mod
from fleet.perception.gauge import (RANGE_BAR, RED_ZONE, ClassicalGaugeReader, CropParams, GaugeReading,
                                    GeminiGaugeReader, render_gauge_crop)
from fleet.perception.lens import lens_obscuration, render_lens
from fleet.perception.trend import detect_change, rounds_to_limit, synth_condition_history

Q = {RobotKind.QUADRUPED}
DISAGREE_FRACTION = 0.05          # readers further apart than 5 % of full scale -> second look
NORMAL_BAND = (0.15, RED_ZONE[0])  # dial fraction; below = low alarm, red band = high alarm


def _xyz(p) -> tuple[float, float, float]:
    p = list(p) + [0.0] * (3 - len(p))
    return (float(p[0]), float(p[1]), float(p[2]))


def _walk_s(robot: Robot, points: list) -> float:
    pts = [robot.position[:2]] + [tuple(p[:2]) for p in points]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:])) / max(robot.speed_mps, 0.1)


def _result(job: Job, robot: Robot, world: World, duration: float, findings: list[Finding], log: list[str],
            used_gemini: bool = False, status: Optional[JobStatus] = None) -> JobResult:
    if status is None:
        status = JobStatus.FINDING if any(f.severity != "info" or f.follow_up for f in findings) else JobStatus.DONE
    log.append(f"done in {duration:.0f} s -> {status.value}, {len(findings)} finding(s)")
    return JobResult(job.id, robot.id, status, world.t, world.t + duration, findings, log, used_gemini)


def _follow(job: Job, kind: str, level: Level, suffix: str, *, target: Optional[str] = None, position=None,
            needs: Optional[set] = None, kinds: Optional[set] = None, priority: int = 3, est: float = 120.0,
            **params) -> Job:
    return Job(id=f"{job.id}.{suffix}", kind=kind, target=target or job.target,
               position=_xyz(position if position is not None else job.position), level=level,
               needs=set(needs or ()), robot_kinds=set(kinds or ()), priority=priority, est_duration_s=est,
               params=params, parent=job.id)


# ------------------------------------------------------------------------------------------------ gauge_read

class GaugeReadJob:
    kind = "gauge_read"
    level = Level.AUTO
    needs = {"camera"}
    robot_kinds = Q
    est_duration_s = 300.0      # 7 gauges on two pipe racks ~30 m apart, ~20 s per gauge at the dial

    def __init__(self, classical: Optional[ClassicalGaugeReader] = None, gemini: Any = "auto"):
        self.classical = classical or ClassicalGaugeReader()
        self._gemini = gemini

    @property
    def gemini(self):
        if self._gemini == "auto":
            self._gemini = GeminiGaugeReader()
        return self._gemini

    def _frame(self, gauge: dict, index: int, truth: float, world: World, second_look: bool) -> np.ndarray:
        frames = world.state.get("gauge_frames") or {}
        if gauge["id"] in frames:
            return np.asarray(frames[gauge["id"]])
        rng = np.random.default_rng(1000 * index + int(world.t) % 997)
        view = world.state.get("gauge_views", {}).get(gauge["id"])
        if second_look:
            # Robot steps square to the dial and shades the lens: small tilt, no glare.
            params = CropParams(yaw_deg=float(rng.uniform(-8, 8)), pitch_deg=float(rng.uniform(-8, 8)),
                                blur_sigma=0.4, noise_std=3.0)
        elif isinstance(view, CropParams):
            params = view
        else:
            params = CropParams.random(rng)
        return render_gauge_crop(truth, params, seed=index)

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        gauges = sorted(world.of_type("gauge"), key=lambda g: g["id"])
        if job.target in world.assets and world.assets[job.target].get("type") == "gauge":
            gauges = [world.assets[job.target]]
        second_look = bool(job.params.get("second_look"))
        use_gemini = job.params.get("use_gemini", True)
        gemini = self.gemini if use_gemini else None
        log = [f"{robot.id}: gauge round over {len(gauges)} gauge(s)" + (" (second look, square-on)" if second_look else "")]
        findings: list[Finding] = []
        used_gemini = False
        truths = world.state.get("gauge_values") or {}
        gauge_log = world.state.setdefault("gauge_log", {})
        logged: set[str] = set()

        for index, g in enumerate(gauges):
            gid = g["id"]
            lo_bar, hi_bar = g.get("range_bar", [0.0, RANGE_BAR])
            span = hi_bar - lo_bar
            truth = float(truths.get(gid, (g.get("reading_bar", span / 2) - lo_bar) / span))
            frame = self._frame(g, int(gid.split("_")[-1]) if gid.split("_")[-1].isdigit() else index,
                                truth, world, second_look)
            c = self.classical.read(frame)
            gm: Optional[GaugeReading] = gemini.read(frame) if gemini is not None else None
            if gm is not None:
                used_gemini = True
            elif gemini is not None and getattr(gemini, "last_error", None) and gemini.last_error not in logged:
                logged.add(gemini.last_error)
                log.append(f"{gid}: no Gemini reading ({gemini.last_error[:80]}); classical reader only")
            readings = [r for r in (c, gm) if r is not None and r.value is not None]
            evidence: dict[str, Any] = {"classical": c.to_dict(), "gemini": gm.to_dict() if gm else None,
                                        "sim_truth_bar": round(lo_bar + truth * span, 2)}
            pos = _xyz(g["position_m"])

            if not readings or (len(readings) == 1 and readings[0].confidence < 0.3):
                log.append(f"{gid}: no confident reading (classical conf {c.confidence:.2f})")
                if second_look:
                    findings.append(Finding("gauge_unreadable", "warning",
                                            f"{gid} unreadable after a square-on second look; manual read needed",
                                            pos, evidence, [_follow(job, "manual_gauge_read", Level.HUMAN,
                                                                    f"manual_{gid}", target=gid, position=pos,
                                                                    priority=2, est=300.0)]))
                else:
                    findings.append(Finding("gauge_second_look", "info",
                                            f"{gid} not readable from this view; re-reading square-on", pos,
                                            evidence, [self._second_look(job, gid, pos)]))
                continue

            if len(readings) == 2 and abs(readings[0].value - readings[1].value) > DISAGREE_FRACTION:
                diff = abs(readings[0].value - readings[1].value)
                best = max(readings, key=lambda r: r.confidence)
                evidence["disagreement_fraction"] = round(diff, 4)
                log.append(f"{gid}: readers disagree by {diff * 100:.1f} % FS "
                           f"(classical {c.reading_bar} bar, Gemini {gm.reading_bar} bar)")
                if second_look:
                    follow = [_follow(job, "manual_gauge_read", Level.HUMAN, f"manual_{gid}", target=gid,
                                      position=pos, priority=2, est=300.0)]
                    summary = f"{gid}: readers still disagree by {diff * 100:.1f} % after a second look; manual read"
                else:
                    follow = [self._second_look(job, gid, pos)]
                    summary = f"{gid}: classical and Gemini readings disagree by {diff * 100:.1f} % FS; second look queued"
                findings.append(Finding("gauge_reading_disagreement", "warning", summary, pos, evidence, follow))
                value, conf = best.value, best.confidence * 0.5
            else:
                w = np.array([max(r.confidence, 1e-3) for r in readings])
                value = float(np.average([r.value for r in readings], weights=w))
                conf = float(w.max())
            bar = round(lo_bar + value * span, 2)
            evidence["fused_bar"] = bar
            gauge_log.setdefault(gid, []).append({"t": world.t, "bar": bar, "confidence": round(conf, 3)})
            log.append(f"{gid}: {bar:.2f} bar (classical {c.reading_bar}, "
                       f"Gemini {gm.reading_bar if gm else 'n/a'}, conf {conf:.2f})")

            band = job.params.get("normal_band", g.get("normal_band", NORMAL_BAND))
            if value >= band[1] or value < band[0]:
                high = value >= band[1]
                findings.append(Finding(
                    "gauge_out_of_range", "critical" if high else "warning",
                    f"{gid} reads {bar:.1f} bar, {'in the red high-alarm band' if high else 'below the normal band'} "
                    f"({lo_bar + band[0] * span:.1f}-{lo_bar + band[1] * span:.1f} bar)",
                    pos, evidence,
                    [_follow(job, "operator_process_check", Level.HUMAN, f"process_{gid}", target=gid, position=pos,
                             priority=1 if high else 2, est=600.0, reading_bar=bar)]))

        stops = [g["position_m"] for g in gauges]
        duration = _walk_s(robot, stops) + 20.0 * len(gauges)
        return _result(job, robot, world, duration, findings, log, used_gemini)

    def _second_look(self, job: Job, gid: str, pos) -> Job:
        return _follow(job, "gauge_read", Level.AUTO, f"relook_{gid}", target=gid, position=pos, needs={"camera"},
                       kinds=Q, priority=2, est=60.0, second_look=True)


# ---------------------------------------------------------------------------------- thermal_acoustic_survey

TEMP_RISE_CRITICAL_C = 10.0
VIB_CRITICAL_MM_S = 7.1           # ISO 10816-3 group 2 rigid: above 7.1 mm/s is "unacceptable"
VIB_ALERT_MM_S = 4.5


class ThermalAcousticSurveyJob:
    kind = "thermal_acoustic_survey"
    level = Level.AUTO
    needs = {"thermal"}
    robot_kinds = Q
    est_duration_s = 360.0        # 6 valves, ~40 s each (thermal frame + 30 s acoustic/vibration capture)

    asset_types = ("valve", "pump", "compressor", "motor")

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        assets = sorted((a for a in world.assets.values() if a.get("type") in self.asset_types),
                        key=lambda a: a["id"])
        if job.target in world.assets:
            assets = [world.assets[job.target]]
        history = world.state.get("condition_history")
        if history is None:
            history = synth_condition_history([a["id"] for a in sorted(
                (a for a in world.assets.values() if a.get("type") in self.asset_types), key=lambda a: a["id"])])
            world.state["condition_history"] = history
        now = world.state.get("condition_now") or {}
        baseline_n = int(job.params.get("baseline_rounds", 10))
        log = [f"{robot.id}: thermal + acoustic survey of {len(assets)} asset(s), baseline = first {baseline_n} rounds"]
        findings: list[Finding] = []
        for a in assets:
            aid = a["id"]
            h = history.setdefault(aid, {"temp_c": [], "vib_mm_s": []})
            if aid in now:
                h["temp_c"].append(float(now[aid]["temp_c"]))
                h["vib_mm_s"].append(float(now[aid]["vib_mm_s"]))
            if not h["temp_c"]:
                log.append(f"{aid}: no history yet")
                continue
            log.append(f"{aid}: {h['temp_c'][-1]:.1f} C, {h['vib_mm_s'][-1]:.2f} mm/s (round {len(h['temp_c'])})")
            for channel, unit, min_sd, min_effect in (("temp_c", "C", 0.3, 2.0), ("vib_mm_s", "mm/s", 0.08, 0.5)):
                tr = detect_change(h[channel], baseline_n=baseline_n, min_sd=min_sd, min_effect=min_effect)
                if tr is None:
                    continue
                f = self._finding(job, a, channel, unit, tr)
                findings.append(f)
                log.append(f"{aid}: {f.summary}")
        duration = _walk_s(robot, [a["position_m"] for a in assets]) + 40.0 * len(assets)
        return _result(job, robot, world, duration, findings, log)

    def _finding(self, job: Job, asset: dict, channel: str, unit: str, tr) -> Finding:
        what = "temperature" if channel == "temp_c" else "vibration"
        ev = {"channel": channel, "unit": unit, **tr.to_dict()}
        if channel == "temp_c":
            critical = tr.direction == "up" and tr.delta >= TEMP_RISE_CRITICAL_C
            ev["rounds_to_critical"] = rounds_to_limit(tr, tr.baseline_mean + TEMP_RISE_CRITICAL_C)
        else:
            critical = tr.current_mean >= VIB_CRITICAL_MM_S
            ev["rounds_to_critical"] = rounds_to_limit(tr, VIB_CRITICAL_MM_S)
            ev["iso10816_zone"] = ("D" if tr.current_mean >= VIB_CRITICAL_MM_S else
                                   "C" if tr.current_mean >= VIB_ALERT_MM_S else "B" if tr.current_mean >= 2.8 else "A")
        sev = "critical" if critical else "warning"
        rate = f", {tr.slope_per_round:+.2f} {unit}/round" if tr.kind == "drift" else ""
        summary = (f"{asset['id']} {what} {tr.kind} {tr.direction}: {tr.baseline_mean:.2f} -> {tr.current_mean:.2f} "
                   f"{unit} ({tr.delta_sigma:+.1f} sigma{rate}) since round {tr.change_round}")
        pos = _xyz(asset["position_m"])
        follow = [_follow(job, "maintenance_work_order" if critical else "condition_closeup",
                          Level.HUMAN if critical else Level.ASSIST, f"{what}_{asset['id']}", target=asset["id"],
                          position=pos, kinds=None if critical else Q, needs=None if critical else {"thermal"},
                          priority=2 if critical else 3, est=1800.0 if critical else 180.0, channel=channel)]
        return Finding(f"{what}_{tr.kind}", sev, summary, pos, ev, follow)


# ---------------------------------------------------------------------------------------- escape_route_check

ROUTE_HALF_WIDTH_M = 0.6          # 1.2 m escape-route width
LOOSE_MAX_SIZE_M = 0.6
LOOSE_MAX_MASS_KG = 10.0          # what the UR3 gripper may pick up and carry (payload 3 kg at full reach x lift)


class EscapeRouteCheckJob:
    kind = "escape_route_check"
    level = Level.AUTO
    needs = {"camera"}
    robot_kinds = Q
    est_duration_s = 150.0        # ~115 m top-deck loop at 1 m/s plus corners

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        route = [tuple(p[:2]) for p in world.state.get("patrol_route") or route_mod.patrol_route()]
        length = route_mod.route_length(route)
        z = float(job.position[2]) if job.position and job.position[2] else 42.5
        log = [f"{robot.id}: walking the escape route, {len(route) - 1} legs, {length:.0f} m"]
        findings: list[Finding] = []

        samples = list(world.state.get("route_blocked_samples") or route_mod.meta_blocked_samples())
        if world.state.get("deck_triangles") is not None:
            samples += route_mod.route_blockages(world.state["deck_triangles"], route)
        for k, group in enumerate(route_mod.cluster_samples(samples)):
            centre = np.mean(np.array(group), axis=0)
            _, chain = route_mod.project_on_route(route, centre)
            pos = (float(centre[0]), float(centre[1]), z)
            findings.append(Finding(
                "escape_route_obstructed", "critical",
                f"fixed structure within 0.45 m of the escape route at chainage {chain:.0f} m ({len(group)} samples)",
                pos, {"samples": group, "chainage_m": round(chain, 1), "source": "route_blockages"},
                [_follow(job, "clear_obstruction", Level.HUMAN, f"structure_{k}", target="escape_route",
                         position=pos, priority=1, est=3600.0)]))
            log.append(f"blockage cluster {k} at ({centre[0]:.1f}, {centre[1]:.1f}), chainage {chain:.0f} m")

        for obj in world.state.get("dropped_objects") or []:
            size = float(obj.get("size_m", 0.3))
            mass = float(obj.get("mass_kg", 1.0))
            dist, chain = route_mod.project_on_route(route, obj["position"])
            pos = _xyz(list(obj["position"][:2]) + [obj["position"][2] if len(obj["position"]) > 2 else z])
            label = obj.get("label", obj.get("id", "object"))
            ev = {"object": obj, "distance_to_route_m": round(dist, 2), "chainage_m": round(chain, 1)}
            if dist > ROUTE_HALF_WIDTH_M + size / 2:
                log.append(f"{label}: {dist:.1f} m off the route, logged as a dropped object only")
                findings.append(Finding("dropped_object_off_route", "info",
                                        f"{label} seen {dist:.1f} m off the escape route", pos, ev))
                continue
            small = size <= LOOSE_MAX_SIZE_M and mass <= LOOSE_MAX_MASS_KG
            oid = obj.get("id", label)
            if small:
                follow = [_follow(job, "remove_loose_item", Level.SUPV, f"remove_{oid}", target=oid, position=pos,
                                  needs={"arm"}, kinds=Q, priority=2, est=90.0, object=obj,
                                  drop_off="nearest laydown / tool store")]
                findings.append(Finding("loose_item_on_route", "warning",
                                        f"loose {label} ({size:.2f} m, {mass:.1f} kg) on the escape route at chainage "
                                        f"{chain:.0f} m; robot can remove it once approved", pos, ev, follow))
            else:
                follow = [_follow(job, "clear_obstruction", Level.HUMAN, f"clear_{oid}", target=oid, position=pos,
                                  priority=1, est=1800.0, object=obj)]
                findings.append(Finding("escape_route_blocked", "critical",
                                        f"{label} ({size:.1f} m, {mass:.0f} kg) blocks the escape route at chainage "
                                        f"{chain:.0f} m; crew must clear it", pos, ev, follow))
            log.append(f"{label}: on route at chainage {chain:.0f} m -> {'SUPV remove_loose_item' if small else 'HUMAN clear_obstruction'}")
        corners = max(0, len(route) - 2)
        duration = (_walk_s(robot, [route[0]]) + length / max(robot.speed_mps, 0.1) + 2.0 * corners
                    + 10.0 * len(world.state.get("dropped_objects") or []))
        return _result(job, robot, world, duration, findings, log)


# -------------------------------------------------------------------------------------------- spill_response

SOCK_LENGTH_M = 1.2
SOCK_OVERLAP_M = 0.15
MAX_ROBOT_SPILL_RADIUS_M = 1.5


class SpillResponseJob:
    kind = "spill_response"
    level = Level.SUPV
    needs = {"arm"}
    robot_kinds = Q
    est_duration_s = 240.0        # ~6 socks at ~25 s each plus approach

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        spills = world.state.get("spills") or {}
        log = [f"{robot.id}: spill response approved for {job.target}"]
        sid = job.target if job.target in spills else None
        if sid is None and spills:
            sid = min(spills, key=lambda k: math.dist(spills[k]["position"][:2], job.position[:2]))
        if sid is None:
            log.append("no spill at the target; nothing to contain")
            return _result(job, robot, world, _walk_s(robot, [job.position]) + 15.0, [], log)
        spill = spills[sid]
        r = float(spill.get("radius_m", 0.5))
        pos = _xyz(spill["position"])
        ring_r = r + 0.25
        n = math.ceil(2 * math.pi * ring_r / (SOCK_LENGTH_M - SOCK_OVERLAP_M))
        stock = int(world.state.get("absorbent_socks", 8))
        ev = {"spill_id": sid, "radius_m": r, "fluid": spill.get("fluid", "unknown"), "socks_needed": n,
              "socks_carried": stock}
        approach = _walk_s(robot, [pos])
        if r > MAX_ROBOT_SPILL_RADIUS_M or n > stock:
            why = "too large for the robot" if r > MAX_ROBOT_SPILL_RADIUS_M else f"needs {n} socks, robot carries {stock}"
            log.append(f"{sid}: {why}; handing to the spill crew")
            f = Finding("spill_beyond_robot", "critical", f"{sid} ({r:.1f} m radius) {why}; spill crew called",
                        pos, ev, [_follow(job, "spill_cleanup_crew", Level.HUMAN, f"crew_{sid}", target=sid,
                                          position=pos, priority=1, est=1800.0)])
            return _result(job, robot, world, approach + 30.0, [f], log, status=JobStatus.HANDED_TO_HUMAN)

        placements = []
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n
            placements.append([round(pos[0] + ring_r * math.cos(a), 2), round(pos[1] + ring_r * math.sin(a), 2),
                               round(math.degrees(a + math.pi / 2) % 180, 1)])
        ev["sock_placements_xy_yaw"] = placements
        world.state["absorbent_socks"] = stock - n
        spill["contained"] = True
        log.append(f"{sid}: {ev['fluid']} spill r={r:.2f} m; laying {n} x {SOCK_LENGTH_M} m socks on a {ring_r:.2f} m ring")
        log.append(f"socks left on robot: {stock - n}")
        f = Finding("spill_contained", "warning", f"{sid} ringed with {n} absorbent socks; crew to recover and find the source",
                    pos, ev, [_follow(job, "spill_waste_disposal", Level.HUMAN, f"dispose_{sid}", target=sid,
                                      position=pos, priority=3, est=900.0)])
        return _result(job, robot, world, approach + 25.0 * n + 20.0, [f], log)


# ------------------------------------------------------------------------------------------- fg_optics_clean

DEFAULT_FG_DETECTORS = {
    "fg_ir_01": {"position": [-14.0, 22.6, 44.6], "type": "IR flame", "salt": 0.35},
    "fg_ogd_02": {"position": [21.0, -9.9, 44.7], "type": "open-path gas", "salt": 0.22},
    "fg_ir_03": {"position": [30.0, -9.9, 44.7], "type": "IR flame", "salt": 0.08},
}
CLEAN_OBSCURATION = 0.05          # window is fit for duty below 5 % covered
WIPE_EFFICIENCY = 0.75            # fresh-water spray + microfibre removes ~75 % of the crust per pass
MAX_WIPES = 3


class FgOpticsCleanJob:
    kind = "fg_optics_clean"
    level = Level.AUTO
    needs = {"arm", "camera"}
    robot_kinds = Q
    est_duration_s = 180.0

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        dets = world.state.setdefault("fg_detectors", {k: dict(v) for k, v in DEFAULT_FG_DETECTORS.items()})
        did = job.target if job.target in dets else min(
            dets, key=lambda k: math.dist(dets[k]["position"][:2], job.position[:2]))
        det = dets[did]
        pos = _xyz(det["position"])
        log = [f"{robot.id}: {det.get('type', 'F&G')} detector {did}; requesting F&G inhibit from the control room"]
        seed = zlib.crc32(did.encode()) % 1000
        before = lens_obscuration(render_lens(det["salt"], seed=seed))
        log.append(f"{did}: window {before * 100:.0f} % obscured by salt (camera)")
        measured, wipes = before, 0
        while measured > CLEAN_OBSCURATION and wipes < MAX_WIPES:
            wipes += 1
            det["salt"] = det["salt"] * (1.0 - WIPE_EFFICIENCY)
            measured = lens_obscuration(render_lens(det["salt"], seed=seed + wipes))
            log.append(f"wipe {wipes}: spray + microfibre pass, now {measured * 100:.1f} % obscured")
        log.append(f"{did}: inhibit released")
        ev = {"detector": did, "type": det.get("type"), "obscuration_before": round(before, 3),
              "obscuration_after": round(measured, 3), "wipes": wipes}
        duration = _walk_s(robot, [pos]) + 60.0 + 30.0 * wipes
        if measured > CLEAN_OBSCURATION:
            f = Finding("fg_window_still_obscured", "warning",
                        f"{did} still {measured * 100:.0f} % obscured after {wipes} wipes; window needs replacing",
                        pos, ev, [_follow(job, "fg_window_replace", Level.HUMAN, f"replace_{did}", target=did,
                                          position=pos, priority=2, est=2400.0)])
            return _result(job, robot, world, duration, [f], log)
        if wipes == 0:
            log.append(f"{did}: window already clean, no wipe needed")
        f = Finding("fg_optics_cleaned", "info",
                    f"{did} window {before * 100:.0f} % -> {measured * 100:.1f} % obscured after {wipes} wipe(s); "
                    f"bump test queued", pos, ev,
                    [_follow(job, "fg_bump_test", Level.ASSIST, f"bump_{did}", target=did, position=pos,
                             needs={"arm"}, kinds=Q, priority=2, est=300.0,
                             note="robot holds the test gas cell / flame simulator; technician confirms the alarm "
                                  "at the F&G panel")])
        return _result(job, robot, world, duration, [f], log)


EXECUTORS = [register(GaugeReadJob()), register(ThermalAcousticSurveyJob()), register(EscapeRouteCheckJob()),
             register(SpillResponseJob()), register(FgOpticsCleanJob())]

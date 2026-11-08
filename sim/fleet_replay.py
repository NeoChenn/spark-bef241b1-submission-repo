"""Replay a fleet timeline on the full rig and render it to an MP4 with a HUD.

    C:\\isaacsim\\python.bat sim\\fleet_replay.py [--timeline runs/fleet/timeline.jsonl] [--max-frames 48]

Reads runs/fleet/timeline.jsonl, written by the fleet coordinator: one line per robot per second
    {"t", "robot", "kind", "position": [x, y, z], "yaw", "state", "job"}
plus event lines
    {"t", "event", "job", "robot", "detail"}      (detail may hold {"position": [x, y, z]}, e.g. a gas leak)
If the file is missing, a fake 60 s timeline in exactly that format is written there first (--fake forces it).

Everything above `main()` is plain Python + numpy (timeline parsing, pose interpolation, shot planning, HUD,
camera projection), so it is unit-tested on CPU in tests/test_fleet_replay.py. Writes
sim/output/fleet/fleet_replay.mp4 and preview stills.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RIG_META = REPO / "sim/output/full_rig_meta.json"
DEFAULT_TIMELINE = REPO / "runs/fleet/timeline.jsonl"
OUT_DIR = REPO / "sim/output/fleet"

DECKS = (20.0, 31.25, 42.5)
TOP_DECK = 42.5
HELIDECK = (-22.0, -22.0, 51.37)       # pad surface 51.27 m; the drone origin sits 0.1 m above it
DRONE_MIN_Z = TOP_DECK + 4.0          # flying drones stay this high over the topsides
GROUND_KINDS = ("Q", "C")
HORIZONTAL_APERTURE = 20.955          # mm, USD camera default
EVENT_HOLD_S = 8.0                    # how long an event stays in the HUD
MIN_SHOT_S = 4.0
ESTABLISH_S = 6.0
URGENT_GAP_S = 3.0                    # a safety event may cut this soon after the previous cut
STALE_CUE_S = 6.0                     # a cue pushed back further than this is dropped
MAX_SHOT_S = 12.0

PATROL_ROUTE = [
    (-21.0, 20.5), (-10.0, 20.5), (-10.0, 11.0), (-1.0, 11.0), (-1.0, -4.5), (24.0, -4.5), (24.0, -9.0),
    (9.0, -9.0), (9.0, -4.5), (-1.0, -4.5), (-1.0, 11.0), (-21.0, 11.0), (-21.0, 20.5),
]

STATE_COLORS = {
    "IDLE": (150, 150, 150), "DOCKED": (150, 150, 150), "CHARGING": (90, 200, 120),
    "PATROL": (70, 170, 255), "TRANSIT": (70, 170, 255), "MOVING": (70, 170, 255), "RETURN": (70, 140, 230),
    "INSPECTING": (255, 190, 40), "WORKING": (255, 150, 30), "RUNNING": (255, 170, 40),
    "WAITING_APPROVAL": (220, 120, 255), "ASSIGNED": (120, 220, 255),
    "ALERT": (255, 60, 50), "FAILED": (255, 60, 50), "HANDED_TO_HUMAN": (255, 120, 80),
    "DONE": (90, 220, 110), "FINDING": (255, 90, 60),
}


# On-screen names and colours (the timeline keeps the ids). Unknown ids show as themselves.
ROBOT_NAMES = {"Q1": "Dog-1", "Q2": "Dog-2", "A1": "Drone", "C1": "Crawler"}
ROBOT_COLORS = {"Q1": (0x2E, 0xE6, 0xD6), "Q2": (0xFF, 0xB0, 0x20), "A1": (0xF3, 0xF5, 0xF8), "C1": (0xA7, 0x8B, 0xFA)}


def display_name(rid: Optional[str]) -> str:
    return ROBOT_NAMES.get(rid, rid or "")


def state_color(state: str) -> tuple[int, int, int]:
    state = (state or "").upper()
    if state in STATE_COLORS:
        return STATE_COLORS[state]
    for key, color in STATE_COLORS.items():
        if key in state:
            return color
    return (200, 200, 200)


# ------------------------------------------------------------------------------------- timeline
@dataclass
class Sample:
    t: float
    position: np.ndarray
    yaw: float
    state: str
    job: Optional[str]


@dataclass
class Event:
    t: float
    event: str
    job: Optional[str]
    robot: Optional[str]
    detail: dict = field(default_factory=dict)

    @property
    def position(self) -> Optional[np.ndarray]:
        p = self.detail.get("position") if isinstance(self.detail, dict) else None
        return np.asarray(p, float) if p is not None and len(p) == 3 else None

    def text(self) -> str:
        if self.event.startswith("detector_") and isinstance(self.detail, dict) and self.detail.get("detector"):
            d = self.detail
            line = f"{d['detector']} {self.event[9:]}"
            if d.get("reading") is not None:
                line += f" {float(d['reading']):.2f} {d.get('unit', 'LEL')}"
            return d.get("summary") or line
        who = f"{display_name(self.robot)} " if self.robot else ""
        what = self.event.replace("_", " ")
        job = f" [{self.job}]" if self.job else ""
        note = self.detail.get("summary") or self.detail.get("note") if isinstance(self.detail, dict) else None
        return f"{who}{what}{job}" + (f": {note}" if note else "")


@dataclass
class Timeline:
    tracks: dict[str, "Track"]
    events: list[Event]

    @property
    def duration(self) -> float:
        ends = [tr.t[-1] for tr in self.tracks.values()] + [e.t for e in self.events]
        return max(ends) if ends else 0.0


def deck_height(z: float) -> float:
    """Deck a ground robot stands on: the highest deck at or below z + 1 m (coordinators may log a trunk/sensor z)."""
    below = [d for d in DECKS if d <= z + 1.0]
    return max(below) if below else DECKS[0]


def ground_position(kind: str, p: np.ndarray, state: str = "") -> np.ndarray:
    """Pin Q/C to a deck surface and keep a flying drone above the topsides."""
    p = np.asarray(p, float).copy()
    if kind in GROUND_KINDS:
        p[2] = deck_height(p[2])
    elif kind == "A":
        landed = (state or "").upper() in ("IDLE", "DOCKED", "CHARGING", "LANDED")
        p[2] = max(p[2], TOP_DECK + 0.3) if landed else max(p[2], DRONE_MIN_Z)
    return p


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    """One robot's samples with per-frame interpolation.

    Position is linear between samples (the coordinator logs at 1 Hz along straight segments; a spline would
    cut corners into deck clutter). Yaw takes the shortest way round. State and job switch at the sample that
    reports them. Distance travelled is integrated so gaits can phase on it."""

    def __init__(self, robot: str, kind: str, samples: list[Sample]):
        samples = sorted(samples, key=lambda s: s.t)
        self.robot, self.kind = robot, kind
        self.samples = samples
        self.t = np.array([s.t for s in samples], float)
        self.p = np.array([ground_position(kind, s.position, s.state) for s in samples], float).reshape(-1, 3)
        yaws = np.array([s.yaw for s in samples], float)
        self.yaw = np.unwrap(yaws) if len(yaws) else yaws
        steps = np.linalg.norm(np.diff(self.p, axis=0), axis=1) if len(self.p) > 1 else np.zeros(0)
        self.dist = np.concatenate([[0.0], np.cumsum(steps)])

    def _segment(self, t: float) -> tuple[int, float]:
        if len(self.t) == 1 or t <= self.t[0]:
            return 0, 0.0
        if t >= self.t[-1]:
            return len(self.t) - 2, 1.0
        i = bisect.bisect_right(self.t, t) - 1
        span = self.t[i + 1] - self.t[i]
        return i, (t - self.t[i]) / span if span > 0 else 0.0

    def position(self, t: float) -> np.ndarray:
        if len(self.t) == 1:
            return self.p[0].copy()
        i, u = self._segment(t)
        return self.p[i] + (self.p[i + 1] - self.p[i]) * u

    def yaw_at(self, t: float) -> float:
        if len(self.t) == 1:
            return float(_wrap(self.yaw[0]))
        i, u = self._segment(t)
        return float(_wrap(self.yaw[i] + (self.yaw[i + 1] - self.yaw[i]) * u))

    def distance(self, t: float) -> float:
        if len(self.t) == 1:
            return 0.0
        i, u = self._segment(t)
        return float(self.dist[i] + (self.dist[i + 1] - self.dist[i]) * u)

    def speed(self, t: float) -> float:
        if len(self.t) == 1:
            return 0.0
        i, _ = self._segment(t)
        span = self.t[i + 1] - self.t[i]
        return float(np.linalg.norm(self.p[i + 1] - self.p[i]) / span) if span > 0 else 0.0

    def velocity(self, t: float) -> np.ndarray:
        if len(self.t) == 1:
            return np.zeros(3)
        i, _ = self._segment(t)
        span = self.t[i + 1] - self.t[i]
        return (self.p[i + 1] - self.p[i]) / span if span > 0 else np.zeros(3)

    def latest(self, t: float) -> Sample:
        i = max(bisect.bisect_right(self.t, t) - 1, 0)
        return self.samples[i]

    def state(self, t: float) -> str:
        return self.latest(t).state

    def job(self, t: float) -> Optional[str]:
        return self.latest(t).job

    def job_changes(self) -> list[tuple[float, Optional[str]]]:
        out, last = [], object()
        for s in self.samples:
            if s.job != last:
                out.append((s.t, s.job))
                last = s.job
        return out


def _yaw(value) -> float:
    return float(value) if value is not None else 0.0


def parse_timeline(lines) -> Timeline:
    """Parse JSONL lines (strings or dicts). Yaw is radians; a file whose yaws exceed 2*pi is read as degrees."""
    rows, events = [], []
    for line in lines:
        if isinstance(line, str):
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
        if "event" in line:
            detail = line.get("detail") or {}
            if not isinstance(detail, dict):
                detail = {"note": str(detail)}
            events.append(Event(float(line["t"]), str(line["event"]), line.get("job"), line.get("robot"), detail))
        elif "robot" in line and "position" in line:
            rows.append(line)
    degrees = any(abs(_yaw(r.get("yaw"))) > 2 * math.pi + 0.01 for r in rows)
    by_robot: dict[str, list[Sample]] = {}
    kinds: dict[str, str] = {}
    for r in rows:
        yaw = _yaw(r.get("yaw"))
        yaw = math.radians(yaw) if degrees else yaw
        rid = str(r["robot"])
        kinds.setdefault(rid, str(r.get("kind") or rid[:1]).upper()[:1])
        by_robot.setdefault(rid, []).append(
            Sample(float(r["t"]), np.asarray(r["position"], float), yaw, str(r.get("state") or ""), r.get("job")))
    tracks = {rid: Track(rid, kinds[rid], s) for rid, s in sorted(by_robot.items())}
    return Timeline(tracks, sorted(events, key=lambda e: e.t))


def load_timeline(path: Path) -> Timeline:
    return parse_timeline(Path(path).read_text(encoding="utf-8").splitlines())


def rig_assets(path: Path = RIG_META) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    return {p["id"]: p for p in meta.get("inspection_points", []) if isinstance(p, dict) and "id" in p}


# --------------------------------------------------------------------------------- fake timeline
def _keyframed(keys: list[tuple], t_end: float) -> list[tuple[float, np.ndarray, str, Optional[str]]]:
    """keys: (t, (x, y, z), state, job). Linear between keys, sampled at 1 Hz."""
    out = []
    times = [k[0] for k in keys]
    for t in range(int(t_end) + 1):
        i = max(bisect.bisect_right(times, t) - 1, 0)
        if i >= len(keys) - 1:
            p = np.asarray(keys[-1][1], float)
        else:
            t0, t1 = keys[i][0], keys[i + 1][0]
            u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            p = np.asarray(keys[i][1], float) + (np.asarray(keys[i + 1][1], float) - np.asarray(keys[i][1], float)) * u
        out.append((float(t), p, keys[i][2], keys[i][3]))
    return out


def _with_yaw(samples, default: float = 0.0, faces: Optional[dict] = None) -> list[tuple]:
    """Heading from motion; while stopped keep the last heading (or face `faces[t_from]` -> yaw)."""
    out, yaw = [], default
    for k, (t, p, state, job) in enumerate(samples):
        nxt = samples[min(k + 1, len(samples) - 1)][1]
        d = nxt[:2] - p[:2]
        if np.linalg.norm(d) > 0.05:
            yaw = math.atan2(d[1], d[0])
        elif faces:
            for t_from, y in sorted(faces.items()):
                if t >= t_from:
                    yaw = y
        out.append((t, p, yaw, state, job))
    return out


def fake_timeline(duration: float = 60.0) -> list[dict]:
    """A plausible 60 s shift: drone gas survey finds a leak, a quadruped isolates it, crawler gauges a line."""
    z = TOP_DECK
    hz = HELIDECK
    leak = [-14.0, 24.3, 43.6]                   # on the blue process line (rig_details.add_pipes), by valve_03
    tracks = {
        ("Q1", "Q"): _with_yaw(_keyframed([
            (0, (-21.0, 11.0, z), "PATROL", None),
            (2, (-21.0, 11.0, z), "TRANSIT", "J1 gauge_read gauge_02"),
            (10, (-21.0, 20.5, z), "TRANSIT", "J1 gauge_read gauge_02"),
            (13, (-17.5, 20.5, z), "INSPECTING", "J1 gauge_read gauge_02"),
            (21, (-17.5, 20.5, z), "DONE", "J1 gauge_read gauge_02"),
            (23, (-17.5, 20.5, z), "PATROL", None),
            (27, (-21.0, 20.5, z), "PATROL", None),
            (37, (-21.0, 11.0, z), "PATROL", None),
            (49, (-10.0, 11.0, z), "PATROL", None),
            (58, (-1.0, 11.0, z), "PATROL", None),
            (60, (-1.0, 9.0, z), "PATROL", None),
        ], duration), faces={13: math.pi / 2}),
        ("Q2", "Q"): _with_yaw(_keyframed([
            (0, (-1.0, -3.0, z), "PATROL", None),
            (12, (-1.0, 11.0, z), "PATROL", None),
            (19, (-5.5, 11.0, z), "TRANSIT", "J3 valve_isolate valve_03"),
            (22, (-10.0, 11.0, z), "TRANSIT", "J3 valve_isolate valve_03"),
            (30, (-10.0, 20.5, z), "TRANSIT", "J3 valve_isolate valve_03"),
            (35, (-15.5, 20.5, z), "WORKING", "J3 valve_isolate valve_03"),
            (46, (-15.5, 20.5, z), "DONE", "J3 valve_isolate valve_03"),
            (48, (-15.5, 20.5, z), "RETURN", None),
            (53, (-10.0, 20.5, z), "RETURN", None),
            (60, (-10.0, 14.0, z), "RETURN", None),
        ], duration), default=math.pi / 2, faces={35: math.pi / 2}),
        ("C1", "C"): _with_yaw(_keyframed([
            (0, (24.0, -9.0, z), "IDLE", None),
            (3, (24.0, -9.0, z), "INSPECTING", "J2 ut_thickness orange_line"),
            (42, (13.0, -9.0, z), "FINDING", "J2 ut_thickness orange_line"),
            (48, (13.0, -9.0, z), "DONE", "J2 ut_thickness orange_line"),
            (60, (11.0, -9.0, z), "IDLE", None),
        ], duration), default=math.pi),
        ("A1", "A"): _with_yaw(_keyframed([
            (0, hz, "IDLE", None),
            (1, hz, "TRANSIT", "J0 gas_survey topsides"),
            (4, (hz[0], hz[1], 56.0), "TRANSIT", "J0 gas_survey topsides"),
            # survey the north pipe rack from outside the rig: west edge, then along the north face
            (8, (-42.0, -8.0, 55.0), "INSPECTING", "J0 gas_survey topsides"),
            (13, (-40.0, 28.0, 52.0), "INSPECTING", "J0 gas_survey topsides"),
            (16.5, (-18.0, 29.0, 48.0), "INSPECTING", "J0 gas_survey topsides"),
            (19, (-14.0, 23.5, 46.8), "ALERT", "J0 gas_survey topsides"),
            (30, (-14.0, 23.5, 46.8), "INSPECTING", "J0 gas_survey topsides"),
            (33, (-14.0, 26.0, 60.0), "INSPECTING", "J0 gas_survey topsides"),   # climb clear of the modules
            (41, (14.0, -2.0, 58.0), "INSPECTING", "J0 gas_survey topsides"),
            (50, (-10.0, -14.0, 56.0), "RETURN", None),
            (56, (hz[0], hz[1], 55.0), "RETURN", None),
            (60, hz, "IDLE", None),
        ], duration), default=math.pi / 4),
    }
    out = []
    for (rid, kind), rows in tracks.items():
        for t, p, yaw, state, job in rows:
            out.append({"t": t, "robot": rid, "kind": kind, "position": [round(float(v), 3) for v in p],
                        "yaw": round(float(yaw), 4), "state": state, "job": job})
    events = [
        {"t": 0.0, "event": "shift_start", "job": None, "robot": None, "detail": {"summary": "4 robots on watch"}},
        {"t": 1.0, "event": "job_assigned", "job": "J0 gas_survey topsides", "robot": "A1", "detail": {}},
        {"t": 2.0, "event": "job_assigned", "job": "J1 gauge_read gauge_02", "robot": "Q1", "detail": {}},
        {"t": 3.0, "event": "job_assigned", "job": "J2 ut_thickness orange_line", "robot": "C1", "detail": {}},
        {"t": 21.0, "event": "job_done", "job": "J1 gauge_read gauge_02", "robot": "Q1",
         "detail": {"summary": "gauge_02 reads 4.6 bar, in range"}},
        {"t": 18.0, "event": "gas_leak", "job": "J0 gas_survey topsides", "robot": "A1",
         "detail": {"position": leak, "summary": "CH4 840 ppm-m on the blue line"}},
        {"t": 19.0, "event": "job_assigned", "job": "J3 valve_isolate valve_03", "robot": "Q2",
         "detail": {"summary": "priority 1, operator approved"}},
        {"t": 42.0, "event": "finding", "job": "J2 ut_thickness orange_line", "robot": "C1",
         "detail": {"summary": "wall 7.1 mm (nominal 9.5)", "position": [13.0, -10.7, 43.4]}},
        {"t": 43.0, "event": "leak_isolated", "job": "J3 valve_isolate valve_03", "robot": "Q2",
         "detail": {"summary": "valve_03 closed, CH4 falling", "position": leak}},
        {"t": 50.0, "event": "job_done", "job": "J0 gas_survey topsides", "robot": "A1",
         "detail": {"summary": "survey complete, 1 leak"}},
    ]
    out.extend(events)
    out.sort(key=lambda r: (r["t"], "event" in r))
    return out


def write_fake_timeline(path: Path, duration: float = 60.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in fake_timeline(duration):
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------------- scene state
def resolve_job_target(job: Optional[str], timeline: Timeline, assets: dict, t: float) -> Optional[np.ndarray]:
    """Where a job's marker goes: an event position for that job, else a rig asset named in the job string."""
    if not job:
        return None
    for e in reversed(timeline.events):
        if e.job == job and e.t <= t + 1e-6 and e.position is not None:
            return e.position
    for token in job.replace(":", " ").replace("/", " ").split():
        if token in assets:
            return np.asarray(assets[token]["position_m"], float)
    return None


def active_jobs(timeline: Timeline, t: float, assets: dict) -> list[dict]:
    out = []
    for rid, tr in timeline.tracks.items():
        job = tr.job(t)
        if not job:
            continue
        target = resolve_job_target(job, timeline, assets, t)
        if target is None:
            continue
        out.append({"job": job, "robot": rid, "state": tr.state(t), "target": target})
    return out


def gas_plumes(timeline: Timeline, t: float) -> list[dict]:
    """Leaks reported so far, with a strength that fades out once a later event isolates them."""
    plumes = []
    for e in timeline.events:
        if "gas" not in e.event and "leak" not in e.event or e.t > t or e.position is None:
            continue
        if "alarm" in e.event:          # an alarm is placed at the detector that raised it, not at the source
            continue
        if any(w in e.event for w in ("isolated", "cleared", "closed", "stopped")):
            continue
        strength = min((t - e.t) / 2.0, 1.0)
        for later in timeline.events:
            if later.t > e.t and later.t <= t and any(w in later.event for w in ("isolated", "cleared", "closed")):
                near = later.position is None or np.linalg.norm(later.position - e.position) < 3.0
                if near:
                    strength *= max(0.0, 1.0 - (t - later.t) / 6.0)
        if strength > 0.01:
            plumes.append({"position": e.position, "strength": float(strength), "since": float(t - e.t)})
    return plumes


def fixed_detectors() -> list[dict]:
    """The installed F&G detectors (fleet/sensors.py): id, head position, type. Empty if fleet is not importable."""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from fleet.sensors import DECK_Z, DETECTORS
    except Exception:
        return []
    return [{"id": d.id, "position": np.asarray(d.position, float), "type": d.type, "deck_z": DECK_Z}
            for d in DETECTORS]


def detector_states(timeline: Timeline, t: float) -> dict[str, str]:
    """Detectors in alarm at t (latched from `detector_alarm` until a later `detector_clear`)."""
    out: dict[str, str] = {}
    for e in timeline.events:
        if e.t > t:
            break
        did = e.detail.get("detector") if isinstance(e.detail, dict) else None
        if not did:
            continue
        if e.event == "detector_alarm":
            out[did] = "alarm"
        elif e.event == "detector_clear":
            out.pop(did, None)
    return out


def recent_events(timeline: Timeline, t: float, hold: float = EVENT_HOLD_S, limit: int = 4) -> list[Event]:
    return [e for e in timeline.events if t - hold <= e.t <= t][-limit:]


# ----------------------------------------------------------------------------------------- shots
@dataclass
class Shot:
    start: float
    end: float
    kind: str                      # "establish" | "follow" | "event"
    robot: Optional[str] = None
    focus: Optional[np.ndarray] = None
    label: str = ""


def plan_shots(timeline: Timeline, duration: Optional[float] = None, min_shot: float = MIN_SHOT_S,
               establish: float = ESTABLISH_S) -> list[Shot]:
    """Establishing shot, then cut on each event or job change. Cuts closer than `min_shot` are dropped,
    except safety events (gas, leak, alert), which always cut and push the next cut back."""
    duration = timeline.duration if duration is None else duration
    cues = []
    for e in timeline.events:
        if e.event == "shift_start" or e.robot is None and e.position is None:
            continue
        urgent = any(w in e.event for w in ("gas", "leak", "alert", "finding", "alarm"))
        cues.append((e.t, 0 if urgent else 1, "event" if e.position is not None else "follow", e.robot,
                     e.position, e.text()))
    for rid, tr in timeline.tracks.items():
        for t, job in tr.job_changes()[1:]:
            if job:
                cues.append((t, 1, "follow", rid, None, f"{rid} -> {job}"))
    cues.sort(key=lambda c: (c[0], c[1]))
    shots = [Shot(0.0, min(establish, duration), "establish", label="establishing")]
    urgent_of = {}
    for t, prio, kind, robot, focus, label in cues:
        if robot is not None and robot not in timeline.tracks:
            robot = None
        if robot is None and focus is None:
            continue
        last = shots[-1]
        gap = URGENT_GAP_S if prio == 0 and not urgent_of.get(id(last)) else min_shot
        start = max(t, establish, last.start + gap)
        if start >= duration or start - t > STALE_CUE_S:
            continue                                   # too late to be news
        if abs(start - last.start) < 1e-6 and last.kind != "establish":
            if prio <= (0 if urgent_of.get(id(last)) else 1):     # a later cue for the same slot wins
                last.kind, last.robot, last.focus, last.label = kind, robot, focus, label
                urgent_of[id(last)] = prio == 0
            continue
        last.end = start
        shot = Shot(start, duration, kind, robot, focus, label)
        urgent_of[id(shot)] = prio == 0
        shots.append(shot)
    shots[-1].end = duration
    shots = _fill_long_shots(shots, timeline)
    return [s for s in shots if s.end - s.start > 1e-6]


def _fill_long_shots(shots: list[Shot], timeline: Timeline, longest: float = MAX_SHOT_S) -> list[Shot]:
    """Split shots longer than `longest` by cutting to another robot that is on a job (or just moving)."""
    out = []
    for s in shots:
        out.append(s)
        while out[-1].end - out[-1].start > longest:
            cur = out[-1]
            t = cur.start + longest * 0.6
            busy = [rid for rid, tr in timeline.tracks.items() if rid != cur.robot and tr.job(t)]
            moving = [rid for rid, tr in timeline.tracks.items() if rid != cur.robot and tr.speed(t) > 0.2]
            pick = (busy or moving or [rid for rid in timeline.tracks if rid != cur.robot] or [None])[0]
            if pick is None:
                break
            if len(busy) > 1:                       # rotate: the one shown least recently
                seen = {sh.robot: sh.start for sh in out}
                pick = min(busy, key=lambda r: seen.get(r, -1.0))
            job = timeline.tracks[pick].job(t)
            nxt = Shot(t, cur.end, "follow", pick, None, f"{pick} {job or timeline.tracks[pick].state(t)}")
            cur.end = t
            out.append(nxt)
    return out


def shot_at(shots: list[Shot], t: float) -> Shot:
    for s in shots:
        if s.start <= t < s.end:
            return s
    return shots[-1]


# Follow-camera framing per robot kind: (back distance, height above subject, focal length).
FRAMING = {"Q": (5.0, 1.5, 24.0), "C": (3.8, 1.1, 26.0), "A": (8.0, 1.5, 26.0), "R": (6.0, 1.5, 24.0)}
RIG_CENTER = np.array([2.0, -2.0, 38.0])
DRONE_CAMERA_MIN_Z = TOP_DECK + 15.0


def _smoothed(track: Track, t: float, window: float = 1.2) -> np.ndarray:
    ts = np.linspace(t - window / 2, t + window / 2, 7)
    return np.mean([track.position(x) for x in ts], axis=0)


def camera_for(shot: Shot, timeline: Timeline, t: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Eye, target and focal length. The camera drifts slowly around its subject during a shot."""
    u = t - shot.start
    if shot.kind == "establish":
        a = math.radians(-60.0 + 4.0 * u)
        r = 175.0 - 6.0 * u
        eye = RIG_CENTER + np.array([r * math.cos(a), r * math.sin(a), 46.0 - 2.0 * u])
        return eye, RIG_CENTER + np.array([0.0, 0.0, 4.0]), 30.0
    track = timeline.tracks.get(shot.robot) if shot.robot else None
    if track is not None:
        subject = _smoothed(track, t)
        kind = track.kind
    else:
        subject, kind = np.asarray(shot.focus, float), "Q"
    back, up, focal = FRAMING.get(kind, FRAMING["Q"])
    if shot.kind == "event":
        focus = np.asarray(shot.focus, float) if shot.focus is not None else subject
        spread = float(np.linalg.norm(focus - subject)) if track is not None else 0.0
        near = track is not None and spread < 8.0
        target = focus + np.array([0.0, 0.0, 1.0]) + ((subject - focus) * 0.45 if near else 0.0)
        back, up, focal = 12.0 + (spread * 0.6 if near else 0.0), 2.6, 22.0
    else:
        target = subject + np.array([0.0, 0.0, 0.6 if kind != "A" else 0.0])
    anchor = track.position(shot.start) if track is not None else target
    # Ground robots are filmed from inboard looking out (no deck-edge railings in front of the lens, sea
    # behind the subject); drones from outboard looking in, with the topsides behind them.
    side = RIG_CENTER[:2] - anchor[:2] if kind in GROUND_KINDS else anchor[:2] - RIG_CENTER[:2]
    base = math.atan2(side[1], side[0]) if np.linalg.norm(side) > 1.0 else -math.pi / 2
    a = base + math.radians(-25.0 + 3.0 * u)
    eye = target + np.array([back * math.cos(a), back * math.sin(a), up])
    if kind in GROUND_KINDS:
        eye[2] = max(eye[2], deck_height(target[2]) + 2.2)
    elif kind == "A" and shot.kind == "follow":
        # A drone crosses the whole topsides in one shot; a chase camera at its height would pass through
        # modules and tanks, so it rides high above them and looks down at the drone.
        eye[2] = max(eye[2], target[2] + 2.5, DRONE_CAMERA_MIN_Z)
    return eye, target, focal


# ------------------------------------------------------------------------------------ projection
def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world 4x4 (column vectors), USD convention: camera looks down -Z, +Y up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    return m


def project(points, cam_to_world: np.ndarray, focal: float, width: int, height: int,
            aperture: float = HORIZONTAL_APERTURE) -> list[Optional[tuple[float, float]]]:
    """Pixel coordinates of world points (None if behind the camera)."""
    inv = np.linalg.inv(cam_to_world)
    out = []
    fx = focal / aperture * width
    for p in np.atleast_2d(np.asarray(points, float)):
        c = inv @ np.append(p, 1.0)
        if c[2] > -0.1:
            out.append(None)
            continue
        out.append((width / 2 + fx * c[0] / -c[2], height / 2 - fx * c[1] / -c[2]))
    return out


# ------------------------------------------------------------------------------------------- HUD
def draw_fleet_hud(rgb: np.ndarray, timeline: Timeline, t: float, shot: Shot,
                   tags: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    d = ImageDraw.Draw(im, "RGBA")
    scale = im.height / 720.0
    try:
        font = ImageFont.truetype("consola.ttf", max(11, int(15 * scale)))
        small = ImageFont.truetype("consola.ttf", max(10, int(13 * scale)))
    except OSError:
        font = small = ImageFont.load_default()
    line = int(20 * scale)
    pad = int(10 * scale)
    rows = list(timeline.tracks.items())
    w = int(430 * scale)
    d.rectangle([pad, pad, pad + w, pad + line * (len(rows) + 1) + pad], fill=(0, 0, 0, 140))
    d.text((pad * 2, pad + 4), f"FLEET  t={t:5.1f}s", fill=(120, 255, 160), font=font)
    for k, (rid, tr) in enumerate(rows):
        y = pad + line * (k + 1) + 6
        color = state_color(tr.state(t))
        r = int(5 * scale)
        d.ellipse([pad * 2, y + 3, pad * 2 + 2 * r, y + 3 + 2 * r], fill=color + (255,))
        job = tr.job(t) or "-"
        style = ROBOT_COLORS.get(rid) or ((255, 255, 255) if rid == shot.robot else (215, 215, 215))
        d.text((pad * 2 + 3 * r, y), f"{display_name(rid):<7} {tr.state(t):<11} {job}"[:52], fill=style, font=font)
    events = recent_events(timeline, t)
    if events:
        h = line * len(events) + pad
        y0 = im.height - pad - h
        d.rectangle([pad, y0, pad + int(620 * scale), im.height - pad], fill=(0, 0, 0, 140))
        for k, e in enumerate(events):
            alert = any(word in e.event for word in ("gas", "leak", "alert", "finding", "fail", "alarm"))
            col = (255, 90, 60) if alert else (235, 235, 235)
            d.text((pad * 2, y0 + pad // 2 + k * line), f"{e.t:5.1f}s  {e.text()}"[:78], fill=col, font=small)
    for rid, (x, y) in (tags or {}).items():
        if not (0 <= x < im.width and 0 <= y < im.height):
            continue
        color = state_color(timeline.tracks[rid].state(t))
        name = display_name(rid)
        own = ROBOT_COLORS.get(rid, color)
        tw = int((8 * len(name) + 14) * scale)
        d.rectangle([x - tw / 2, y - line, x + tw / 2, y - 3], fill=(0, 0, 0, 150), outline=own + (255,))
        d.text((x - tw / 2 + 4, y - line + 2), name, fill=own if rid in ROBOT_COLORS else (255, 255, 255), font=small)
        d.line([x, y - 3, x, y + 6], fill=color + (220,), width=2)
    return np.asarray(im)


# ===================================================================================== Isaac main
def main() -> None:
    import argparse
    import time as clock

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeline", default=str(DEFAULT_TIMELINE))
    parser.add_argument("--fake", action="store_true", help="overwrite the timeline with the fake 60 s one")
    parser.add_argument("--out", default=str(OUT_DIR / "fleet_replay.mp4"))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--subframes", type=int, default=4, help="RTX subframes per video frame")
    parser.add_argument("--max-frames", type=int, default=0, help="stop early, for quick tests")
    parser.add_argument("--start", type=float, default=0.0, help="timeline second to start at")
    parser.add_argument("--stills", type=int, default=8, help="number of sample frames saved as PNG")
    parser.add_argument("--at", default="", help="comma-separated times: render only these as stills")
    parser.add_argument("--preset", default="golden", help="cinematic preset (night/golden/storm) or 'none'")
    parser.add_argument("--no-ocean", action="store_true", help="skip the per-frame ocean animation")
    parser.add_argument("--bitrate", type=float, default=4.0, help="MP4 bitrate in Mbit/s (default 4, ~30 MB/min)")
    args = parser.parse_args()

    timeline_path = Path(args.timeline)
    if args.fake or not timeline_path.exists():
        write_fake_timeline(timeline_path)
        print(f"FAKE_TIMELINE {timeline_path}", flush=True)
    timeline = load_timeline(timeline_path)
    assets = rig_assets()
    shots = plan_shots(timeline)
    for s in shots:
        print(f"SHOT {s.start:5.1f}-{s.end:5.1f} {s.kind:<9} {s.robot or '-':<3} {s.label}", flush=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "sim"))
    deps = Path(os.environ.get("ISAAC_DEPS", Path.home() / ".isaac-deps"))
    if deps.exists():
        sys.path.insert(0, str(deps))

    import omni.kit.app
    from PIL import Image

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("omni.replicator.core", True)
    manager.set_extension_enabled_immediate("omni.videoencoding", True)
    import omni.replicator.core as rep
    from video_encoding import get_video_encoding_interface

    import fleet_scene

    scene = fleet_scene.FleetScene.open(app, timeline, preset=None if args.preset == "none" else args.preset,
                                        ocean=not args.no_ocean)
    product = rep.create.render_product(fleet_scene.CAMERA_PATH, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def frame_at(t: float) -> np.ndarray:
        shot = shot_at(shots, t)
        eye, target, focal = camera_for(shot, timeline, t)
        scene.update(t, assets)
        scene.set_camera(eye, target, focal)
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        data = np.asarray(annotator.get_data())
        if not data.size:
            return np.zeros((args.height, args.width, 3), np.uint8)
        cam = look_at(eye, target)
        tags = {}
        for rid, tr in timeline.tracks.items():
            pos = scene.tag_point(rid, t)
            px = project([pos], cam, focal, args.width, args.height)[0]
            if px is not None and np.linalg.norm(pos - eye) < 120.0:
                tags[rid] = px
        return draw_fleet_hud(data[..., :3].astype(np.uint8), timeline, t, shot, tags)

    for _ in range(3):
        frame_at(args.start)                               # shader / texture warm-up

    if args.at:
        for value in args.at.split(","):
            t = float(value)
            for _ in range(2):
                rgb = frame_at(t)
            path = OUT_DIR / f"still_t{t:04.1f}.png"
            Image.fromarray(rgb).save(path)
            print(f"STILL {path}", flush=True)
        rep.orchestrator.wait_until_complete()
        app.close()
        return

    frames = int((timeline.duration - args.start) * args.fps)
    if args.max_frames:
        frames = min(frames, args.max_frames)
    out = Path(args.out)
    import carb.settings
    carb.settings.get_settings().set_int("/exts/omni.videoencoding/bitrate", int(args.bitrate * 1e6))
    encoder = get_video_encoding_interface()
    if not encoder.start_encoding(str(out), float(args.fps), frames, True):
        raise RuntimeError(f"could not start video encoding for {out}")
    every = max(frames // max(args.stills, 1), 1)
    written = 0
    started = clock.time()
    for k in range(frames):
        t = args.start + k / args.fps
        rgb = frame_at(t)
        rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
        if encoder.encode_next_frame_from_buffer(np.ascontiguousarray(rgba), args.width, args.height):
            written += 1
        if args.stills and k % every == 0:
            Image.fromarray(rgb).save(OUT_DIR / f"replay_{k:04d}.png")
        if k % args.fps == 0:
            print(f"FRAME {k}/{frames} t={t:.1f}s elapsed={clock.time() - started:.0f}s", flush=True)
    encoder.finalize_encoding()
    seconds = clock.time() - started
    if not out.exists() or out.stat().st_size == 0 or written == 0:
        raise RuntimeError(f"no video written to {out}")
    print(f"VIDEO_WRITTEN {out} frames={written}/{frames} seconds={seconds:.0f} "
          f"per_frame={seconds / max(written, 1):.2f}s size_mb={out.stat().st_size / 1e6:.1f}", flush=True)
    rep.orchestrator.wait_until_complete()
    app.close()


if __name__ == "__main__":
    main()
    os._exit(0)

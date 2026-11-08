import base64
import io
import json
import random
from pathlib import Path

import numpy as np

from rig.captions import robot_entry

HOOKS = ("spawn_patch", "set_glare", "occlude_camera", "force_spike", "person_at")
EXTRA = ("camera",)
CONTROL_EVENTS = ("cut_link", "intent")
MISSION = ("scenario", "disturb", "select", "views", "force")
SCENARIO_IDS = ("gas", "patrol", "repair", "flyby")
SCENARIO = ("set_weather", "drop_obstacle", "drop_can", "close_walkway", "close_stairs", "clear", "gas_leak",
            "control_action", "battery_fault", "approve", "reject", "weld_job")
EVENTS = HOOKS + EXTRA + CONTROL_EVENTS + SCENARIO + MISSION
WEATHER = ("calm", "wind", "rain", "fog", "storm")
FLEET = ("drop_obstacle", "drop_can", "close_walkway", "close_stairs", "clear", "set_weather", "weld_job")
TIMES = ("dusk", "night")
RATES = {"SOAK": 0.15, "SCRUB": 0.35}
DWELL = 1.3
THUMB_W = 320
PATCH_OFFSET = (0.45, 0.1)


def _bits(mask) -> list[str]:
    m = np.asarray(mask, bool)
    return ["".join("1" if v else "0" for v in row) for row in m]


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def thumb(img: np.ndarray, mask=None) -> dict:
    from PIL import Image
    rgb = np.ascontiguousarray(np.asarray(img)[..., :3]).astype(np.uint8).copy()
    if mask is not None and np.shape(mask) == rgb.shape[:2]:
        m = np.asarray(mask, bool)
        inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
        edge = m & ~inner
        edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        rgb[edge] = (255, 59, 48)
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb).resize((THUMB_W, round(THUMB_W * h / w)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return {"jpg": base64.b64encode(buf.getvalue()).decode(), "w": w, "h": h}


def evidence(orch, h: dict) -> dict:
    o = orch
    st = o.state.value
    last = o.records[-1] if o.records else None
    ev = dict(last.evidence) if last is not None and last.to == st else {}
    force = h.get("force_n")
    if st == "BACK_OFF":
        force = max(float(ev.get("force") or 0.0), float(force or 0.0))
    ev.update(force=force, force_limit=h.get("force_limit"), rust_fraction=h.get("rust_fraction"),
              detector_conf=h.get("detector_conf"), gemini=h.get("gemini"), patches=len(getattr(o, "patches", ())),
              soak_left=(h.get("soak") or {}).get("left"))
    if hasattr(o.bridge, "person_distance"):
        ev["person_m"] = o.bridge.person_distance()
    if hasattr(o, "_gas"):
        ev.update(gas=o._gas(), gas_stop=o.lcfg.gas_stop, flange=o.flange_id, bolt=o.target_bolt,
                  job="gas" if o.flange_id is not None else "rust")
    return ev


class HudFeed:
    def __init__(self, orch, rid: str = "unit-1", name: str = "Unit 1 · rust", pose=None, camera=None, peers=(),
                 control=None, zone: str = "A", extra=None, extras=None):
        self.orch = orch
        self.extras = dict(extras or {})
        self.rid, self.name, self.pose, self.camera, self.peers = rid, name, pose, camera, list(peers)
        self.seen = 0
        self.last_perceived = None
        self.verdict_seen = None
        self.source = "EDGE"
        self.control, self.zone, self.extra = control, zone, extra

    def header(self) -> dict:
        o = self.orch
        world, pipe = o.world, o.pipe
        records = o.records[self.seen:]
        self.seen = len(o.records)
        verdict = getattr(o, "advisor_last", None)
        if records:
            self.source = "CLOUD" if verdict is not None and verdict is not self.verdict_seen else "EDGE"
            self.verdict_seen = verdict
        conf = None
        if o.last is not None:
            cs = [p.mean_conf for p in o.last.result.patches]
            conf = round(float(max(cs) if cs else o.last.result.max_conf), 3)
        theta, z = o.arm.ee_theta_z()
        i, j = pipe.theta_z_to_cell(theta, z)
        out = {
            "t": round(float(o.t), 2),
            "state": o.state.value,
            "hud": o.overlay_text(),
            "events": [json.loads(json.dumps(r.to_json(), default=_plain)) for r in records],
            "rust_grid": np.round(np.clip(_np(world.rust[0]), 0, 1), 2).tolist(),
            "sprayed": _bits(_np(world.sprayed[0])),
            "painted": _bits(_np(world.painted[0])),
            "target": _bits(o.target),
            "brush": [int(i), int(j)],
            "force_n": round(float(o.arm.current_force()), 2),
            "force_limit": float(o.cfg.force_limit),
            "detector_conf": conf,
            "rust_fraction": None if o.rust_frac is None else round(float(o.rust_frac), 4),
            "gemini": self._gemini(verdict),
            "source": self.source,
            "soak": {"left": round(max(0.0, o.soak_until - o.t), 2), "total": float(o.cfg.soak_s)}
            if o.state.value == "SOAK" else None,
        }
        if hasattr(o, "gas_telemetry"):
            out["gas"] = o.gas_telemetry()
        rob = self.robot(out)
        out["caption"] = rob["caption"]
        out["robots"] = [rob] + [p.robot() for p in self.peers]
        if self.camera is not None:
            cam = self.camera()
            if cam is not None:
                out["camera"] = cam
        for k, fn in self.extras.items():
            v = fn()
            if v is not None:
                out[k] = v
        if o.last is not None and o.last is not self.last_perceived and o.last_image is not None:
            self.last_perceived = o.last
            out["inspection"] = thumb(o.last_image, o.last.result.mask)
        if self.control is not None:
            from policy.control.adapters import from_header, merge
            snap = from_header(out, self.rid, self.zone)
            out["control"] = self.control.tick(merge(snap, self.extra() if self.extra else None))
        return out

    def robot(self, h: dict | None = None) -> dict:
        h = self._vitals() if h is None else h
        pos = None if self.pose is None else self.pose()
        return robot_entry(self.rid, self.name, self.orch.state.value, evidence(self.orch, h), pos)

    def _vitals(self) -> dict:
        o = self.orch
        return {"force_n": float(o.arm.current_force()), "force_limit": float(o.cfg.force_limit),
                "rust_fraction": o.rust_frac,
                "soak": {"left": max(0.0, o.soak_until - o.t)} if o.state.value == "SOAK" else None}

    def _gemini(self, verdict):
        if verdict is None:
            return None
        calls = getattr(self.orch.advisor, "calls", None) or [{}]
        return {**verdict.summary(), "question": calls[-1].get("question")}


def _plain(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def check_event(event: dict) -> dict:
    name = event.get("name")
    if name not in EVENTS:
        raise ValueError(f"unknown event {name!r}; expected one of {EVENTS}")
    args = dict(event.get("args") or {})
    lv = args.get("level", "storm")
    if name == "set_weather" and not isinstance(lv, (int, float)) and lv not in WEATHER:
        raise ValueError(f"unknown weather {lv!r}; expected one of {WEATHER} or 0..1")
    if name == "set_weather" and args.get("time") is not None and args["time"] not in TIMES:
        raise ValueError(f"unknown time {args['time']!r}; expected one of {TIMES}")
    if name == "scenario" and args.get("id") not in SCENARIO_IDS:
        raise ValueError(f"unknown scenario {args.get('id')!r}; expected one of {SCENARIO_IDS}")
    return {"name": name, "args": args}


def is_fleet_event(name: str, args: dict) -> bool:
    return name in FLEET or (name == "person_at" and any(k in args for k in ("xy", "xyz", "at")))


def apply_event(bridge, event: dict, control=None, fleet=None) -> bool:
    ev = check_event(event)
    name, args = ev["name"], ev["args"]
    if is_fleet_event(name, args):
        if fleet is not None:
            fleet.event(name, args)
        if name == "person_at":
            return fleet is not None
    if name in CONTROL_EVENTS:
        if control is None:
            raise ValueError("no control centre attached")
        if name == "cut_link":
            control.cut_link(bool(args.get("on", True)))
        else:
            control.intent(str(args.get("text") or ""))
        return True
    if name in SCENARIO and not hasattr(bridge, name):
        if hasattr(bridge, "scenario"):
            bridge.scenario(name, **args)
            return True
        return False
    if name == "spawn_patch" and ("theta" not in args or "z" not in args):
        theta, z = bridge.ee_theta_z()
        args.setdefault("theta", float(theta) + PATCH_OFFSET[0])
        args.setdefault("z", float(z) + PATCH_OFFSET[1])
    getattr(bridge, "set_shot" if name == "camera" else name)(**args)
    return True


def playback_times(snaps: list[dict]) -> list[float]:
    pts, pt = [], 0.0
    for k, s in enumerate(snaps):
        if k:
            prev = snaps[k - 1]
            pt += (s["t"] - prev["t"]) * RATES.get(prev["state"], 1.0)
            if s["state"] != prev["state"]:
                pt += DWELL
            if s.get("gemini") and s["gemini"] != prev.get("gemini"):
                pt += s["gemini"].get("latency_ms", 0) / 1000
        pts.append(round(pt, 3))
    return pts


class ScriptedAdvisor:
    model = "gemini-robotics-er-2-preview (scripted)"

    def __init__(self, veto: int = 1):
        self.veto, self.calls, self.last_error = veto, [], None
        self.rng = random.Random(7)

    def verify(self, image, question="rust"):
        from policy.gemini_verifier import GeminiVerdict
        img = np.asarray(image, np.int32)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        score = r - b
        ys, xs = np.nonzero((score > 55) & (r > g + 10))
        if not len(xs):
            ys, xs = np.nonzero(score >= np.sort(score, axis=None)[-40])
        rust = question == "rust" or self.veto > 0
        if question == "clean" and self.veto > 0:
            self.veto -= 1
        points = []
        if rust and len(xs):
            order = np.argsort(xs)
            for q in (0.15, 0.5, 0.85):
                k = order[int(q * (len(order) - 1))]
                points.append((float(xs[k]), float(ys[k])))
        verdict = GeminiVerdict("rust" if rust else "clean", 0.82 if rust else 0.91,
                                "orange-brown residue still visible along the scrubbed band" if rust
                                else "uniform bare steel, no oxide texture left", points,
                                latency_ms=self.rng.uniform(900, 1500), model=self.model)
        self.calls.append({"question": question, **verdict.summary()})
        return verdict

    def confident(self, verdict):
        return verdict is not None and verdict.verdict != "cannot_tell" and verdict.confidence >= 0.6


FLEET_GAP = 5.0
FLEET_CAM = dict(eye=(7.5, 8.5, 6.0), target=(0.0, 0.0, 0.0))
FLEET_SCRIPT = {
    "unit-1": ("Unit 1 · rust", [(0, 540, 540), (3, 700, 470), (6, 820, 425), (24, 820, 425)],
               [(0, "PATROL"), (6, "DETECT"), (8, "APPROACH"), (9.5, "SPRAY"), (11, "SOAK"), (14, "SCRUB"),
                (22, "VERIFY")], (2.0, 4.2)),
    "unit-2": ("Unit 2 · gas", [(0, -320, 260), (5, 300, 340), (8, 410, 380), (24, 410, 380)],
               [(0, "GAS_CHECK"), (2, "LEAK_APPROACH"), (8.5, "LEAK_INSPECT"), (12, "RETORQUE"), (17, "LEAK_VERIFY"),
                (21, "DONE")], None),
    "unit-3": ("Unit 3", [(0, 1010, 300), (9, 1100, 262), (24, 1100, 262)],
               [(0, "PATROL"), (9, "TO_DOCK"), (13, "CHARGING")], None),
}
FLEET_WHY = {"PATROL": "deck 2 patrol assigned", "DETECT": "at patrol pose -> inspect", "APPROACH": "rust confirmed -> approach",
             "SPRAY": "at standoff -> spray remover", "SOAK": "sprayed -> soak", "SCRUB": "soak complete",
             "VERIFY": "rust below target -> verify", "GAS_CHECK": "gas 0.08 LEL above clear 0.05 -> find the leak",
             "LEAK_APPROACH": "gas rising upwind -> approach flange F2", "LEAK_INSPECT": "at flange F2 -> read bolts",
             "RETORQUE": "bolt 3 loose by 0.11 rad -> re-torque", "LEAK_VERIFY": "re-torqued -> verify gas",
             "DONE": "gas back to normal", "TO_DOCK": "battery 22 % below 25 % -> dock",
             "CHARGING": "on pad 1 -> charge"}
GAS_SENSORS = (("GD-1", "Flange F2", 1.0), ("GD-2", "Separator", 0.42), ("GD-3", "Compressor", 0.12),
               ("GD-4", "Wellhead", 0.05))


def _at(keys, t):
    for k in range(len(keys) - 1, -1, -1):
        if t >= keys[k][0]:
            return keys[k][1]
    return keys[0][1]


def _fleet_ev(state, t, t0):
    u = (t - t0)
    if state == "SCRUB":
        return {"force": 30.5 + 2.5 * np.sin(3.1 * t), "rust_fraction": max(0.04, 0.62 - 0.07 * u)}
    if state == "SOAK":
        return {"soak_left": max(0.0, 30.0 - 10.0 * u)}
    if state == "DETECT":
        return {"detector_conf": 0.88}
    if state == "VERIFY":
        return {"rust_fraction": 0.03}
    if state in ("TO_DOCK", "CHARGING"):
        return {"soc": 0.22 + (0.01 * u if state == "CHARGING" else 0.0)}
    gas = {"GAS_CHECK": 0.08 + 0.02 * u, "LEAK_APPROACH": min(0.18, 0.12 + 0.012 * u), "LEAK_INSPECT": 0.18,
           "RETORQUE": max(0.06, 0.18 - 0.025 * u), "LEAK_VERIFY": max(0.02, 0.05 - 0.008 * u)}.get(state)
    return {"gas": gas, "flange": "F2", "bolt": 3} if gas is not None else {"job": "gas"} if state == "DONE" else {}


def mock_gas(g: float) -> dict:
    g = round(float(g), 3)
    sensors = [{"id": i, "at": at, "lel": round(max(0.01, g * k), 3)} for i, at, k in GAS_SENSORS]
    return {"sensors": sensors, "max": max(s["lel"] for s in sensors), "alarm": g >= 0.10,
            "wind": {"dir_deg": 225, "speed_ms": 6.0}}


def mock_control(gas: dict) -> dict:
    g = gas["max"]
    if gas["alarm"]:
        situation = [f"Gas at {100 * g:.0f} % LEL near flange F2, wind from the south-west.",
                     "Unit 2 is on the leak; Unit 1 keeps scrubbing outside the 8 m zone."]
        acts = [("retorque_f2", "Re-torque flange F2 bolts", "unit-2", 0.92, "Loose bolt 3 is the likely source."),
                ("hold_unit1", "Pause Unit 1 until gas clears", "unit-1", 0.71, "It is downwind if the plume grows."),
                ("call_human", "Call the deck supervisor", None, 0.38, "Only if the gas passes 20 % LEL.")]
        key = "fleet_gas"
    else:
        situation = ["All units nominal. No gas above background; Unit 3 charging."]
        acts = [("continue", "Continue the current plan", None, 0.9, "Nothing has changed."),
                ("survey", "Queue a survey of deck 2", "unit-3", 0.46, "Unit 3 is idle once charged.")]
        key = "fleet_nominal"
    return {"mode": "CLOUD", "model": "gemini-4-argon (scripted)", "replay": True, "banner": None, "key": key,
            "situation": situation, "anomalies": [], "vetoed": [], "link": "UP",
            "actions": [{"id": i, "label": l, "robot": r, "confidence": c, "rationale": why, "source": "cloud",
                         "kind": i, "rank": n + 1} for n, (i, l, r, c, why) in enumerate(acts)]}


def mock_sensors(robots: list[dict], evs: dict, t: float) -> dict:
    out = {}
    for k, r in enumerate(robots):
        ev = evs[r["id"]]
        out[r["id"]] = {"force_n": round(float(ev.get("force") or 0.0), 1), "gas_lel": round(float(ev.get("gas") or 0.01), 3),
                        "thermal_c": round(31.0 + 4 * k + 0.6 * np.sin(0.7 * t + k), 1),
                        "battery": round(float(ev.get("soc") or (0.82 - 0.17 * k)), 3),
                        "reach": round(0.35 + 0.3 * float(r["state"] in ("SCRUB", "RETORQUE", "SPRAY")), 2),
                        "highlight": ["force_n"] if r["state"] == "SCRUB" else []}
    return out


def mock_orchestrator(states: dict, gas: dict) -> dict:
    jobs = [{"id": "rust_repair-001", "kind": "rust_repair", "target": "pipe_rack_A", "level": "SUPV", "priority": 3,
             "status": "DONE" if states["unit-1"] == "VERIFY" else "RUNNING", "robot": "unit-1"},
            {"id": "gas_survey-002", "kind": "gas_survey", "target": "flange F2", "level": "AUTO", "priority": 1,
             "status": "DONE" if states["unit-2"] == "DONE" else "RUNNING", "robot": "unit-2"},
            {"id": "gauge_read-003", "kind": "gauge_read", "target": "gauge_03", "level": "AUTO", "priority": 4,
             "status": "QUEUED", "robot": None},
            {"id": "valve_isolate-004", "kind": "valve_isolate", "target": "valve_05", "level": "SUPV", "priority": 2,
             "status": "WAITING_APPROVAL" if gas["alarm"] else "QUEUED", "robot": None}]
    approvals = [{"job": "valve_isolate-004", "kind": "valve_isolate", "target": "valve_05",
                  "text": "Gemini: close valve_05 to starve the F2 leak. Unit 1 stays outside the 8 m zone; "
                          "nobody is on the walkway. Needs your approval (SUPV)."}] if gas["alarm"] else []
    return {"jobs": jobs, "approvals": approvals}


FLEET_RIG = {"unit-1": [(0, -1.0, 4.0), (24, -1.0, 4.0)],
             "unit-2": [(0, 9.0, -4.5), (2, -1.0, -4.5), (5, -1.0, 11.0), (6.5, -10.0, 11.0), (7.5, -10.0, 20.5), (8.5, -6.5, 20.5), (24, -6.5, 20.5)],
             "unit-3": [(0, 20.0, -9.0), (3, 9.0, -9.0), (4, 9.0, -4.5), (6, -1.0, -4.5), (9, -1.0, 11.0), (11, -10.0, 11.0), (13, -12.0, 9.4), (24, -12.0, 9.4)]}
F2 = [-5.88, 23.98, 43.6]


def mock_rig(robots: list[dict], evs: dict, t: float, gas: dict) -> dict:
    out = []
    for k, r in enumerate(robots):
        keys = FLEET_RIG[r["id"]]
        ts = [p[0] for p in keys]
        x, y = float(np.interp(t, ts, [p[1] for p in keys])), float(np.interp(t, ts, [p[2] for p in keys]))
        x2, y2 = float(np.interp(t + 0.5, ts, [p[1] for p in keys])), float(np.interp(t + 0.5, ts, [p[2] for p in keys]))
        yaw = float(np.arctan2(y2 - y, x2 - x)) if abs(x2 - x) + abs(y2 - y) > 1e-6 else 0.0
        soc = evs[r["id"]].get("soc") or round(0.82 - 0.17 * k, 3)
        out.append({"id": r["id"], "name": r["name"], "state": r["state"], "job_state": r["state"],
                    "pose": {"x": round(x, 2), "y": round(y, 2), "z": 42.5, "yaw": round(yaw, 3)}, "deck": "top_deck",
                    "battery": {"soc": soc}, "task": {"unit-1": "rust:pipe_rack_A", "unit-2": "gas:F2"}.get(r["id"])})
    zones = [{"kind": "gas", "id": "gas:F2", "xyz": F2, "radius": 8.0}] if gas["alarm"] else []
    tasks = [{"id": "rust:pipe_rack_A", "kind": "rust", "xyz": [-1.0, 5.0, 43.6], "status": "active", "robot": "unit-1"},
             {"id": "gas:F2", "kind": "gas", "xyz": F2, "status": "active" if gas["alarm"] else "done", "robot": "unit-2"}]
    return {"schema": "fleet_state.v1", "t": round(t, 2), "robots": out, "zones": zones, "tasks": tasks}


def mock_fleet(pt0: float, t_start: float, dt: float = 0.5, length: float = 24.0) -> list[dict]:
    from rig.camera import look_at, unproject_ground
    cam = look_at(**FLEET_CAM)
    snaps, prev = [], {}
    for k in range(int(length / dt) + 1):
        t = k * dt
        robots, events, gas, evs = [], [], 0.01, {}
        for rid, (name, path, states, hidden) in FLEET_SCRIPT.items():
            ts = [p[0] for p in path]
            u, v = np.interp(t, ts, [p[1] for p in path]), np.interp(t, ts, [p[2] for p in path])
            state = _at(states, t)
            since = max(s[0] for s in states if s[0] <= t)
            occluded = hidden is not None and hidden[0] <= t < hidden[1]
            ev = evs[rid] = _fleet_ev(state, t, since)
            gas = max(gas, ev.get("gas") or 0.0)
            robots.append(robot_entry(rid, name, state, ev, unproject_ground(cam, u, v), occluded))
            if prev.get(rid) != state:
                events.append({"t": round(t_start + t, 2), "from": prev.get(rid, "IDLE"), "to": state,
                               "reason": f"{name.split(' ·')[0]}: {FLEET_WHY[state]}"})
                prev[rid] = state
        snaps.append({"t": round(t_start + t, 2), "pt": round(pt0 + t, 3), "state": robots[0]["state"], "shot": 2,
                      "caption": robots[0]["caption"], "robots": robots, "camera": cam, "events": events,
                      "gas": mock_gas(gas), "control": mock_control(mock_gas(gas)), "sensors": mock_sensors(robots, evs, t),
                      "orchestrator": mock_orchestrator({r["id"]: r["state"] for r in robots}, mock_gas(gas)),
                      "fleet": mock_rig(robots, evs, t, mock_gas(gas))})
    return snaps


def record_mock(path, log_path: str = "runs/repair/decisions.jsonl") -> dict:
    from policy.demo_mock import build
    from policy.orchestrator import S
    orch, bridge, world, perception, arm, pipe = build(log_path, verbose=False)
    orch.advisor = ScriptedAdvisor()
    feed = HudFeed(orch)
    snaps: list[dict] = []
    last = {"t": -1.0}

    def snap(force=False):
        if force or orch.t - last["t"] >= 0.5:
            last["t"] = orch.t
            snaps.append(feed.header())

    advance = bridge.advance

    def tracked(dt):
        advance(dt)
        snap()
    bridge.advance = tracked
    arm.spike_at_step = 40
    flags = set()
    snap(True)
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN) and len(orch.records) < 400:
        prev = orch.state
        orch.step()
        if prev == S.SPRAY and "patch" not in flags:
            flags.add("patch")
            th = bridge.dmg_theta + 0.75
            bridge.events.append((bridge.t + 12.0, "new rust", lambda: world.inject_patch(th, 0.13, size=0.04, level=0.8)))
        if orch.state == S.VERIFY and prev == S.SCRUB and "occl" not in flags:
            flags.add("occl")
            perception.occluded_frames = 2
            bridge.person_window = (bridge.t + 0.5, bridge.t + 2.5)
        snap(True)
    for k, pt in enumerate(playback_times(snaps)):
        snaps[k]["pt"] = pt
    for k in range(len(snaps) - 1, 0, -1):
        for key in ("rust_grid", "sprayed", "painted", "target"):
            if snaps[k][key] == snaps[k - 1][key]:
                del snaps[k][key]
    from policy.control.mock import attach_control
    attach_control(snaps, Path(path).parent / "control_calls.jsonl")
    session = {"snapshots": snaps, "duration": snaps[-1]["pt"]}
    session["fleet"] = mock_fleet(snaps[-1]["pt"] + FLEET_GAP, snaps[-1]["t"] + 20.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(session, separators=(",", ":")))
    return session


if __name__ == "__main__":
    import sys
    s = record_mock(sys.argv[1] if len(sys.argv) > 1 else "runs/hud/mock_session.json")
    print(f"{len(s['snapshots'])} snapshots, {s['duration']:.1f} s playback")

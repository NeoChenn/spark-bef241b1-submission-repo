from __future__ import annotations

import copy
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

from policy.fleet.fleet import Event
from policy.fleet.jobs import SETUP_S
from rig.captions import caption, fleet_robots
from rig.mission.gas import ALARM_LEL, FLOW_GS, METHODS, SUB_S, WALK_MS, GasLeak, wind_from_weather
from rig.mission.jobs import MissionPort
from rig.mission.obsviews import ObsViews
from rig.mission.scenarios import Scenarios, gust_force, highlight, shielded

SCEN_EVENTS = ("scenario", "disturb", "select", "camera", "demo")
FLEET_EVENTS = ("drop_obstacle", "drop_can", "close_walkway", "close_stairs", "clear", "set_weather", "weld_job")
ROLE = {"pick_can": "pick-up", "weld": "weld", "rust": "rust", "gas": "gas", "leak_check": "leak check", "inspect": "inspection", "crack": "crack"}
RUST = ("DETECT", "SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT", "APPROACH", "BACK_OFF", "GRIND", "WELD", "INSPECT_WELD")
GAS = ("LEAK_APPROACH", "LEAK_INSPECT", "RETORQUE", "LEAK_VERIFY", "GAS_CHECK")
RANK = {"stop": 4, "warn": 3, "act": 2, "ok": 1}
DEFAULT_FLANGE = "F2"
FAULT_SOC = 0.12
PERSON_M = 2.0
ISOLATE_S = 6.0
FLASH_S = 7.0
GAS_KINDS = ("torque", "valve", "isolat", "gas", "leak", "flange")
ARGON_S = 6.0
LOG_KEEP = 300
HOLD_MARGIN = 1.0
SEEK_SPEED = 1.5
DECK_TOL = 1.5


def unit_name(rid: str) -> str:
    d = "".join(c for c in str(rid) if c.isdigit())
    return f"Unit {d}" if d else str(rid)


def robot_id(rid, ids) -> str | None:
    if rid in ids:
        return rid
    d = "".join(c for c in str(rid or "") if c.isdigit())
    return next((i for i in ids if "".join(c for c in i if c.isdigit()) == d), None) if d else None


def _ev(t, who, what, why) -> dict:
    return {"t": round(float(t), 2), "from": who, "to": what, "reason": why}


class MissionLive:
    def __init__(self, fleet=None, control=None, scenario: str = "judges", speed: float = 4.0, seed: int = 0,
                 field=None, today=date.today, obs=True):
        if fleet is None:
            from policy.fleet.live import FleetLive
            fleet = FleetLive(scenario, speed)
        self.fleet, self.today = fleet, today
        self.port = MissionPort(fleet)
        self.control = control(self.port) if callable(control) else control
        m = fleet.map
        self.leak = GasLeak({x["id"]: x["xyz"] for x in m["flanges"]}, m.get("bounds"), seed, field)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="mission")
        self.override: dict[str, dict] = {}
        self.out: list[dict] = []
        self.seen = len(fleet.f.event_log)
        self.last_t = fleet.f.t
        self.isolate_at = None
        self.valve_job = None
        self.ctl: dict | None = None
        self.notes: list[dict] = []
        self.busy = False
        self.sim_flange = lambda: None
        self.view_seq: dict = {}
        self.obs = ObsViews() if obs else None
        self._flash = None
        self.plan_key = None
        self.agent_key = None
        self.sim_out: list[dict] = []
        self.valve_by = self.valve_ft = None
        self.wait_isolation = False
        self.valves: dict = {}
        self.gusts: list = []
        self.flow_gs = FLOW_GS
        self.scen = Scenarios(self)

    @property
    def t(self) -> float:
        return float(self.fleet.f.t)

    def ids(self) -> list[str]:
        return [r.id for r in self.fleet.f.robots]

    def note(self, who, what, why) -> None:
        e = _ev(self.t, who, what, why)
        self.out.append(e)
        self.notes = (self.notes + [e])[-LOG_KEEP:]

    def advance(self, wall_dt: float) -> None:
        with self.lock:
            self.fleet.advance(wall_dt)
            dt = max(0.0, self.t - self.last_t)
            self.last_t = self.t
            w = (self.fleet.state().get("hazards") or {}).get("weather")
            self.leak.set_wind(*wind_from_weather(w))
            gdt = dt if self.leak.seek is None or self.leak.seek.done else min(dt, wall_dt * SEEK_SPEED)
            n = int(math.ceil(gdt / SUB_S)) if gdt > 0 else 0
            for _ in range(n):
                self.leak.step(gdt / n)
                self._walk_back(gdt / n)
            self._gas_logic()
            self.port.sync()
            self.scen.tick()
            self._fleet_events()

    def _fleet_events(self) -> None:
        log = self.fleet.f.event_log
        if len(log) < self.seen:
            self.seen = 0
        for e in log[self.seen:]:
            self.out.append(_ev(e["t"], unit_name(e["robot"]) if e.get("robot") else "Fleet",
                                e["kind"].upper(), e["message"]))
        self.seen = len(log)

    def _robot(self, rid):
        return next((r for r in self.fleet.f.robots if r.id == rid), None)

    def _gas_task(self):
        if self.leak.fid is None:
            return None
        return next((t for t in self.fleet.f.tasks.values() if t.kind == "gas" and t.poi == self.leak.fid
                     and t.status in ("pending", "assigned", "active")), None)

    def _hold(self) -> float:
        tr = self.fleet.jobs.get("gas", "loose")
        segs = [s[0] for s in tr.segments if s[0] > 0]
        return max(0.0, SETUP_S.get("gas", 0.0) + (segs[0] if segs else 10.0) - HOLD_MARGIN)

    def _iso_hold(self) -> float:
        tr = self.fleet.jobs.get("gas", "loose")
        seg = next((s[0] for s in tr.segments if s[1] == "RETORQUE"), None)
        return self._hold() if seg is None else max(0.0, SETUP_S.get("gas", 0.0) + seg - HOLD_MARGIN)

    def _gas_logic(self) -> None:
        lk = self.leak
        f = self.fleet.f
        if not lk.on and lk.queued:
            fid = lk.promote()
            self.note("Fleet", "NEXT_LEAK", f"{fid} is next -> {unit_name('R2')} localises it")
        if self.valve_ft is not None:
            vt = f.tasks.get(self.valve_ft)
            if vt is None or vt.status == "done":
                self.valve_ft = None
                if lk.on:
                    lk.stop("isolated")
                    self.note(unit_name(self.valve_by), "ISOLATED", f"isolation valve upstream of {lk.fid} closed, leak starved")
                self._close_valve_job("DONE")
        task = self._gas_task()
        if task is not None and self.wait_isolation and lk.on and lk.found is not None:
            with self.fleet.lock:
                task.progress = min(task.progress, self._iso_hold())
        if lk.on and lk.found is None and task is not None and task.robot:
            r = self._robot(task.robot)
            src = lk.source()
            if lk.seek is None and r is not None and abs(r.pos[2] - (src[2] - 1.1)) < DECK_TOL and \
                    (float(np.linalg.norm(r.pos[:2] - src[:2])) < 11.0 or r.state == "WORK"):
                lk.begin_seek(r.id, r.pos, float(r.pos[2]))
                self.note(unit_name(r.id), "GAS_SEEK", f"gas on the trunk sensor near {lk.fid} -> {METHODS[lk.method]}")
            if lk.seek is not None:
                with self.fleet.lock:
                    task.progress = min(task.progress, self._hold())
        sk = lk.seek
        if sk is not None and lk.robot_id:
            r = self._robot(lk.robot_id)
            if not sk.done:
                self.override[lk.robot_id] = {"xy": sk.pos.copy(), "yaw": sk.yaw, "z": float(r.pos[2]) if r else sk.deck_z,
                                              "mode": sk.mode}
            elif self.override.get(lk.robot_id, {}).get("mode") not in (None, "RETURN"):
                fd = lk.found
                self.override[lk.robot_id]["mode"] = "RETURN"
                self.note(unit_name(lk.robot_id), "LEAK_FOUND",
                          f"leak localised at {fd['flange']} in {fd['seek_s']:.0f} s, {fd['err_m']:.1f} m from the flange "
                          f"-> read bolts and re-torque")
                self.valve_job = self.port.queue("valve_isolate", lk.fid, "SUPV", 1, by="coordinator",
                                                 why=f"Close the isolation valve upstream of {lk.fid} to starve the leak; "
                                                     + (f"{unit_name(self.valve_by)} turns the valve, {unit_name(lk.robot_id)} "
                                                        "re-torques once it is shut." if self.valve_by else
                                                        f"{unit_name(lk.robot_id)} re-torques meanwhile.")
                                                     + " Needs your approval.")
                self._finding()
        if lk.on and lk.found is not None and task is None:
            lk.stop("fixed")
            self.note("Fleet", "LEAK_FIXED", f"{lk.fid} re-torqued, gas falling back to background")
            self._close_valve_job("DONE")
        if not lk.on and lk.fid is not None:
            self._supersede(lk.fid)
        if self.isolate_at is not None and self.t >= self.isolate_at:
            self.isolate_at = None
            if lk.on:
                lk.stop("isolated")
                self.note("Operator", "ISOLATED", f"valve upstream of {lk.fid} closed, leak starved")
            self._close_valve_job("DONE")

    def _supersede(self, fid: str) -> None:
        for j in self.port.extra.values():
            if j["target"] == fid and j["status"] in ("WAITING_APPROVAL", "QUEUED") and not j.get("fleet_task") \
                    and any(k in j["kind"].lower() for k in GAS_KINDS):
                j["status"] = "CANCELLED"
                self.port.log("superseded", j["id"], None, why=f"{fid} leak already stopped")
                self.note("Coordinator", "SUPERSEDED", f"{j['id']} {j['kind']} {fid}: leak already stopped")

    def _close_valve_job(self, status: str) -> None:
        j = self.port.extra.get(self.valve_job or "")
        if j is not None and j["status"] in ("QUEUED", "RUNNING", "WAITING_APPROVAL", "ASSIGNED"):
            j["status"] = status if j["status"] != "WAITING_APPROVAL" else "CANCELLED"

    def _walk_back(self, dt: float) -> None:
        for rid, o in list(self.override.items()):
            if o.get("mode") != "RETURN":
                continue
            r = self._robot(rid)
            if r is None:
                del self.override[rid]
                continue
            d = r.pos[:2] - o["xy"]
            n = float(np.linalg.norm(d))
            if n < WALK_MS * dt + 1e-6:
                del self.override[rid]
                continue
            o["xy"] = o["xy"] + d / n * WALK_MS * dt
            o["yaw"] = math.atan2(d[1], d[0])

    def _finding(self) -> None:
        c = self.control
        if c is None or getattr(c, "agent", None) is None or not self._budget_ok():
            return
        fd = dict(self.leak.found or {})
        self._agent(lambda: c.finding({"kind": "gas_leak", "severity": "critical", "target": fd.get("flange"),
                                       "zone": fd.get("flange"), "detail": fd}, now=self.t))

    def _budget_ok(self) -> bool:
        c = self.control
        return c is not None and c.cloud and not c.link_cut and c.budget.allows(self.today())

    def _agent(self, fn) -> bool:
        if self.busy:
            return False
        self.busy = True

        def run():
            try:
                rec = fn()
                if rec and rec.get("cost_usd"):
                    self.control.budget.spend(float(rec["cost_usd"]), self.today())
            finally:
                self.busy = False
        self.pool.submit(run)
        return True

    def event(self, name: str, args: dict | None = None) -> dict:
        args = dict(args or {})
        with self.lock:
            if name in SCEN_EVENTS:
                out = self.scen.start(str(args.get("id") or "")) if name == "scenario" else \
                    self.scen.disturb() if name == "disturb" else self.scen.set_camera(args) if name == "camera" else \
                    self.scen.demo(str(args.get("id") or "")) if name == "demo" else \
                    self.scen.select(args.get("robot"))
                self._fleet_events()
                return out
            out = self._event(name, args)
            self.scen.toast(name, args)
            self._fleet_events()
            self._react(name, args, out)
            return out

    def flash(self, lead: str, rest: str = "", tone: str = "warn", state: str = "EVENT", secs: float = FLASH_S,
              prio: int = 2) -> None:
        if self._flash and self._flash["until"] > time.monotonic() and self._flash["prio"] > prio:
            return
        self._flash = {"until": time.monotonic() + secs, "prio": prio,
                       "cap": {"state": state, "lead": lead, "rest": rest, "text": lead + rest, "tone": tone}}

    def _react(self, name, args, out) -> None:
        cap = self._event_caption(name, args, out)
        if cap is None:
            return
        self.flash(*cap)
        c = self.control
        if c is None or name in ("approve", "reject", "cut_link", "intent"):
            return
        if hasattr(c, "nudge"):
            c.nudge(f"event:{name}:{self.t:.0f}")
        if getattr(c, "agent", None) is not None and self._budget_ok():
            detail = {k: v for k, v in {**args, **out}.items() if isinstance(v, (str, int, float, bool, list, dict))}
            self._agent(lambda: c.finding({"kind": name, "severity": "critical" if name == "gas_leak" else "warning",
                                           "detail": detail}, now=self.t))

    def _event_caption(self, name, args, out):
        if name == "battery_fault":
            r = self._robot(out["battery_fault"])
            return (f"{unit_name(r.id)} · Battery fault", f" — {r.battery.soc:.0%}, walking to the dock; its job goes back "
                    "to the queue", "warn", "TO_DOCK")
        if name == "person_at":
            if "fleet" in out:
                return ("Worker on deck", " — any unit within 3 m pauses until clear", "stop", "PAUSE_PERSON")
            if not out.get("person"):
                return ("Worker clear", " — paused units resume their jobs", "ok", "DONE")
            return (f"{unit_name(out['person']['robot'])} · Paused", " — worker on the walkway, holding 3 m clear",
                    "stop", "PAUSE_PERSON")
        if name == "set_weather":
            w = (self.fleet.state().get("hazards") or {}).get("weather") or {}
            lv = w.get("level", args.get("level", "storm"))
            when = f" at {w['time']}" if w.get("time") else ""
            if lv == "calm":
                return (f"Weather calm{when}", " — normal walking speed, plume settles", "ok", "DONE")
            return (f"Weather · {lv}{when}", f" — walk x{(w.get('speed') or {}).get('walk', 1):.2f}, sight "
                    f"{w.get('perception_range_m', 12):.0f} m; the gas plume turns with the wind", "warn", "WAIT_LULL")
        if name in ("drop_obstacle", "close_walkway", "close_stairs"):
            log = ((out.get("fleet") or {}).get("log") or [{}])[-1]
            return ("Walkway blocked", f" — {log.get('message', 'units re-route')}", "warn", "BLOCKED")
        if name == "drop_can":
            log = ((out.get("fleet") or {}).get("log") or [{}])[-1]
            return ("Can on the walkway", f" — {log.get('message', 'nearest gripper unit sent to pick it up')}", "warn",
                    "BLOCKED")
        if name == "clear":
            return ("Routes clear", " — hazards removed, units take the short way again", "ok", "DONE")
        if name == "gas_leak":
            if not out.get("leak"):
                return ("Leak stopped", " — gas falling back to background", "ok", "DONE")
            return (f"Gas leak at {out['leak']}", " — nearest unit pre-empted to localise it, hot work stopped nearby",
                    "stop", "GAS_CHECK")
        if name == "weld_job":
            return ("Weld repair requested", f" at {args.get('poi', 'the riser')} — hot-work permit checks gas first",
                    "act", "SETUP")
        if name == "cut_link":
            on = args.get("on", True)
            return ("Gemini link cut", " — robots continue on edge rules", "warn", "HOLD") if on else \
                ("Gemini link restored", " — Argon planning again", "ok", "DONE")
        if name in ("approve", "reject") and out.get("ok"):
            return (f"Operator {'approved' if name == 'approve' else 'rejected'}", f" {out[name]}", "ok", "DONE")
        if name == "intent":
            return ("Operator intent", f" — “{args.get('text', '')}”", "act", "WORK")
        return None

    def _event(self, name, args) -> dict:
        f, ids = self.fleet, self.ids()
        if name in FLEET_EVENTS or (name == "person_at" and any(k in args for k in ("xy", "xyz", "at"))):
            return {"fleet": f.event(name, args)}
        if name == "person_at":
            if args.get("on") is False or ("distance" in args and args["distance"] is None):
                with f.lock:
                    f.f.persons.clear()
                    f.f._event_tick = True
                self.note("Judge", "PERSON_LEFT", "person left the deck -> robots resume")
                return {"person": None}
            rid = self._pick(args.get("robot"), ids)
            r = self._robot(rid)
            d = float(args.get("distance") or PERSON_M)
            with f.lock:
                try:
                    xyz = f.f._ahead(r, d)
                except ValueError:
                    xyz = [float(r.pos[0]) + d, float(r.pos[1]), float(r.pos[2])]
            with f.lock:
                f.f._apply(Event(f.f.t, "person", {"xyz": xyz, "duration": float(args.get("duration", 90))}))
                f.snap = f.f.snapshot()
            self.note("Judge", "PERSON", f"worker {d:.1f} m from {unit_name(rid)} -> it pauses until clear")
            return {"person": {"robot": rid, "xyz": xyz}}
        if name == "gas_leak":
            if args.get("on", True) is False:
                if self.leak.on:
                    self.leak.stop("isolated")
                    self.note("Judge", "LEAK_OFF", f"{self.leak.fid} leak stopped")
                return {"leak": None}
            fid = args.get("flange") or args.get("fid") or self.sim_flange() or DEFAULT_FLANGE
            if self.leak.on and self.leak.fid == fid:
                return {"leak": fid}
            self.leak.start(fid, float(args.get("flow_gs") or 0) or self.leak_flow())
            self.override.clear()
            with f.lock:
                f.f.schedule(f.f.t, "gas_alarm", flange=fid, scenario="loose")
            self.note("Judge", "GAS_LEAK", f"flange {fid} leaking -> detectors rising, nearest unit pre-empted")
            return {"leak": fid}
        if name == "battery_fault":
            rid = self._pick(args.get("robot"), ids)
            r = self._robot(rid)
            with f.lock:
                r.battery.wh = min(r.battery.wh, FAULT_SOC * r.battery.capacity_wh)
                f.f._event_tick = True
            self.note("Judge", "BATTERY_FAULT", f"{unit_name(rid)} battery fault at {r.battery.soc:.0%} -> dock, job back to the queue")
            return {"battery_fault": rid}
        if name in ("approve", "reject"):
            jid = str(args.get("job") or args.get("job_id") or "")
            ok = self.port.approve_request(jid, "operator", "approved by the operator in the HUD") if name == "approve" \
                else self.port.reject(jid)
            if ok and name == "approve" and jid == self.valve_job:
                self.port.extra[jid]["status"] = "RUNNING"
                if self.valve_by:
                    self._valve_task(jid)
                else:
                    self.isolate_at = self.t + ISOLATE_S
            if ok:
                self.note("Operator", name.upper(), jid)
            return {name: jid, "ok": ok}
        if name == "cut_link":
            if self.control is not None:
                self.control.cut_link(bool(args.get("on", True)))
            self.note("Operator", "LINK_CUT" if args.get("on", True) else "LINK_UP",
                      "Gemini link cut -> edge rules only" if args.get("on", True) else "Gemini link restored")
            return {"link": "CUT" if args.get("on", True) else "UP"}
        if name == "intent":
            text = str(args.get("text") or "")
            self.note("Operator", "INTENT", text)
            c = self.control
            if c is None or getattr(c, "agent", None) is None:
                return {"intent": "no planner"}
            if not self._budget_ok():
                if c.cloud and not c.link_cut:
                    c.agent.last = {**c.agent.last, "intent": text, "status": "capped"}
                else:
                    c.intent(text, now=self.t)
                return {"intent": "edge"}
            return {"intent": "planning" if self._agent(lambda: c.intent(text, now=self.t)) else "busy"}
        return {}

    def _valve_task(self, jid: str) -> None:
        fid = self.leak.fid
        xyz = self.valves.get(fid) or self.leak.flanges[fid].tolist()
        with self.fleet.lock:
            t = self.fleet.f.add_task("valve", f"valve_{fid}", xyz, only=self.valve_by)
            self.fleet.snap = self.fleet.f.snapshot()
        self.valve_ft = t.id
        self.note("Coordinator", "VALVE", f"{unit_name(self.valve_by)} sent to close the isolation valve upstream of {fid}")

    def take_sim(self) -> list[dict]:
        with self.lock:
            out, self.sim_out = self.sim_out, []
            return out

    def brush_force(self, rid: str, t: float | None = None) -> float:
        t = self.t if t is None else t
        raw = 30.5 + 2.5 * math.sin(3.1 * t) + float(gust_force([t], self.gusts)[0])
        return float(shielded([raw])[0])

    def leak_flow(self) -> float:
        return self.flow_gs

    def _pick(self, rid, ids) -> str:
        if rid in (None, ""):
            return self._busiest()
        out = robot_id(rid, ids)
        if out is None:
            raise ValueError(f"unknown robot {rid!r}; expected one of {ids}")
        return out

    def _busiest(self) -> str:
        rs = [r for r in self.fleet.f.robots if not r.failed]
        return max(rs, key=lambda r: (r.state == "WORK", r.task is not None)).id

    def _jobinfo(self, r) -> dict:
        t = r.task
        if t is None:
            return {}
        tr = self.fleet.jobs.get(t.kind, t.scenario)
        e = t.progress - SETUP_S.get(t.kind, 0.0)
        starts = [s[0] for s in tr.segments] + [tr.duration_s]
        i = max(0, sum(1 for s in starts[:-1] if s <= e) - 1)
        span = max(starts[i + 1] - starts[i], 1e-6)
        return {"kind": t.kind, "u": min(1.0, max(0.0, e - starts[i]) / span), "left": starts[i + 1] - e,
                "poi": t.poi, "scenario": t.scenario}

    def _evidence(self, r, state: str, info: dict, gas: float | None) -> dict:
        u, tt = info.get("u", 0.0), self.t
        ev = {"job": "gas" if info.get("kind") in ("gas", "leak_check") else info.get("kind")}
        if state == "SCRUB":
            ev.update(force=round(self.brush_force(r.id, tt), 1), rust_fraction=round(max(0.04, 0.62 * (1 - u)), 3))
        elif state == "SOAK":
            ev["soak_left"] = round(info.get("left", 0.0), 1)
        elif state == "DETECT":
            ev["detector_conf"] = 0.88
        elif state == "VERIFY":
            ev["rust_fraction"] = 0.03
        elif state == "SPRAY":
            ev["patches"] = 1
        if state in GAS or info.get("kind") in ("gas", "leak_check"):
            ev.update(gas=None if gas is None else round(gas, 3), flange=info.get("poi"), bolt=3)
        if self.fleet.f.persons:
            d = min(float(np.linalg.norm(r.pos[:2] - p["xyz"][:2])) for p in self.fleet.f.persons)
            if d < 6.0:
                ev["person_m"] = round(d, 1)
        return ev

    def _robot_gas(self, rid, xyz) -> float | None:
        sk = self.leak.seek
        if sk is not None and self.leak.robot_id == rid and not sk.done:
            return float(sk.trunk.lel)
        if self.leak.plume is None:
            return None
        return float(self.leak.lel_at([[xyz[0], xyz[1], xyz[2] + 0.45]])[0])

    def snapshot_fleet(self) -> dict:
        with self.fleet.lock:
            s = copy.deepcopy(self.fleet.snap)
            info = {r.id: (self._jobinfo(r), r) for r in self.fleet.f.robots}
            welds = weld_views(self.fleet.f)
        s["views"] = {}
        for d in s.get("robots") or []:
            inf, r = info.get(d["id"], ({}, None))
            o = self.override.get(d["id"])
            if o is not None:
                d["pose"] = {"x": round(float(o["xy"][0]), 2), "y": round(float(o["xy"][1]), 2), "z": round(o["z"], 2),
                             "yaw": round(float(o["yaw"]), 3)}
                d["path"] = [[d["pose"]["x"], d["pose"]["y"], d["pose"]["z"]]]
                d["prev_path"] = None
                if o["mode"] == "DEMO":
                    d.update({k: o.get(k) for k in ("state", "task", "job_state", "rollout", "rust", "weld")})
                elif o["mode"] != "RETURN":
                    d["job_state"] = "LEAK_APPROACH"
                    d["seek"] = o["mode"]
            kind = inf.get("kind")
            role = ROLE.get(kind, kind.replace("_", " ")) if kind else "dock" if d["state"] in (
                "CHARGING", "DOCKED", "TO_DOCK", "DOCK_QUEUE") else "standby"
            d["name"] = f"{unit_name(d['id'])} · {role}"
            p = d["pose"]
            g = self._robot_gas(d["id"], (p["x"], p["y"], p["z"]))
            st = d.get("job_state") or d["state"]
            d["evidence"] = self._evidence(r, st, inf, g) if r is not None else {}
            d["gas_lel"] = g
            wv = welds.get(d["id"])
            if wv is not None:
                d["job_state"] = wv.get("state") or d.get("job_state")
                d["evidence"] = {**d["evidence"], **(wv.get("evidence") or {}), "job": "weld"}
                if wv.get("pip") and (d.get("task") or "").startswith("weld"):
                    s["views"][d["id"]] = {"policy": wv["pip"]}
        return s

    def gas(self) -> dict:
        lk = self.leak
        det = lk.detectors()
        sensors = [{k: x[k] for k in ("id", "at", "lel", "ppm")} for x in det]
        sk = lk.seek
        out = {}
        if sk is not None and lk.robot_id:
            name = unit_name(lk.robot_id)
            sensors += [{"id": "IR-T", "at": f"{name} trunk IR", "lel": round(sk.trunk.lel, 4), "ppm": round(sk.trunk.reading)},
                        {"id": "CAT-A", "at": f"{name} arm tip", "lel": round(sk.tip.lel, 4), "ppm": round(sk.tip.reading)}]
            m, sig = sk.estimate if sk.estimate is not None else (sk.pos, 9.9)
            out.update(unit=lk.robot_id, mode=sk.mode, robot={"x": round(float(sk.pos[0]), 2), "y": round(float(sk.pos[1]), 2),
                                                              "yaw_deg": round(math.degrees(sk.yaw), 1)},
                       source={"x": round(float(m[0]), 2), "y": round(float(m[1]), 2), "sigma_m": round(float(sig), 2)},
                       track=sk.path[-80:], hits=sk.hits)
        mx = max(s["lel"] for s in sensors)
        level = "RETREAT" if mx >= 0.20 else "ALARM" if mx >= ALARM_LEL or lk.on else "CLEAR"
        out.update(method=METHODS[lk.method], sensors=sensors, max=round(mx, 4), alarm=bool(lk.on or mx >= ALARM_LEL), level=level,
                   wind={"dir_deg": round((lk.toward + 180.0) % 360.0, 1), "speed_ms": round(lk.speed, 1)},
                   flange=lk.fid if (lk.found or lk.on) else None, leak=lk.on, found=lk.found,
                   isolated=lk.isolated)
        if lk.on and lk.plume is not None:
            out["flow_gs"] = round(lk.plume.q, 2)
            out["flow_kg_h"] = round(lk.plume.q * 3.6, 2)
        return out

    def header(self, events: bool = True) -> dict:
        with self.lock:
            fl = self.snapshot_fleet()
            robots = fleet_robots(fl)
            gas = self.gas()
            views = self._fresh_views(fl.get("views") or {})
            pol = {rid for rid, v in (fl.get("views") or {}).items() if "policy" in v}
            for rid, kv in (self.obs.update(self, fl, pol) if self.obs is not None else {}).items():
                views.setdefault(rid, {}).update(kv)
            out = {"t": fl["t"], "fleet": {k: v for k, v in fl.items() if k not in ("events", "views")},
                   "robots": robots, "views": views,
                   "gas": gas, "sensors": self.sensors(fl), "orchestrator": self.orchestrator(),
                   "caption": self.caption(robots, gas), **self.scen.header(),
                   "mission": {"t": fl["t"], "rounds": self.fleet.rounds, "speed": self.fleet.speed,
                               "leak": {"on": self.leak.on, "flange": self.leak.fid, "found": self.leak.found}}}
            if self.ctl is not None:
                out["control"] = self.ctl
            if events:
                out["events"], self.out = self.out, []
            return out

    def _fresh_views(self, views: dict) -> dict:
        out = {}
        for rid, v in views.items():
            key = (v["policy"].get("seq"), v["policy"].get("caption"))
            if self.view_seq.get(rid) != key:
                self.view_seq[rid] = key
                out[rid] = v
        return out

    def orchestrator(self) -> dict:
        o = self.port.header()
        c = self.control
        o["planner"] = c.agent.summary() if c is not None and getattr(c, "agent", None) is not None else None
        return o

    def sensors(self, fl: dict) -> dict:
        out = {}
        for k, d in enumerate(fl.get("robots") or []):
            ev = d.get("evidence") or {}
            st = d.get("job_state") or d["state"]
            out[d["id"]] = {"force_n": round(float(ev.get("force") or 0.0), 1),
                            "gas_lel": round(float(d.get("gas_lel") or 0.0), 3),
                            "thermal_c": round(31.0 + 4 * k + 0.6 * math.sin(0.7 * self.t + k) + (6.0 if st == "SCRUB" else 0.0), 1),
                            "battery": round(float((d.get("battery") or {}).get("soc") or 0.0), 3),
                            "reach": round(0.35 + 0.3 * float(st in ("SCRUB", "RETORQUE", "SPRAY", "PAINT")), 2),
                            "highlight": highlight(d)}
        return out

    def caption(self, robots: list[dict], gas: dict) -> dict | None:
        if self._flash and self._flash["until"] > time.monotonic():
            return dict(self._flash["cap"])
        if not robots:
            return None

        def rank(r):
            c = r.get("caption") or {}
            hot = 2.5 if gas.get("alarm") and any(k in r["state"] for k in ("GAS", "LEAK", "RETORQUE")) else 0.0
            return RANK.get(c.get("tone"), 0) + hot - (1.5 if any(k in r["state"] for k in ("DOCK", "CHARG")) else 0.0) \
                + (0.5 if r["state"] in RUST else 0.0)
        focus = self.scen.focus()
        r = next((x for x in robots if x["id"] == focus), None) or max(robots, key=rank)
        c = dict(r["caption"])
        c["lead"] = f"{unit_name(r['id'])} · {c['lead']}"
        c["text"] = c["lead"] + c["rest"]
        c["robot"] = r["id"]
        return c

    def control_input(self) -> dict:
        fl = self.snapshot_fleet()
        robots = []
        for d in fl.get("robots") or []:
            ev = d.get("evidence") or {}
            st = d.get("job_state") or d["state"]
            robots.append({"id": d["id"], "state": st, "caption": caption(st, ev)["text"], "pose": d["pose"],
                           "battery": d.get("battery"), "task": d.get("task"), "zone": ev.get("flange") or d.get("deck") or "",
                           "force_n": ev.get("force"), "person_m": ev.get("person_m")})
        gas = self.gas()
        det = [s for s in gas["sensors"] if s["id"].startswith("GD-")]
        g = {"sensors": [{"id": s["id"], "at": s["id"][3:], "lel": s["lel"]} for s in det], "wind": gas["wind"]}
        if gas.get("found"):
            g["flange"], g["flow_kg_h"] = gas["found"]["flange"], gas.get("flow_kg_h")
            g["source"] = {"x": gas["found"]["x"], "y": gas["found"]["y"], "sigma_m": gas["found"]["sigma_m"]}
        jobs = [{**j, "zone": j.get("target")} for j in self.port.jobs() if j["status"] in
                ("QUEUED", "WAITING_APPROVAL", "ASSIGNED", "RUNNING", "BLOCKED")][:12]
        w = (fl.get("hazards") or {}).get("weather") or {}
        return {"t": fl["t"], "robots": robots, "gas": g, "tasks": jobs, "events": self.notes[-6:],
                "storm_level": float(w.get("intensity") or 0.0)}

    def control_tick(self, now: float | None = None) -> dict | None:
        if self.control is None:
            return None
        from policy.control.adapters import from_header
        with self.lock:
            h = self.control_input()
        self.ctl = self.control.tick(from_header(h), now=now if now is not None else time.monotonic())
        with self.lock:
            self._argon_news()
        return self.ctl

    def _argon_news(self) -> None:
        c = self.ctl or {}
        sit = c.get("situation") or []
        key = (tuple(sit), tuple(a.get("id") for a in c.get("actions") or []))
        if c.get("mode") == "CLOUD" and sit and key != self.plan_key:
            top = next((a for a in c.get("actions") or [] if a.get("source") != "edge"), None)
            self.note("Argon", "PLAN", sit[0] + (f" -> {top['label']}" if top and top.get("label") else ""))
            self.flash("Argon", f" · {_short(sit[0])}", "act", "PLAN", ARGON_S, prio=1)
        self.plan_key = key
        ag = (c.get("agent") or {}) if c else {}
        calls = ag.get("calls") or []
        akey = (len(calls), calls[-1].get("label") if calls else None, ag.get("text"))
        if calls and akey != self.agent_key and self.agent_key is not None:
            last = calls[-1]
            self.note("Argon", last.get("label", "CALL").upper().replace(" ", "_")[:24],
                      (last.get("why") or "") + (f" (refused: {last['refused']})" if last.get("refused") else ""))
            self.flash("Argon", f" · {_short(last.get('label', ''))}" + (f" — {_short(last.get('why', ''), 70)}"
                                                                         if last.get("why") else ""), "act", "PLAN",
                       ARGON_S, prio=1)
        self.agent_key = akey

    def for_sim(self) -> dict:
        s = self.snapshot_fleet()
        lk = self.leak
        w = lk.wind_unit
        leak = {"xyz": [round(float(v), 2) for v in lk.source()], "on": lk.on, "wind": [round(float(w[0]), 3), round(float(w[1]), 3)]} \
            if lk.fid is not None else None
        return {"t": s["t"], "hazards": s["hazards"], "leak": leak,
                "robots": [{k: r.get(k) for k in ("id", "pose", "path", "prev_path", "reroute_t", "state", "task", "job_state", "rollout", "rust", "weld")}
                           for r in s["robots"]]}

    def state(self) -> dict:
        return self.snapshot_fleet()


def weld_views(f) -> dict:
    try:
        from policy.weld.live import fleet_weld_views
    except ImportError:
        return {}
    return fleet_weld_views(f)


def _short(text: str, n: int = 110) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

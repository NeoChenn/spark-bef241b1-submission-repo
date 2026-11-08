from __future__ import annotations

import random
import threading

from policy.fleet import barrel
from policy.fleet.jobs import JobLibrary, JobTrace, run_orchestrator
from policy.fleet.scenario import SCENARIOS

LOG_KEEP = 300
BARREL_EVERY_S = 60.0
QUICK = {"rust": [(0, "DETECT", "rust confirmed"), (6, "SPRAY", "spray converter"), (14, "SOAK", "soak 30 s"),
                  (44, "SCRUB", "scrub, force-limited"), (90, "VERIFY", "on-board + second opinion"),
                  (100, "PAINT", "paint"), (120, "DONE", "repair complete")],
         "leak_check": [(0, "LEAK_APPROACH", "approach flange"), (10, "LEAK_INSPECT", "gas + bolt check"),
                        (35, "DONE", "flange checked")],
         "inspect": [(0, "DETECT", "read gauge"), (15, "DONE", "gauge in range")],
         "gas": [(0, "LEAK_APPROACH", "approach leak"), (10, "LEAK_INSPECT", "localise leak"),
                 (30, "RETORQUE", "re-torque bolts"), (55, "DONE", "leak stopped")],
         "crack": [(0, "LEAK_INSPECT", "inspect crack"), (25, "ESCALATE_TO_HUMAN", "crack -> human crew")],
         "pick_can": [(0, "DETECT", "can found on camera"), (1.35, "APPROACH", "arm to pre-grasp"),
                      (3.6, "GRASP", "close gripper on the can"), (5.85, "LIFT", "lift clear of the grating"),
                      (7.2, "STOW", "drop the can in the bin"), (9.0, "DONE", "can stowed, walkway clear")],
         "weld": [(0, "INSPECT", "eddy + laser seam scan"), (16, "GRIND", "grind out crack"),
                  (37, "WELD", "seam-tracked weld"), (68, "INSPECT_WELD", "verify weld"), (77, "DONE", "weld verified")]}


def quick_run(kind: str, scenario: str) -> JobTrace:
    try:
        return run_orchestrator(kind, scenario)
    except ImportError:
        segs = QUICK[kind]
        return JobTrace(kind, scenario, float(segs[-1][0]), segs[-1][1], segs)


class FleetLive:
    def __init__(self, scenario: str = "judges", speed: float = 4.0, jobs=None, max_steps: int = 40,
                 barrel_every_s: float = BARREL_EVERY_S, seed=None):
        jobs = jobs or JobLibrary(quick_run)
        self.barrel_every_s, self.rng = barrel_every_s, random.Random(seed)
        self.scenario, self.speed, self.jobs, self.max_steps = scenario, speed, jobs, max_steps
        self.lock = threading.Lock()
        self.f = SCENARIOS[scenario](3, jobs=jobs, disturb=False) if scenario == "judges" else \
            SCENARIOS[scenario](3, jobs=jobs)
        self.specs = [(t.kind, t.poi, t.scenario, t.severity, t.exposure) for t in self.f.tasks.values()]
        self.map = self.f.rm.to_json()
        self.acc = 0.0
        self.rounds = 1
        self.snap = self.f.snapshot()

    def event(self, name: str, args: dict | None = None) -> dict:
        with self.lock:
            n = len(self.f.event_log)
            self.f.event(name, **(args or {}))
            self.snap = self.f.snapshot()
            return {"t": self.f.t, "log": self.f.event_log[n:]}

    def advance(self, wall_dt: float) -> dict:
        with self.lock:
            self.acc += wall_dt * self.speed
            rolled = self.f.roll(wall_dt)
            self._auto_barrel(wall_dt)
            k = 0
            while self.acc >= self.f.dt and k < self.max_steps:
                self.f.step()
                self.acc -= self.f.dt
                k += 1
                if self.f.done():
                    self.new_round()
            self.acc = min(self.acc, self.f.dt)
            if k or rolled:
                self.snap = self.f.snapshot()
            return self.snap

    def _auto_barrel(self, dt: float) -> None:
        hz = self.f.hz
        if self.barrel_every_s <= 0 or hz.weather["wind"] < barrel.AUTO_WIND:
            return
        if self.rng.random() >= dt / self.barrel_every_s:
            return
        if sum(1 for o in hz.obstacles.values() if o.get("auto")) >= barrel.AUTO_MAX:
            return
        movers = [r.id for r in self.f.robots if not r.failed and r.legs]
        if not movers:
            return
        try:
            self.f.event("drop_obstacle", kind="barrel", at=self.rng.choice(movers), auto=True)
        except ValueError:
            pass

    def new_round(self) -> None:
        f = self.f
        for tid in [tid for tid, t in f.tasks.items() if t.status == "done"]:
            del f.tasks[tid]
        for kind, poi, scenario, sev, exp in self.specs:
            f.add_task(kind, poi, scenario=scenario, severity=sev, exposure=exp)
        f.event_log = f.event_log[-LOG_KEEP:]
        self.rounds += 1

    def state(self) -> dict:
        with self.lock:
            return self.snap

    def for_sim(self) -> dict:
        s = self.state()
        return {"t": s["t"], "hazards": s["hazards"],
                "robots": [{k: r[k] for k in ("id", "pose", "path", "prev_path", "reroute_t", "state")}
                           for r in s["robots"]]}

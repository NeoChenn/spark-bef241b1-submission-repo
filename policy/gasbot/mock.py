from __future__ import annotations

import math

import numpy as np
import torch

from detection.leak import LeakLocalizer
from detection.localize import RustLocalizer, fake_colour_classifier
from policy.demo_mock import GreedyScrub, MockRustWorld, scene_geometry
from policy.gasbot import deck
from policy.gasbot.bot import GasBotConfig, GasBotMixin, GasBotOrchestrator
from policy.gasbot.locate import unit
from policy.gasbot.plume import Wind
from policy.leak.job import LeakConfig, LeakJob
from policy.leak.mock import LeakArm, LeakPerception
from policy.leak.world import LeakWorld
from policy.orchestrator import TERMINAL, OrchestratorConfig, S


class GasMockBridge(GasBotMixin):
    def __init__(self, leak_world, wind, perception, pipe, seed=0, flow_full=None, h2s_ratio=0.0, start=None,
                 gas_stop=0.20, err_db=0.0):
        self.t = 0.0
        self.perception = perception
        self._pipe = pipe
        self.route = deck.Route()
        self.xy = np.asarray(start if start is not None else deck.ROUTE[0], float)
        self.route.nearest(self.xy)
        d = self.route.target() - self.xy
        self.yaw = math.atan2(d[1], d[0])
        self.lap_done = False
        self.events: list = []
        self.gas_stop = gas_stop
        self.path = [self.xy.copy()]
        self.gas_init(leak_world, wind, seed, flow_full, h2s_ratio, machinery_err_db=err_db)

    def sim_time(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
        while self.events and self.events[0][0] <= self.t:
            self.events.pop(0)[2]()
        self.gas_sync(self.t)

    def pipe(self):
        return self._pipe

    def publish_overlay(self, lines) -> None:
        self.hud = lines

    def base_xy(self) -> np.ndarray:
        return self.xy

    def base_yaw(self) -> float:
        return self.yaw

    def _set(self, xy, yaw=None) -> None:
        self.xy = np.asarray(xy, float)
        if yaw is not None:
            self.yaw = float(yaw)
        self.path.append(self.xy.copy())

    def patrol_step(self, dist: float, dt: float) -> None:
        p, yaw, lap = self.route.step(self.xy, dist)
        self._set(p, yaw)
        self.lap_done |= lap
        self.advance(dt)

    def seek_move(self, v, dt: float) -> bool:
        v = np.asarray(v, float)
        q, blocked = deck.clamp(self.xy + v * dt)
        self._set(q, math.atan2(v[1], v[0]) if np.linalg.norm(v) > 1e-6 else None)
        self.advance(dt)
        return blocked

    def walk_to(self, target, yaw=None, speed=0.6, dt=0.5, guard=True) -> bool:
        target, _ = deck.clamp(target)
        for _ in range(400):
            d = target - self.xy
            n = float(np.linalg.norm(d))
            if n < 1e-3:
                break
            step = min(n, speed * dt)
            self._set(self.xy + d / n * step, math.atan2(d[1], d[0]))
            self.advance(dt * step / (speed * dt))
            g = self.safety_reading()
            if guard and (g is None or g >= self.gas_stop):
                return False
        if yaw is not None:
            self._set(self.xy, yaw)
        return True

    def flange(self, fid):
        return self.leak_world.flanges[fid].geom

    def nearest_flange(self):
        f = self.nearest_flange_to(self.xy)
        return f[0] if f else None

    def approach_flange(self, fid) -> bool:
        g = self.flange(fid)
        stand = g.center[:2] + 1.1 * g.normal[:2]
        if not self.walk_to(stand, math.atan2(-g.normal[1], -g.normal[0])):
            return False
        self.at_flange = fid
        self.perception.focus(fid)
        self.advance(1.5)
        return True

    def retreat(self, distance: float = 3.0) -> None:
        self.at_flange = None
        self.walk_to(self.upwind_target(distance), guard=False)

    def return_to_work(self) -> None:
        self.at_flange = None
        self.route.nearest(self.xy)

    def go_to_patrol_pose(self, pose) -> bool:
        return True

    def approach(self, *a) -> bool:
        return True


SCENARIOS = ("none", "loose", "crack", "big")


def make_leak(rng, lw: LeakWorld, scenario: str, fid: str) -> float:
    if scenario == "loose":
        bolts = rng.choice(8, size=int(rng.integers(1, 3)), replace=False)
        for b in bolts:
            lw.loosen(fid, int(b), float(rng.uniform(0.12, 0.25)))
        return float(10 ** rng.uniform(math.log10(1.5), math.log10(12.0)))
    if scenario == "crack":
        lw.crack(fid, 0.3)
        return float(10 ** rng.uniform(math.log10(3.0), math.log10(12.0)))
    if scenario == "big":
        lw.gas_spike(fid)
        return float(rng.uniform(80.0, 200.0))
    return 0.0


def build(seed=0, scenario="loose", log_path=None, fid=None, wind_speed=None, wind_dir=None, h2s_ratio=None,
          start=None, gas_cfg=None, seek_cfg=None):
    rng = np.random.default_rng(seed)
    geoms = deck.flanges()
    lw = LeakWorld(geoms)
    fid = fid or str(rng.choice([g.id for g in geoms]))
    flow_full = {g.id: 0.0 for g in geoms}
    flow_full[fid] = make_leak(rng, lw, scenario, fid)
    lw.settle()
    wind = Wind(float(wind_speed if wind_speed is not None else rng.uniform(2.0, 9.0)),
                float(wind_dir if wind_dir is not None else rng.uniform(-math.pi, math.pi)), seed=seed + 5)
    if start is None:
        a, b = deck.ROUTE[int(rng.integers(len(deck.ROUTE)))], deck.ROUTE[int(rng.integers(len(deck.ROUTE)))]
        start = np.asarray(a) + rng.uniform() * (np.asarray(b) - np.asarray(a))
        start, _ = deck.clamp(start)
    pipe, _ = scene_geometry()
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    world.set_rust(torch.zeros(1, pipe.n_z, pipe.n_theta))
    perc = LeakPerception(lw)
    h2s = float(h2s_ratio if h2s_ratio is not None else (rng.choice([0.0, 0.0, 2e-4, 8e-4])))
    bridge = GasMockBridge(lw, wind, perc, pipe, seed, flow_full, h2s, start, err_db=float(rng.normal(0.0, 1.5)))
    arm = LeakArm(world, pipe, lw, seed=seed)
    loc = RustLocalizer(fake_colour_classifier, patch=16, stride=8, threshold=0.5, temperature=1.0)
    orch = GasBotOrchestrator(bridge, perc, loc, world, arm, GreedyScrub(pipe), OrchestratorConfig(log_path=log_path),
                              leak=LeakJob(LeakLocalizer(), LeakConfig(max_settle_reads=15)),
                              gas_cfg=gas_cfg or GasBotConfig(), seek_cfg=seek_cfg)
    truth = {"fid": fid, "scenario": scenario, "source": deck.leak_point(lw.flanges[fid].geom).tolist(),
             "flow_gs": round(flow_full[fid] * lw.flanges[fid].level, 3), "wind_dir": wind.dir0,
             "wind_speed": wind.speed0, "h2s_ratio": h2s}
    return orch, bridge, truth


def run(orch, bridge, max_t=900.0, until_fixed=True, on_step=None) -> None:
    while orch.state not in TERMINAL and bridge.t < max_t:
        orch.step()
        if on_step is not None:
            on_step(orch)
        if until_fixed and orch.state == S.PATROL and orch.resolved:
            break


def episode(seed, scenario="loose", max_t=900.0, **kw) -> dict:
    orch, bridge, truth = build(seed, scenario, **kw)
    run(orch, bridge, max_t)
    src = np.asarray(truth["source"][:2])
    track = [(t - orch.detect_t, float(np.hypot(x - src[0], y - src[1])), s) for t, x, y, s in orch.est_track]
    err = None
    if orch.found_t is not None:
        near = [e for t, x, y, s in orch.est_track if t <= orch.found_t
                for e in [float(np.hypot(x - src[0], y - src[1]))]]
        err = near[-1] if near else None
    return {"seed": seed, **truth, "detected": orch.detect_t is not None, "t_detect": orch.detect_t,
            "t_found": orch.found_t, "t_fixed": orch.fixed_t, "loc_err": err, "flange_ok": orch.flange_id == truth["fid"]
            if orch.found_t is not None else None, "final": orch.state.value, "flow_est": orch.flow, "flow_conc": orch.flow_conc,
            "alerts": [a["kind"] for a in orch.alerts], "track": track, "t_end": bridge.t,
            "path": np.round(np.array(bridge.path), 2).tolist()}

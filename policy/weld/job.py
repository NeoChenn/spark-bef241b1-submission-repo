from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from policy.weld.env import WeldEnv
from policy.weld.policy import OpenLoop, SeamTracker
from policy.weld.seam import Seam, heat_input
from policy.weld.sensors import eddy_array, groove_center, laser_profile

LEL_LIMIT = 10.0
SCAN_MM_S = 20.0
GRIND_MM_S = 4.0
VERIFY_MM_S = 10.0
GRIND_W_MM = 9.0
GRIND_N = 30.0
CRACK_THR = 0.5
MARGIN_MM = 6.0
FILL_MIN = 0.9
DT = 0.25


@dataclass
class Rec:
    t: float
    frm: str
    to: str
    reason: str
    evidence: dict = field(default_factory=dict)


class Hold(Exception):
    pass


class WeldJob:
    def __init__(self, seam: Seam, seed: int = 0, lel=2.0, policy_factory=SeamTracker, hold_max_s: float = 90.0,
                 max_rework: int = 1):
        self.seam, self.rng = seam, np.random.default_rng(seed)
        self.seed, self.lel_fn = seed, lel if callable(lel) else (lambda t, v=float(lel): v)
        self.policy_factory, self.hold_max_s, self.max_rework = policy_factory, hold_max_s, max_rework
        self.t, self.state = 0.0, "IDLE"
        self.records: list[Rec] = []
        self.alerts: list[dict] = []
        self.on_state = None
        self.on_frame = None
        self.centers: dict[float, float] = {}
        self.scan: list[tuple[float, float]] = []
        self.paths: dict[str, list] = {"INSPECT": [], "GRIND": [], "WELD": [], "INSPECT_WELD": []}
        self.view: dict = {}
        self.ev: dict = {"job": "weld"}
        self.result: dict = {}
        self.his: list[float] = []

    def lel(self) -> float:
        return float(self.lel_fn(self.t))

    def go(self, to: str, reason: str, **ev) -> None:
        self.ev = {**self.ev, **ev, "lel_pct": round(self.lel(), 1)}
        self.records.append(Rec(round(self.t, 2), self.state, to, reason, dict(self.ev)))
        self.state = to
        if self.on_state:
            self.on_state(to, self.t)

    def frame(self, x, y, ys=None, z=None, c=None) -> None:
        self.view = {"state": self.state, "x": x, "y": y, "profile_y": ys, "profile_z": z, "center_est": c,
                     "t": self.t, "ev": dict(self.ev)}
        if self.on_frame:
            self.on_frame(self)

    def tick(self, dt: float) -> None:
        self.t += dt

    def seam_y(self, x: float) -> float:
        xs = np.array(sorted(self.centers))
        return float(np.interp(x, xs, [self.centers[k] for k in xs])) if len(xs) else 0.0

    def inspect(self) -> tuple[float, float] | None:
        self.go("INSPECT", "eddy-current + laser scan along the seam")
        y, step = 0.0, SCAN_MM_S * DT / 5
        for x in np.arange(0.0, self.seam.cfg.length_mm, step):
            ys, z = laser_profile(self.seam, x, y, self.rng)
            c = groove_center(ys, z)
            if c is not None:
                y = c
                self.centers[float(x)] = c
            s = eddy_array(self.seam, x, y, self.rng)
            self.scan.append((float(x), s))
            self.paths["INSPECT"].append((float(x), y))
            self.ev["crack_score"] = round(s, 2)
            self.tick(DT / 5)
            self.frame(float(x), y, ys, z, c)
        hits = [x for x, s in self.scan if s > CRACK_THR]
        if len(hits) < 2:
            return None
        L = self.seam.cfg.length_mm
        return max(min(hits) - MARGIN_MM, 0.0), min(max(hits) + MARGIN_MM, L)

    def permit(self, resume: str | None = None) -> None:
        if self.lel() <= LEL_LIMIT:
            return
        g = self.lel()
        self.go("HOT_WORK_HOLD", f"gas {g:.0f}% LEL > {LEL_LIMIT:.0f}% -> arc off, hot work blocked")
        t0 = self.t
        while self.lel() > LEL_LIMIT:
            if self.t - t0 >= self.hold_max_s:
                msg = (f"hot-work permit needed: gas {self.lel():.0f}% LEL > {LEL_LIMIT:.0f}% LEL for "
                       f"{self.hold_max_s:.0f} s -> human crew")
                self.alerts.append({"t": round(self.t, 2), "kind": "hot_work_blocked", "message": msg})
                self.go("ESCALATE_TO_HUMAN", msg)
                raise Hold
            self.seam.cool(1.0)
            self.tick(1.0)
            self.ev["temp_c"] = round(self.seam.temp_c, 1)
            self.frame(self.view.get("x", 0.0), self.view.get("y", 0.0))
        if resume:
            self.go(resume, f"gas {self.lel():.0f}% LEL <= {LEL_LIMIT:.0f}% -> hot work resumes")

    def grind(self, x0: float, x1: float) -> None:
        self.permit()
        self.go("GRIND", f"grind out crack {x0:.0f}-{x1:.0f} mm", force=GRIND_N)
        y = self.seam_y(x0)
        for x in np.arange(x0, x1, GRIND_MM_S * DT):
            self.permit("GRIND")
            ys, z = laser_profile(self.seam, x, y, self.rng)
            c = groove_center(ys, z)
            y = c if c is not None else self.seam_y(x)
            self.seam.grind(float(x), y, GRIND_W_MM)
            self.seam.grind(float(x) + 0.5, y, GRIND_W_MM)
            self.paths["GRIND"].append((float(x), y))
            self.ev.update(force=round(GRIND_N + float(self.rng.normal(0, 2.0)), 1))
            self.seam.cool(DT)
            self.tick(DT)
            self.frame(float(x), y, ys, z, c)
        self.ev["force"] = 0.0

    def weld(self, x0: float, x1: float) -> None:
        self.permit()
        self.go("WELD", f"seam-tracked weld {x0:.0f}-{x1:.0f} mm, heat input held at target")
        env = WeldEnv(self.seam, x0, x1, seed=self.seed + len(self.records))
        pol = self.policy_factory()
        obs, self.his, err = env.obs(), [], []
        while True:
            if self.lel() > LEL_LIMIT:
                self.permit("WELD")
                obs = env.obs()
            dy, v, i = pol.act(obs)
            obs, done = env.step(dy, v, i)
            tl = obs["tele"]
            hi = heat_input(tl["current_a"], tl["voltage_v"], tl["speed_mm_s"], self.seam.cfg.eta)
            self.his.append(hi)
            yc = self.seam_y(env.x)
            err.append(env.y - yc)
            self.paths["WELD"].append((env.x, env.y))
            self.ev.update(heat_kj_mm=round(hi, 2), current_a=round(tl["current_a"], 1),
                           voltage_v=round(tl["voltage_v"], 1), speed_mm_s=round(tl["speed_mm_s"], 2),
                           temp_c=round(obs["temp_c"], 1), track_err_mm=round(abs(err[-1]), 2), force=0.0)
            self.tick(env.dt)
            self.frame(env.x, env.y, obs["profile_y"], obs["profile_z"], obs["center_est"])
            if done:
                break
        self.ev["coverage"] = round(self.seam.metrics(x0, x1)["coverage"], 3)

    def verify(self, x0: float, x1: float) -> list[float]:
        self.go("INSPECT_WELD", "laser fill check + eddy re-scan + heat log")
        bad, cfg = [], self.seam.cfg
        for x in np.arange(x0, x1, VERIFY_MM_S * DT / 2):
            yc = self.seam_y(x)
            ys, z = laser_profile(self.seam, x, yc, self.rng)
            band = (np.abs(ys - yc) <= cfg.groove_half_mm) & np.isfinite(z)
            fill = float((z[band] > 0.3).mean()) if band.any() else 0.0
            s = eddy_array(self.seam, x, yc, self.rng)
            if fill < FILL_MIN or s > CRACK_THR:
                bad.append(float(x))
            self.paths["INSPECT_WELD"].append((float(x), yc))
            self.ev.update(crack_score=round(s, 2), fill=round(fill, 2))
            self.tick(DT / 2)
            self.frame(float(x), yc, ys, z, None)
        his = np.asarray(self.his)
        heat_bad = float(((his < cfg.hi_lo) | (his > cfg.hi_hi)).mean()) if len(his) else 0.0
        self.ev["heat_out_of_band"] = round(heat_bad, 3)
        if heat_bad > 0.1:
            bad = bad or [x0, x1]
        return bad

    def run(self) -> str:
        ext = None
        try:
            ext = self.inspect()
            if ext is None:
                self.go("DONE", "seam sound: no crack indication")
                return self.finish(None)
            x0, x1 = ext
            self.view.update(x=(x0 + x1) / 2, y=self.seam_y((x0 + x1) / 2), profile_y=None, profile_z=None)
            self.ev["crack_mm"] = round(x1 - x0 - 2 * MARGIN_MM, 1)
            self.grind(x0, x1)
            self.weld(x0, x1)
            for k in range(self.max_rework + 1):
                bad = self.verify(x0, x1)
                if not bad:
                    self.go("DONE", "weld verified: groove filled, no crack indication, heat input in band")
                    break
                if k == self.max_rework:
                    msg = f"weld failed verification over {len(bad)} stations after rework -> human welder"
                    self.alerts.append({"t": round(self.t, 2), "kind": "weld_reject", "message": msg})
                    self.go("ESCALATE_TO_HUMAN", msg)
                    break
                self.weld(max(min(bad) - 3, x0), min(max(bad) + 3, x1))
            return self.finish(ext)
        except Hold:
            return self.finish(ext)

    def finish(self, ext) -> str:
        self.result = {"outcome": self.state, "crack_found": ext is not None, "t": round(self.t, 2),
                       "repair_x": list(ext) if ext else None}
        if ext and self.paths["WELD"]:
            m = self.seam.metrics(*ext)
            p = [(x, y) for x, y in self.paths["WELD"] if ext[0] <= x <= ext[1]]
            e = [y - self.seam.center_at(x) for x, y in p] or [0.0]
            self.result.update(m, track_rmse_mm=float(np.sqrt(np.mean(np.square(e)))))
        return self.state


SCENARIOS = {"crack": {}, "clean": {"crack": False}, "gas": {"lel": 25.0},
             "gas_clears": {"lel": lambda t: 22.0 if t < 45.0 else 3.0}, "open_loop": {"policy_factory": OpenLoop}}


def make_job(scenario: str = "crack", seed: int = 1) -> WeldJob:
    kw = dict(SCENARIOS.get(scenario, {}))
    return WeldJob(Seam(seed=seed, crack=kw.pop("crack", True)), seed=seed, **kw)


def weld_trace(scenario: str = "crack", seed: int = 1):
    from policy.fleet.jobs import JobTrace
    j = make_job(scenario, seed)
    j.run()
    segs = []
    for r in j.records:
        if segs and segs[-1][0] == r.t:
            segs[-1] = (r.t, r.to, r.reason)
        else:
            segs.append((r.t, r.to, r.reason))
    return JobTrace("weld", scenario, float(j.t), j.state, segs, list(j.alerts), len(j.records))

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from policy.gasbot.plume import RHO_CH4_G_M3
from policy.gasbot.sensors import LEL_PPM
from rig.mission.gas import TIP_Z, TRUNK_Z, GasLeak, wind_from_weather
from rig.mission.scenarios import DEMO_WEATHER, GAS_FLOW_GS, DemoGas, QuietSeek, walk
from rig.sim.gait import GaitLoop, Walker

PHASES = ("SEEK", "WALK", "TURN", "VERIFY")
GRID = {"x": (11.0, 23.0), "y": (-13.0, -7.5), "z": (0.15, 1.55), "res": 0.12, "nz": 8}
FLUCT_TAU = 0.3


class Run:
    def __init__(self, seed=0, dt=0.05, stop=None, turn_s=4.0, walk_ms=DemoGas.walk_ms, near_m=DemoGas.near_m,
                 seek_max_s=24.0, alarm_s=DemoGas.alarm_s, wait_s=DemoGas.seek_wait_s, n_particles=400):
        D = DemoGas
        self.dt, self.turn_s, self.walk_ms, self.near_m, self.seek_max_s = dt, turn_s, walk_ms, near_m, seek_max_s
        self.deck, self.src, self.n_particles = D.deck, np.asarray(D.src, float), n_particles
        self.lk = lk = GasLeak({D.fid: list(D.src)}, D.box, seed, None)
        lk.method = "infotaxis"
        lk.set_wind(*wind_from_weather(DEMO_WEATHER))
        lk.start(D.fid, GAS_FLOW_GS)
        g = GaitLoop.load()
        zm = g.zmean if g else 0.36
        if stop is None:
            from rig.sim.taskmotion import scrub_base

            w = D.wheel
            x, y, yaw = scrub_base((w[0], w[1] - 0.01, w[2]), w[1], w[2], 0.15, self.deck + 0.32 - 0.2)
            stop = (x, y, round(self.deck + 0.32 - zm, 2), yaw)
        self.stop = tuple(float(v) for v in stop)
        self.fe = 1.0
        e, trip = 0.0, None
        while True:
            self._leak_step()
            e += dt
            if trip is None and e >= alarm_s - 1e-9:
                trip = e
            if trip is not None and e >= trip + wait_s - 1e-9:
                break
        self.warm_s = e
        self.sk = lk.seek = QuietSeek(lk, [*D.start, self.deck], self.deck, D.box, lk.seed)
        self.rng = np.random.default_rng(seed + 101)
        self.k, self.t = 0, 0.0
        self.t_found = self.t_at = self.t_close = None
        self.found_by = self.estimate = self.from_xy = None
        self.samples, self.target = [], None
        self.sigma0 = self.sk.pf.estimate()[1]
        self._particles()
        self.belief = self._belief()
        self.pose = (*D.start, self.deck, 0.0)

    def _leak_step(self):
        self.lk.step(self.dt)
        self.fe += (self.lk.plume.fluct - self.fe) * (1 - math.exp(-self.dt / FLUCT_TAU))

    def _particles(self):
        pf = self.sk.pf
        idx = self.rng.choice(pf.n, size=min(self.n_particles, pf.n), replace=False, p=pf.w)
        w = pf.w[idx]
        self.parts = np.c_[pf.p[idx, :2], w / w.max()].astype(np.float32)

    def _belief(self):
        m, s, _ = self.sk.pf.estimate()
        return {"mean": [float(m[0]), float(m[1])], "sigma": float(s)}

    @property
    def phase(self):
        if self.t_found is None:
            return "SEEK"
        if self.t_at is None:
            return "WALK"
        return "TURN" if self.t_close is None else "VERIFY"

    def step(self):
        sk, lk = self.sk, self.lk
        seeking = self.t_found is None
        if seeking:
            pos, tip, nxt = sk.pos.copy(), sk.tip_xy(), sk.next_obs
        self._leak_step()
        self.k += 1
        self.t = self.k * self.dt
        if seeking:
            if sk.next_obs != nxt:
                self.samples += [[*pos, self.deck + TRUNK_Z, float(sk.trunk.lel)], [*tip, self.deck + TIP_Z, float(sk.tip.lel)]]
                self.target = None if sk.plan.target is None else np.asarray(sk.plan.target, float).copy()
                self._particles()
            self.belief = self._belief()
            self.pose = (*sk.pos, self.deck, sk.yaw)
            near = float(np.linalg.norm(sk.pos - self.src[:2])) < self.near_m
            if sk.hit is not None or sk.t >= self.seek_max_s or near:
                sk.done, self.t_found, self.from_xy = True, self.t, sk.pos.copy()
                self.found_by = "hit" if sk.hit is not None else "near" if near else "timeout"
                m = np.array([sk.hit["x"], sk.hit["y"]]) if sk.hit is not None else sk.pf.estimate()[0]
                self.estimate = [float(m[0]), float(m[1]), float(self.src[2])]
            return
        x, y, z, yaw = self.stop
        if self.t_at is None:
            d = float(np.linalg.norm(np.array([x, y]) - self.from_xy))
            u = (self.t - self.t_found) / max(1.0, d / self.walk_ms)
            if u < 1:
                h = math.atan2(y - self.from_xy[1], x - self.from_xy[0])
                self.pose = (*walk(self.from_xy, (x, y), u), self.deck, h)
                return
            self.t_at = self.t
        if self.t_close is None and self.t >= self.t_at + self.turn_s - 1e-9 and lk.on:
            lk.stop("isolated")
            self.t_close = self.t
        self.pose = (x, y, z, yaw)

    def snap(self):
        sk, lk = self.sk, self.lk
        arrow = None
        if self.t_found is None and self.target is not None:
            arrow = [[float(sk.pos[0]), float(sk.pos[1])], [float(self.target[0]), float(self.target[1])]]
        return {"t": self.t, "phase": self.phase, "pose": tuple(float(v) for v in self.pose),
                "valve_s": None if self.t_at is None else self.t - self.t_at, "leak_on": bool(lk.on),
                "flow": float(lk.plume.q), "wind": lk.wind.unit().tolist(), "wind_speed": float(lk.wind.u),
                "n_samples": len(self.samples), "particles": self.parts, "belief": dict(self.belief), "arrow": arrow,
                "found": self.t_found is not None, "obs": int(sk.pf.n_obs)}

    def field(self, pts):
        c = self.lk.plume.concentration(pts) * self.fe
        return c / RHO_CH4_G_M3 * 1e6 / LEL_PPM


def grid_axes(deck, g=GRID):
    xs = np.arange(g["x"][0], g["x"][1] + 1e-9, g["res"])
    ys = np.arange(g["y"][0], g["y"][1] + 1e-9, g["res"])
    zs = deck + np.linspace(g["z"][0], g["z"][1], g["nz"])
    return xs, ys, zs


def plan(frames=285, fps=30, cuts=(75, 90, 165, 180), seed=0, stop=None, turn_s=4.0, dt=0.05, seek_end=None,
         close_frame=None, turn_frames=None, tail_s=2.5, walk_ms=DemoGas.walk_ms, near_m=DemoGas.near_m,
         seek_max_s=24.0, n_particles=400, grid=None) -> dict:
    kw = dict(seed=seed, dt=dt, stop=stop, turn_s=turn_s, walk_ms=walk_ms, near_m=near_m, seek_max_s=seek_max_s,
              n_particles=n_particles)
    probe = Run(**kw)
    while probe.t_close is None:
        probe.step()
    seek_end = cuts[2] - 8 if seek_end is None else seek_end
    close_frame = frames - 40 if close_frame is None else close_frame
    kf, kt = [0, seek_end], [0.0, probe.t_found]
    if turn_frames:
        kf.append(close_frame - turn_frames)
        kt.append(probe.t_at)
    kf += [close_frame, frames - 1]
    kt += [probe.t_close, probe.t_close + tail_s]
    ft = np.interp(np.arange(frames), kf, kt)
    steps = np.round(ft / dt).astype(int)
    g = grid or GRID
    run = Run(**kw)
    xs, ys, zs = grid_axes(run.deck, g)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    gait = GaitLoop.load()
    walker = None
    snaps, fields, legs, roots = [], [], [], []
    for i, s in enumerate(steps):
        while run.k < s:
            run.step()
        sn = run.snap()
        tv = i / fps
        if gait is not None:
            if walker is None:
                walker = Walker(gait, sn["pose"], tv)
            walker.target(sn["pose"], tv)
            (x, y, z, yaw), q, bob = walker.step(tv)
            roots.append([x, y, z + gait.zmean + bob, yaw])
            legs.append(q)
        else:
            x, y, z, yaw = sn["pose"]
            roots.append([x, y, z + 0.36, yaw])
            legs.append(np.zeros(12))
        snaps.append(sn)
        fields.append(run.field(pts).reshape(Z.shape).astype(np.float16))
    t = np.array([s["t"] for s in snaps])
    ph = [s["phase"] for s in snaps]
    found_frame = ph.index("WALK")
    at_frame = ph.index("TURN") if "TURN" in ph else None
    leak = [s["leak_on"] for s in snaps]
    close_f = leak.index(False) if False in leak else None
    sig = [s["belief"]["sigma"] for s in snaps]
    est = run.estimate
    arr = {
        "t": t, "dt": np.r_[t[1] - t[0], np.diff(t)], "phase": np.array([PHASES.index(p) for p in ph], np.int8),
        "root": np.array(roots, np.float64), "legs": np.array(legs, np.float32),
        "valve_s": np.array([np.nan if s["valve_s"] is None else s["valve_s"] for s in snaps]),
        "leak_on": np.array(leak), "flow": np.array([s["flow"] for s in snaps]),
        "wind": np.array([s["wind"] for s in snaps]), "wind_speed": np.array([s["wind_speed"] for s in snaps]),
        "n_samples": np.array([s["n_samples"] for s in snaps], np.int32),
        "samples": np.array(run.samples, np.float64).reshape(-1, 4),
        "particles": np.stack([s["particles"] for s in snaps]),
        "belief": np.array([[*s["belief"]["mean"], s["belief"]["sigma"]] for s in snaps]),
        "arrow": np.array([np.full((2, 2), np.nan) if s["arrow"] is None else s["arrow"] for s in snaps]),
        "found": np.array([s["found"] for s in snaps]), "obs": np.array([s["obs"] for s in snaps], np.int32),
        "field": np.stack(fields), "gx": xs, "gy": ys, "gz": zs,
    }
    metrics = {"seek_s": round(probe.t_found, 2), "found_by": run.found_by,
               "err_m": round(float(np.hypot(est[0] - run.src[0], est[1] - run.src[1])), 3),
               "sigma0": round(run.sigma0, 3), "sigma_found": round(sig[found_frame], 3), "n_obs": int(run.sk.pf.n_obs),
               "n_samples": len(run.samples), "found_frame": found_frame, "at_frame": at_frame, "close_frame": close_f,
               "t_at": round(probe.t_at, 2), "t_close": round(probe.t_close, 2), "t_end": round(float(t[-1]), 2),
               "warm_s": round(run.warm_s, 2), "dt": dt, "stop": [round(v, 4) for v in run.stop]}
    meta = {"src": run.src.tolist(), "wheel": list(DemoGas.wheel), "deck": run.deck, "seed": seed, "fps": fps,
            "frames": frames, "cuts": list(cuts), "estimate": est, "metrics": metrics}
    return build(arr, meta)


def build(arr: dict, meta: dict) -> dict:
    fr = []
    for i in range(len(arr["t"])):
        a = arr["arrow"][i]
        vs = float(arr["valve_s"][i])
        b = arr["belief"][i]
        found = bool(arr["found"][i])
        fr.append({"i": i, "t": float(arr["t"][i]), "dt": float(arr["dt"][i]), "phase": PHASES[int(arr["phase"][i])],
                   "root": arr["root"][i].tolist(), "legs": arr["legs"][i].tolist(), "valve_s": None if np.isnan(vs) else vs,
                   "leak_on": bool(arr["leak_on"][i]), "flow": float(arr["flow"][i]), "wind": arr["wind"][i].tolist(),
                   "wind_speed": float(arr["wind_speed"][i]), "samples": arr["samples"][:int(arr["n_samples"][i])],
                   "particles": arr["particles"][i], "belief": {"mean": b[:2].tolist(), "sigma": float(b[2])},
                   "arrow": None if np.isnan(a).any() else a.tolist(), "found": found,
                   "estimate": meta["estimate"] if found else None, "obs": int(arr["obs"][i]), "field": arr["field"][i]})
    return {**meta, "grid": {"x": arr["gx"], "y": arr["gy"], "z": arr["gz"]}, "frames": fr, "arrays": arr}


def save(p: dict, path) -> None:
    meta = {k: v for k, v in p.items() if k not in ("grid", "frames", "arrays")}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **p["arrays"])


def load(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return build(arr, meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/gas/plan.npz")
    ap.add_argument("--frames", type=int, default=285)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cuts", default="75,90,165,180")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turn_s", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seek_end", type=int)
    ap.add_argument("--close_frame", type=int)
    ap.add_argument("--turn_frames", type=int)
    ap.add_argument("--tail_s", type=float, default=2.5)
    ap.add_argument("--stop", help="x,y,z_pose,yaw")
    a = ap.parse_args(argv)
    import os

    t0 = time.perf_counter()
    p = plan(a.frames, a.fps, tuple(int(c) for c in a.cuts.split(",")), a.seed,
             tuple(float(v) for v in a.stop.split(",")) if a.stop else None, a.turn_s, a.dt, a.seek_end, a.close_frame,
             a.turn_frames, a.tail_s)
    el = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save(p, a.out)
    print(json.dumps({**p["metrics"], "estimate": p["estimate"], "plan_s": round(el, 2),
                      "mb": round(os.path.getsize(a.out) / 1e6, 2)}))


if __name__ == "__main__":
    main()

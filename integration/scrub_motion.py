from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from integration.kinematics import ARM_JOINTS, ARM_STOWED, LIFT_JOINT, LIFT_URDF, RobotModel, pose
from policy.brush_squish import compression, pad_thickness
from policy.interfaces import PipeGeometry

REPO = Path(__file__).resolve().parent.parent
META = REPO / "sim/output/rig_merged_meta.json"
TIP = "brush_tip"
JOINTS = [LIFT_JOINT, *ARM_JOINTS]
WEAR = {"k": 0.4, "min_force": 5.0, "mult_soaked": 4.0, "mult_sprayed": 1.5, "max_slide_speed": 0.5,
        "rust_eps": 0.02}
SPRING_K = 4000.0
PAD_RADIUS = 0.036
FACE_GAP = 0.004
CONE = 0.275
AIM = (0.7, 2.0)
SEAT = 0.08
DEPLOY_S = 3.2
FORCE_RATE = 40.0
CLEANUP_S = 3.2
CLEAN_EPS = 1e-4
LIFT_SPEED = 0.15
ARM_LINKS = ("ur3_upper_arm_link", "ur3_forearm_link", "ur3_wrist_1_link", "ur3_wrist_2_link", "ur3_wrist_3_link")
ARM_LINK_RADIUS = 0.045
SEEDS = tuple((lift, dpan, sl, el) for lift in (0.3, 0.42) for dpan in (-0.6, -0.3, 0.0, 0.3)
              for sl, el in ((-1.7, 1.5), (-2.1, 2.2)))
STANCE_OFFSETS = tuple((step, slide) for step in (0.0, 0.03, 0.06) for slide in (0.0, 0.03))
PHASE_STATES = (("deploy", ("DETECT", "APPROACH")), ("spray", ("SPRAY",)), ("soak", ("SOAK",)),
                ("scrub", ("SCRUB",)), ("verify", ("VERIFY",)), ("paint", ("PAINT",)),
                ("stow", ("VERIFY_PAINT", "DONE", "RETURN")))


def wrap(a):
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi


def smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def load_meta(path=META) -> dict:
    return json.loads(Path(path).read_text())


def pipe_geometry(meta: dict, shape=(40, 64)) -> PipeGeometry:
    w = meta["world"]
    mp = w["main_pipe"]
    return PipeGeometry(w["segment_origin_m"], mp["axis"], mp["ref"], float(mp["radius_m"]),
                        float(w["segment_length_m"]), int(shape[0]), int(shape[1]))


def damage_theta_z(meta: dict, pipe: PipeGeometry) -> tuple[float, float]:
    d = meta["world"]["damage"]
    n = np.asarray(d["surface_normal"], float)
    theta = math.atan2(float(n @ pipe.binormal), float(n @ pipe.ref)) % (2 * math.pi)
    return theta, float((np.asarray(d["position_m"], float) - pipe.origin) @ pipe.axis)


def base_pose(meta: dict, scrub_pose) -> np.ndarray:
    w = meta["world"]
    x, y, yaw = (float(v) for v in scrub_pose)
    return pose(x, y, float(w["deck_z_m"]) + float(w["robot_spawn"]["trunk_height_above_deck_m"]), yaw)


def cell_centres(pipe: PipeGeometry):
    i, j = np.meshgrid(np.arange(pipe.n_z), np.arange(pipe.n_theta), indexing="ij")
    return pipe.cell_to_theta_z(i, j)


def rust_blob(pipe: PipeGeometry, theta: float, z: float, size: float = 0.05, level: float = 0.85) -> np.ndarray:
    th, zz = cell_centres(pipe)
    f = np.exp(-0.5 * ((wrap(th - theta) * pipe.radius / size) ** 2 + ((zz - z) / (0.8 * size)) ** 2))
    return np.clip((f - 0.3) / 0.25, 0.0, 1.0) * level


def first_rust_grid(rows, unit: str = "unit1"):
    for r in rows:
        s = (r.get("robots") or {}).get(unit, {}).get("surface")
        if isinstance(s, dict) and s.get("rust_grid") is not None:
            return np.asarray(s["rust_grid"], float)
    return None


def rust_track_from_timeline(rows, unit: str = "unit1") -> list:
    out = []
    for r in rows:
        v = (r.get("robots") or {}).get(unit, {}).get("rust")
        if isinstance(v, (int, float)) and (not out or out[-1][1] != float(v)):
            out.append((float(r["t"]), float(v)))
    return out


def windows_from_timeline(rows, unit: str = "unit1") -> dict:
    seq = []
    for r in rows:
        st = (r.get("robots") or {}).get(unit, {}).get("state")
        if st and (not seq or seq[-1][1] != st):
            seq.append((float(r["t"]), str(st).upper()))
    scrub = next((t for t, s in seq if s == "SCRUB"), None)
    out = {}
    for name, states in PHASE_STATES:
        after = [t for t, s in seq if s in states and (scrub is None or name != "stow" or t > scrub)]
        if after:
            out[name] = after[0]
    return out


def boustrophedon(u0, u1, z0, z1, step, ds=0.002) -> np.ndarray:
    n = max(1, int(math.ceil((u1 - u0) / step - 1e-9)))
    rows = np.linspace(u0, u1, n + 1)
    pts = []
    for k, u in enumerate(rows):
        za, zb = (z0, z1) if k % 2 == 0 else (z1, z0)
        m = max(2, int(math.ceil(abs(zb - za) / ds)))
        pts += [(u, za + (zb - za) * s) for s in np.linspace(0, 1, m)]
        if k + 1 < len(rows):
            rad = (rows[k + 1] - u) / 2
            sgn = 1.0 if zb >= za else -1.0
            for a in np.linspace(0, math.pi, max(4, int(math.ceil(math.pi * rad / ds))))[1:-1]:
                pts.append((u + rad * (1 - math.cos(a)), zb + sgn * rad * math.sin(a)))
    return np.asarray(pts, float)


def arclength(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def along(path: np.ndarray, s: np.ndarray) -> np.ndarray:
    L = arclength(path)
    return np.stack([np.interp(s, L, path[:, k]) for k in range(path.shape[1])], -1)


@dataclass
class Plan:
    t: np.ndarray
    phase: list
    q: np.ndarray
    tip: np.ndarray
    approach: np.ndarray
    theta: np.ndarray
    z: np.ndarray
    h: np.ndarray
    force: np.ndarray
    nozzle_on: np.ndarray
    rust: np.ndarray
    sprayed: np.ndarray
    painted: np.ndarray
    footprint: np.ndarray
    speeds: dict
    windows: dict
    clean_t: float | None
    warnings: list = field(default_factory=list)


class ScrubJob:

    def __init__(self, scene_meta=None, scrub_pose=None, patch=None, urdf=LIFT_URDF, shape=(40, 64),
                 dt: float = 1 / 60, spray_standoff: float = 0.11, paint_standoff: float = 0.10,
                 hover: float = 0.08, force=(28.0, 4.0), scrub_speed: float = 0.11, step_in: bool = True):
        from integration.fleet_paths import SCRUB

        self.meta = load_meta() if scene_meta is None else (load_meta(scene_meta) if isinstance(scene_meta, (str, Path))
                                                            else scene_meta)
        rust0 = None if patch is None else np.asarray(patch, float)
        if rust0 is not None:
            shape = rust0.shape
        self.pipe = pipe_geometry(self.meta, shape)
        self.theta0, self.z0 = damage_theta_z(self.meta, self.pipe)
        self.rust0 = rust_blob(self.pipe, self.theta0, self.z0) if rust0 is None else rust0
        self.model = RobotModel.from_urdf(urdf)
        self.chain = self.model.chain(TIP)
        self.slot = {j: k for k, j in enumerate(JOINTS)}
        self.lo = np.array([self.model.joints[j].lower for j in JOINTS])
        self.hi = np.array([self.model.joints[j].upper for j in JOINTS])
        self.weight = np.array([0.05, 1, 1, 1, 1, 1, 0.3])
        self.dt, self.hover, self.force_mean, self.force_amp = dt, hover, force[0], force[1]
        self.spray_h, self.paint_h, self.scrub_speed = spray_standoff, paint_standoff, scrub_speed
        self.thickness = pad_thickness(SPRING_K)
        self.cell_theta, self.cell_z = cell_centres(self.pipe)
        self.cell_p, self.cell_n = self.pipe.surface_point(self.cell_theta, self.cell_z)
        self._extents()
        self.requested_pose = tuple(float(v) for v in (SCRUB if scrub_pose is None else scrub_pose))
        x, y, yaw = self.requested_pose
        rel = np.array([x, y, self.pipe.origin[2]]) - self.pipe.origin
        radial = rel - (rel @ self.pipe.axis) * self.pipe.axis
        inward = -radial / max(np.linalg.norm(radial), 1e-9)
        self.seed = None
        ax = self.pipe.axis
        best = None
        for step, slide in (STANCE_OFFSETS if step_in else ((0.0, 0.0),)):
            stance = (x + step * inward[0] + slide * ax[0], y + step * inward[1] + slide * ax[1], yaw)
            self.base = base_pose(self.meta, stance)
            got = self._branch()
            if got is not None and (best is None or got[0] - 0.5 * step > best[0]):
                best = (got[0] - 0.5 * step, stance, got[1])
        if best is not None:
            self.scrub_pose, self.seed = best[1], best[2]
            self.base = base_pose(self.meta, self.scrub_pose)
        if self.seed is None:
            raise RuntimeError("no elbow-up arm branch covers the rust patch from the scrub pose")
        self.step_in = float(np.hypot(self.scrub_pose[0] - x, self.scrub_pose[1] - y))
        self.plan_ = None

    def _extents(self):
        r = self.pipe.radius
        m = self.rust0 > WEAR["rust_eps"]
        du = wrap(self.cell_theta[m] - self.theta0) * r
        dz = self.cell_z[m]
        hu, hz = math.pi * r / self.pipe.n_theta, self.pipe.length / self.pipe.n_z / 2
        self.patch_u = (float(du.min()) - hu, float(du.max()) + hu)
        self.patch_z = (float(dz.min()) - hz, float(dz.max()) + hz)

    def surface(self, theta, z, h):
        theta, z, h = (np.asarray(v, float) for v in (theta, z, h))
        p, _ = self.pipe.surface_point(theta, z)
        tilt = np.clip(h / SEAT, 0.0, 1.0)
        d = self.pipe.surface_point(theta - tilt * AIM[0] * wrap(theta - self.theta0), z)[1]
        d = d - (tilt * AIM[1] * (z - self.z0))[..., None] * self.pipe.axis
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return p + (FACE_GAP + h)[..., None] * d, d

    def paths(self) -> dict:
        (u0, u1), (z0, z1) = self.patch_u, self.patch_z
        out = {}
        for name, h, margin in (("spray", self.spray_h, 0.012), ("paint", self.paint_h, 0.008)):
            fp = CONE * (h + FACE_GAP)
            out[name] = self.fitted_rows(u0 - margin + 0.5 * fp, u1 + margin - 0.5 * fp, 1.75 * fp,
                                         margin - 0.5 * fp, 1.75 * fp)
        inset = 0.25 * PAD_RADIUS
        out["scrub"] = boustrophedon(u0 + inset, u1 - inset, z0 + inset, z1 - inset, 0.6 * PAD_RADIUS)
        return out

    def fitted_rows(self, u0, u1, step, margin, reach, ds=0.002) -> np.ndarray:
        n = max(1, int(math.ceil((u1 - u0) / step - 1e-9)))
        rows = np.linspace(u0, u1, n + 1)
        m = self.rust0 > WEAR["rust_eps"]
        cu = wrap(self.cell_theta[m] - self.theta0) * self.pipe.radius
        cz = self.cell_z[m]
        ends = []
        for u in rows:
            near = np.abs(cu - u) <= reach
            zz = cz[near] if near.any() else cz
            ends.append((float(zz.min()) - margin, float(zz.max()) + margin))
        pts = []
        for k, u in enumerate(rows):
            za, zb = ends[k] if k % 2 == 0 else ends[k][::-1]
            m_ = max(2, int(math.ceil(abs(zb - za) / ds)))
            pts += [(u, za + (zb - za) * s_) for s_ in np.linspace(0, 1, m_)]
            if k + 1 < len(rows):
                zn = ends[k + 1][1] if k % 2 == 0 else ends[k + 1][0]
                rad = (rows[k + 1] - u) / 2
                sgn = 1.0 if zb >= za else -1.0
                for a in np.linspace(0, math.pi, max(4, int(math.ceil(math.pi * rad / ds))))[1:-1]:
                    pts.append((u + rad * (1 - math.cos(a)), zb + (zn - zb) * a / math.pi + sgn * rad * math.sin(a)))
        return np.asarray(pts, float)

    def natural_windows(self, t0: float = 0.0, soak_s: float = 5.7, deploy_s: float = 2.2) -> dict:
        P = {k: arclength(v)[-1] for k, v in self.paths().items()}
        w = {"deploy": t0}
        w["spray"] = w["deploy"] + deploy_s
        w["soak"] = w["spray"] + P["spray"] / 0.3 + 0.6
        w["scrub"] = w["soak"] + soak_s
        w["verify"] = w["scrub"] + 2 * P["scrub"] / self.scrub_speed + 1.2
        w["paint"] = w["verify"] + 1.5
        w["stow"] = w["paint"] + P["paint"] / 0.3 + 0.6
        return w

    def fk(self, qv) -> np.ndarray:
        T = self.base.copy()
        for j in self.chain:
            T = T @ j.local(float(qv[self.slot[j.name]]) if j.name in self.slot else 0.0)
        return T

    def ik(self, qv, target, approach, pref=None, iterations: int = 60):
        qv = np.array(qv, float)
        e = np.zeros(6)
        for _ in range(iterations):
            T = self.fk(qv)
            e = np.concatenate([target - T[:3, 3], approach - T[:3, 0]])
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            J = np.zeros((6, len(JOINTS)))
            for k in range(len(JOINTS)):
                d = qv.copy()
                d[k] += 1e-6
                Td = self.fk(d)
                J[:, k] = np.concatenate([Td[:3, 3] - T[:3, 3], Td[:3, 0] - T[:3, 0]]) / 1e-6
            wt = self.weight.copy()
            for _k in range(2):
                Jw = J * wt
                A = Jw @ J.T + 1e-5 * np.eye(6)
                step = Jw.T @ np.linalg.solve(A, e)
                if pref is not None:
                    step += (np.eye(len(JOINTS)) - Jw.T @ np.linalg.solve(A, J)) @ (0.05 * wt * (pref - qv))
                stuck = ((qv <= self.lo + 1e-9) & (step < 0)) | ((qv >= self.hi - 1e-9) & (step > 0))
                if not stuck.any():
                    break
                wt = np.where(stuck, 0.0, wt)
            qv = np.clip(qv + np.clip(step, -0.2, 0.2), self.lo, self.hi)
        T = self.fk(qv)
        return qv, float(np.linalg.norm(target - T[:3, 3])), float(math.acos(np.clip(T[:3, 0] @ approach, -1, 1)))

    def _solve(self, q, ks, tip, appr, seed, t, fixed_lift=False):
        cur = seed.copy()
        saved = self.weight.copy()
        if fixed_lift:
            self.weight = self.weight * np.r_[0.0, np.ones(len(JOINTS) - 1)]
        try:
            for n_, k in enumerate(ks):
                if fixed_lift:
                    cur[0] = q[k, 0]
                cur, pe, ae = self.ik(cur, tip[k], appr[k], seed, 200 if n_ == 0 else 40)
                if pe > 2e-3 or ae > 0.02:
                    raise RuntimeError(f"IK lost the brush at t={t[k]:.2f}: {pe:.4f} m")
                q[k] = cur
        finally:
            self.weight = saved

    def qdict(self, qv) -> dict:
        return {j: float(v) for j, v in zip(JOINTS, qv)}

    def _reach(self, qv, pose_from, pose_to, pref, steps: int = 12):
        worst = (np.inf, np.inf)
        for s in np.linspace(0, 1, steps + 1)[1:]:
            p = [a + s * (b - a) for a, b in zip(pose_from, pose_to)]
            tgt, n = self.surface(*p)
            qv, pe, ae = self.ik(qv, tgt, -n, pref, 80)
            if pe > 1e-3 or ae > 0.01:
                return None
            b = self.branch_ok(qv)
            worst = (min(worst[0], b[0]), min(worst[1], b[1]))
        return qv, worst

    def branch_ok(self, qv) -> tuple[float, float]:
        q = self.qdict(qv)
        return min(q["ur3_elbow_joint"], -q["ur3_wrist_2_joint"], 10 * self.elbow_up(q) + 0.8), self.arm_clearance(q)

    def _branch(self):
        r = self.pipe.radius
        centre = (self.theta0 + float(np.mean(self.patch_u)) / r, float(np.mean(self.patch_z)), 0.0)
        goals = []
        for name, path in self.paths().items():
            h = {"spray": self.spray_h, "paint": self.paint_h, "scrub": 0.0}[name]
            for u in (path[:, 0].min(), path[:, 0].max()):
                for z in (path[:, 1].min(), path[:, 1].max()):
                    goals.append((self.theta0 + u / r, z, h))
        goals.append((centre[0], centre[1], self.hover))
        tgt, d = self.surface(*centre)
        rel = np.linalg.inv(self.base) @ np.append(tgt, 1.0)
        phi = math.atan2(rel[1], rel[0])
        best = None
        for lift, dpan, sl, el in SEEDS:
            q0 = {**self.model.default_q(), **ARM_STOWED, LIFT_JOINT: lift, "ur3_elbow_joint": el,
                  "ur3_shoulder_pan_joint": float(wrap(phi + dpan)), "ur3_shoulder_lift_joint": sl}
            qv, pe, ae = self.ik(np.array([q0[j] for j in JOINTS]), tgt, -d, None, 200)
            if pe > 1e-3 or ae > 0.01 or self.branch_ok(qv)[0] < 0.3:
                continue
            worst = self.branch_ok(qv)
            for g in goals:
                got = self._reach(qv.copy(), centre, g, qv)
                if got is None:
                    worst = None
                    break
                worst = (min(worst[0], got[1][0]), min(worst[1], got[1][1]))
            if worst is not None and worst[0] > 0.15 and worst[1] > 0.01:
                score = min(worst[0], 0.8) + min(worst[1], 0.1)
                if best is None or score > best[0]:
                    best = (score, qv)
        return best

    def link_points(self, q: dict) -> dict:
        P = self.model.link_poses(self.base, q)
        return {k: P[k][:3, 3] for k in ("ur3_upper_arm_link", "ur3_forearm_link", "ur3_wrist_1_link",
                                         "ur3_wrist_2_link", "ur3_wrist_3_link", TIP)}

    def elbow_up(self, q: dict) -> float:
        P = self.link_points(q)
        s, e, w = P["ur3_upper_arm_link"], P["ur3_forearm_link"], P["ur3_wrist_1_link"]
        d = w - s
        foot = s + float((e - s) @ d) / float(d @ d) * d
        return float((e - foot)[2])

    def arm_clearance(self, q: dict) -> float:
        P = self.link_points(q)
        pts = [P[a] + (P[b] - P[a]) * s for a, b in zip(ARM_LINKS[:-1], ARM_LINKS[1:]) for s in np.linspace(0, 1, 8)]
        rel = np.asarray(pts) - self.pipe.origin
        radial = rel - (rel @ self.pipe.axis)[:, None] * self.pipe.axis
        return float(np.linalg.norm(radial, axis=1).min() - self.pipe.radius - ARM_LINK_RADIUS)

    def plan(self, t_spray=None, t_soak=None, t_scrub=None, t_paint=None, t_done=None, t_deploy=None,
             t_verify=None, stow_s: float = 2.0, soak_required=None, rust_track=None) -> Plan:
        nat = self.natural_windows()
        given = {"deploy": t_deploy, "spray": t_spray, "soak": t_soak, "scrub": t_scrub, "verify": t_verify,
                 "paint": t_paint, "stow": t_done}
        if t_spray is None:
            w = dict(nat)
        else:
            w = {"spray": float(t_spray)}
            prev = "spray"
            for k in ("soak", "scrub", "verify", "paint", "stow"):
                w[k] = float(given[k]) if given[k] is not None else w[prev] + nat[k] - nat[prev]
                prev = k
            w["deploy"] = float(t_deploy) if t_deploy is not None else w["spray"] - (nat["spray"] - nat["deploy"])
            w["deploy"] = min(w["deploy"], w["spray"] - DEPLOY_S)
        if w["verify"] <= w["scrub"]:
            w["verify"] = w["scrub"] + nat["verify"] - nat["scrub"]
        r = self.pipe.radius
        th_of = lambda u: self.theta0 + u / r
        seed = self.seed
        t_lift = 1.875 * seed[0] / LIFT_SPEED * 1.05
        w["lift_up"] = min(w["deploy"], w["spray"] - t_lift)
        w["end"] = w["stow"] + max(stow_s, t_lift)
        warnings = []
        for a, b in zip(("deploy", "spray", "soak", "scrub", "verify", "paint", "stow"),
                        ("spray", "soak", "scrub", "verify", "paint", "stow", "end")):
            if w[b] <= w[a]:
                raise ValueError(f"phase {a} has no time: {w[a]} -> {w[b]}")
        paths = self.paths()
        n = int(round((w["end"] - w["lift_up"]) / self.dt)) + 1
        t = w["lift_up"] + np.arange(n) * self.dt
        U, Z, H = np.zeros(n), np.zeros(n), np.full(n, self.hover)
        F, noz, cart = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
        phase = [""] * n
        speeds = {}

        def idx(a, b):
            return np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9))

        def move(ii, p0, p1):
            if len(ii):
                s = smooth((t[ii] - t[ii[0]]) / max(t[ii[-1]] - t[ii[0]], self.dt))[:, None]
                v = np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0))
                U[ii], Z[ii], H[ii] = v[:, 0], v[:, 1], v[:, 2]
                cart[ii] = True

        def trace(ii, path, h, name, passes=1):
            full = path
            for k in range(1, passes):
                full = np.vstack([full, (path if k % 2 == 0 else path[::-1])[1:]])
            L = arclength(full)[-1]
            dur = max(t[ii[-1]] - t[ii[0]], self.dt)
            s = (t[ii] - t[ii[0]]) / dur
            p = along(full, s * L)
            U[ii], Z[ii] = p[:, 0], p[:, 1]
            H[ii] = h
            cart[ii] = True
            speeds[name] = L / dur
            return full

        tr = 0.35
        sp = paths["spray"]
        ii = idx(w["spray"], w["soak"] - tr)
        trace(ii, sp, self.spray_h, "spray")
        noz[ii] = True
        for k in ii:
            phase[k] = "spray"
        sc = self.scrub_rows()
        p_scrub0 = (sc[0], sc[1], self.hover)
        ii = idx(w["soak"] - tr, w["soak"])
        move(ii, (sp[-1, 0], sp[-1, 1], self.spray_h), (sp[-1, 0], sp[-1, 1], self.hover))
        down = 1.3
        ii2 = idx(w["soak"], w["scrub"] - down)
        move(ii2[: max(1, min(len(ii2), int(1.5 / self.dt)))], (sp[-1, 0], sp[-1, 1], self.hover), p_scrub0)
        tail = ii2[max(1, min(len(ii2), int(1.5 / self.dt))):]
        U[tail], Z[tail], H[tail], cart[tail] = p_scrub0[0], p_scrub0[1], self.hover, True
        for k in np.concatenate([ii, ii2]):
            phase[k] = "soak" if t[k] >= w["soak"] else "spray"
        ii_s = idx(w["scrub"], w["verify"] - down)
        th_s = th_of(U)
        k_spr = [k for k in range(n) if noz[k] and phase[k] == "spray"]
        tip_s, d_s = self.surface(th_s[k_spr], Z[k_spr], H[k_spr])
        spr = np.zeros(self.rust0.shape, bool)
        for a, d in zip(tip_s, d_s):
            spr |= self.cone(a, -d)
        target = self.rust_target(rust_track, w) if rust_track is not None else None
        Us, Zs, Fs = self._scrub_control(t[ii_s], spr, target)
        U[ii_s], Z[ii_s], cart[ii_s] = Us, Zs, True
        F[ii_s] = Fs
        H[ii_s] = -Fs / SPRING_K
        speeds["scrub"] = float(np.mean(np.hypot(np.diff(Us), np.diff(Zs))) / self.dt)
        press = Fs[0] / FORCE_RATE
        ii = idx(w["scrub"] - down, w["scrub"])
        move(ii[t[ii] < w["scrub"] - press], p_scrub0, (Us[0], Zs[0], 0.0))
        ip = ii[t[ii] >= w["scrub"] - press]
        U[ip], Z[ip], cart[ip] = Us[0], Zs[0], True
        H[ip] = -(t[ip] - (w["scrub"] - press)) * FORCE_RATE / SPRING_K
        for k in ii:
            phase[k] = "soak"
        full = np.stack([Us, Zs], -1)
        pt = paths["paint"]
        p_paint0 = (pt[0, 0], pt[0, 1], self.hover)
        ii_u = idx(w["verify"] - down, w["verify"])
        move(ii_u, (full[-1, 0], full[-1, 1], H[ii_s[-1]]), (full[-1, 0], full[-1, 1], self.hover))
        ii_v = idx(w["verify"], w["paint"])
        move(ii_v, (full[-1, 0], full[-1, 1], self.hover), p_paint0)
        for k in np.concatenate([ii, ii_s, ii_u]):
            phase[k] = "scrub"
        for k in ii_v:
            phase[k] = "verify"
        ii = idx(w["paint"], w["paint"] + 0.4)
        move(ii, p_paint0, (pt[0, 0], pt[0, 1], self.paint_h))
        ii_p = idx(w["paint"] + 0.4, w["stow"] - 0.4)
        trace(ii_p, pt, self.paint_h, "paint")
        noz[ii_p] = True
        ii_q = idx(w["stow"] - 0.4, w["stow"])
        move(ii_q, (pt[-1, 0], pt[-1, 1], self.paint_h), (pt[-1, 0], pt[-1, 1], self.hover))
        for k in np.concatenate([ii, ii_p, ii_q]):
            phase[k] = "paint"
        hard = H < 0
        F = np.where(hard, -H * SPRING_K, 0.0)
        theta = th_of(U)
        tip, nrm = self.surface(theta, Z, H)
        appr = -nrm

        q = np.zeros((n, len(JOINTS)))
        first = int(np.flatnonzero(cart)[0])
        ks = np.flatnonzero(cart)
        self._solve(q, ks, tip, appr, seed, t)
        lift = q[ks, 0].copy()
        sig = int(0.5 / self.dt)
        kern = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
        lift = np.convolve(np.pad(lift, 3 * sig, mode="edge"), kern / kern.sum(), mode="valid")
        cap = LIFT_SPEED * 0.9 * self.dt
        for rng in (range(1, len(lift)), range(len(lift) - 2, -1, -1)):
            for i in rng:
                j = i - 1 if rng.step > 0 else i + 1
                lift[i] = np.clip(lift[i], lift[j] - cap, lift[j] + cap)
        q[ks, 0] = np.minimum(lift, self.hi[0])
        self._solve(q, ks, tip, appr, seed, t, fixed_lift=True)
        stowed = np.array([ARM_STOWED[j] for j in JOINTS])
        ii_d = idx(w["lift_up"], w["spray"])
        qa = q[first]
        s_arm = smooth((t[ii_d] - w["deploy"]) / (w["spray"] - w["deploy"]))
        s_lift = smooth((t[ii_d] - w["lift_up"]) / (w["spray"] - w["lift_up"]))
        q[ii_d] = stowed + s_arm[:, None] * (qa - stowed)
        q[ii_d, 0] = stowed[0] + s_lift * (qa[0] - stowed[0])
        for k in ii_d:
            phase[k] = "deploy"
        ii_e = idx(w["stow"], w["end"] + self.dt)
        last = q[ii_e[0] - 1]
        s_arm = smooth((t[ii_e] - w["stow"]) / stow_s)
        s_lift = smooth((t[ii_e] - w["stow"]) / (w["end"] - w["stow"]))
        q[ii_e] = last + s_arm[:, None] * (stowed - last)
        q[ii_e, 0] = last[0] + s_lift * (stowed[0] - last[0])
        for k in ii_e:
            phase[k] = "stow"
        for k, nm in enumerate(phase):
            if not nm:
                phase[k] = "deploy" if t[k] < w["spray"] else "stow"
        for k in np.concatenate([ii_d, ii_e]):
            T = self.model.tool_pose(self.base, dict(zip(JOINTS, q[k])), TIP)
            tip[k], appr[k] = T[:3, 3], T[:3, 0]
        lift_v = np.abs(np.diff(q[:, 0])).max() / self.dt
        if lift_v > LIFT_SPEED * 1.01:
            warnings.append(f"lift speed {lift_v:.3f} m/s over the URDF limit {LIFT_SPEED}")
        for name, v in speeds.items():
            lo, hi = (0.1, 0.2) if name == "scrub" else (0.1, 0.5)
            if not lo - 1e-6 <= v <= hi + 1e-6:
                warnings.append(f"{name} speed {v:.3f} m/s outside {lo}-{hi}")
        rust, sprayed, painted, foot, clean_t = self._simulate(t, theta, Z, H, F, noz, phase, tip, appr, w,
                                                                soak_required)
        if clean_t is None:
            warnings.append("scrub window ends with rust above 5 %")
        self.plan_ = Plan(t, phase, q, tip, appr, theta, Z, H, F, noz, rust, sprayed, painted, foot, speeds, w,
                          clean_t, warnings)
        return self.plan_

    def cone(self, tip, direction) -> np.ndarray:
        v = self.cell_p - np.asarray(tip, float)
        d = np.linalg.norm(v, axis=-1)
        a = np.asarray(direction, float)
        return ((v @ a) >= math.cos(math.atan(CONE)) * d) & ((self.cell_n @ a) < 0) & (d < 2.5 * self.spray_h)

    def wear(self, cur, theta, z, force, vel, sprayed, soaked):
        d2 = ((wrap(self.cell_theta - theta) * self.pipe.radius) ** 2 + (self.cell_z - z) ** 2) / PAD_RADIUS ** 2
        wgt = np.clip(1.0 - d2, 0.0, None)
        if force < WEAR["min_force"]:
            return cur, wgt
        mult = np.where(soaked, WEAR["mult_soaked"], np.where(sprayed, WEAR["mult_sprayed"], 1.0))
        rate = WEAR["k"] * force * min(vel, WEAR["max_slide_speed"])
        return cur - np.minimum(cur, rate * mult * wgt * self.dt), wgt

    def scrub_rows(self):
        (u0, u1), (z0, z1) = self.patch_u, self.patch_z
        inset = 0.25 * PAD_RADIUS
        n = max(1, int(math.ceil((u1 - u0 - 2 * inset) / (0.6 * PAD_RADIUS) - 1e-9)))
        self.rows = np.linspace(u0 + inset, u1 - inset, n + 1)
        self.stroke = ((z0 + z1) / 2, max(0.02, (z1 - z0) / 2 - inset))
        return (self.rows[0], self.stroke[0] - self.stroke[1])

    def rust_target(self, track, w):
        pts = sorted(track)
        before = [v for t, v in pts if t <= w["scrub"]]
        tr = np.asarray([(w["scrub"], before[-1] if before else 1.0)] + [(t, v) for t, v in pts if t > w["scrub"]],
                        float)
        t_fin = w["verify"] - CLEANUP_S
        t = np.arange(w["scrub"], w["verify"] + self.dt, self.dt)
        base = np.interp(t, tr[:, 0], tr[:, 1])
        low = np.flatnonzero(base <= 0.077)
        t0 = min(t[low[0]] if len(low) else t_fin, t_fin)
        tail = np.interp(t, [t0, t_fin], [0.077, 0.05]) * (t <= t_fin)
        return t, np.maximum(base, tail), base, t_fin

    def _scrub_control(self, ts, sprayed, target):
        r, (zc, amp) = self.pipe.radius, self.stroke
        rows = self.rows
        cu = wrap(self.cell_theta - self.theta0) * r
        owner = np.abs(cu[..., None] - rows).argmin(-1)
        cur = self.rust0.astype(np.float64).copy()
        total0 = max(float(cur.sum()), 1e-9)
        omega = math.pi * self.scrub_speed / (2 * amp)
        u, z, phi, k, f_prev = rows[0], zc - amp, 0.0, 0, 16.0
        U, Z, F = [], [], []
        for i, t in enumerate(ts):
            if i:
                phi_new = phi + omega * self.dt
                if math.floor(phi_new / math.pi) > math.floor(phi / math.pi):
                    clean_up = target is not None and t >= target[3]
                    lim = CLEAN_EPS if clean_up else 0.03
                    dirty = [j for j in range(len(rows)) if cur[owner == j].max(initial=0.0) > lim]
                    if k not in dirty and dirty:
                        k = min(dirty, key=lambda j: (abs(j - k), -j))
                phi = phi_new
                u += float(np.clip(rows[k] - u, -0.04 * self.dt, 0.04 * self.dt))
                z = zc - amp * math.cos(phi)
            if target is None:
                f = self.force_mean + self.force_amp * math.sin(2 * math.pi * 1.3 * (t - ts[0]))
            elif t >= target[3]:
                f = 20.0 if cur.max() > CLEAN_EPS else 3.0
            else:
                goal = float(np.interp(t, target[0], target[1]))
                err = cur.sum() / total0 - goal
                base = 5.5 + 10.5 * float(np.clip((goal - 0.08) / 0.1, 0.0, 1.0))
                f = float(np.clip(base + 500.0 * err, 3.0, 30.0))
            f = min(f, FORCE_RATE * 0.8 * float(ts[-1] - t))
            f = float(np.clip(f, f_prev - FORCE_RATE * self.dt, f_prev + FORCE_RATE * self.dt))
            f_prev = f
            if i:
                vel = math.hypot(u - U[-1], z - Z[-1]) / self.dt
                cur, _ = self.wear(cur, self.theta0 + u / r, z, f, vel, sprayed, sprayed)
            U.append(u)
            Z.append(z)
            F.append(f)
        return np.array(U), np.array(Z), np.array(F)

    def _simulate(self, t, theta, Z, H, F, noz, phase, tip, appr, w, soak_required):
        r = self.pipe.radius
        soak_at = w["scrub"] if soak_required is None else None
        n = len(t)
        rust = np.empty((n,) + self.rust0.shape, np.float32)
        sprayed = np.zeros((n,) + self.rust0.shape, bool)
        painted = np.zeros_like(sprayed)
        foot = np.zeros_like(sprayed)
        cur = self.rust0.astype(np.float64).copy()
        wet_t = np.full(cur.shape, np.inf)
        spr, pnt = np.zeros(cur.shape, bool), np.zeros(cur.shape, bool)
        total0 = max(float(self.rust0.sum()), 1e-9)
        clean_t = None
        for k in range(n):
            if noz[k]:
                cone = self.cone(tip[k], appr[k])
                if phase[k] == "spray":
                    wet_t = np.where(cone & ~spr, t[k], wet_t)
                    spr |= cone
                elif phase[k] == "paint":
                    pnt |= cone
            if F[k] >= WEAR["min_force"] and k > 0:
                soaked = spr & ((t[k] >= soak_at) if soak_at is not None else (t[k] - wet_t >= soak_required))
                vel = math.hypot((theta[k] - theta[k - 1]) * r, Z[k] - Z[k - 1]) / self.dt
                cur, wgt = self.wear(cur, theta[k], Z[k], F[k], vel, spr, soaked)
                foot[k] = wgt > 0
            rust[k], sprayed[k], painted[k] = cur, spr, pnt
            if clean_t is None and phase[k] == "scrub" and cur.sum() / total0 < 0.05:
                clean_t = float(t[k])
        return rust, sprayed, painted, foot, clean_t

    def index(self, t: float) -> int:
        p = self.plan_
        return int(np.clip(round((t - p.t[0]) / self.dt), 0, len(p.t) - 1))

    def at(self, t: float) -> dict:
        p = self.plan_
        if p is None:
            p = self.plan()
        k = self.index(t)
        before, after = t < p.t[0] - 1e-9, t > p.t[-1] + 1e-9
        th, z = float(p.theta[k]), float(p.z[k])
        surf, n = self.pipe.surface_point(th, z)
        force = 0.0 if before or after else float(p.force[k])
        on = bool(p.nozzle_on[k]) and not (before or after)
        q = {j: float(v) for j, v in zip(JOINTS, p.q[k])}
        if before or after:
            q = {j: ARM_STOWED[j] for j in JOINTS}
        T = self.model.tool_pose(self.base, q, TIP)
        phase = "idle" if before else "done" if after else p.phase[k]
        return {
            "t": float(t), "phase": phase, "q": q, "base": self.base.copy(),
            "tip": {"xyz": T[:3, 3].tolist(), "rot": T[:3, :3].tolist(), "approach": T[:3, 0].tolist()},
            "brush_contact": {"xyz": surf.tolist(), "normal": n.tolist(), "force_N": force,
                              "squish": float(compression(force)), "in_contact": force > 0.0},
            "nozzle": {"xyz": T[:3, 3].tolist(), "dir": T[:3, 0].tolist(), "on": on,
                       "length": float(p.h[k] + FACE_GAP), "mode": phase if on else ""},
            "rust": p.rust[0 if before else k], "sprayed": p.sprayed[0 if before else k],
            "painted": p.painted[0 if before else k],
            "rust_fraction": float(p.rust[0 if before else k].sum() / max(self.rust0.sum(), 1e-9)),
        }


def scrub_track(job: ScrubJob, fps: int = 30) -> dict:
    p = job.plan_
    rows = []
    for t in np.arange(p.t[0], p.t[-1] + 1e-9, 1.0 / fps):
        k = job.index(t)
        rows.append({"t": round(float(t), 4), "rust_fraction": round(float(p.rust[k].sum() / job.rust0.sum()), 4),
                     "force_N": round(float(p.force[k]), 2), "phase": p.phase[k]})
    return {"source": "integration/scrub_motion.py", "unit": "unit1", "fps": fps, "clean_t": p.clean_t,
            "windows": {k: round(float(v), 4) for k, v in p.windows.items()}, "rows": rows}


def film_job(timeline, scrub_pose=(18.225, -5.262, 0.0)) -> ScrubJob:
    rows = [json.loads(ln) for ln in Path(timeline).read_text().splitlines() if ln.strip()]
    w = windows_from_timeline(rows)
    job = ScrubJob(patch=first_rust_grid(rows), scrub_pose=scrub_pose)
    job.plan(t_deploy=w.get("deploy"), t_spray=w["spray"], t_soak=w["soak"], t_scrub=w["scrub"],
             t_verify=w.get("verify"), t_paint=w["paint"], t_done=w.get("stow"),
             rust_track=rust_track_from_timeline(rows))
    return job


def check_png(job: ScrubJob, out) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p, r = job.plan_, job.pipe.radius
    ext = [job.pipe.length / -2, job.pipe.length / 2, 0, 2 * math.pi * r]
    u = lambda th: np.mod(th, 2 * math.pi) * r
    w = p.windows
    picks = [("start of scrub", w["scrub"]), ("scrub +2 s", w["scrub"] + 2), ("scrub +6 s", w["scrub"] + 6), ("scrub +10 s", w["scrub"] + 10),
             ("end of scrub", w["verify"] - 0.05), ("after paint", w["stow"])]
    fig, ax = plt.subplots(1, len(picks) + 1, figsize=(4 * (len(picks) + 1), 4.6), sharey=True)
    for a in ax:
        a.set_xlim(job.z0 - 0.13, job.z0 + 0.13)
        a.set_ylim(u(job.theta0) - 0.14, u(job.theta0) + 0.14)
        a.set_xlabel("z along pipe (m)")
    ax[0].set_ylabel("arc r·θ (m)")
    ax[0].imshow(job.rust0.T[::-1], extent=ext, origin="upper", cmap="Oranges", vmin=0, vmax=1,
                 aspect="auto", interpolation="nearest")
    for name, col in (("spray", "tab:blue"), ("scrub", "black"), ("paint", "tab:green")):
        m = np.array([ph == name for ph in p.phase]) & (p.nozzle_on if name != "scrub" else p.force > 0)
        ax[0].plot(p.z[m], u(p.theta[m]), lw=0.8, color=col, label=name)
    ax[0].legend(loc="upper right", fontsize=7)
    ax[0].set_title("rust at start + tool paths")
    for a, (label, t) in zip(ax[1:], picks):
        k = job.index(t)
        img = np.clip(p.rust[k] / 0.85, 0, 1)
        rgb = np.ones(img.shape + (3,))
        rgb[p.sprayed[k]] = (0.75, 0.85, 0.95)
        rgb = rgb * (1 - img[..., None]) + np.array([0.72, 0.33, 0.1]) * img[..., None]
        rgb[p.painted[k]] = (0.42, 0.55, 0.42)
        a.imshow(np.transpose(rgb, (1, 0, 2))[::-1], extent=ext, origin="upper", aspect="auto",
                 interpolation="nearest")
        m = (p.t <= t) & (p.force > 0)
        a.plot(p.z[m], u(p.theta[m]), lw=0.5, color="black", alpha=0.5)
        if p.force[k] > 0:
            a.add_patch(plt.Circle((p.z[k], u(p.theta[k])), PAD_RADIUS, fill=False, color="magenta", lw=1.5))
        a.set_title(f"{label} t={t:.1f}  rust {p.rust[k].sum() / job.rust0.sum():.0%}")
    fig.tight_layout()
    a_png = out / "scrub_coverage.png"
    fig.savefig(a_png, dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    ticks = [("deploy", w["deploy"] + 0.6), ("spray", w["spray"] + 1.2), ("scrub low row", w["scrub"] + 1.0),
             ("scrub top row", w["scrub"] + 4.6), ("paint", w["paint"] + 2.0)]
    o = job.pipe.origin
    for a, (i0, i1, lab) in zip(ax, ((1, 2, "side: y (m)"), (0, 1, "top: x (m)"))):
        if i0 == 1:
            a.add_patch(plt.Circle((o[1], o[2]), r, color="0.6"))
        else:
            a.add_patch(plt.Rectangle((o[0] - 0.6, o[1] - r), 1.2, 2 * r, color="0.6"))
        a.set_aspect("equal")
        a.set_xlabel(lab)
    for (label, t), col in zip(ticks, plt.cm.viridis(np.linspace(0, 0.9, len(ticks)))):
        q = job.qdict(p.q[job.index(t)])
        P = job.model.link_poses(job.base, q)
        names = ["base", "lift_housing", "ur3_base_link", "ur3_shoulder_link", "ur3_upper_arm_link",
                 "ur3_forearm_link", "ur3_wrist_1_link", "ur3_wrist_2_link", "ur3_wrist_3_link", TIP]
        pts = np.array([P[n][:3, 3] for n in names if n in P])
        for a, (i0, i1) in zip(ax, ((1, 2), (0, 1))):
            a.plot(pts[:, i0], pts[:, i1], "-o", ms=3, color=col, label=f"{label} {t:.1f}s")
            T = P[TIP]
            a.plot([T[i0, 3], T[i0, 3] + 0.05 * T[i0, 0]], [T[i1, 3], T[i1, 3] + 0.05 * T[i1, 0]], color="red")
    ax[0].set_ylabel("z (m)")
    ax[1].set_ylabel("y (m)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Unit 1 arm, looking along the pipe")
    ax[1].set_title("Unit 1 arm from above")
    fig.tight_layout()
    b_png = out / "scrub_arm.png"
    fig.savefig(b_png, dpi=110)
    plt.close(fig)
    return [a_png, b_png]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=str(REPO / "runs_sample/film/timeline.jsonl"))
    ap.add_argument("--out", default=str(REPO / "runs/scrub_motion"))
    ap.add_argument("--track", default="")
    a = ap.parse_args()
    j = film_job(a.timeline)
    if a.track:
        Path(a.track).write_text(json.dumps(scrub_track(j)))
    print(json.dumps({"stance": j.scrub_pose, "windows": j.plan_.windows, "speeds": j.plan_.speeds,
                      "clean_t": j.plan_.clean_t, "warnings": j.plan_.warnings,
                      "png": [str(x) for x in check_png(j, a.out)]}, indent=1, default=float))

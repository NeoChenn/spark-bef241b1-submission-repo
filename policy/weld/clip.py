from __future__ import annotations

import json
import sys

import numpy as np

from integration.kinematics import ARM_JOINTS, TRUNK_HEIGHT, RobotModel, pose
from policy.weld.env import LEAD_MM, WeldEnv
from policy.weld.policy import OpenLoop, SeamTracker
from policy.weld.seam import Seam, SeamConfig, seg_dist
from policy.weld.sensors import groove_center, laser_profile

ORIGIN = (0.0, -0.5, 0.73)
REF, UP, AXIS = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
RADIUS, THETA_S, LIFT = 0.15, 0.9, 0.0


def make_seam(seed: int, wander_mm: float, x0: float, x1: float) -> Seam:
    seam = Seam(SeamConfig(wander_mm=wander_mm), seed=seed)
    rng = np.random.default_rng(seed + 101)
    n = 36
    px = np.sort(np.linspace(x0 - 8, x1 + 8, n) + rng.normal(0, 1.0, n))
    py = np.interp(px, seam.xs, seam.center) + np.clip(rng.normal(0, 0.8, n), -2, 2)
    seam.crack = np.stack([px, py], 1)
    gx, gy = np.meshgrid(seam.xs, seam.ys, indexing="ij")
    d = np.min([seg_dist(gx, gy, seam.crack[i], seam.crack[i + 1]) for i in range(n - 1)], axis=0)
    seam.crack_mask = d < seam.cfg.crack_w_mm
    return seam


def rollout(seam: Seam, policy, x0: float, x1: float, seed: int, record: bool = False):
    y0 = groove_center(*laser_profile(seam, x0 - LEAD_MM, 0.0, np.random.default_rng(seed + 3)))
    env = WeldEnv(seam, x0, x1, seed=seed, y0=y0 or 0.0)
    obs, err, steps = env.obs(), [], []
    for _ in range(4000):
        h = float(np.median(seam.height(env.x, env.y + np.array([-1.0, 0.0, 1.0]))))
        dy, v, i = policy.act(obs)
        prev = obs
        obs, done = env.step(dy, v, i)
        if x0 <= env.x <= x1:
            err.append(env.y - seam.center_at(env.x))
        if record:
            steps.append({"x": env.x, "y": env.y, "h": h, "width": env.width, "true": seam.center_at(env.x),
                          "target": policy.target, "x_look": prev["x_look"], "ys": prev["profile_y"],
                          "z": prev["profile_z"], "center_est": prev["center_est"]})
        if done:
            break
    err = np.asarray(err)
    return float(np.sqrt(np.mean(err ** 2))), float(np.abs(err).max()), steps


def world(x_mm, y_mm, h_mm, x0: float):
    x_mm, y_mm, h_mm = np.broadcast_arrays(np.asarray(x_mm, float), np.asarray(y_mm, float), np.asarray(h_mm, float))
    th = THETA_S + y_mm / (1000 * RADIUS)
    rho = RADIUS + h_mm / 1000
    return (np.asarray(ORIGIN) + AXIS * (x0 + x_mm[..., None] / 1000)
            + rho[..., None] * (np.cos(th)[..., None] * REF + np.sin(th)[..., None] * UP))


def normal(y_mm):
    th = THETA_S + np.asarray(y_mm, float) / (1000 * RADIUS)
    return np.cos(th)[..., None] * REF + np.sin(th)[..., None] * UP


def tool_axis(y_mm, drag_deg: float):
    a = np.radians(drag_deg)
    v = -normal(y_mm) * np.cos(a) - AXIS * np.sin(a)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def refine(model, base, q, target, approach, iters: int = 6):
    q = dict(q)
    for _ in range(iters):
        T = model.tool_pose(base, q)
        e = np.concatenate([target - T[:3, 3], approach - T[:3, 0]])
        if np.linalg.norm(e[:3]) < 2e-5 and np.linalg.norm(e[3:]) < 2e-5:
            break
        J = np.zeros((6, len(ARM_JOINTS)))
        for k, name in enumerate(ARM_JOINTS):
            dq = dict(q)
            dq[name] += 1e-6
            Td = model.tool_pose(base, dq)
            J[:, k] = (np.concatenate([Td[:3, 3], Td[:3, 0]]) - np.concatenate([T[:3, 3], T[:3, 0]])) / 1e-6
        step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        for k, name in enumerate(ARM_JOINTS):
            q[name] += float(step[k])
        q = model.clamp(q)
    return q


def clean(a):
    a = np.asarray(a, float)
    return np.where(np.isfinite(a), np.round(a, 5), np.nan).tolist()


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return jsonable(o.item())
    return o


def sub(a, n: int = 300):
    a = np.asarray(a)
    return a if len(a) <= n else a[np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))]


def plan(frames: int = 210, fps: int = 30, weld=(14, 176), seed: int = 10, wander_mm: float = 10.0,
         x_start: float = 50.0, x_end: float = 260.0, standoff_mm: float = 11.0, drag_deg: float = 12.0,
         hover_m: float = 0.06, lift_m: float = 0.06) -> dict:
    x0 = -(x_start + x_end) / 2000
    seam = make_seam(seed, wander_mm, x_start, x_end)
    rmse, tmax, steps = rollout(seam, SeamTracker(), x_start, x_end, seed, record=True)
    ol_rmse, ol_max, _ = rollout(make_seam(seed, wander_mm, x_start, x_end), OpenLoop(), x_start, x_end, seed)
    d_rmse, _, _ = rollout(make_seam(seed, 3.5, x_start, x_end), SeamTracker(), x_start, x_end, seed)
    d_ol, _, _ = rollout(make_seam(seed, 3.5, x_start, x_end), OpenLoop(), x_start, x_end, seed)

    K = len(steps)
    sx = np.array([s["x"] for s in steps])
    sy = np.array([s["y"] for s in steps])
    sh = np.convolve(np.pad([s["h"] for s in steps], 4, mode="edge"), np.ones(9) / 9, mode="valid")
    sw = np.array([s["width"] for s in steps])
    w0, w1 = weld
    step_t = (w0 + np.arange(K) / max(K - 1, 1) * (w1 - w0)) / fps
    crown = seam.cfg.crown_mm
    bead_pts = world(sx, sy, crown, x0)
    track_pts = world(sx, sy, 0.3, x0)

    model = RobotModel.from_urdf()
    base = pose(0.0, 0.0, TRUNK_HEIGHT)
    q = model.default_q()
    q["lift_joint"] = LIFT

    def tip_at(s):
        x, y, h = (np.interp(s, np.arange(K), v) for v in (sx, sy, sh))
        return world(x, y, h + standoff_mm, x0), tool_axis(y, drag_deg), x, y

    start_tip, start_ax, _, _ = tip_at(0)
    end_tip, end_ax, _, _ = tip_at(K - 1)
    hover = start_tip + normal(sy[0]) * hover_m
    off = end_tip + normal(sy[-1]) * lift_m
    out, errs, qs = [], [], []
    for f in range(frames):
        arc = w0 <= f <= w1
        s = float(np.clip((f - w0) / max(w1 - w0, 1) * (K - 1), 0, K - 1))
        if f < w0:
            u = smoothstep(f / max(w0, 1))
            tip, ax, tx, ty = hover + (start_tip - hover) * u, start_ax, sx[0], sy[0]
        elif f > w1:
            u = smoothstep((f - w1) / 12.0)
            tip, ax, tx, ty = end_tip + (off - end_tip) * u, end_ax, sx[-1], sy[-1]
        else:
            tip, ax, tx, ty = tip_at(s)
        res = model.solve_ik(base, q, tip, ax, iterations=300 if f == 0 else 80)
        q = refine(model, base, res.q, tip, ax)
        T = model.tool_pose(base, q)
        err = float(np.linalg.norm(T[:3, 3] - tip))
        errs.append((arc, err, float(np.arccos(np.clip(T[:3, 0] @ ax, -1, 1)))))
        qs.append([q[j] for j in ARM_JOINTS])
        k = int(np.floor(s + 1e-9)) if f >= w0 else -1
        st = steps[min(max(int(round(s)), 0), K - 1)]
        rec = {"q": {n: float(v) for n, v in q.items()}, "tip": T[:3, 3].tolist(), "axis": T[:3, 0].tolist(),
               "arc": bool(arc), "progress_mm": float(tx - sx[0]), "torch_seam_xy": [float(tx), float(ty)],
               "laser_world": None, "laser_dir": None, "center_est_world": None, "target_y_mm": None,
               "true_center_y_mm": float(seam.center_at(tx)), "bead_width_mm": float(sw[max(k, 0)]) if arc else 0.0}
        if arc:
            th = THETA_S + st["ys"].mean() / (1000 * RADIUS)
            rec["laser_world"] = clean(world(st["x_look"], st["ys"], st["z"], x0))
            rec["laser_dir"] = (-np.sin(th) * REF + np.cos(th) * UP).tolist()
            c = st["center_est"]
            if c is not None:
                rec["center_est_world"] = world(st["x_look"], c, float(seam.height(st["x_look"], [c])[0]), x0).tolist()
            rec["target_y_mm"] = st["target"]
        n = k + 1
        if n > 0:
            idx = np.unique(np.linspace(0, n - 1, min(n, 300)).round().astype(int))
            rec["track_world"] = clean(track_pts[idx])
            rec["bead_world"] = clean(bead_pts[idx])
            rec["bead_width_m"] = clean(sw[idx] / 1000)
            rec["bead_age_s"] = clean(np.maximum(f / fps - step_t[idx], 0.0))
        else:
            rec["track_world"], rec["bead_world"], rec["bead_width_m"], rec["bead_age_s"] = [], [], [], []
        out.append(rec)

    errs = np.array(errs)
    qs = np.array(qs)
    dq = np.abs(np.diff(qs, axis=0))
    ddq = np.abs(np.diff(qs, 2, axis=0))
    weld_err = errs[errs[:, 0] == 1]
    tips = np.array([o["tip"] for o in out])
    welded = [o for o in out if o["arc"]]
    dev = [np.linalg.norm(np.array(o["tip"]) - world(*o["torch_seam_xy"], 0.0, x0)) * 1000 - standoff_mm
           for o in welded]
    seam_dev = [abs(o["torch_seam_xy"][1] - seam.center_at(o["torch_seam_xy"][0])) for o in welded]
    metrics = {"tracker_rmse_mm": rmse, "tracker_max_mm": tmax, "open_loop_rmse_mm": ol_rmse,
               "open_loop_max_mm": ol_max, "tracker_rmse_default_wander_mm": d_rmse,
               "open_loop_rmse_default_wander_mm": d_ol, "wander_mm": wander_mm, "policy_steps": K,
               "steps_per_frame": K / max(w1 - w0, 1), "weld_length_mm": float(sx[-1] - sx[0]),
               "ik_max_err_mm_weld": float(weld_err[:, 1].max() * 1000),
               "ik_max_err_mm_all": float(errs[:, 1].max() * 1000),
               "ik_max_axis_err_rad": float(errs[:, 2].max()),
               "max_dq_rad_per_frame": float(dq.max()), "max_ddq_rad_per_frame2": float(ddq.max()),
               "max_tip_step_mm": float(np.linalg.norm(np.diff(tips, axis=0), axis=1).max() * 1000),
               "max_standoff_dev_mm": float(np.max(np.abs(dev))),
               "max_torch_seam_dev_mm": float(np.max(seam_dev))}
    seam_x = sub(seam.xs, 160)
    return jsonable({
        "fps": fps, "n_frames": frames, "weld": list(weld), "seed": seed,
        "base": base.tolist(),
        "pipe": {"origin": list(ORIGIN), "axis": AXIS.tolist(), "ref": REF.tolist(), "up": UP.tolist(),
                 "radius": RADIUS, "theta_s": THETA_S, "x0": x0, "groove_half_m": seam.cfg.groove_half_mm / 1000,
                 "groove_depth_m": seam.cfg.groove_depth_mm / 1000, "seam_length_m": seam.cfg.length_mm / 1000},
        "seam_world": clean(world(seam_x, np.interp(seam_x, seam.xs, seam.center), 0.0, x0)),
        "crack_world": clean(world(seam.crack[:, 0], seam.crack[:, 1], 0.0, x0)),
        "metrics": metrics,
        "frames": out,
    })


if __name__ == "__main__":
    d = plan()
    with open(sys.argv[1] if len(sys.argv) > 1 else "weld_clip.json", "w") as fh:
        json.dump(d, fh, allow_nan=False)
    print(json.dumps(d["metrics"], indent=1))

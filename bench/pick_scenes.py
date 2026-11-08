from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

W, H = 640, 480
K = np.array([[520.0, 0, W / 2], [0, 520.0, H / 2], [0, 0, 1]])
LIGHT = np.array([-0.35, 0.45, 1.0]) / np.linalg.norm([-0.35, 0.45, 1.0])
COLOURS = {
    "can": [("red can", (0.78, 0.08, 0.1)), ("green can", (0.1, 0.55, 0.2)), ("blue can", (0.1, 0.25, 0.75)),
            ("silver can", (0.72, 0.73, 0.75)), ("white can", (0.9, 0.9, 0.88))],
    "bottle": [("clear bottle", (0.82, 0.86, 0.9)), ("green glass bottle", (0.12, 0.42, 0.2)),
               ("brown bottle", (0.42, 0.24, 0.1)), ("white bottle", (0.93, 0.93, 0.92))],
    "box": [("cardboard box", (0.6, 0.46, 0.3)), ("grey metal box", (0.5, 0.52, 0.55)),
            ("white box", (0.88, 0.88, 0.86)), ("red box", (0.75, 0.12, 0.1))],
}
DISTRACTORS = ("red ball", "yellow tool case", "pipe offcut", "blue rag", "orange hard hat")


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def look_at(eye, target):
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, np.cross(f, r), f])
    T[:3, 3] = eye
    return T


@dataclass
class Prim:
    kind: str
    c: np.ndarray
    R: np.ndarray
    size: tuple
    rgb: tuple
    inst: int


def _hit(p: Prim, O, D):
    o, d = p.R.T @ (O - p.c), D @ p.R
    n = np.zeros_like(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        if p.kind == "box":
            h = np.asarray(p.size)
            t1, t2 = (-h - o) / d, (h - o) / d
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            tmin, tmax = lo.max(1), hi.min(1)
            t = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)
            ax = lo.argmax(1)
            n[np.arange(len(d)), ax] = -np.sign(d[np.arange(len(d)), ax])
        elif p.kind == "sphere":
            r = p.size[0]
            b, cc = d @ o, o @ o - r * r
            a = (d * d).sum(1)
            disc = b * b - a * cc
            t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
            t = np.where(t > 0, t, np.inf)
            n = (o + t[:, None] * d) / r
        else:
            r, hh = p.size
            a = d[:, 0] ** 2 + d[:, 1] ** 2
            b = 2 * (o[0] * d[:, 0] + o[1] * d[:, 1])
            cc = o[0] ** 2 + o[1] ** 2 - r * r
            disc = b * b - 4 * a * cc
            ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
            zs = o[2] + ts * d[:, 2]
            ts = np.where((disc >= 0) & (ts > 0) & (np.abs(zs) <= hh), ts, np.inf)
            t, n = ts, np.zeros_like(d)
            ps = o + np.where(np.isfinite(ts), ts, 0)[:, None] * d
            n[:, :2] = ps[:, :2] / r
            for s in (-1.0, 1.0):
                tc = (s * hh - o[2]) / d[:, 2]
                pc = o + np.where(np.isfinite(tc), tc, 0)[:, None] * d
                ok = (tc > 0) & (pc[:, 0] ** 2 + pc[:, 1] ** 2 <= r * r) & (tc < t)
                t = np.where(ok, tc, t)
                n[ok] = [0, 0, s]
    t = np.where(np.isfinite(t), t, np.inf)
    return t, n @ p.R.T


def _cyl(c, R, r, length, rgb, inst):
    return Prim("cyl", np.asarray(c, float), R, (r, length / 2), rgb, inst)


def make_object(name: str, xy, yaw: float, lying: bool, rgb, inst: int, rng):
    x, y = xy
    if name == "can":
        r, L = 0.033, 0.12
        if lying:
            R = rot_z(yaw) @ rot_y(math.pi / 2)
            return [_cyl([x, y, r], R, r, L, rgb, inst)], [x, y, 2 * r]
        return [_cyl([x, y, L / 2], np.eye(3), r, L, rgb, inst),
                _cyl([x, y, L + 0.003], np.eye(3), r * 0.85, 0.006, (0.75, 0.76, 0.78), inst)], [x, y, L + 0.006]
    if name == "bottle":
        r, Lb, rn, Ln = 0.036, 0.15, 0.014, 0.05
        cap = (0.1, 0.3, 0.8) if rng.random() < 0.5 else (0.9, 0.9, 0.9)
        if lying:
            R = rot_z(yaw) @ rot_y(math.pi / 2)
            ax = R[:, 2]
            base = np.array([x, y, r])
            return [_cyl(base, R, r, Lb, rgb, inst),
                    _cyl(base + ax * (Lb / 2 + Ln / 2), R, rn, Ln, rgb, inst),
                    _cyl(base + ax * (Lb / 2 + Ln + 0.01), R, rn * 1.2, 0.02, cap, inst)], [x, y, 2 * r]
        return [_cyl([x, y, Lb / 2], np.eye(3), r, Lb, rgb, inst),
                _cyl([x, y, Lb + Ln / 2], np.eye(3), rn, Ln, rgb, inst),
                _cyl([x, y, Lb + Ln + 0.01], np.eye(3), rn * 1.2, 0.02, cap, inst)], [x, y, Lb + Ln + 0.02]
    if name == "box":
        hx, hy, hz = rng.uniform(0.05, 0.08), rng.uniform(0.035, 0.06), rng.uniform(0.025, 0.06)
        return [Prim("box", np.array([x, y, hz]), rot_z(yaw), (hx, hy, hz), rgb, inst)], [x, y, 2 * hz]
    if name == "red ball":
        return [Prim("sphere", np.array([x, y, 0.04]), np.eye(3), (0.04,), (0.8, 0.1, 0.08), inst)], [x, y, 0.08]
    if name == "yellow tool case":
        return [Prim("box", np.array([x, y, 0.03]), rot_z(yaw), (0.08, 0.05, 0.03), (0.85, 0.7, 0.05), inst)], \
            [x, y, 0.06]
    if name == "pipe offcut":
        return [_cyl([x, y, 0.025], rot_z(yaw) @ rot_y(math.pi / 2), 0.025, 0.24, (0.35, 0.36, 0.38), inst)], \
            [x, y, 0.05]
    if name == "blue rag":
        return [Prim("box", np.array([x, y, 0.006]), rot_z(yaw), (0.07, 0.06, 0.006), (0.15, 0.3, 0.7), inst)], \
            [x, y, 0.012]
    if name == "orange hard hat":
        return [Prim("sphere", np.array([x, y, 0.0]), np.eye(3), (0.075,), (0.95, 0.45, 0.05), inst)], [x, y, 0.075]
    raise KeyError(name)


def render(prims: list, T, noise: float = 0.012, seed: int = 0):
    vs, us = np.mgrid[0:H, 0:W]
    dc = np.stack([(us.ravel() + 0.5 - K[0, 2]) / K[0, 0], (vs.ravel() + 0.5 - K[1, 2]) / K[1, 1],
                   np.ones(W * H)], -1)
    O, D = T[:3, 3], dc @ T[:3, :3].T
    with np.errstate(divide="ignore"):
        tf = np.where(D[:, 2] < 0, -O[2] / D[:, 2], np.inf)
    t, inst = tf.copy(), np.zeros(W * H, int)
    nrm = np.tile([0, 0, 1.0], (W * H, 1))
    col = np.zeros((W * H, 3))
    for p in prims:
        tp, n = _hit(p, O, D)
        m = tp < t
        t[m], inst[m], nrm[m], col[m] = tp[m], p.inst, n[m], p.rgb
    P = O + np.where(np.isfinite(t), t, 0)[:, None] * D
    fl = inst == 0
    gx, gy = np.floor(P[:, 0] / 0.03), np.floor(P[:, 1] / 0.03)
    bar = (np.abs(P[:, 0] / 0.03 - np.round(P[:, 0] / 0.03)) < 0.12) | (np.abs(P[:, 1] / 0.09 - np.round(P[:, 1] / 0.09)) < 0.05)
    col[fl] = (0.36 + 0.05 * ((gx + gy) % 2)[fl, None] - 0.12 * bar[fl, None]) * np.array([1.0, 1.0, 0.97])
    shade = 0.32 + 0.68 * np.clip(nrm @ LIGHT, 0, 1)
    so = P + nrm * 1e-4
    shadow = np.zeros(W * H, bool)
    idx = np.nonzero(np.isfinite(t))[0]
    for p in prims:
        tl, _ = _hit_many(p, so[idx], LIGHT)
        shadow[idx] |= np.isfinite(tl) & (inst[idx] != p.inst)
    shade = np.where(shadow, 0.32, shade)
    spec = np.clip((nrm @ (LIGHT + np.array([0, 0, 1.0])) / np.linalg.norm(LIGHT + [0, 0, 1])), 0, 1) ** 40
    rgb = col * shade[:, None] + 0.25 * spec[:, None] * (inst != 0)[:, None]
    rgb += np.random.default_rng(seed).normal(0, noise, rgb.shape)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(H, W, 3)
    depth = t.reshape(H, W).copy()
    depth[~np.isfinite(depth)] = np.nan
    return rgb, depth, inst.reshape(H, W)


def _hit_many(p: Prim, O, d):
    oo = (O - p.c) @ p.R
    dd = np.asarray(d, float) @ p.R
    t = np.full(len(O), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        if p.kind == "box":
            h = np.asarray(p.size)
            t1, t2 = (-h - oo) / dd, (h - oo) / dd
            tmin, tmax = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
            t = np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0), np.inf)
        elif p.kind == "sphere":
            r = p.size[0]
            b, cc = oo @ dd, (oo * oo).sum(1) - r * r
            disc = b * b - cc
            t = np.where((disc >= 0) & (-b + np.sqrt(np.maximum(disc, 0)) > 0), 1.0, np.inf)
        else:
            r, hh = p.size
            a = dd[0] ** 2 + dd[1] ** 2
            for s in (-1, 1):
                if a > 1e-12:
                    b = 2 * (oo[:, 0] * dd[0] + oo[:, 1] * dd[1])
                    cc = oo[:, 0] ** 2 + oo[:, 1] ** 2 - r * r
                    disc = b * b - 4 * a * cc
                    ts = (-b + s * np.sqrt(np.maximum(disc, 0))) / (2 * a)
                    zs = oo[:, 2] + ts * dd[2]
                    t = np.where((disc >= 0) & (ts > 0) & (np.abs(zs) <= hh), np.minimum(t, ts), t)
                if abs(dd[2]) > 1e-12:
                    tc = (s * hh - oo[:, 2]) / dd[2]
                    pc = oo + tc[:, None] * dd
                    t = np.where((tc > 0) & (pc[:, 0] ** 2 + pc[:, 1] ** 2 <= r * r), np.minimum(t, tc), t)
    return t, None


@dataclass
class Scene:
    seed: int
    kind: str
    label: str
    rgb: np.ndarray
    depth: np.ndarray
    inst: np.ndarray
    K: np.ndarray
    T: np.ndarray
    target: int
    gt_xyz: np.ndarray
    gt_px: tuple
    objects: list = field(default_factory=list)


def make_scene(seed: int, kind: str, n_distract: int = 2) -> Scene:
    rng = np.random.default_rng(seed)
    for _ in range(50):
        eye = [-0.55 + rng.uniform(-0.05, 0.05), rng.uniform(-0.08, 0.08), 0.62 + rng.uniform(-0.05, 0.08)]
        T = look_at(eye, [0.08, 0.0, 0.0])
        label, rgb = COLOURS[kind][rng.integers(len(COLOURS[kind]))]
        names = [kind] + list(rng.choice(DISTRACTORS, n_distract, replace=False))
        xys = []
        while len(xys) < len(names):
            xy = (rng.uniform(-0.12, 0.28), rng.uniform(-0.2, 0.2))
            if all(math.hypot(xy[0] - a, xy[1] - b) > 0.17 for a, b in xys):
                xys.append(xy)
        prims, objs, gt = [], [], None
        for i, (nm, xy) in enumerate(zip(names, xys), 1):
            lying = nm in ("can", "bottle") and rng.random() < 0.4
            ps, top = make_object(nm, xy, float(rng.uniform(-math.pi, math.pi)), lying,
                                  rgb if i == 1 else None, i, rng)
            prims += ps
            objs.append({"name": label if i == 1 else str(nm), "inst": i, "lying": bool(lying), "top": top})
            gt = top if i == 1 else gt
        im, depth, inst = render(prims, T, seed=seed)
        from policy.pickgrasp.vlm_grasp import project
        u, v = project(gt, K, T)
        ok = 8 < u < W - 8 and 8 < v < H - 8 and inst[int(round(v)), int(round(u))] == 1
        vis = all((inst == o["inst"]).sum() > 300 for o in objs)
        if ok and vis:
            return Scene(seed, kind, label, im, depth, inst, K.copy(), T, 1, np.asarray(gt, float), (u, v), objs)
        seed += 10007
        rng = np.random.default_rng(seed)
    raise RuntimeError("no valid scene")

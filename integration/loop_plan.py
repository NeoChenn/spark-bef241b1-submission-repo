from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from integration.kinematics import ARM_JOINTS, LIFT_JOINT

BRUSH_OFFSET = (-0.025, 0.0, 0.0)
GROUPS = ((LIFT_JOINT,), (ARM_JOINTS[0],), (ARM_JOINTS[1],), (ARM_JOINTS[2],), tuple(ARM_JOINTS[3:]))


@dataclass(frozen=True)
class PipeCyl:
    origin: tuple
    axis: tuple
    radius: float
    half_length: float


def brush_fk(model, base: np.ndarray, offset=BRUSH_OFFSET) -> Callable[[dict], np.ndarray]:
    off = np.array([*offset, 1.0])
    full = model.default_q()

    def fk(q: dict) -> np.ndarray:
        return (model.tool_pose(base, {**full, **q}) @ off)[:3]

    return fk


def _split(p, pipe: PipeCyl):
    o, a = np.asarray(pipe.origin, float), np.asarray(pipe.axis, float)
    rel = np.asarray(p, float) - o
    z = float(rel @ a)
    return z, rel - z * a


def clearance(p, pipe: PipeCyl, brush_r: float) -> float:
    z, radial = _split(p, pipe)
    rho = float(np.linalg.norm(radial))
    out = max(0.0, abs(z) - pipe.half_length)
    if out > 0:
        return float(np.hypot(out, max(0.0, rho - pipe.radius))) - brush_r
    return rho - pipe.radius - brush_r


def retract_target(p, pipe: PipeCyl, brush_r: float, goal: float) -> np.ndarray:
    p = np.asarray(p, float)
    if clearance(p, pipe, brush_r) >= goal:
        return p
    z, radial = _split(p, pipe)
    n = radial / max(float(np.linalg.norm(radial)), 1e-9)
    return np.asarray(pipe.origin, float) + z * np.asarray(pipe.axis, float) + (pipe.radius + brush_r + goal) * n


def _lerp(q0: dict, q1: dict, s: float) -> dict:
    return {k: q0[k] + (q1[k] - q0[k]) * s for k in q1}


def path_clearance(fk, waypoints: list[dict], pipe: PipeCyl, brush_r: float, n: int = 40) -> float:
    best = np.inf
    for q0, q1 in zip(waypoints, waypoints[1:]):
        for k in range(n + 1):
            best = min(best, clearance(fk(_lerp(q0, q1, k / n)), pipe, brush_r))
    return float(best)


def _stage(q: dict, q_to: dict, names) -> dict:
    return {**q, **{k: q_to[k] for k in names if k in q_to}}


def plan_staged(fk, q_from: dict, q_to: dict, pipe: PipeCyl, brush_r: float, groups=GROUPS) -> tuple[list[dict], float]:
    q_from = {k: float(q_from[k]) for k in q_to}
    q_to = {k: float(v) for k, v in q_to.items()}
    best, best_c = [q_to], path_clearance(fk, [q_from, q_to], pipe, brush_r)
    for order in itertools.permutations(groups):
        stages, q = [], dict(q_from)
        for names in order:
            q = _stage(q, q_to, names)
            stages.append(dict(q))
        c = path_clearance(fk, [q_from] + stages, pipe, brush_r)
        if c > best_c + 1e-6:
            best, best_c = stages, c
    return best, best_c


def stage_steps(q0: dict, q1: dict, dt: float, joint_rate: float = 2.0, lift_rate: float = 0.4,
                min_steps: int = 6) -> int:
    t = 0.0
    for k, v in q1.items():
        rate = lift_rate if k == LIFT_JOINT else joint_rate
        t = max(t, abs(v - q0.get(k, v)) / rate)
    return max(min_steps, int(round(t / dt)))


def plan_reposition(err: Callable, targets, dxs=(0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3),
                    lifts=(0.42, 0.45, 0.38), lift0: float = 0.42, tol: float = 0.03) -> dict:
    def score(dx, lift):
        return sum(err(dx, lift, t) < tol for t in targets)

    before = score(0.0, lift0)
    if before == len(targets):
        return {"dx": 0.0, "lift": float(lift0), "reached": before, "before": before, "targets": len(targets)}
    cands = sorted(((score(dx, l), -(abs(dx) + abs(l - lift0)), dx, l) for dx in dxs for l in lifts), reverse=True)
    n, _, dx, lift = cands[0]
    if n <= before:
        dx, lift, n = 0.0, lift0, before
    return {"dx": float(dx), "lift": float(lift), "reached": int(n), "before": int(before), "targets": len(targets)}


def ik_error(model, base: np.ndarray, q: dict, target, joints, iterations: int = 60, damping: float = 0.05) -> float:
    fk = brush_fk(model, base)
    q = dict(q)
    target = np.asarray(target, float)
    for _ in range(iterations):
        p = fk(q)
        e = target - p
        n = float(np.linalg.norm(e))
        if n < 2e-3:
            break
        J = np.zeros((3, len(joints)))
        for k, name in enumerate(joints):
            d = dict(q)
            d[name] += 1e-5
            J[:, k] = (fk(d) - p) / 1e-5
        e = e * min(1.0, 0.05 / n)
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), e)
        for k, name in enumerate(joints):
            j = model.joints[name]
            q[name] = float(np.clip(q[name] + dq[k], j.lower, j.upper))
    return float(np.linalg.norm(target - fk(q)))


def cyl_lerp(p0, p1, pipe: PipeCyl, s: float) -> np.ndarray:
    o, a = np.asarray(pipe.origin, float), np.asarray(pipe.axis, float)
    z0, r0 = _split(p0, pipe)
    z1, r1 = _split(p1, pipe)
    rho0, rho1 = float(np.linalg.norm(r0)), float(np.linalg.norm(r1))
    e = r0 / max(rho0, 1e-9)
    b = np.cross(a, e)
    th1 = float(np.arctan2(r1 @ b, r1 @ e))
    th = th1 * s
    rho = rho0 + (rho1 - rho0) * s
    return o + (z0 + (z1 - z0) * s) * a + rho * (np.cos(th) * e + np.sin(th) * b)

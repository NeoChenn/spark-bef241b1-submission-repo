"""Deterministic lidar detector for objects lying in the patrol path (numpy only, runs anywhere).

A return counts as "in the path" when it projects onto the path centre-line within `tube` metres
laterally and between `ahead_min` and `ahead_max` metres ahead of the robot along the path. The walls
(pipes) sit outside the tube by construction, so whatever is left is an object on the walkway.
Returns are clustered in beam order; each cluster becomes one detection with a size estimate, and
it is pickable when that size fits the gripper and the container. A detection is confirmed after
`confirm` consecutive frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectCfg:
    tube: float = 0.36
    ahead_min: float = 0.2
    ahead_max: float = 3.0
    lidar_range: float = 4.0
    cluster_gap: float = 0.12
    min_points: int = 1
    confirm: int = 1
    match_dist: float = 0.25
    pick_max_width: float = 0.11
    width_pad_beams: float = 1.0


@dataclass
class Detection:
    xy: np.ndarray
    width: float
    range: float
    bearing: float
    ahead: float
    lateral: float
    pickable: bool
    hits: int = 1


class PathRef:
    """Polyline path resampled at `ds`; projects points to (index, lateral, s)."""

    def __init__(self, path_xy, ds: float = 0.1):
        p = np.asarray(path_xy, dtype=np.float64)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(s[-1] / ds) + 1, 2)
        si = np.linspace(0.0, s[-1], n)
        self.P = np.stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])], 1)
        d = np.gradient(self.P, axis=0)
        self.h = np.arctan2(d[:, 1], d[:, 0])
        self.nrm = np.stack([-np.sin(self.h), np.cos(self.h)], 1)
        self.s = si
        self.ds = si[1] - si[0]
        self.length = si[-1]

    def reversed(self) -> "PathRef":
        return PathRef(self.P[::-1].copy(), self.ds)

    def project(self, pts, around: int | None = None, window: int = 60):
        pts = np.atleast_2d(pts)
        lo, hi = (0, len(self.P)) if around is None else (max(around - window, 0), min(around + window + 1, len(self.P)))
        P = self.P[lo:hi]
        d2 = ((pts[:, None] - P[None]) ** 2).sum(-1)
        i = d2.argmin(1) + lo
        lat = ((pts - self.P[i]) * self.nrm[i]).sum(-1)
        return i, lat, self.s[i]

    def along_residual(self, pts, i):
        """Tangential offset from the projected path point; large when a point lies past the window or path end."""
        tan = np.stack([np.cos(self.h[i]), np.sin(self.h[i])], -1)
        return ((np.atleast_2d(pts) - self.P[i]) * tan).sum(-1)


def hits_world(ranges, angles, pose, lidar_range):
    r = np.asarray(ranges, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    ok = np.isfinite(r) & (r < lidar_range - 1e-3) & (r > 0.05)
    x, y, yaw = pose
    th = yaw + a[ok]
    pts = np.stack([x + r[ok] * np.cos(th), y + r[ok] * np.sin(th)], 1)
    return pts, r[ok], a[ok], np.flatnonzero(ok)


def detect_frame(ranges, angles, pose, path: PathRef, robot_idx: int | None = None, cfg: DetectCfg = DetectCfg()):
    pts, r, a, beam = hits_world(ranges, angles, pose, cfg.lidar_range)
    if len(pts) == 0:
        return []
    ri, _, s_robot = path.project(np.array(pose[:2])[None], robot_idx)
    hi, lat, s = path.project(pts, int(ri[0]), window=int(cfg.ahead_max / path.ds) + 10)
    ahead = s - s_robot[0]
    on_path = np.abs(path.along_residual(pts, hi)) < 1.5 * path.ds
    keep = on_path & (np.abs(lat) < cfg.tube) & (ahead > cfg.ahead_min) & (ahead < cfg.ahead_max)
    if not keep.any():
        return []
    pts, r, a, lat, ahead = pts[keep], r[keep], a[keep], lat[keep], ahead[keep]
    order = np.argsort(a)
    pts, r, a, lat, ahead = pts[order], r[order], a[order], lat[order], ahead[order]
    dbeam = (2 * math.pi / len(ranges)) if len(ranges) > 1 else 0.0
    groups, cur = [], [0]
    for k in range(1, len(pts)):
        if np.linalg.norm(pts[k] - pts[k - 1]) < cfg.cluster_gap:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    if len(groups) > 1 and np.linalg.norm(pts[groups[0][0]] - pts[groups[-1][-1]]) < cfg.cluster_gap:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    out = []
    for g in groups:
        if len(g) < cfg.min_points:
            continue
        gp = pts[g]
        chord = float(np.linalg.norm(gp[0] - gp[-1])) if len(g) > 1 else 0.0
        rng = float(r[g].min())
        width = chord + cfg.width_pad_beams * rng * dbeam
        x, y, yaw = pose
        c = gp.mean(0)
        ray = (c - np.array([x, y])) / max(np.linalg.norm(c - np.array([x, y])), 1e-6)
        c = c + ray * width / 2
        bearing = float(math.atan2(c[1] - y, c[0] - x) - yaw)
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        out.append(Detection(xy=c, width=width, range=float(np.linalg.norm(c - np.array([x, y]))),
                             bearing=bearing, ahead=float(ahead[g].mean()), lateral=float(lat[g].mean()),
                             pickable=width <= cfg.pick_max_width))
    return out


@dataclass
class Tracker:
    cfg: DetectCfg = field(default_factory=DetectCfg)
    tracks: list = field(default_factory=list)

    def update(self, dets: list[Detection]) -> list[Detection]:
        new = []
        for d in dets:
            best = min(self.tracks, key=lambda t: np.linalg.norm(t.xy - d.xy), default=None)
            if best is not None and np.linalg.norm(best.xy - d.xy) < self.cfg.match_dist:
                d.hits = best.hits + 1
                d.width = max(d.width, 0.7 * best.width + 0.3 * d.width)
                d.pickable = d.width <= self.cfg.pick_max_width
            new.append(d)
        self.tracks = new
        return [t for t in self.tracks if t.hits >= self.cfg.confirm]

    def reset(self):
        self.tracks = []

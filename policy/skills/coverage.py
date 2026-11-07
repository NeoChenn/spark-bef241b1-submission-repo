"""Boustrophedon coverage over a binary mask on the pipe (theta, z) grid.

Lanes run along the pipe axis (z) at fixed theta; consecutive lanes alternate
direction and step in theta by ``lane_width`` rows. Theta wraps around the pipe.
"""
from __future__ import annotations

import numpy as np

from policy.interfaces import PipeGeometry, Waypoint


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Square dilation by k cells; wraps in theta (rows), clips in z (cols)."""
    m = np.asarray(mask, bool)
    if k <= 0:
        return m.copy()
    out = m.copy()
    for di in range(-k, k + 1):
        rolled = np.roll(m, di, axis=0)
        for dj in range(-k, k + 1):
            if dj == 0:
                out |= rolled
            elif dj > 0:
                out[:, dj:] |= rolled[:, :-dj]
            else:
                out[:, :dj] |= rolled[:, -dj:]
    return out


def boustrophedon_cells(mask: np.ndarray, lane_width: int = 1) -> list[tuple[int, int]]:
    """Ordered (row, col) cells visiting every lane that intersects ``mask``.

    Each lane is ``lane_width`` rows tall; the tool is assumed to cover the whole
    lane height, so a waypoint at lane-centre row covers all rows of the lane.
    Starts at the first occupied theta row after the largest empty theta gap so a
    patch straddling theta = 0 is swept contiguously.
    """
    m = np.asarray(mask, bool)
    nT = m.shape[0]
    rows_occ = m.any(axis=1)
    if not rows_occ.any():
        return []
    # start after the largest run of empty rows (handles wrap-around)
    start = 0
    if not rows_occ.all():
        best, run, best_end = -1, 0, 0
        for k in range(2 * nT):
            if not rows_occ[k % nT]:
                run += 1
                if run > best:
                    best, best_end = run, k
            else:
                run = 0
        start = (best_end + 1) % nT
    order = [(start + k) % nT for k in range(nT)]
    cells, forward = [], True
    for li in range(0, nT, lane_width):
        lane_rows = order[li:li + lane_width]
        cols = np.nonzero(m[lane_rows].any(axis=0))[0]
        if len(cols) == 0:
            continue
        centre = lane_rows[len(lane_rows) // 2]
        seq = cols if forward else cols[::-1]
        cells.extend((int(centre), int(c)) for c in seq)
        forward = not forward
    return cells


def lane_footprint(cell: tuple[int, int], lane_width: int, n_theta: int, n_z: int,
                   z_half: int = 0) -> list[tuple[int, int]]:
    i, j = cell
    h = lane_width // 2
    rows = [(i + d) % n_theta for d in range(-h, lane_width - h)]
    cols = range(max(0, j - z_half), min(n_z, j + z_half + 1))
    return [(r, c) for r in rows for c in cols]


def covered_by(cells, lane_width: int, shape, z_half: int = 0) -> np.ndarray:
    cov = np.zeros(shape, bool)
    for c in cells:
        for r, cc in lane_footprint(c, lane_width, shape[0], shape[1], z_half):
            cov[r, cc] = True
    return cov


def coverage_path(mask: np.ndarray, pipe: PipeGeometry, margin: int = 1, lane_width: int = 1,
                  standoff: float = 0.0, flow_rate: float = 0.0):
    """Mask -> (waypoints, target mask after margin dilation)."""
    target = dilate(mask, margin)
    cells = boustrophedon_cells(target, lane_width)
    wps = []
    for (i, j) in cells:
        th, z = pipe.cell_to_theta_z(i, j)
        p, n = pipe.surface_point(th, z)
        wps.append(Waypoint(position=p + standoff * n, approach=-n, theta=float(th), z=float(z),
                            cell=(i, j), standoff=standoff, flow_rate=flow_rate))
    return wps, target

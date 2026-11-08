from __future__ import annotations

import numpy as np

from policy.interfaces import PipeGeometry, Waypoint


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    m = np.asarray(mask, bool)
    if k <= 0:
        return m.copy()
    out = m.copy()
    for dj in range(-k, k + 1):
        rolled = np.roll(m, dj, axis=1)
        for di in range(-k, k + 1):
            if di == 0:
                out |= rolled
            elif di > 0:
                out[di:] |= rolled[:-di]
            else:
                out[:di] |= rolled[-di:]
    return out


def boustrophedon_cells(mask: np.ndarray, lane_width: int = 1) -> list[tuple[int, int]]:
    m = np.asarray(mask, bool)
    nT = m.shape[1]
    cols_occ = m.any(axis=0)
    if not cols_occ.any():
        return []
    start = 0
    if not cols_occ.all():
        best, run, best_end = -1, 0, 0
        for k in range(2 * nT):
            if not cols_occ[k % nT]:
                run += 1
                if run > best:
                    best, best_end = run, k
            else:
                run = 0
        start = (best_end + 1) % nT
    order = [(start + k) % nT for k in range(nT)]
    cells, forward = [], True
    for li in range(0, nT, lane_width):
        lane = order[li:li + lane_width]
        rows = np.nonzero(m[:, lane].any(axis=1))[0]
        if len(rows) == 0:
            continue
        centre = lane[len(lane) // 2]
        seq = rows if forward else rows[::-1]
        cells.extend((int(r), int(centre)) for r in seq)
        forward = not forward
    return cells


def lane_footprint(cell: tuple[int, int], lane_width: int, n_z: int, n_theta: int,
                   z_half: int = 0) -> list[tuple[int, int]]:
    i, j = cell
    h = lane_width // 2
    cols = [(j + d) % n_theta for d in range(-h, lane_width - h)]
    rows = range(max(0, i - z_half), min(n_z, i + z_half + 1))
    return [(r, c) for r in rows for c in cols]


def covered_by(cells, lane_width: int, shape, z_half: int = 0) -> np.ndarray:
    cov = np.zeros(shape, bool)
    for c in cells:
        for r, cc in lane_footprint(c, lane_width, shape[0], shape[1], z_half):
            cov[r, cc] = True
    return cov


def coverage_path(mask: np.ndarray, pipe: PipeGeometry, margin: int = 1, lane_width: int = 1,
                  standoff: float = 0.0, flow_rate: float = 0.0):
    target = dilate(mask, margin)
    cells = boustrophedon_cells(target, lane_width)
    wps = []
    for (i, j) in cells:
        th, z = pipe.cell_to_theta_z(i, j)
        p, n = pipe.surface_point(th, z)
        wps.append(Waypoint(position=p + standoff * n, approach=-n, theta=float(th), z=float(z),
                            cell=(i, j), standoff=standoff, flow_rate=flow_rate))
    return wps, target

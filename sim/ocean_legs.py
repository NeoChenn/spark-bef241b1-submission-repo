from __future__ import annotations

import numpy as np


def waterline_points(points: np.ndarray, counts: np.ndarray, indices: np.ndarray, z: float) -> np.ndarray:
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.arange(indices.size) - starts
    nxt = starts + (pos + 1) % np.repeat(counts, counts)
    a, b = points[indices], points[indices[nxt]]
    da, db = a[:, 2] - z, b[:, 2] - z
    cross = (da * db < 0)
    t = (da[cross] / (da[cross] - db[cross]))[:, None]
    return (a[cross, :2] + t * (b[cross, :2] - a[cross, :2])).astype(np.float64)


def cluster_circles(xy: np.ndarray, link: float = 2.0, min_points: int = 6) -> list[tuple[float, float, float]]:
    if len(xy) == 0:
        return []
    xy = np.unique(np.round(xy, 3), axis=0)
    parent = np.arange(len(xy))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells: dict[tuple[int, int], list[int]] = {}
    keys = np.floor(xy / link).astype(int)
    for i, (kx, ky) in enumerate(keys):
        cells.setdefault((kx, ky), []).append(i)
    for (kx, ky), members in cells.items():
        near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cells.get((kx + dx, ky + dy), [])]
        for i in members:
            for j in near:
                if j > i and np.hypot(*(xy[i] - xy[j])) < link:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
    roots = np.array([find(i) for i in range(len(xy))])
    circles = []
    for r in np.unique(roots):
        pts = xy[roots == r]
        if len(pts) < min_points:
            continue
        c = pts.mean(0)
        circles.append((float(c[0]), float(c[1]), float(np.hypot(*(pts - c).T).max())))
    return circles

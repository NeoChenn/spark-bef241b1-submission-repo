from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

CACHE = Path(tempfile.gettempdir()) / "rig-plume"
SKIP = ("/World/Leak", "/World/Cinematic", "/World/Environment", "/World/Ocean", "/Environment", "/RigStream")


def mark_triangles(tris: np.ndarray, origin, shape, dx: float) -> np.ndarray:
    solid = np.zeros(shape, bool)
    if len(tris) == 0:
        return solid
    o = np.asarray(origin, np.float64)
    n = np.asarray(shape)
    edge = np.max(np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2), axis=1)
    steps = np.clip(np.ceil(edge / (0.45 * dx)).astype(int), 1, 400)
    for s in np.unique(steps):
        t = tris[steps == s]
        a, b = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing="ij")
        keep = a + b <= s
        u, v = a[keep] / s, b[keep] / s
        pts = t[:, None, 0] + u[None, :, None] * (t[:, None, 1] - t[:, None, 0]) + v[None, :, None] * (t[:, None, 2] - t[:, None, 0])
        idx = np.floor((pts.reshape(-1, 3) - o) / dx).astype(int)
        ok = np.all((idx >= 0) & (idx < n), axis=1)
        idx = idx[ok]
        solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return solid


def stage_triangles(stage, lo, hi) -> np.ndarray:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or path.startswith(SKIP):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty() or np.any(np.asarray(rng.GetMax()) < lo) or np.any(np.asarray(rng.GetMin()) > hi):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        m = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64)
        p = np.asarray(pts, np.float64) @ m[:3, :3] + m[3, :3]
        counts, idx = np.asarray(counts), np.asarray(idx)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tri = []
        for k in np.unique(counts):
            s = starts[counts == k]
            tri += [np.stack([idx[s], idx[s + j], idx[s + j + 1]], 1) for j in range(1, k - 1)]
        if not tri:
            continue
        t = p[np.concatenate(tri)]
        tl, th = t.min(axis=1), t.max(axis=1)
        keep = np.all(th >= lo, axis=1) & np.all(tl <= hi, axis=1)
        out.append(t[keep])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def voxelize(stage, origin, shape, dx: float) -> np.ndarray:
    lo = np.asarray(origin, np.float64)
    hi = lo + np.asarray(shape) * dx
    return mark_triangles(stage_triangles(stage, lo, hi), origin, shape, dx)


def obstacles_for(stage_or_path, origin, shape, dx: float) -> np.ndarray | None:
    key = hashlib.sha1(repr((str(stage_or_path) if isinstance(stage_or_path, (str, Path)) else "live",
                            np.round(origin, 3).tolist(), list(shape), dx)).encode()).hexdigest()[:16]
    path = CACHE / f"solid_{key}.npz"
    if path.exists():
        return np.unpackbits(np.load(path)["bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    stage = stage_or_path
    if isinstance(stage_or_path, (str, Path)):
        try:
            from pxr import Usd
        except ImportError:
            return None
        if not Path(stage_or_path).exists():
            return None
        stage = Usd.Stage.Open(str(stage_or_path))
    solid = voxelize(stage, origin, shape, dx)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bits=np.packbits(solid.ravel()))
    return solid

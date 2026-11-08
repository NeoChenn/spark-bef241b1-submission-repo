import numpy as np

from rig.views.geom import rays, scaled


def fan(counts, indices) -> np.ndarray:
    counts = np.asarray(counts, np.int64)
    idx = np.asarray(indices, np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = []
    for k in range(1, int(counts.max(initial=0)) - 1):
        m = counts > k + 1
        s = starts[m]
        out.append(np.stack([idx[s], idx[s + k], idx[s + k + 1]], 1))
    if not out:
        return np.zeros((0, 3), np.int64)
    tris = np.concatenate(out)
    order = np.argsort(np.concatenate([starts[counts > k + 1] * 64 + k for k in range(1, len(out) + 1)]), kind="stable")
    return tris[order]


def ray_grid(cam: dict, w: int, h: int):
    c = scaled(cam, w, h)
    o, d = rays(c)
    return o, d / np.linalg.norm(d, axis=-1, keepdims=True)


def depth_from_hits(t, d, cam: dict) -> np.ndarray:
    T = np.asarray(cam["T"], np.float64)
    fwd = T[:3, 2]
    t = np.asarray(t, np.float64).reshape(d.shape[:2])
    z = t * (d @ fwd)
    return np.where(t > 0, z, np.inf)


def cast_numpy(o, dirs, verts, tris, max_t: float) -> np.ndarray:
    v0, v1, v2 = (np.asarray(verts, np.float64)[np.asarray(tris)[:, k]] for k in range(3))
    e1, e2 = v1 - v0, v2 - v0
    out = np.full(len(dirs), -1.0)
    for i, d in enumerate(np.asarray(dirs, np.float64)):
        p = np.cross(d, e2)
        det = (e1 * p).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0
        u = (s * p).sum(1) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < max_t)
        if hit.any():
            out[i] = t[hit].min()
    return out


class WarpCaster:
    def __init__(self, verts, tris, device: str = "cuda:0"):
        import warp as wp

        self.wp, self.device = wp, device
        self.points = wp.array(np.asarray(verts, np.float32), dtype=wp.vec3, device=device)
        self.indices = wp.array(np.asarray(tris, np.int32).reshape(-1), dtype=int, device=device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices)
        self.kernel = _kernel(wp)

    def cast(self, o, dirs, max_t: float) -> np.ndarray:
        wp = self.wp
        d = wp.array(np.asarray(dirs, np.float32).reshape(-1, 3), dtype=wp.vec3, device=self.device)
        out = wp.zeros(len(d), dtype=float, device=self.device)
        wp.launch(self.kernel, dim=len(d), inputs=[self.mesh.id, wp.vec3(*map(float, o)), d, float(max_t), out], device=self.device)
        return out.numpy()


_KERNEL = {}


def _kernel(wp):
    if "k" not in _KERNEL:
        @wp.kernel
        def cast_rays(mesh: wp.uint64, o: wp.vec3, dirs: wp.array(dtype=wp.vec3), max_t: float, out: wp.array(dtype=float)):
            i = wp.tid()
            q = wp.mesh_query_ray(mesh, o, dirs[i], max_t)
            out[i] = wp.where(q.result, q.t, -1.0)

        _KERNEL["k"] = cast_rays
    return _KERNEL["k"]

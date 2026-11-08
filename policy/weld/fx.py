from __future__ import annotations

import math

import numpy as np

ARC_RGB = (0.75, 0.85, 1.0)
POOL_RGB = (1.0, 0.45, 0.12)
AGE_EDGES = (0.3, 0.8, 1.6, 2.6, 3.6)
HOT = (((1.0, 0.88, 0.6), 1.0), ((1.0, 0.6, 0.2), 0.55), ((1.0, 0.34, 0.07), 0.28),
       ((0.9, 0.14, 0.02), 0.11), ((0.55, 0.05, 0.01), 0.035), ((0.0, 0.0, 0.0), 0.0))


def _hash(i, seed):
    x = (np.asarray(i, np.int64) * 374761393 + int(seed) * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def _noise(t, rate, seed):
    x = t * rate
    i = math.floor(x)
    f = x - i
    f = f * f * (3 - 2 * f)
    return float(_hash(i, seed) * (1 - f) + _hash(i + 1, seed) * f)


def arc_flicker(t: float, seed: int = 0) -> float:
    fast = _noise(t, 47.0, seed) * 0.45 + _noise(t, 113.0, seed + 1) * 0.25
    slow = _noise(t, 6.0, seed + 2)
    v = 0.72 + fast + (slow - 0.5) * 0.2
    gate = _noise(t, 9.0, seed + 3)
    if gate > 0.86:
        v *= 1.0 - min((gate - 0.86) / 0.14, 1.0) * 0.35
    return float(np.clip(v, 0.6, 1.3))


def pipe_normals(points, origin, axis):
    p = np.asarray(points, np.float64)
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    d = p - np.asarray(origin, np.float64)
    r = d - np.outer(d @ a, a)
    return r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def bead_mesh(points, normals, widths, ages, crown=0.002, segs=9, ripple=0.18, pitch=0.0025,
              sink=0.0003, edges=AGE_EDGES, with_normals=False):
    p = np.asarray(points, np.float64)
    n = len(p)
    if n < 2:
        e = (np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32))
        return e + (np.zeros((0, 3)),) if with_normals else e
    nrm = _unit(np.asarray(normals, np.float64))
    w = np.broadcast_to(np.asarray(widths, np.float64), (n,))
    age = np.broadcast_to(np.asarray(ages, np.float64), (n,))
    tan = np.gradient(p, axis=0)
    tan = _unit(tan - (tan * nrm).sum(-1, keepdims=True) * nrm)
    lat = _unit(np.cross(nrm, tan))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=-1))])
    mod = 1.0 - ripple * np.abs(np.sin(np.pi * s / pitch)) ** 0.7
    taper = np.minimum(1.0, np.minimum(s, s[-1] - s) / max(0.6 * float(w.max()), 1e-9) + 0.35)
    th = np.linspace(0.0, np.pi, segs)
    half = (0.5 * w * taper)[:, None]
    hgt = (crown * mod * taper)[:, None]
    base = p - nrm * sink
    v = (base[:, None, :] + lat[:, None, :] * (half * np.cos(th))[..., None]
         + nrm[:, None, :] * (hgt * np.sin(th) + sink * np.sin(th) ** 0.25)[..., None])
    verts = v.reshape(-1, 3)
    i = np.arange(n - 1)[:, None] * segs
    j = np.arange(segs - 1)[None, :]
    a = i + j
    quads = np.stack([a, a + 1, a + 1 + segs, a + segs], -1).reshape(-1, 4)
    cap0 = np.arange(segs)[::-1]
    cap1 = (n - 1) * segs + np.arange(segs)
    indices = np.concatenate([quads.ravel(), cap0, cap1]).astype(np.int32)
    counts = np.concatenate([np.full(len(quads), 4), [segs, segs]]).astype(np.int32)
    qa = np.repeat(0.5 * (age[:-1] + age[1:]), segs - 1)
    fa = np.concatenate([qa, [age[0], age[-1]]])
    face_bin = np.digitize(fa, edges).astype(np.int32)
    out = (verts, counts, indices, face_bin)
    if not with_normals:
        return out
    cn = (lat[:, None, :] * (np.cos(th)[None, :] / np.maximum(half, 1e-9))[..., None]
          + nrm[:, None, :] * (np.sin(th)[None, :] / np.maximum(hgt, 1e-9))[..., None])
    return out + (_unit(cn).reshape(-1, 3),)


def _blur(a, r, passes=3):
    r = int(max(r, 1))
    for _ in range(passes):
        for ax in (0, 1):
            pad = [(0, 0)] * a.ndim
            pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
            hi = np.take(c, np.arange(2 * r + 1, c.shape[ax]), axis=ax)
            lo = np.take(c, np.arange(0, c.shape[ax] - 2 * r - 1), axis=ax)
            a = (hi - lo) / (2 * r + 1)
    return a


def _upsample(a, h, w):
    from PIL import Image

    a = np.ascontiguousarray(a, np.float32)
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a, "F").resize((w, h), Image.BILINEAR))
    return np.stack([_upsample(a[..., c], h, w) for c in range(a.shape[-1])], -1)


def _bloom(layer, radius, gain, f=4):
    h, w = layer.shape[:2]
    if h < 2 * f or w < 2 * f:
        return layer
    hh, ww = h // f, w // f
    a = layer[:hh * f, :ww * f].reshape(hh, f, ww * f, 3)
    a = sum(a[:, k] for k in range(f)).reshape(hh, ww, f, 3)
    small = sum(a[:, :, k] for k in range(f)) / (f * f)
    layer += gain * _upsample(_blur(small, max(radius / f, 1)), h, w)
    return layer


def _composite(img, layer, y0, x0):
    out = np.array(img, copy=True)
    h, w = layer.shape[:2]
    view = out[y0:y0 + h, x0:x0 + w]
    p = np.multiply(layer, 255.0, dtype=np.float32)
    p += view
    over = np.maximum(np.maximum(p[..., 0], p[..., 1]), p[..., 2])
    over -= 255.0
    np.maximum(over, 0, out=over)
    over *= 0.6
    p += over[..., None]
    np.clip(p, 0, 255, out=p)
    view[...] = p
    return out


def _depth_ok(depth, x, y, z, eps):
    if depth is None:
        return np.ones(np.shape(z), bool)
    d = depth[np.clip(y, 0, depth.shape[0] - 1), np.clip(x, 0, depth.shape[1] - 1)]
    d = np.where(np.isfinite(d) & (d > 0), d, np.inf)
    return z <= d + eps


def project(p, K, T):
    q = (np.asarray(p, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = q[..., 2]
    uv = np.stack([K[0, 0] * q[..., 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * q[..., 1] / np.maximum(z, 1e-6) + K[1, 2]], -1)
    return uv, z


def _temp_rgb(temp):
    stops = np.array([[0.55, 0.06, 0.01], [1.0, 0.28, 0.04], [1.0, 0.6, 0.18], [1.0, 0.92, 0.7]], np.float32)
    x = np.clip(temp, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


class Sparks:
    def __init__(self, seed=0, rate=28, speed=(1.0, 4.0), life=(0.15, 0.6), cone=0.75, back=0.6,
                 drag=1.2, restitution=0.35, kill_on_hit=0.3, cap=1500):
        self.rng = np.random.default_rng(seed)
        self.rate, self.speed, self.life, self.cone, self.back = rate, speed, life, cone, back
        self.drag, self.rest, self.kill, self.cap = drag, restitution, kill_on_hit, cap
        self.pos = np.zeros((0, 3))
        self.prev = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.lifetime = np.zeros(0)
        self.heat = np.zeros(0)

    def __len__(self):
        return len(self.pos)

    def _emit(self, k, tip, normal, travel):
        r = self.rng
        n = _unit(np.asarray(normal, np.float64))
        tr = np.asarray(travel, np.float64)
        tr = _unit(tr - (tr @ n) * n) if np.linalg.norm(tr) > 1e-9 else np.zeros(3)
        axis = _unit(n - self.back * tr)
        u = _unit(np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0]))
        v = np.cross(axis, u)
        ang = self.cone * np.sqrt(r.random(k))
        phi = r.random(k) * 2 * np.pi
        d = (np.cos(ang)[:, None] * axis + np.sin(ang)[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v))
        sp = r.uniform(*self.speed, k) * (0.6 + 0.4 * r.random(k))
        p0 = np.asarray(tip, np.float64) + n * 0.001 + r.normal(0, 0.0008, (k, 3))
        self.pos = np.concatenate([self.pos, p0])
        self.prev = np.concatenate([self.prev, p0])
        self.vel = np.concatenate([self.vel, d * sp[:, None]])
        self.age = np.concatenate([self.age, r.random(k) * 0.01])
        self.lifetime = np.concatenate([self.lifetime, r.uniform(*self.life, k)])
        self.heat = np.concatenate([self.heat, r.uniform(0.8, 1.15, k)])

    def step(self, dt, tip=None, normal=(0, 0, 1), travel_dir=(1, 0, 0), on=True, pipe=None, ground=0.0, rate=None):
        if on and tip is not None:
            k = int(self.rng.poisson(self.rate if rate is None else rate))
            if k:
                self._emit(k, tip, normal, travel_dir)
        if not len(self):
            return self
        self.prev = self.pos.copy()
        self.vel *= math.exp(-self.drag * dt)
        self.vel[:, 2] -= 9.81 * dt
        self.pos = self.pos + self.vel * dt
        self.age += dt
        hit = np.zeros(len(self), bool)
        if pipe is not None:
            o, a, rad = np.asarray(pipe[0], np.float64), _unit(np.asarray(pipe[1], np.float64)), float(pipe[2])
            d = self.pos - o
            along = d @ a
            radial = d - along[:, None] * a
            dist = np.linalg.norm(radial, axis=-1)
            inside = (dist < rad) & (self.age > 0.02)
            if len(pipe) > 3:
                inside &= np.abs(along) < float(pipe[3]) * 0.5
            if inside.any():
                nn = radial[inside] / np.maximum(dist[inside], 1e-9)[:, None]
                self.pos[inside] = o + along[inside, None] * a + nn * rad * 1.001
                vn = (self.vel[inside] * nn).sum(-1, keepdims=True)
                vt = self.vel[inside] - np.minimum(vn, 0) * nn
                self.vel[inside] = vt * 0.6 - np.minimum(vn, 0) * nn * self.rest
                hit |= inside
        if ground is not None:
            g = self.pos[:, 2] < ground
            if g.any():
                self.pos[g, 2] = ground + 1e-4
                self.vel[g, :2] *= 0.5
                self.vel[g, 2] = -self.vel[g, 2] * self.rest
                hit |= g
        dead = (self.age >= self.lifetime) | (hit & (self.rng.random(len(self)) < self.kill))
        if len(self) > self.cap:
            dead[: len(self) - self.cap] = True
        keep = ~dead
        for k in ("pos", "prev", "vel", "age", "lifetime", "heat"):
            setattr(self, k, getattr(self, k)[keep])
        return self

    def temperature(self):
        return np.clip(1.0 - self.age / np.maximum(self.lifetime, 1e-6), 0, 1) ** 0.8 * self.heat

    def draw(self, img, K, T, depth=None, shutter=1 / 50, gain=1.6, bloom=0.9, bloom_radius=10, eps=0.03, width=1.0):
        if not len(self):
            return img
        H, W = img.shape[:2]
        tail = self.pos - self.vel * shutter
        (uv1, z1), (uv0, z0) = project(self.pos, K, T), project(tail, K, T)
        ok = (z1 > 0.02) & (z0 > 0.02)
        if not ok.any():
            return img
        uv1, uv0, z1, z0 = uv1[ok], uv0[ok], z1[ok], z0[ok]
        temp = self.temperature()[ok]
        col = _temp_rgb(np.minimum(temp, 1.0)) * (gain * (0.25 + temp ** 1.5))[:, None]
        L = np.linalg.norm(uv1 - uv0, axis=-1)
        S = int(np.clip(np.ceil(L.max()) + 1, 2, 160))
        s = np.linspace(0.0, 1.0, S, dtype=np.float32)
        uv = uv0[:, None, :] + (uv1 - uv0)[:, None, :] * s[None, :, None]
        z = z0[:, None] + (z1 - z0)[:, None] * s[None, :]
        wgt = (0.25 + 0.75 * s[None, :] ** 1.5) * np.minimum(1.0, 3.0 / np.maximum(L, 1.0) ** 0.35)[:, None]
        c = (col[:, None, :] * wgt[..., None]).reshape(-1, 3)
        uv, z = uv.reshape(-1, 2), z.reshape(-1)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        uv, z, c = uv[inb], z[inb], c[inb]
        if not len(uv):
            return img
        xi, yi = uv[:, 0].astype(int), uv[:, 1].astype(int)
        vis = _depth_ok(depth, xi, yi, z, eps)
        uv, c, xi, yi = uv[vis], c[vis], xi[vis], yi[vis]
        if not len(uv):
            return img
        pad = int(bloom_radius * 3 + 4)
        x0, y0 = max(xi.min() - pad, 0), max(yi.min() - pad, 0)
        x1, y1 = min(xi.max() + pad + 2, W), min(yi.max() + pad + 2, H)
        h, w = y1 - y0, x1 - x0
        fx, fy = uv[:, 0] - xi, uv[:, 1] - yi
        lx, ly = xi - x0, yi - y0
        flat = np.concatenate([(ly + dy) * w + lx + dx for dy in (0, 1) for dx in (0, 1)])
        wts = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
        layer = np.stack([np.bincount(flat, np.tile(c[:, ch], 4) * wts, h * w) for ch in range(3)], -1).astype(np.float32)
        layer = layer.reshape(h, w, 3)
        if width > 1.0:
            layer = _blur(layer, 1, 1) * 1.5
        layer = _bloom(layer, bloom_radius, bloom)
        return _composite(img, layer, y0, x0)


def _grid(x0, x1, y0, y1, u, v, f=1):
    dx = (np.arange(x0, x1, f, dtype=np.float32) + (f - 1) * 0.5 - u)[None, :]
    dy = (np.arange(y0, y1, f, dtype=np.float32) + (f - 1) * 0.5 - v)[:, None]
    return dx, dy


def glare(img, uv, z, depth=None, strength=1.0, flicker=1.0, size=1.0, streak=1.0, eps=0.03, warm=0.35, halo=1.0):
    H, W = img.shape[:2]
    u, v = float(uv[0]), float(uv[1])
    if not (z > 0.02) or strength <= 0:
        return img
    vis = 1.0
    if depth is not None and 0 <= u < W and 0 <= v < H:
        vis = float(_depth_ok(depth, np.array([int(u)]), np.array([int(v)]), np.array([z]), eps)[0])
    k = strength * flicker
    sc = size * (0.75 + 0.25 * flicker) * H / 1080.0
    r_halo, r_warm, sx = 70 * sc, 190 * sc, 330 * sc * streak
    rh = int(r_warm * 3)
    x0, x1 = max(int(u) - rh, 0), min(int(u) + rh + 1, W)
    y0, y1 = max(int(v) - rh, 0), min(int(v) + rh + 1, H)
    if x1 <= x0 or y1 <= y0:
        return img
    leak = (1.0 if vis else 0.3) * halo
    f = 4
    dx, dy = _grid(x0, x1, y0, y1, u, v, f)
    r2 = dx * dx + dy * dy
    hb = 1.0 / (1.0 + r2 * (1 / r_halo ** 2))
    hb = hb * hb * np.sqrt(hb)
    hw = 1.0 / (1.0 + r2 * (1 / r_warm ** 2))
    hw = hw * np.sqrt(hw)
    blue = np.array(ARC_RGB, np.float32)
    amber = np.array((1.0, 0.62, 0.3), np.float32)
    small = (hb[..., None] * (blue * 1.5 * leak) + hw[..., None] * (amber * warm * leak)).astype(np.float32)
    layer = _upsample(small, y1 - y0, x1 - x0)
    if vis:
        rc = int(30 * sc) + 2
        cx0, cx1, cy0, cy1 = max(int(u) - rc, x0), min(int(u) + rc + 1, x1), max(int(v) - rc, y0), min(int(v) + rc + 1, y1)
        if cx1 > cx0 and cy1 > cy0:
            ex, ey = _grid(cx0, cx1, cy0, cy1, u, v)
            q = ex * ex + ey * ey
            core = np.exp(-q / (2 * (4.5 * sc) ** 2)) * 12.0 + np.exp(-q / (2 * (11 * sc) ** 2)) * 1.8
            layer[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] += core[..., None]
        bw = int(16 * sc) + 2
        by0, by1 = max(int(v) - bw, y0), min(int(v) + bw + 1, y1)
        sx0, sx1 = max(int(u - 2.5 * sx), 0), min(int(u + 2.5 * sx) + 1, W)
        if by1 > by0 and sx1 > sx0:
            ex, ey = _grid(sx0, sx1, by0, by1, u, v)
            st = np.exp(-ex * ex / (2 * sx ** 2)) * np.exp(-ey * ey / (2 * (1.6 * sc) ** 2)) * 0.9
            st += np.exp(-ex * ex / (2 * (sx * 0.35) ** 2)) * np.exp(-ey * ey / (2 * (5 * sc) ** 2)) * 0.25
            band = (st[..., None] * (blue * 0.7 * streak)).astype(np.float32)
            gx0, gx1 = min(x0, sx0), max(x1, sx1)
            full = np.zeros((y1 - y0, gx1 - gx0, 3), np.float32)
            full[:, x0 - gx0:x1 - gx0] = layer
            full[by0 - y0:by1 - y0, sx0 - gx0:sx1 - gx0] += band
            layer, x0 = full, gx0
    layer *= k
    return _composite(img, layer, y0, x0)


class Smoke:
    def __init__(self, seed=0, rate=2.5, life=(1.5, 3.0), rise=(0.06, 0.16), wind=(0.05, 0.02, 0.0), alpha=0.07,
                 tone=(120, 122, 130), cap=200):
        self.rng = np.random.default_rng(seed)
        self.rate, self.life, self.rise, self.wind, self.alpha = rate, life, rise, np.asarray(wind, np.float64), alpha
        self.tone, self.cap = np.asarray(tone, np.float32), cap
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.lifetime = np.zeros(0)
        self.size0 = np.zeros(0)

    def __len__(self):
        return len(self.pos)

    def step(self, dt, tip=None, on=True):
        r = self.rng
        if on and tip is not None:
            k = int(r.poisson(self.rate))
            if k:
                self.pos = np.concatenate([self.pos, np.asarray(tip, np.float64) + r.normal(0, 0.003, (k, 3))])
                v = r.normal(0, 0.02, (k, 3))
                v[:, 2] = r.uniform(*self.rise, k)
                self.vel = np.concatenate([self.vel, v])
                self.age = np.concatenate([self.age, np.zeros(k)])
                self.lifetime = np.concatenate([self.lifetime, r.uniform(*self.life, k)])
                self.size0 = np.concatenate([self.size0, r.uniform(0.006, 0.012, k)])
        if not len(self):
            return self
        self.vel += (self.wind - self.vel) * min(dt * 0.8, 1.0) * np.array([1.0, 1.0, 0.0])
        self.vel += r.normal(0, 0.02, self.vel.shape) * dt
        self.pos = self.pos + self.vel * dt
        self.age += dt
        keep = self.age < self.lifetime
        if keep.sum() > self.cap:
            keep[np.flatnonzero(keep)[: keep.sum() - self.cap]] = False
        for k in ("pos", "vel", "age", "lifetime", "size0"):
            setattr(self, k, getattr(self, k)[keep])
        return self

    def draw(self, img, K, T, depth=None, eps=0.05, f=4):
        if not len(self):
            return img
        H, W = img.shape[:2]
        h, w = -(-H // f), -(-W // f)
        uv, z = project(self.pos, K, T)
        uv = (uv - (f - 1) * 0.5) / f
        x = self.age / self.lifetime
        rad = K[0, 0] * (self.size0 + 0.05 * x) / np.maximum(z, 1e-3) / f
        a = self.alpha * np.sin(np.pi * np.clip(x, 0, 1)) ** 0.8
        dlow = None if depth is None else depth[::f, ::f]
        acc = np.zeros((h, w), np.float32)
        for i in range(len(z)):
            if z[i] < 0.03 or rad[i] < 0.25:
                continue
            u, v, rr = uv[i, 0], uv[i, 1], rad[i] * 2.2 + 1
            x0, x1 = max(int(u - rr), 0), min(int(u + rr) + 1, w)
            y0, y1 = max(int(v - rr), 0), min(int(v + rr) + 1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            dx, dy = _grid(x0, x1, y0, y1, u, v)
            m = np.exp(-(dx * dx + dy * dy) * (0.5 / rad[i] ** 2)) * a[i]
            if dlow is not None:
                d = dlow[y0:y1, x0:x1]
                m = m * ((~np.isfinite(d)) | (d <= 0) | (z[i] <= d + eps))
            acc[y0:y1, x0:x1] -= np.log1p(-np.minimum(m, 0.95))
        if not acc.any():
            return img
        ys, xs = np.nonzero(acc > 1e-3)
        Y0, Y1, X0, X1 = ys.min() * f, min((ys.max() + 1) * f, H), xs.min() * f, min((xs.max() + 1) * f, W)
        A = 1.0 - np.exp(-_upsample(acc, h * f, w * f)[Y0:Y1, X0:X1])
        out = np.array(img, copy=True)
        patch = out[Y0:Y1, X0:X1].astype(np.float32)
        patch += (self.tone - patch) * A[..., None]
        out[Y0:Y1, X0:X1] = patch
        return out


def _translate_op(prim):
    from pxr import UsdGeom

    x = UsdGeom.Xformable(prim)
    for op in x.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return x.AddTranslateOp()


def arc_light(stage, path="/World/Weld/arc", radius=0.004, color=ARC_RGB, intensity=1.0e8, scale=1.0,
              pool=True, pool_radius=0.008, pool_color=POOL_RGB, pool_intensity=6.0e6, pool_lift=0.002):
    from pxr import Gf, UsdLux

    def sphere(p, r, c):
        lt = UsdLux.SphereLight.Define(stage, p)
        lt.CreateRadiusAttr(float(r))
        lt.CreateColorAttr(Gf.Vec3f(*map(float, c)))
        lt.CreateEnableColorTemperatureAttr(False)
        lt.CreateIntensityAttr(0.0)
        return lt, _translate_op(lt.GetPrim())

    arc, arc_op = sphere(path, radius, color)
    pl = sphere(path + "_pool", pool_radius, pool_color) if pool else None
    state = {"glow": 0.0}

    def set(pos, on=True, flicker=1.0, normal=None, dt=1 / 30):
        p = np.asarray(pos, np.float64)
        arc_op.Set(Gf.Vec3d(*map(float, p)))
        arc.GetIntensityAttr().Set(float(intensity * scale * flicker) if on else 0.0)
        if pl is not None:
            q = p + (np.asarray(normal, np.float64) * pool_lift if normal is not None else 0.0)
            pl[1].Set(Gf.Vec3d(*map(float, q)))
            target = 1.0 if on else 0.0
            state["glow"] += (target - state["glow"]) * (1 - math.exp(-dt / (0.08 if on else 0.6)))
            pl[0].GetIntensityAttr().Set(float(pool_intensity * scale * state["glow"] * (0.85 + 0.15 * flicker)))
        return arc

    return set


def _surface(stage, path, diffuse, metallic, roughness, emissive):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, diffuse)))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, emissive)))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat, sh


class WeldBead:
    def __init__(self, stage, path="/World/Weld/bead", looks="/World/Looks/weld", emissive=60.0, crown=0.002,
                 segs=9, ripple=0.18, pitch=0.0025, edges=AGE_EDGES):
        from pxr import UsdGeom, UsdShade, Vt

        self.stage, self.crown, self.segs, self.ripple, self.pitch, self.edges = stage, crown, segs, ripple, pitch, edges
        self.mesh = UsdGeom.Mesh.Define(stage, path)
        self.mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self.shaders, self.subsets = [], []
        api = UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim())
        for k, (rgb, e) in enumerate(HOT):
            cooled = k == len(HOT) - 1
            mat, sh = _surface(stage, f"{looks}_{k}", (0.22, 0.21, 0.2) if cooled else (0.08, 0.05, 0.04),
                               0.8 if cooled else 0.5, 0.45 if cooled else 0.55, tuple(c * e * emissive for c in rgb))
            sub = UsdGeom.Subset.CreateGeomSubset(self.mesh, f"age_{k}", UsdGeom.Tokens.face, Vt.IntArray(),
                                                  UsdShade.Tokens.materialBind)
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
            self.shaders.append(sh)
            self.subsets.append(sub)
            if cooled:
                api.Bind(mat)
        api.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)

    def set_emissive(self, emissive):
        from pxr import Gf

        for sh, (rgb, e) in zip(self.shaders, HOT):
            sh.GetInput("emissiveColor").Set(Gf.Vec3f(*(float(c * e * emissive) for c in rgb)))

    def update(self, points, normals=None, widths=0.008, ages=0.0, pipe=None, time=None):
        from pxr import Vt

        p = np.asarray(points, np.float64).reshape(-1, 3)
        if normals is None and pipe is not None:
            normals = pipe_normals(p, pipe[0], pipe[1])
        if normals is None:
            normals = np.tile([0.0, 0.0, 1.0], (len(p), 1))
        v, c, i, b, n = bead_mesh(p, normals, widths, ages, self.crown, self.segs, self.ripple, self.pitch,
                                  edges=self.edges, with_normals=True)
        kw = {} if time is None else {"time": time}
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)), **kw)
        self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(c), **kw)
        self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(i), **kw)
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)), **kw)
        if len(v):
            self.mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([v.min(0), v.max(0)]).astype(np.float32)), **kw)
        for k, sub in enumerate(self.subsets):
            sub.GetIndicesAttr().Set(Vt.IntArray.FromNumpy(np.flatnonzero(b == k).astype(np.int32)), **kw)
        return b

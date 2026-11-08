from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from policy.pickcan.layers import CORAL, DARK, project

STOPS = np.array([0.0, 10.0, 50.0, 100.0])
RAMP = np.array([[38, 84, 214], [0, 206, 230], [250, 214, 50], [235, 45, 40]], np.float32)
ICE = np.array([196, 228, 255], np.float32)
CYAN = (150, 228, 255)
WHITE = (236, 243, 255)
NOISE = np.random.default_rng(7).random((64, 64)).astype(np.float32)
FLOOR_PCT, TOP_PCT, KAPPA = 1.2, 45.0, 1.5
FADE_F, COLLAPSE_F, PULSE_S = 6, 14, 1.3


def ramp(pct) -> np.ndarray:
    p = np.clip(np.asarray(pct, np.float64), 0, 100)
    return np.stack([np.interp(p, STOPS, RAMP[:, c]) for c in range(3)], -1).astype(np.float32)


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _bilerp(a, fx, fy):
    ny, nx = a.shape[-2:]
    fx = np.clip(fx, 0, nx - 1.001)
    fy = np.clip(fy, 0, ny - 1.001)
    x0, y0 = fx.astype(np.int32), fy.astype(np.int32)
    u, v = fx - x0, fy - y0
    return (a[..., y0, x0] * (1 - u) * (1 - v) + a[..., y0, x0 + 1] * u * (1 - v)
            + a[..., y0 + 1, x0] * (1 - u) * v + a[..., y0 + 1, x0 + 1] * u * v)


def _noise(x, y):
    n = NOISE.shape[0]
    fx, fy = x % n, y % n
    x0, y0 = fx.astype(np.int32), fy.astype(np.int32)
    u, v = fx - x0, fy - y0
    u, v = u * u * (3 - 2 * u), v * v * (3 - 2 * v)
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    return (NOISE[y0, x0] * (1 - u) * (1 - v) + NOISE[y0, x1] * u * (1 - v)
            + NOISE[y1, x0] * (1 - u) * v + NOISE[y1, x1] * u * v)


def plume(K, T, depth, field, grid, wind=(1.0, 0.0), t=0.0, speed=4.0, ds=4, nv=18):
    H, W = depth.shape
    h, w = H // ds, W // ds
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 1e6)[:h * ds, :w * ds]
    dlow = d.reshape(h, ds, w, ds).min(axis=(1, 3)).astype(np.float32)
    R, c = T[:3, :3], T[:3, 3]
    u = (np.arange(w) + 0.5) * ds - 0.5
    v = (np.arange(h) + 0.5) * ds - 0.5
    xn = ((u - K[0, 2]) / K[0, 0]).astype(np.float32)
    yn = ((v - K[1, 2]) / K[1, 1]).astype(np.float32)
    ray = xn[None, :, None] * R[:, 0] + yn[:, None, None] * R[:, 1] + R[:, 2]
    ray = ray.astype(np.float32)
    gx, gy, gz = (np.asarray(grid[k], np.float64) for k in ("x", "y", "z"))
    res = float(gx[1] - gx[0])
    f = np.asarray(field, np.float32)
    zv = np.linspace(gz[0], gz[-1], nv)
    k = np.interp(zv, gz, np.arange(len(gz)))
    k0 = np.minimum(k.astype(int), len(gz) - 2)
    fz = f[k0] * (1 - (k - k0))[:, None, None] + f[k0 + 1] * (k - k0)[:, None, None]
    if fz.max() * 100 < FLOOR_PCT:
        return np.zeros((H, W), np.float32), np.zeros((H, W, 3), np.float32)
    dz = float(zv[1] - zv[0])
    rz = np.where(np.abs(ray[..., 2]) < 1e-4, -1e-4, ray[..., 2])
    path = np.minimum(dz * np.linalg.norm(ray, axis=-1) / np.abs(rz), 4 * dz)
    wd = np.asarray(wind, np.float32) / max(float(np.linalg.norm(wind)), 1e-6)
    adv = float(t) * 0.5 * float(speed)
    A = np.zeros((h, w), np.float32)
    C = np.zeros((h, w, 3), np.float32)
    lo, hi = math.log10(FLOOR_PCT), math.log10(TOP_PCT)
    order = range(nv - 1, -1, -1) if c[2] > zv[-1] else range(nv)
    for j in order:
        s = (zv[j] - c[2]) / rz
        ok = (s > 0) & (s < dlow + 0.4)
        if not ok.any():
            continue
        x = c[0] + s * ray[..., 0]
        y = c[1] + s * ray[..., 1]
        ok &= (x > gx[0]) & (x < gx[-1]) & (y > gy[0]) & (y < gy[-1])
        if not ok.any():
            continue
        iy, ix = np.nonzero(ok)
        xs, ys, ss = x[ok], y[ok], s[ok]
        pct = _bilerp(fz[j], (xs - gx[0]) / res, (ys - gy[0]) / res) * 100
        live = pct > FLOOR_PCT
        if not live.any():
            continue
        iy, ix, xs, ys, ss, pct = iy[live], ix[live], xs[live], ys[live], ss[live], pct[live]
        edge = np.clip(np.minimum.reduce([xs - gx[0], gx[-1] - xs, ys - gy[0], gy[-1] - ys]) / 0.9, 0, 1)
        occ = np.clip((dlow[iy, ix] - ss) / 0.3 + 0.5, 0, 1)
        al = (xs * wd[0] + ys * wd[1] - adv) / 1.1
        cr = (ys * wd[0] - xs * wd[1]) / 0.32
        n = 0.62 * _noise(al, cr) + 0.38 * _noise(al * 2.1 + 17.0, cr * 1.7 + 5.0)
        dens = np.clip((np.log10(pct) - lo) / (hi - lo), 0, 1) ** 1.5
        a = 1 - np.exp(-KAPPA * dens * (0.15 + 2.2 * n * n) * path[iy, ix])
        a *= edge * occ * 0.9
        t_ = 1 - A[iy, ix]
        C[iy, ix] += (t_ * a)[:, None] * ramp(pct * (1.2 + 0.5 * n))
        A[iy, ix] += t_ * a
    A = np.minimum(A, 0.75)
    up = lambda im: im.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(ds * 0.7))
    a_ = np.asarray(up(Image.fromarray(np.clip(A * 255, 0, 255).astype(np.uint8))), np.float32) / 255
    c_ = np.asarray(up(Image.fromarray(np.clip(C, 0, 255).astype(np.uint8))), np.float32)
    return a_, c_


def _visible(uv, z, depth, tol):
    H, W = depth.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    d = depth[y, x]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0.05)
    return inside, ~np.isfinite(d) | (d <= 0) | (z <= d + tol)


def _disc(img, x, y, r, col, a, soft=0.9, glow=False):
    H, W = img.shape[:2]
    e = r * (2.2 if glow else 1.0) + 2
    x0, x1, y0, y1 = int(max(x - e, 0)), int(min(x + e + 1, W)), int(max(y - e, 0)), int(min(y + e + 1, H))
    if x0 >= x1 or y0 >= y1 or a <= 0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    m = np.exp(-d2 / (2 * r * r)) if glow else np.clip((r - np.sqrt(d2)) / soft + 0.5, 0, 1)
    sub = img[y0:y1, x0:x1]
    sub += (np.asarray(col, np.float32) - sub) * (m * a)[..., None]


def _strokes(img, segs, col, width, glow=0.0, ss=2):
    if not segs:
        return
    H, W = img.shape[:2]
    pts = np.concatenate([np.asarray(s[0], float).reshape(-1, 2) for s in segs])
    pad = width * 2 + glow * 3 + 4
    x0, y0 = int(max(pts[:, 0].min() - pad, 0)), int(max(pts[:, 1].min() - pad, 0))
    x1, y1 = int(min(pts[:, 0].max() + pad, W)), int(min(pts[:, 1].max() + pad, H))
    if x0 >= x1 or y0 >= y1:
        return
    m = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    dr = ImageDraw.Draw(m)
    for p, a, kind in segs:
        q = [((u - x0 + 0.5) * ss, (v - y0 + 0.5) * ss) for u, v in np.asarray(p, float).reshape(-1, 2)]
        if kind == "poly":
            dr.polygon(q, fill=int(255 * a))
        else:
            dr.line(q, fill=int(255 * a), width=max(int(round(width * ss)), 1), joint="curve")
    m = m.reduce(ss)
    a_ = np.asarray(m, np.float32) / 255
    if glow:
        a_ = np.maximum(a_, 0.45 * np.asarray(m.filter(ImageFilter.GaussianBlur(glow)), np.float32) / 255)
    sub = img[y0:y1, x0:x1]
    sub += (np.asarray(col, np.float32) - sub) * a_[..., None]


def _poly3(pts, K, T, depth, a, tol=0.15, dim=0.35):
    uv, z = project(pts, K, T)
    _, vis = _visible(uv, z, depth, tol)
    segs = []
    for k in range(len(pts) - 1):
        if z[k] > 0.05 and z[k + 1] > 0.05:
            segs.append(([uv[k], uv[k + 1]], a * (1.0 if vis[k] and vis[k + 1] else dim), "line"))
    return segs


def _ring(cx, cy, cz, r, n=72):
    th = np.linspace(0, 2 * math.pi, n)
    return np.c_[cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, cz)]


def _obs_start(plan, i):
    fr = plan["frames"]
    j = i
    while j > 0 and fr[j - 1]["obs"] == fr[i]["obs"]:
        j -= 1
    return j


def draw(geo_rgb, K, T, depth, fr, plan, scale=None) -> np.ndarray:
    img = np.asarray(geo_rgb, np.float32)[..., :3].copy()
    H, W = img.shape[:2]
    s = W / 1920 if scale is None else scale
    depth = np.asarray(depth, np.float32).reshape(H, W)
    deck, src = float(plan["deck"]), np.asarray(plan["src"], float)
    fps, i = float(plan.get("fps", 30)), int(fr["i"])
    tv = i / fps
    A, C = plume(K, T, depth, fr["field"], plan["grid"], fr["wind"], fr["t"], fr["wind_speed"], ds=4 if W > 1400 else 3)
    img = img * (1 - A[..., None]) + C * 1.1
    found_f = plan["metrics"]["found_frame"]
    col = max(i - found_f, 0) / COLLAPSE_F if fr["found"] else 0.0
    c = ease(col)
    j0 = _obs_start(plan, i)
    u = ease((i - j0) / FADE_F) if j0 > 0 else 1.0
    est = np.asarray(fr["estimate"] if fr["estimate"] is not None else fr["belief"]["mean"] + [src[2]], float)

    trail = np.array([[*f["root"][:2], deck + 0.45] for f in plan["frames"][:i + 1:2]] + [[*fr["root"][:2], deck + 0.45]])
    if len(trail) > 1:
        _strokes(img, _poly3(trail, K, T, depth, 0.22), WHITE, 1.3 * s)

    sets = [(fr["particles"], u)]
    if u < 1:
        sets.append((plan["frames"][j0 - 1]["particles"], 1 - u))
    if c < 1:
        for P, wa in sets:
            P = np.asarray(P, np.float32)
            xy = P[:, :2] + (est[:2] - P[:, :2]) * c
            pts = np.c_[xy, np.full(len(xy), src[2])]
            uv, z = project(pts, K, T)
            inside, vis = _visible(uv, z, depth, 0.2)
            al = (0.12 + 0.55 * P[:, 2]) * wa * (1 - c) * np.where(vis, 1.0, 0.35)
            for (x, y), a in zip(uv[inside], al[inside]):
                _disc(img, x, y, 3.0 * s, ICE, a * 0.45, glow=True)
                _disc(img, x, y, 1.35 * s, ICE, min(a * 1.3, 1.0))

    if c < 1:
        b = fr["belief"]
        m1, s1 = np.asarray(b["mean"], float), float(b["sigma"])
        if u < 1:
            b0 = plan["frames"][j0 - 1]["belief"]
            m1 = np.asarray(b0["mean"], float) + (m1 - np.asarray(b0["mean"], float)) * u
            s1 = float(b0["sigma"]) + (s1 - float(b0["sigma"])) * u
        m1 = m1 + (est[:2] - m1) * c
        s1 = s1 + (0.3 - s1) * c
        ring = _ring(m1[0], m1[1], src[2], s1, 96)
        _strokes(img, _poly3(ring, K, T, depth, 0.85 * (1 - c)), WHITE, 1.5 * s, glow=3 * s)

    if fr["phase"] == "SEEK" and fr["arrow"] is not None:
        a0, a1 = np.asarray(fr["root"][:2], float), np.asarray(fr["arrow"][1], float)
        if np.linalg.norm(a1 - a0) > 0.08:
            seg = np.c_[np.linspace(a0, a1, 12), np.full(12, deck + 0.45)]
            uv, z = project(seg, K, T)
            if (z > 0.05).all():
                aa = 0.95 * u
                tip, back = uv[-1], uv[-3]
                dvec = tip - back
                dvec /= max(np.linalg.norm(dvec), 1e-6)
                nrm = np.array([-dvec[1], dvec[0]])
                hl, hw = 13 * s, 6 * s
                head = [tip + dvec * 2 * s, tip - dvec * hl + nrm * hw, tip - dvec * hl * 0.72, tip - dvec * hl - nrm * hw]
                body = [uv[0], tip - dvec * hl * 0.75]
                segs = [(body, aa, "line"), (head, aa, "poly")]
                _strokes(img, segs, CYAN, 2.0 * s, glow=4 * s)

    S = np.asarray(fr["samples"], float).reshape(-1, 4)
    if len(S):
        lens = np.array([len(f["samples"]) for f in plan["frames"]])
        uv, z = project(S[:, :3], K, T)
        inside, vis = _visible(uv, z, depth, 0.15)
        cols = ramp(np.maximum(S[:, 3] * 100, 0) * 2.2) * 0.72 + 0.28 * np.float32(255)
        for k in np.argsort(-z):
            if not inside[k]:
                continue
            born = int(np.searchsorted(lens, k + 1))
            age = (i - born) / fps
            newest = k >= len(S) - 2
            pop = 1 + 0.6 * math.exp(-age * 6)
            r = (6.2 if newest else 4.4) * s * pop
            a = 1.0 if vis[k] else 0.35
            if newest:
                _disc(img, *uv[k], r * 2.0, cols[k], 0.28 * a, glow=True)
            _disc(img, *uv[k], r + 1.7 * s, DARK, 0.85 * a)
            _disc(img, *uv[k], r, cols[k], a)
            _disc(img, uv[k][0] - r * 0.25, uv[k][1] - r * 0.3, r * 0.35, (255, 255, 255), 0.35 * a)

    if fr["found"]:
        k = ease(col * 1.4)
        mk = np.asarray(fr["estimate"], float)
        p = (tv - found_f / fps) % PULSE_S / PULSE_S
        rings = [(_ring(mk[0], mk[1], mk[2], 0.3), 1.0, 2.2), (_ring(mk[0], mk[1], mk[2], 0.3 + 0.55 * ease(p)), 0.7 * (1 - p), 1.6)]
        for ring, a, wdt in rings:
            _strokes(img, _poly3(ring, K, T, depth, a * k, tol=0.3, dim=0.45), CORAL, wdt * s, glow=4 * s)
        uv, z = project(mk[None], K, T)
        if z[0] > 0.05:
            _, vis = _visible(uv, z, depth, 0.3)
            a = k * (1.0 if vis[0] else 0.5)
            r = 7.5 * s * (1 + 0.25 * math.sin(2 * math.pi * p)) * (1 + 0.5 * (1 - k))
            _disc(img, *uv[0], r * 2.6, CORAL, 0.35 * a, glow=True)
            _disc(img, *uv[0], r + 2.0 * s, (255, 236, 226), 0.95 * a)
            _disc(img, *uv[0], r, CORAL, a)
    return np.clip(img, 0, 255).astype(np.uint8)

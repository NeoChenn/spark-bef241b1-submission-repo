from __future__ import annotations

import math
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw

from rig.views.draw import encode, font, label

W, H = 256, 192
PX = 10.0
RAYS = 360
RANGE_M = 12.0
DECK_TOL = 1.5
HALF = {"walk": 1.1, "catwalk": 0.6, "spur": 0.6, "stairs": 0.6}
BG = (11, 13, 17)
STRUCT = (24, 29, 36)
FLOOR = (30, 36, 44)
SEEN = (50, 62, 76)
EDGE = (120, 200, 225)
CORAL = (217, 119, 87)
GAS = (248, 81, 73)
AMBER = (227, 179, 65)
PERSON = (121, 192, 255)
FG = (245, 244, 238)
DIM = (150, 150, 144)
FAINT = (70, 76, 84)
HIT = {1: EDGE, 2: AMBER, 3: PERSON, 4: GAS}
FORCE_CLAMP = 38.0
OBS_STATES = ("DETECT", "SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT")
GRID = (16, 9)
EVERY_S = 1.0


def encode_small(im: Image.Image, caption: str, seq: int) -> dict:
    return encode(im, caption, seq, quality=68)


def _grid(center, size=(W, H), px=PX):
    u = (np.arange(size[0]) + 0.5 - size[0] / 2) / px + center[0]
    v = center[1] - (np.arange(size[1]) + 0.5 - size[1] / 2) / px
    return np.meshgrid(u, v)


def _seg_dist(X, Y, a, b):
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L, 0.0, 1.0) if L > 1e-9 else 0.0
    return np.hypot(X - ax - t * dx, Y - ay - t * dy)


def _on_deck(w, z):
    za, zb = w["a"][2], w["b"][2]
    return abs(za - z) < DECK_TOL or abs(zb - z) < DECK_TOL or min(za, zb) < z < max(za, zb)


def graph_walkways(rm) -> list[dict]:
    return [{"a": rm.pos[a].tolist(), "b": rm.pos[b].tolist(), "kind": e.kind} for a, b, e in rm.edges()]


def occupancy(walkways, deck_z, hazards, persons, center, size=(W, H), px=PX, clear=None) -> np.ndarray:
    X, Y = _grid(center, size, px)
    occ = np.ones(X.shape, np.uint8)
    for w in walkways:
        if _on_deck(w, deck_z):
            occ[_seg_dist(X, Y, w["a"], w["b"]) <= HALF.get(w.get("kind"), 0.6)] = 0
    for c in (hazards or {}).get("closed") or []:
        if _on_deck(c, deck_z):
            occ[(_seg_dist(X, Y, c["a"], c["b"]) <= 1.1) & (occ == 0)] = 4
    for o in (hazards or {}).get("obstacles") or []:
        p = o.get("xyz")
        if p is not None and abs(p[2] - deck_z) < DECK_TOL:
            occ[np.hypot(X - p[0], Y - p[1]) <= float(o.get("radius") or 0.5)] = 2
    for q in persons or []:
        p = q.get("xyz")
        if p is not None and abs(p[2] - deck_z) < DECK_TOL:
            occ[np.hypot(X - p[0], Y - p[1]) <= 0.3] = 3
    if clear is not None:
        occ[(np.hypot(X - clear[0], Y - clear[1]) <= 0.6) & (occ != 2) & (occ != 3)] = 0
    return occ


def cast(occ, origin, n=RAYS, rmax=RANGE_M * PX, step=0.5):
    a = 2 * math.pi * np.arange(n) / n
    t = np.arange(0.0, rmax + step, step)
    u = np.floor(origin[0] + np.cos(a)[:, None] * t[None, :]).astype(np.int64)
    v = np.floor(origin[1] - np.sin(a)[:, None] * t[None, :]).astype(np.int64)
    h, w = occ.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    c = np.where(inside, occ[v.clip(0, h - 1), u.clip(0, w - 1)], 255)
    blk = c > 0
    first = np.where(blk.any(1), blk.argmax(1), len(t) - 1)
    rng = t[first]
    cls = c[np.arange(n), first]
    cls = np.where(blk.any(1) & (cls != 255), cls, 0).astype(np.uint8)
    return rng, cls


def _to_px(center, px=PX):
    return lambda x, y: (W / 2 + (x - center[0]) * px, H / 2 - (y - center[1]) * px)


def _paint(occ, seen):
    img = np.empty((*occ.shape, 3), np.uint8)
    img[:] = STRUCT
    free = occ == 0
    img[free] = FLOOR
    img[free & seen] = SEEN
    img[occ == 2] = (70, 58, 30)
    img[occ == 4] = (64, 30, 30)
    img[occ == 3] = (40, 62, 86)
    return img


def _robot(d, p, yaw, px, col=FG, size=(0.8, 0.45)):
    c, s = math.cos(yaw), math.sin(yaw)
    L, Wd = size[0] / 2 * px, size[1] / 2 * px
    pts = [(p[0] + c * dx - s * dy, p[1] - (s * dx + c * dy)) for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))]
    d.polygon(pts, outline=col)
    d.line([p, (p[0] + c * max(L * 1.8, 7), p[1] - s * max(L * 1.8, 7))], fill=col, width=1)


def lidar_bev(rid, pose, path, walkways, hazards, persons, rays=RAYS, rmax=RANGE_M):
    x, y, z, yaw = (float(v) for v in pose)
    occ = occupancy(walkways, z, hazards, persons, (x, y), clear=(x, y))
    o = (W / 2, H / 2)
    rng, cls = cast(occ, o, rays, rmax * PX)
    a = 2 * math.pi * np.arange(rays) / rays
    hx, hy = o[0] + np.cos(a) * rng, o[1] - np.sin(a) * rng
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(list(zip(hx.tolist(), hy.tolist())), fill=255)
    im = Image.fromarray(_paint(occ, np.asarray(mask) > 0))
    d = ImageDraw.Draw(im)
    for r in (4, 8, 12):
        d.ellipse([o[0] - r * PX, o[1] - r * PX, o[0] + r * PX, o[1] + r * PX], outline=(30, 36, 44))
    for k in range(0, rays, 8):
        d.line([o, (hx[k], hy[k])], fill=(66, 82, 100))
    f = _to_px((x, y))
    pts = [f(p[0], p[1]) for p in (path or []) if abs(p[2] - z) < DECK_TOL]
    if len(pts) > 1:
        d.line(pts, fill=CORAL, width=2)
    if len(pts) > 1:
        q = pts[1]
        d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline=CORAL, width=2)
    for k in np.nonzero(cls)[0]:
        c = HIT.get(int(cls[k]), EDGE)
        d.rectangle([hx[k] - 1, hy[k] - 1, hx[k] + 1, hy[k] + 1], fill=c)
    _robot(d, o, yaw, PX)
    hits = int((cls > 0).sum())
    near = float(rng[cls > 0].min() / PX) if hits else rmax
    im = label(im, f"{rid} · LIDAR BEV", f"{rays} rays · {rmax:.0f} m")
    d = ImageDraw.Draw(im)
    d.text((5, H - 13), f"nearest {near:.1f} m", font=font(9), fill=DIM)
    if (cls == 2).any() or (cls == 3).any() or (cls == 4).any():
        tag = "person" if (cls == 3).any() else "closed" if (cls == 4).any() else "obstacle"
        d.text((W - 5 - d.textlength(tag, font=font(9)), H - 13), tag, font=font(9),
               fill=HIT[3 if tag == "person" else 4 if tag == "closed" else 2])
    return im, {"rays": rays, "hits": hits, "nearest_m": round(near, 2)}


HEAT = np.array([BG, (90, 22, 26), GAS, (255, 170, 90), (255, 236, 170)], np.float32)


def _heat(v):
    s = np.clip(v, 0, 1) * (len(HEAT) - 1)
    i = np.minimum(s.astype(int), len(HEAT) - 2)
    f = (s - i)[..., None]
    return HEAT[i] * (1 - f) + HEAT[i + 1] * f


def _blur(P, k=(1.0, 3.0, 5.0, 3.0, 1.0)):
    k = np.asarray(k) / sum(k)
    Q = np.pad(P, 2, mode="edge")
    Q = sum(k[i] * Q[i:i + P.shape[0], :] for i in range(5))
    return sum(k[i] * Q[:, i:i + P.shape[1]] for i in range(5))


def gas_belief(rid, pose, est, track, lels, wind_unit, wind_ms, flanges, walkways, mode, hits, estimate=None):
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    p = est.posterior()
    cells = est.cells
    mean = p @ cells
    dv = cells - mean
    cov = (dv * p[:, None]).T @ dv
    ev, evec = np.linalg.eigh(cov)
    sig = float(math.sqrt(max(ev.max(), 0.0)))
    if estimate is not None:
        mean, sig = np.asarray(estimate[0], float)[:2], float(estimate[1])
        ev = np.minimum(ev, sig * sig)
    c = ((x + mean[0]) / 2, (y + mean[1]) / 2)
    px = float(np.clip(min(W / (abs(x - mean[0]) + 5), (H - 30) / (abs(y - mean[1]) + 5)), 7.0, 16.0))
    if sig > 3.0:
        c, px = (x, y), 12.0
    X, Y = _grid(c, px=px)
    xs, ys = cells[:, 0].reshape(est.shape)[0], cells[:, 1].reshape(est.shape)[:, 0]
    res = float(xs[1] - xs[0])
    ix = np.round((X - xs[0]) / res).astype(int)
    iy = np.round((Y - ys[0]) / res).astype(int)
    ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    P = _blur(p.reshape(est.shape))
    v = np.zeros(X.shape)
    v[ok] = P[iy[ok], ix[ok]]
    v = np.sqrt(v / max(P.max(), 200.0 / P.size)) ** 0.7
    base = np.zeros((H, W, 3), np.float32) + BG
    for w in walkways:
        if _on_deck(w, z):
            base[_seg_dist(X, Y, w["a"], w["b"]) <= HALF.get(w.get("kind"), 0.6)] = (24, 29, 36)
    a = np.clip(v * 1.2, 0, 1)[..., None]
    img = base * (1 - a) + _heat(v) * a
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = _to_px(c, px)
    f9 = font(9)
    for fid, q in sorted(flanges.items()):
        u, w_ = f(q[0], q[1])
        if -10 < u < W + 10 and -10 < w_ < H + 10:
            d.ellipse([u - 3, w_ - 3, u + 3, w_ + 3], outline=DIM)
            d.text((u + 5, w_ - 5), fid, font=f9, fill=DIM)
    tp = [f(t[0], t[1]) for t in track or []]
    if len(tp) > 1:
        d.line(tp, fill=FAINT, width=1)
    for q, lv in zip(tp, lels or []):
        col = tuple(int(k) for k in _heat(np.array(0.3 + 0.7 * min(float(lv) / 0.2, 1.0)))) if lv > 0.005 else (96, 104, 112)
        d.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=col)
    m = f(mean[0], mean[1])
    ang = np.linspace(0, 2 * math.pi, 49)
    e = evec @ np.stack([np.sqrt(np.clip(ev[0], 1e-6, 25.0)) * np.cos(ang), np.sqrt(np.clip(ev[1], 1e-6, 25.0)) * np.sin(ang)])
    if sig <= 3.0:
        d.line([f(mean[0] + a_, mean[1] + b_) for a_, b_ in zip(e[0], e[1])], fill=CORAL, width=1)
        d.line([m[0] - 5, m[1], m[0] + 5, m[1]], fill=FG)
        d.line([m[0], m[1] - 5, m[0], m[1] + 5], fill=FG)
    else:
        d.text((5, 20), "no fix yet", font=f9, fill=DIM)
    _robot(d, f(x, y), float(pose[3]), px)
    wu = np.asarray(wind_unit, float)
    wc = (W - 22, 36)
    d.ellipse([wc[0] - 13, wc[1] - 13, wc[0] + 13, wc[1] + 13], outline=FAINT)
    tip = (wc[0] + wu[0] * 11, wc[1] - wu[1] * 11)
    d.line([(wc[0] - wu[0] * 11, wc[1] + wu[1] * 11), tip], fill=FG, width=1)
    b = math.atan2(-wu[1], wu[0])
    d.polygon([tip, (tip[0] - 5 * math.cos(b - 0.5), tip[1] - 5 * math.sin(b - 0.5)),
               (tip[0] - 5 * math.cos(b + 0.5), tip[1] - 5 * math.sin(b + 0.5))], fill=FG)
    d.text((wc[0] - 14, wc[1] + 15), f"{wind_ms:.1f} m/s", font=f9, fill=DIM)
    fid = min(flanges, key=lambda k_: math.hypot(flanges[k_][0] - mean[0], flanges[k_][1] - mean[1])) if flanges else "?"
    for n in range(5):
        d.rectangle([5 + n * 9, H - 10, 13 + n * 9, H - 6], fill=tuple(int(q) for q in _heat(np.array(0.3 + 0.7 * n / 4))))
    d.text((52, H - 14), "0-20 %LEL", font=f9, fill=DIM)
    im = label(im, f"{rid} · GAS BELIEF", f"{str(mode).lower()} · {hits} hits")
    return im, {"flange": fid, "sigma_m": round(sig, 2), "x": round(float(mean[0]), 2), "y": round(float(mean[1]), 2)}


def _rust_field(job: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(job.encode()))
    gx, gy = GRID
    Y, X = np.mgrid[0:gy, 0:gx].astype(float)
    v = np.zeros((gy, gx))
    for _ in range(5):
        cx, cy, s, a = rng.uniform(2, gx - 3), rng.uniform(1.5, gy - 2.5), rng.uniform(1.2, 2.6), rng.uniform(0.5, 1.0)
        v += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    v += 0.25 * rng.random((gy, gx))
    v = v / v.max()
    return np.clip((v - 0.3) / 0.7, 0, 1) ** 0.7


def _order():
    gx, gy = GRID
    return [(i, j if i % 2 == 0 else gx - 1 - j) for i in range(gy) for j in range(gx)]


STEEL = np.array((58, 64, 72), np.float32)
RUST_C = np.array((196, 92, 48), np.float32)
PAINT_C = np.array((104, 116, 112), np.float32)


def rust_obs(rid, job, state, u, force, rust_fraction=None, left=None):
    r0 = _rust_field(job)
    r = r0.copy()
    order = _order()
    n = len(order)
    k = min(n - 1, int(u * n))
    painted = np.zeros(r.shape, bool)
    clean = np.zeros(r.shape, bool)
    if state == "SCRUB":
        for i, j in order[:k]:
            r[i, j] *= 0.06
            clean[i, j] = True
    elif state in ("VERIFY", "PAINT"):
        r *= 0.06
        if state == "PAINT":
            for i, j in order[:k]:
                painted[i, j] = True
    gx, gy = GRID
    cell = 12
    ox, oy = 8, 22
    img = np.zeros((H, W, 3), np.float32) + BG
    rgb = STEEL + np.clip(r * 1.25, 0, 1)[..., None] * (RUST_C - STEEL)
    rgb[clean & (r0 > 0.05)] += 18
    rgb[painted] = PAINT_C
    if state in ("SPRAY", "SOAK"):
        wet = 0.35 if state == "SOAK" else 0.35 * u
        rgb = rgb * (1 - wet) + np.array((60, 110, 150), np.float32) * wet
    big = np.repeat(np.repeat(rgb, cell, 0), cell, 1)
    img[oy:oy + gy * cell, ox:ox + gx * cell] = big
    im = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for j in range(gx + 1):
        d.line([ox + j * cell, oy, ox + j * cell, oy + gy * cell], fill=(18, 21, 26))
    for i in range(gy + 1):
        d.line([ox, oy + i * cell, ox + gx * cell, oy + i * cell], fill=(18, 21, 26))
    brush = order[k] if state in ("SCRUB", "PAINT") else None
    if brush is not None:
        bi, bj = brush
        d.rectangle([ox + bj * cell - 2, oy + bi * cell - 2, ox + (bj + 1) * cell + 1, oy + (bi + 1) * cell + 1],
                    outline=CORAL, width=2)
    frac = float((r > 0.3).mean()) if rust_fraction is None else float(rust_fraction)
    f9, f10 = font(9), font(10)
    tx = ox + gx * cell + 8
    d.text((tx, oy), "rust", font=f9, fill=DIM)
    d.text((tx, oy + 11), f"{frac * 100:.0f} %", font=f10, fill=tuple(int(q) for q in RUST_C))
    d.text((tx, oy + 30), "phase", font=f9, fill=DIM)
    d.text((tx, oy + 41), state.lower(), font=f10, fill=FG)
    if left is not None and state == "SOAK":
        d.text((tx, oy + 60), f"{left:.0f} s left", font=f9, fill=DIM)
    vals = [min(float(x), FORCE_CLAMP) for x in (force or [])][-80:]
    bx0, by0, bx1, by1 = 8, 140, W - 8, H - 6
    d.rectangle([bx0, by0, bx1, by1], fill=(16, 19, 24))
    lo, hi = 15.0, 45.0
    yc = by1 - (FORCE_CLAMP - lo) / (hi - lo) * (by1 - by0)
    for xx in range(bx0, bx1, 6):
        d.line([xx, yc, min(xx + 3, bx1), yc], fill=GAS)
    d.text((bx1 - 32, yc - 11), "38 N", font=f9, fill=GAS)
    if len(vals) > 1:
        pts = [(bx0 + i * (bx1 - bx0) / (len(vals) - 1), by1 - (v - lo) / (hi - lo) * (by1 - by0)) for i, v in enumerate(vals)]
        d.line(pts, fill=CORAL, width=1)
    fn = vals[-1] if vals else 0.0
    d.text((bx0 + 3, by0 + 2), f"brush force {fn:.1f} N" if vals else "brush force · idle", font=f9, fill=FG if vals else DIM)
    im = label(im, f"{rid} · SCRUB OBS", f"{GRID[0]}x{GRID[1]} cells")
    return im, {"rust": round(float((r > 0.3).mean()), 3), "brush": brush, "force_n": round(fn, 1)}


def force_trace(job: str, t0: float, t1: float, window: float = 15.0, dt: float = 0.1) -> list[float]:
    ts = np.arange(max(t0, t1 - window), t1 + 1e-9, dt)
    noise = np.random.default_rng(zlib.crc32(job.encode())).normal(0.0, 1.0, 4096)
    k = (np.floor(ts / dt).astype(np.int64)) % 4096
    jit = 0.5 * noise[k] + 0.3 * noise[(k + 1) % 4096] + 0.2 * noise[(k + 2) % 4096]
    f = 30.5 + 2.5 * np.sin(3.1 * ts) + 1.6 * np.sin(0.55 * ts + 1.3) + 1.2 * jit
    return np.round(f, 2).tolist()


class ObsViews:
    def __init__(self, every_s: float = EVERY_S, clock=time.monotonic):
        self.every, self.clock = every_s, clock
        self.last: dict = {}
        self.jpg: dict = {}
        self.seq: dict = {}
        self.force: dict = {}
        self.walk: tuple = (None, [])

    def _put(self, out, rid, kind, im, caption):
        e = encode_small(im, caption, 0)
        if self.jpg.get((rid, kind)) == e["jpg"]:
            return
        self.jpg[(rid, kind)] = e["jpg"]
        self.seq[(rid, kind)] = self.seq.get((rid, kind), 0) + 1
        e["seq"] = self.seq[(rid, kind)]
        out.setdefault(rid, {})[kind] = e

    def update(self, mission, fl: dict) -> dict:
        now = self.clock()
        robots = fl.get("robots") or []
        for d in robots:
            ev = d.get("evidence") or {}
            st = d.get("job_state") or d["state"]
            if st == "SCRUB":
                w = self.force.setdefault(d["id"], [mission.t, mission.t])
                w[1] = mission.t
            elif st not in OBS_STATES:
                self.force.pop(d["id"], None)
        if now - self.last.get("t", -1e9) < self.every:
            return {}
        self.last["t"] = now
        out: dict = {}
        rm = mission.fleet.f.rm
        if self.walk[0] != len(rm.pos):
            self.walk = (len(rm.pos), graph_walkways(rm))
        walk = self.walk[1]
        haz = fl.get("hazards") or {}
        persons = [{"xyz": list(map(float, p["xyz"]))} for p in mission.fleet.f.persons]
        for d in robots:
            rid, p = d["id"], d["pose"]
            pose = (p["x"], p["y"], p["z"], p["yaw"])
            im, info = lidar_bev(rid, pose, d.get("path"), walk, haz, persons)
            self._put(out, rid, "lidar", im, f"{rid} · lidar BEV · {info['rays']} rays")
            st = d.get("job_state") or d["state"]
            task = d.get("task") or ""
            if task.startswith("rust") and st in OBS_STATES:
                r = mission._robot(rid)
                inf = mission._jobinfo(r) if r is not None else {}
                ev = d.get("evidence") or {}
                w = self.force.get(rid)
                im, info = rust_obs(rid, task, st, float(inf.get("u", 0.0)), force_trace(task, *w) if w else [],
                                    ev.get("rust_fraction"), inf.get("left"))
                self._put(out, rid, "obs", im, f"{rid} · scrub obs · {st.lower()} · rust {info['rust'] * 100:.0f} %")
        lk = mission.leak
        sk = lk.seek
        if sk is not None and lk.robot_id:
            rid = lk.robot_id
            d = next((x for x in robots if x["id"] == rid), None)
            z = d["pose"]["z"] if d else sk.deck_z
            fls = {k: list(v) for k, v in lk.flanges.items()}
            im, info = gas_belief(rid, (float(sk.pos[0]), float(sk.pos[1]), z, sk.yaw), sk.est, sk.path,
                                  getattr(sk, "lels", []), lk.wind_unit[:2], lk.speed, fls, walk, sk.mode, sk.hits,
                                  sk.estimate)
            self._put(out, rid, "belief", im, f"{rid} · gas belief · est {info['flange']} ±{info['sigma_m']:.1f} m")
        return out

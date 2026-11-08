from __future__ import annotations

import math

import numpy as np

from policy.pickcan.layers import project

SRC = (19.16, -9.25, 43.2)
PIPE_R = 0.07
JET = (-0.25, -0.35, 0.9)
WHEEL = (20.5, -9.25, 43.5745)
WHEEL_PROTO = "wheel"
STOP = (20.3, -9.53, 0.0)
TRUNK = 0.32
PARK_Q = (0.0, -1.57, 2.4, -2.4, -1.57, 0.0, 0.0)
HOVER_Q = (0.6005, -1.4188, 1.2242, -1.3763, -1.5708, -0.747, 0.4377)
ENGAGE_Q = (0.6005, -1.4199, 1.232, -1.3829, -1.5708, -0.747, 0.359)
ALIGN_S, REACH_S, TURN_S, SPIN = 0.65, 0.9, 3.6, 2.5 * math.pi
EE = ("Geometry/base/lift_housing/lift_carriage/ur3_base_link/ur3_shoulder_link/ur3_upper_arm_link/ur3_forearm_link/"
      "ur3_wrist_1_link/ur3_wrist_2_link/ur3_wrist_3_link/ur3_ee_link")
KEY = {"shaft": (0.0, 0.09, 0.013), "flange": (0.0, 0.022, 0.024), "socket": (0.072, 0.1, 0.022),
       "arm": (0.088, 0.068, 0.011), "pin": (0.088, 0.125, 0.0075, 0.062)}


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def valve_stop(deck=42.5):
    from rig.sim.gait import GaitLoop

    g = GaitLoop.load()
    return STOP[0], STOP[1], round(deck + TRUNK - (g.zmean if g else 0.36), 2), STOP[2]


def wheel_angle(e, turn_s=TURN_S, spin=SPIN):
    u = min(max((float(e) - REACH_S) / turn_s, 0.0), 1.0)
    return -spin * (u - math.sin(2 * math.pi * u) / (2 * math.pi))


def valve_arm(e, turn_s=TURN_S, spin=SPIN):
    e = max(float(e), 0.0)
    park, hov, eng = (np.array(q, np.float64) for q in (PARK_Q, HOVER_Q, ENGAGE_Q))
    if e < ALIGN_S:
        u = e / ALIGN_S
        q = park + (hov - park) * _ease((u - 0.25) / 0.75)
        q[6] = park[6] + (hov[6] - park[6]) * _ease(u / 0.6)
    else:
        q = hov + (eng - hov) * _ease((e - ALIGN_S) / (REACH_S - ALIGN_S))
    q[5] -= wheel_angle(e, turn_s, spin)
    return q


def _material(stage, path, rgb, metallic, roughness):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def dress_tool(stage, body_path, looks="/World/Looks/gas_key"):
    from pxr import Gf, UsdGeom, UsdShade

    ee = f"{body_path}/{EE}"
    brush = stage.GetPrimAtPath(f"{ee}/scrubber_link")
    if brush.IsValid():
        UsdGeom.Imageable(brush).MakeInvisible()
    steel = _material(stage, f"{looks}_steel", (0.2, 0.21, 0.23), 0.85, 0.4)
    yellow = _material(stage, f"{looks}_yellow", (0.85, 0.6, 0.06), 0.1, 0.5)
    root = f"{ee}/valve_key"
    UsdGeom.Xform.Define(stage, root)

    def cyl(path, x0, x1, r, mat, y=0.0):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateAxisAttr("X")
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(x1 - x0)
        UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, y, 0.0))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)

    for name in ("flange", "shaft", "socket"):
        x0, x1, r = KEY[name]
        cyl(f"{root}/{name}", x0, x1, r, steel)
    ax, reach, half = KEY["arm"]
    px, px1, pr, pd = KEY["pin"]
    for k in range(3):
        arm = UsdGeom.Xform.Define(stage, f"{root}/prong_{k}")
        ops = UsdGeom.Xformable(arm)
        ops.AddTranslateOp().Set(Gf.Vec3d(ax, 0.0, 0.0))
        ops.AddRotateXOp().Set(120.0 * k)
        c = UsdGeom.Cube.Define(stage, f"{root}/prong_{k}/bar")
        c.CreateSizeAttr(1.0)
        cops = UsdGeom.Xformable(c)
        cops.AddTranslateOp().Set(Gf.Vec3d(0.0, reach / 2, 0.0))
        cops.AddScaleOp().Set(Gf.Vec3f(2 * half, reach, 2 * half))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(yellow)
        cyl(f"{root}/prong_{k}/pin", 0.0, px1 - px, pr, yellow, pd)
    return root


class Handwheel:
    def __init__(self, stage, at=WHEEL, tol=0.25):
        from pxr import Gf, Usd, UsdGeom

        best = None
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.PointInstancer):
                continue
            pi = UsdGeom.PointInstancer(prim)
            protos = [str(p).rsplit("/", 1)[-1] for p in pi.GetPrototypesRel().GetTargets()]
            if WHEEL_PROTO not in protos:
                continue
            M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ids = np.asarray(pi.GetProtoIndicesAttr().Get())
            pos = np.asarray(pi.GetPositionsAttr().Get(), np.float64)
            for i in np.flatnonzero(ids == protos.index(WHEEL_PROTO)):
                d = float(np.linalg.norm(np.asarray(M.Transform(Gf.Vec3d(*pos[i]))) - np.asarray(at)))
                if d < tol and (best is None or d < best[0]):
                    best = (d, pi, int(i), M)
        if best is None:
            raise LookupError(f"no handwheel instance within {tol} m of {at}")
        _, self.instancer, self.index, M = best
        self.path = str(self.instancer.GetPath())
        self.attr = self.instancer.GetOrientationsAttr()
        self.base = list(self.attr.Get())
        q = self.base[self.index]
        self.rest = Gf.Quatd(float(q.GetReal()), Gf.Vec3d(*map(float, q.GetImaginary()))).GetNormalized()
        self.axis = M.GetInverse().TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        self.angle = None

    def set(self, angle_rad):
        from pxr import Gf, Vt

        a = round(float(angle_rad), 4)
        if a == self.angle:
            return
        self.angle = a
        q = (Gf.Rotation(self.axis, math.degrees(a)).GetQuat() * self.rest).GetNormalized()
        out = list(self.base)
        out[self.index] = Gf.Quath(float(q.GetReal()), Gf.Vec3h(*map(float, q.GetImaginary())))
        self.attr.Set(Vt.QuathArray(out))


def _fbm(n, seed, octaves=4):
    r = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    k = np.hypot(f[:, None], f[None, :])
    spec = np.fft.fft2(r.standard_normal((n, n))) * np.where(k > 0, k, 1.0) ** -1.6 * (k < 0.25)
    a = np.real(np.fft.ifft2(spec))
    a = (a - a.min()) / max(np.ptp(a), 1e-9)
    return a.astype(np.float32)


def _upsample(a, h, w):
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize((w, h), Image.BILINEAR))


class Vapour:
    KEYS = ("pos", "vel", "age", "life", "trav", "ph", "fr", "r0", "w", "off", "jet")

    def __init__(self, seed=0, jet_rate=260.0, puff_rate=34.0, speed=(4.0, 7.0), jet=JET, pipe_r=PIPE_R, cone=0.11,
                 jet_life=(0.1, 0.28), puff_life=(1.1, 2.0), rise=0.3, drift=(0.6, 1.5), fade=(0.7, 1.2),
                 jet_alpha=0.35, puff_alpha=0.5, lit=(214, 196, 178), shade=(128, 128, 134), shutter=1 / 60,
                 h=1 / 120, cap=1500):
        self.rng = np.random.default_rng(seed)
        self.rates, self.lives = (jet_rate, puff_rate), (jet_life, puff_life)
        self.speed, self.cone, self.rise, self.drift, self.fade = speed, cone, rise, drift, fade
        self.jet_alpha, self.puff_alpha, self.shutter, self.h, self.cap = jet_alpha, puff_alpha, shutter, h, cap
        self.dir = np.asarray(jet, np.float64) / np.linalg.norm(jet)
        self.pipe_r = pipe_r
        self.lit, self.shade = np.asarray(lit, np.float32), np.asarray(shade, np.float32)
        self.tex = _fbm(128, seed + 7)
        self.t, self.on = 0.0, False
        for k in self.KEYS:
            setattr(self, k, np.zeros((0, 3) if k in ("pos", "vel", "ph", "fr") else (0, 2) if k == "off" else 0))
        self.jet = self.jet.astype(bool)

    def __len__(self):
        return len(self.pos)

    def _emit(self, k, src, hh, jet):
        r = self.rng
        n = self.dir
        u = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        g = r.normal(0, self.cone * (1.0 if jet else 1.6), (k, 2))
        d = n + g[:, :1] * u + g[:, 1:] * v
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        sp = r.uniform(*self.speed, k) * (1.0 if jet else 0.8)
        s = r.random(k) * hh
        vel = d * sp[:, None]
        new = {"pos": np.asarray(src, np.float64) + n * self.pipe_r + r.normal(0, 0.003, (k, 3)) + vel * s[:, None],
               "vel": vel, "age": s, "life": r.uniform(*self.lives[0 if jet else 1], k), "trav": sp * s,
               "ph": r.uniform(0, 2 * np.pi, (k, 3)), "fr": r.uniform(0.5, 1.6, (k, 3)),
               "r0": r.uniform(0.003, 0.007, k) if jet else r.uniform(0.008, 0.016, k),
               "w": r.uniform(0.6, 1.3, k), "off": r.uniform(0, 1, (k, 2)), "jet": np.full(k, jet)}
        for key in self.KEYS:
            setattr(self, key, np.concatenate([getattr(self, key), new[key]]))

    def step(self, dt, src, wind_xy=(0.0, 0.0), wind_speed=0.0, on=True):
        dt = float(dt)
        if dt <= 0:
            return self
        on = bool(on)
        if self.on and not on and len(self):
            self.life = np.minimum(self.life, self.age + self.rng.uniform(*self.fade, len(self)))
        self.on = on
        wxy = np.asarray(wind_xy, np.float64)[:2]
        wxy = wxy / max(np.linalg.norm(wxy), 1e-9)
        wind = np.r_[wxy * min(self.drift[0] * float(wind_speed), self.drift[1]), 0.0]
        n = max(1, int(math.ceil(dt / self.h - 1e-9)))
        hh = dt / n
        for _ in range(n):
            if on:
                for jet, rate in zip((True, False), self.rates):
                    k = int(self.rng.poisson(rate * hh))
                    if k:
                        self._emit(k, src, hh, jet)
            if len(self):
                a = self.age
                amp = 0.02 + 0.22 * np.minimum(a / 0.9, 1.0)
                turb = amp[:, None] * np.sin(self.ph + 2 * np.pi * self.fr * self.t) * np.array([1.0, 1.0, 0.7])
                target = wind * np.minimum(a / 0.3, 1.0)[:, None] + turb
                target[:, 2] += self.rise * np.minimum(a / 0.5, 1.0)
                tau = 0.05 + 0.45 * np.minimum(a / 0.8, 1.0)
                self.vel += (target - self.vel) * (1 - np.exp(-hh / tau))[:, None]
                self.pos += self.vel * hh
                self.trav += np.linalg.norm(self.vel, axis=1) * hh
                self.age += hh
            self.t += hh
        keep = self.age < self.life
        if keep.sum() > self.cap:
            keep[np.flatnonzero(keep)[: keep.sum() - self.cap]] = False
        for key in self.KEYS:
            setattr(self, key, getattr(self, key)[keep])
        return self

    def radius(self):
        return self.r0 + 0.17 * np.minimum(self.trav, 0.3) + 0.07 * self.age ** 0.8

    def density(self):
        x = np.clip(self.age / np.maximum(self.life, 1e-6), 0, 1)
        r = self.radius()
        jet = self.jet_alpha * (1 - x) * np.minimum(1.0, 0.02 / r)
        a = np.clip(self.age / 0.35, 0, 1)
        puff = self.puff_alpha * a * a * (3 - 2 * a) * (1 - x) ** 1.4 * np.minimum(1.0, (0.06 / r) ** 2)
        return self.w * np.where(self.jet, jet, puff)

    def draw(self, img, K, T, depth=None, f=None, eps=0.05):
        if not len(self):
            return img
        H, W = img.shape[:2]
        f = f or max(1, int(round(H / 540)))
        h, w = -(-H // f), -(-W // f)
        uv, z = project(self.pos, K, T)
        uv2, _ = project(self.pos + self.vel * self.shutter, K, T)
        rad = self.radius()
        sig = K[0, 0] * rad * 0.5 / np.maximum(z, 1e-3) / f
        tau = self.density()
        uv, uv2 = (uv - (f - 1) * 0.5) / f, (uv2 - (f - 1) * 0.5) / f
        dlow = None if depth is None else np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)[::f, ::f]
        acc = np.zeros((h, w), np.float32)
        tex, tn = self.tex, self.tex.shape[0]
        for i in np.flatnonzero((z > 0.05) & (tau > 2e-4)):
            s = max(sig[i], 0.4)
            d = uv2[i] - uv[i] if self.jet[i] else np.zeros(2)
            L = float(np.hypot(*d))
            sa = math.sqrt(s * s + (0.5 * L) ** 2)
            ex = d / L if L > 1e-6 else np.array([1.0, 0.0])
            rr = 2.6 * sa + 1
            u, v = uv[i, 0] + d[0] * 0.5, uv[i, 1] + d[1] * 0.5
            x0, x1 = max(int(u - rr), 0), min(int(u + rr) + 1, w)
            y0, y1 = max(int(v - rr), 0), min(int(v + rr) + 1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            dx = (np.arange(x0, x1, dtype=np.float32) - u)[None, :]
            dy = (np.arange(y0, y1, dtype=np.float32) - v)[:, None]
            pa = dx * ex[0] + dy * ex[1]
            pp = dy * ex[0] - dx * ex[1]
            m = np.exp(-0.5 * (pa * pa / (sa * sa) + pp * pp / (s * s))) * (tau[i] * s / sa)
            if not self.jet[i] and s > 1.2:
                o = self.off[i] * tn + self.t * np.array([3.0, -5.0])
                tx = (dx * (11.0 / s) + o[0]).astype(np.int32) % tn
                ty = (dy * (11.0 / s) + o[1]).astype(np.int32) % tn
                m = m * np.clip(1.9 * tex[ty, tx] - 0.25, 0.0, 1.6)
            if dlow is not None:
                m = m * np.clip((dlow[y0:y1, x0:x1] - z[i]) / max(rad[i], eps) + 0.5, 0.0, 1.0)
            acc[y0:y1, x0:x1] += m
        ys, xs = np.nonzero(acc > 1e-3)
        if not len(ys):
            return img
        y0, y1, x0, x1 = ys.min(), ys.max() + 2, xs.min(), xs.max() + 2
        Y0, Y1, X0, X1 = y0 * f, min(y1 * f, H), x0 * f, min(x1 * f, W)
        sub = acc[y0:y1, x0:x1]
        tf = (_upsample(sub, (y1 - y0) * f, (x1 - x0) * f) if f > 1 else sub)[: Y1 - Y0, : X1 - X0]
        A = (1.0 - np.exp(-tf))[..., None]
        col = self.shade + (self.lit - self.shade) * np.exp(-1.4 * tf)[..., None]
        out = np.array(img, copy=True)
        patch = out[Y0:Y1, X0:X1].astype(np.float32)
        out[Y0:Y1, X0:X1] = np.clip(patch + (col - patch) * A + 0.5, 0, 255).astype(np.uint8)
        return out

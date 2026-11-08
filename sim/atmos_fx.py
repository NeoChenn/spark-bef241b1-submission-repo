from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import atmos

ROOT = "/World/Atmos"
FLARE_TIP = (-59.5, 8.0, 71.0)
EXHAUST = (32.0, 22.0, 42.5, 10.0)
VENTS = ((-16.0, 15.0, 41.6), (-0.5, -13.0, 37.6), (-2.0, 0.0, 39.2), (-7.0, 22.0, 31.6))


def _material(stage, name, diffuse, emissive=(0.0, 0.0, 0.0), opacity=1.0, texture=None, primvar_emissive=False,
              roughness=0.6):
    path = f"{ROOT}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    em = sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    em.Set(Gf.Vec3f(*emissive))
    op = sh.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    op.Set(opacity)
    if texture:
        st = UsdShade.Shader.Define(stage, f"{path}/uv")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        op.ConnectToSource(tex.CreateOutput("a", Sdf.ValueTypeNames.Float))
        sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.01)
    if primvar_emissive:
        rd = UsdShade.Shader.Define(stage, f"{path}/glow")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        em.ConnectToSource(rd.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def _curves(stage, name, width, mat):
    c = UsdGeom.BasisCurves.Define(stage, f"{ROOT}/{name}")
    c.CreateTypeAttr("linear")
    c.CreateWidthsAttr(Vt.FloatArray([width]))
    c.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    c.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, -1000), Gf.Vec3f(0, 0, -1000.1)]))
    c.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
    _bind(c, mat)
    return c


def _set_curves(c, pts, colors=None):
    if not len(pts):
        pts = np.array([[0, 0, -1000.0], [0, 0, -1000.1]])
    c.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    c.GetCurveVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(pts) // 2, 2, np.int32)))
    if colors is not None and len(colors) == len(pts) // 2:
        pv = UsdGeom.PrimvarsAPI(c).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform)
        pv.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))


def _texture() -> Path:
    path = Path(tempfile.gettempdir()) / "atmos_fog_soft.png"
    if not path.exists():
        a = atmos.soft_texture(256, seed=3)
        Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a])).save(path)
    return path


def leg_sources(stage):
    import cinematic

    legs = np.array(cinematic.leg_centers(stage) or [(0.0, 0.0, 5.0)], float)

    def spawn(rng, k):
        leg = legs[rng.integers(0, len(legs), k)]
        a = rng.uniform(0, math.tau, k)
        d = np.column_stack([np.cos(a), np.sin(a)])
        pos = np.column_stack([leg[:, :2] + d * (leg[:, 2:3] + 0.3), rng.uniform(0.5, 2.5, k)])
        up = rng.uniform(4.0, 14.0, k)
        vel = np.column_stack([d * rng.uniform(1.0, 4.0, (k, 1)), up])
        return pos, vel

    return spawn


def flare_source(rng, k):
    return np.asarray(FLARE_TIP) + rng.normal(0, 1.0, (k, 3)) + [0.0, 0.0, 3.0]


class Puffs:
    def __init__(self, stage, name, particles, colour, size, emissive=0.0, alpha=0.6, levels=4):
        self.p, self.size = particles, size
        root = f"{ROOT}/{name}"
        self.inst = UsdGeom.PointInstancer.Define(stage, root)
        protos = []
        for k in range(levels):
            m = UsdGeom.Mesh.Define(stage, f"{root}/proto_{k}")
            m.CreatePointsAttr([(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)])
            m.CreateFaceVertexCountsAttr([4])
            m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            m.CreateNormalsAttr([(0, -1, 0)] * 4)
            m.CreateDoubleSidedAttr(True)
            m.CreateSubdivisionSchemeAttr("none")
            UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                [(0, 0), (1, 0), (1, 1), (0, 1)])
            mat = _material(stage, f"{name}_{k}", colour, tuple(c * emissive for c in colour), texture=_texture(),
                            roughness=1.0)
            UsdShade.Shader.Get(stage, f"{ROOT}/Looks/{name}_{k}/tex").CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(1, 1, 1, alpha * (k + 1) / levels))
            _bind(m, mat)
            protos.append(m.GetPath())
        self.inst.CreatePrototypesRel().SetTargets(protos)
        self.inst.CreatePositionsAttr(Vt.Vec3fArray())
        self.inst.CreateProtoIndicesAttr(Vt.IntArray())
        self.inst.CreateScalesAttr(Vt.Vec3fArray())
        self.inst.CreateOrientationsAttr(Vt.QuathArray())
        self.levels = levels

    def update(self, dt, wind, eye):
        p = self.p
        p.step(dt, wind)
        u = np.clip(p.age / np.maximum(p.ttl, 1e-6), 0, 1)
        s = self.size[0] + (self.size[1] - self.size[0]) * np.sqrt(u)
        self.inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p.pos.astype(np.float32)))
        self.inst.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(s[:, None], 3, 1).astype(np.float32)))
        self.inst.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(atmos.fade_bucket(p.age, p.ttl, self.levels).astype(np.int32)))
        if eye is not None:
            q = atmos.billboard_quats(p.pos, eye)
            self.inst.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(*map(float, r)) for r in q]))


class Atmosphere:
    def __init__(self, stage, preset: str, tier: str = "live", weather=None, cam_path: str | None = None, flow=None,
                 flow_kw=None):
        self.stage, self.preset, self.tier, self.q = stage, preset, tier, atmos.TIERS[tier]
        self.cam_path, self.t, self.flow_kw = cam_path, None, flow_kw or {}
        self.w = atmos.weather_state(weather or atmos.DEFAULT_KIND.get(preset, "calm"))
        UsdGeom.Scope.Define(stage, ROOT)
        UsdGeom.Scope.Define(stage, f"{ROOT}/Looks")
        self.ember_mat = _material(stage, "ember", (0.0, 0.0, 0.0), primvar_emissive=True)
        self.rain_mat = _material(stage, "rain", (0.6, 0.65, 0.7), (0.05, 0.06, 0.07), opacity=0.5, roughness=0.1)
        self.spray_mat = _material(stage, "spray", (0.85, 0.9, 0.92), (0.02, 0.025, 0.03), roughness=0.3)
        self.embers = atmos.Particles(self.q["embers"], (np.full(3, -1e4), np.full(3, 1e4)), velocity=(0, 0, 9.0),
                                      spread=3.0, gravity=-1.5, drag=0.35, life=(1.2, 3.5), source=flare_source, seed=1)
        self.ember_curves = _curves(stage, "embers", 0.07, self.ember_mat)
        self.rain = atmos.Particles(self.q["rain"], (np.array([-30, -30, -5.0]), np.array([30, 30, 30.0])),
                                    velocity=(0, 0, -11.0), spread=0.6, seed=2)
        self.rain_curves = _curves(stage, "rain", 0.012, self.rain_mat)
        self.spray = atmos.Particles(self.q["spray"], (np.array([-200, -200, -0.5]), np.array([200, 200, 40.0])),
                                     gravity=-9.8, drag=0.25, life=(0.8, 2.4), source=leg_sources(stage), seed=3)
        self.spray_curves = _curves(stage, "spray", 0.05, self.spray_mat)
        self._build_exhaust()
        self.flow = self._build_flow() if (flow if flow is not None else self.q["flow"]) else None
        self.puffs = [] if self.flow is not None else self._build_puffs()
        self.wet = self._collect_roughness()
        self.set_weather(self.w)

    def _build_puffs(self):
        tip = np.asarray(FLARE_TIP) + [-1.5, 0.0, 11.0]
        vents = np.array(list(VENTS) + [(EXHAUST[0], EXHAUST[1], EXHAUST[2] + EXHAUST[3] + 0.3)])
        far = (np.full(3, -1e4), np.full(3, 1e4))
        smoke = atmos.Particles(self.q["smoke"], far, velocity=(0, 0, 3.0), spread=0.8, gravity=0.6, drag=0.18,
                                life=(6.0, 12.0), seed=5, source=lambda r, k: tip + r.normal(0, 1.2, (k, 3)))
        steam = atmos.Particles(self.q["steam"], far, velocity=(0, 0, 2.5), spread=0.4, gravity=0.4, drag=0.3,
                                life=(2.5, 6.0), seed=6,
                                source=lambda r, k: vents[r.integers(0, len(vents), k)] + r.normal(0, 0.25, (k, 3)))
        return [Puffs(self.stage, "smoke", smoke, (0.05, 0.045, 0.04), (3.0, 22.0), alpha=0.7),
                Puffs(self.stage, "steam", steam, (0.75, 0.77, 0.8), (0.8, 7.0), emissive=0.02, alpha=0.5)]

    def _build_exhaust(self):
        x, y, z, h = EXHAUST
        c = UsdGeom.Cylinder.Define(self.stage, f"{ROOT}/exhaust_stack")
        c.CreateRadiusAttr(0.9)
        c.CreateHeightAttr(h)
        c.AddTranslateOp().Set(Gf.Vec3d(x, y, z + h / 2))
        _bind(c, _material(self.stage, "stack", (0.08, 0.08, 0.085), roughness=0.5))

    def _build_flow(self):
        import omni.kit.app

        if not omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.flowusd", True):
            return None
        vents = list(VENTS) + [(EXHAUST[0], EXHAUST[1], EXHAUST[2] + EXHAUST[3] + 0.5)]
        self.flow_layer = Sdf.Layer.CreateAnonymous("atmos_flow.usda")
        self.flow_layer.ImportFromString(atmos.flow_usda(FLARE_TIP, vents, self.w["wind"], self.q["flow_cell"], **self.flow_kw))
        prim = self.stage.DefinePrim(f"{ROOT}/Flow", "Xform")
        prim.GetReferences().AddReference(self.flow_layer.identifier)
        self._settings({"/rtx/flow/enabled": True, "/rtx/flow/pathTracingEnabled": True,
                        "/rtx/flow/rayTracedReflectionsEnabled": True, "/rtx/flow/rayTracedTranslucencyEnabled": True,
                        "/rtx/flow/rayTracedShadowsEnabled": True})
        return prim

    def _collect_roughness(self):
        out = []
        looks = self.stage.GetPrimAtPath("/World/Looks")
        if not looks:
            return out
        for p in Usd.PrimRange(looks):
            sh = UsdShade.Shader(p)
            if p.IsA(UsdShade.Shader) and sh.GetIdAttr().Get() == "UsdPreviewSurface":
                r = sh.GetInput("roughness")
                if r and r.Get() is not None and not r.HasConnectedSource() and "sea" not in str(p.GetPath()):
                    out.append((r, float(r.Get())))
        return out

    @staticmethod
    def _settings(values):
        import carb.settings

        s = carb.settings.get_settings()
        for k, v in values.items():
            s.set(k, v)

    def set_weather(self, level="calm", wind_dir=None, gust=None):
        self.w = level if isinstance(level, dict) else atmos.weather_state(level, wind_dir, gust)
        self._settings(atmos.volumetric_settings(self.preset, self.tier, self.w))
        UsdGeom.Imageable(self.rain_curves).GetVisibilityAttr().Set("inherited" if self.w["rain"] > 0 else "invisible")
        wet = 1.0 - 0.7 * self.w["rain"]
        for attr, base in self.wet:
            attr.Set(max(0.04, base * wet))
        if self.flow is not None:
            wind = Gf.Vec3f(self.w["wind"][0], self.w["wind"][1], 0.0)
            for i in (0, 1):
                a = self.stage.GetPrimAtPath(f"{ROOT}/Flow/wind_{i}").GetAttribute("velocity")
                if a:
                    a.Set(wind)
        return self.w

    def _eye(self):
        if not self.cam_path:
            return None, None
        m = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.cam_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = np.array(m.ExtractTranslation())
        fwd = -np.array([m[2][0], m[2][1], m[2][2]])
        return eye, fwd / (np.linalg.norm(fwd) + 1e-9)

    def update(self, t: float) -> None:
        dt = 1 / 24 if self.t is None else min(0.25, max(1e-3, t - self.t))
        self.t = t
        wind = self.w["wind"]
        eye, fwd = self._eye()
        self.embers.step(dt, wind)
        heat = np.clip(1 - self.embers.age / np.maximum(self.embers.ttl, 1e-3), 0, 1)
        colors = np.column_stack([60 * heat + 4, 22 * heat ** 2 + 0.6, 4 * heat ** 4]) * 2
        _set_curves(self.ember_curves, atmos.streak_points(self.embers.pos, self.embers.vel, 0.03), colors)
        if self.w["rain"] > 0 and eye is not None:
            self.rain.recentre(eye + fwd * 22.0)
            self.rain.step(dt, (wind[0] * 0.6, wind[1] * 0.6))
            k = int(len(self.rain.pos) * self.w["rain"])
            vel = self.rain.vel[:k] + [wind[0] * 0.6, wind[1] * 0.6, 0.0]
            _set_curves(self.rain_curves, atmos.streak_points(self.rain.pos[:k], vel, 0.045))
        for puff in self.puffs:
            puff.update(dt, wind, eye)
        self.spray.step(dt, wind)
        k = int(len(self.spray.pos) * min(1.0, 0.15 + self.w["spray"]))
        _set_curves(self.spray_curves, atmos.streak_points(self.spray.pos[:k], self.spray.vel[:k], 0.05))
        tip = self.stage.GetPrimAtPath("/World/Cinematic/Lamps/lamp_400/light")
        if tip:
            base = tip.GetAttribute("spark:baseIntensity")
            flick = 0.8 + 0.2 * math.sin(t * 13.0) * math.sin(t * 7.3 + 1.0)
            if base.IsValid():
                tip.GetAttribute("inputs:intensity").Set(base.Get() * flick)
        for i in range(5):
            p = self.stage.GetPrimAtPath(f"/World/Cinematic/Flare/flame_{i}")
            if not p:
                continue
            op = p.GetAttribute("xformOp:translate")
            base = p.GetAttribute("spark:baseTranslate")
            if not base.IsValid():
                base = p.CreateAttribute("spark:baseTranslate", Sdf.ValueTypeNames.Double3, custom=True)
                base.Set(op.Get())
            b = base.Get()
            lean = atmos.wind_lean(wind, b[2] - 70.0)
            op.Set(Gf.Vec3d(b[0] + lean[0], b[1] + lean[1], b[2]))

    def post(self, rgb, cam: dict, t: float, lens: bool = False):
        from rig.camera import project
        from rig.post import LensDrops, heat_shimmer

        u, v, z = project(cam, (FLARE_TIP[0], FLARE_TIP[1], FLARE_TIP[2] + 12.0))
        if z > 1.0:
            r = cam["K"][0][0] * 5.0 / z
            if r > 3.0:
                rgb = heat_shimmer(rgb, (u, v), r, t, min(5.0, 0.12 * r))
        if lens:
            if not hasattr(self, "lens"):
                self.lens, self.lens_t = LensDrops(seed=4), t
                for _ in range(40):
                    self.lens.step(0.1, self.w["rain"])
            self.lens.step(max(0.0, t - self.lens_t), self.w["rain"])
            self.lens_t = t
            rgb = self.lens.apply(rgb)
        return rgb

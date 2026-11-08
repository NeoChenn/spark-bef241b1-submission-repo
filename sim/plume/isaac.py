import math

import numpy as np

from sim.plume.colormap import lel_rgb

ROOT = "/World/Plume"
BANDS = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 1e9)
BAND_OPACITY = (0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.5)
PARK = (0.0, 0.0, -1.0e4)


def omnipbr(stage, path: str, color, opacity: float, emissive: float):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    c = Gf.Vec3f(*map(float, color))
    for name, typ, val in (("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, c),
                           ("enable_opacity", Sdf.ValueTypeNames.Bool, True),
                           ("opacity_constant", Sdf.ValueTypeNames.Float, float(opacity)),
                           ("enable_emission", Sdf.ValueTypeNames.Bool, emissive > 0),
                           ("emissive_color", Sdf.ValueTypeNames.Color3f, c),
                           ("emissive_intensity", Sdf.ValueTypeNames.Float, float(emissive)),
                           ("reflection_roughness_constant", Sdf.ValueTypeNames.Float, 1.0),
                           ("specular_level", Sdf.ValueTypeNames.Float, 0.0)):
        sh.CreateInput(name, typ).Set(val)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def cloud(field, lo: float = BANDS[0], hi: float = 1e9, max_points: int = 60000, seed: int = 0):
    pct = field.fraction / field.lel * 100.0
    flat = pct.ravel()
    idx = np.flatnonzero((flat >= lo) & (flat < hi))
    if len(idx) > max_points:
        idx = idx[np.argpartition(flat[idx], -max_points)[-max_points:]]
    ijk = np.stack(np.unravel_index(idx, pct.shape), 1)
    off = np.modf(np.abs(np.sin((idx[:, None] + seed) * np.array([12.9898, 78.233, 37.719])) * 43758.5453))[0] - 0.5
    return (field.origin + (ijk + 0.5 + 0.6 * off) * field.dx).astype(np.float32), flat[idx]


class PlumeVisual:
    def __init__(self, stage, plume, path: str = ROOT + "/Cloud", max_points: int = 8000, size: float = 1.7,
                 emissive: float = 150.0, opacity: float = 1.0):
        from pxr import UsdGeom, UsdShade, Vt

        self.plume, self.max_points = plume, max_points
        self.width = size * plume.dx
        self.bands = []
        for k, (lo, hi) in enumerate(zip(BANDS[:-1], BANDS[1:])):
            prim = UsdGeom.PointInstancer.Define(stage, f"{path}/band{k}")
            proto = UsdGeom.Sphere.Define(stage, f"{path}/band{k}/proto")
            proto.CreateRadiusAttr(0.5 * self.width)
            mid = lo if hi > 1e8 else 0.5 * (lo + hi)
            mat = omnipbr(stage, f"{ROOT}/Looks/band{k}", lel_rgb(mid), min(1.0, BAND_OPACITY[k] * opacity), emissive)
            UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(mat)
            prim.CreatePrototypesRel().SetTargets([proto.GetPath()])
            prim.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.zeros(max_points, np.int32)))
            prim.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.full((max_points, 3), PARK, np.float32)))
            self.bands.append((prim, lo, hi))
        self.count = 0
        self.update()

    def update(self) -> int:
        from pxr import Vt

        field = self.plume.field()
        self.count = 0
        for prim, lo, hi in self.bands:
            pts, _ = cloud(field, lo, hi, self.max_points)
            buf = np.full((self.max_points, 3), PARK, np.float32)
            buf[: len(pts)] = pts
            prim.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(buf))
            self.count += len(pts)
        return self.count


def analysis_pose(plume, side: float = 1.0, distance: float = 6.5, height: float = 2.6, ahead: float = 1.8):
    w = plume.wind.at(plume.t)
    h = math.atan2(w[1], w[0]) if np.linalg.norm(w[:2]) > 1e-6 else math.radians(plume.wind.heading_deg)
    d = np.array([math.cos(h), math.sin(h), 0.0])
    perp = np.array([-d[1], d[0], 0.0]) * side
    s = plume.source
    target = s + d * ahead + np.array([0.0, 0.0, 0.2])
    eye = s + perp * distance + d * (ahead * 0.6) + np.array([0.0, 0.0, height])
    return tuple(eye), tuple(target)


def make_camera(stage, eye, target, path: str = ROOT + "/AnalysisCamera", focal: float = 18.0):
    from pxr import Gf, UsdGeom

    from rig.sim.orbit import look_at_rows

    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 9 / 16)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
    op = UsdGeom.Xformable(cam).AddTransformOp()
    op.Set(Gf.Matrix4d(look_at_rows(tuple(map(float, eye)), tuple(map(float, target)))))
    return path, op


class LeakStream:
    def __init__(self, stage, args):
        import omni.replicator.core as rep

        from sim.plume.solver import Plume, Wind

        self.args = args
        self.plume = Plume.for_flange(args.leak, args.leak_mm, args.leak_bar, args.leak_h2s,
                                      Wind(args.leak_wind, args.leak_heading, args.leak_storm), stage=stage)
        self.visual = PlumeVisual(stage, self.plume)
        self.cam, _ = make_camera(stage, *analysis_pose(self.plume, args.leak_side))
        self.rp = rep.create.render_product(self.cam, tuple(args.pip_size))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach(self.rp)
        self.last = None
        self.stats = None
        self.pip_at = self.field_at = 0.0
        self.sim_ms = 0.0
        self.ticks = 0
        print(f"rig sim: leak {args.leak} {args.leak_mm} mm {args.leak_bar} bar -> {self.plume.flow_rate()} "
              f"solid={self.plume.solid_np.mean():.3f} device={self.plume.device}", flush=True)

    def tick(self, t: float) -> None:
        import time

        dt = 0.0 if self.last is None else min(max(t - self.last, 0.0), 0.1)
        self.last = t
        if dt <= 0:
            return
        t0 = time.perf_counter()
        self.ticks += 1
        due = self.ticks % self.args.leak_vis_every == 0
        st = self.plume.step(dt, stats=due)
        if due:
            self.stats = st
            self.visual.update()
        self.sim_ms = 0.9 * self.sim_ms + 0.1 * (time.perf_counter() - t0) * 1e3

    def header(self, now: float) -> dict | None:
        import base64

        from sim.plume.pip import compose, jpeg_b64

        if self.stats is None:
            return None
        p = self.plume
        out = {"flange": self.args.leak, "stats": self.stats, "flow": p.flow_rate(), "wind": p.wind.to_json(p.t),
               "points": self.visual.count, "sim_ms": round(self.sim_ms, 2)}
        if now >= self.pip_at:
            data = np.asarray(self.ann.get_data())
            if data.ndim == 3 and data.size:
                out["jpg"] = jpeg_b64(compose(data, self.stats, out["flow"], out["wind"], f"GAS PLUME  ·  {self.args.leak}"))
                self.pip_at = now + self.args.pip_every
        if now >= self.field_at:
            out["field"] = base64.b64encode(p.field().to_bytes(2)).decode()
            self.field_at = now + 1.0
        return out

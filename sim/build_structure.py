from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dressing_geometry as dg
import photoreal as pr
import structure as s
import structure_geometry as sg
import structure_textures as stx
import topside
import topside_frames as tf
from photoreal_usd import bind_mdl

LEGS_MESH = "/World/Rig/legs_and_lower_deck_concrete"
SPLASH_BANDS = [f"/World/Details/splash_band_{n:03d}" for n in range(92, 96)]
SIDES = 96
DECK_UNDERSIDE = 18.75
FLOODS = ((2.0, -31.0), (14.0, 8.0), (-16.0, 2.0))
OUTLINE = topside.DECK_EDGES[0][1]
OVERHANG = ((25.0, -37.5), (37.5, 25.0), 30.0)
LANDING_ANGLE = -62.0
RISERS = (
    ((-14.0, 0.32), (2.0, 0.45), (18.0, 0.28), (-40.0, 0.62)),
    ((-150.0, 0.32), (-165.0, 0.5), (-125.0, 0.6)),
    ((-100.0, 0.4), (-84.0, 0.3), (75.0, 0.32)),
    ((60.0, 0.4), (105.0, 0.32), (30.0, 0.55)),
)
KELVINS = (3200, 4300, 2700, 5000)
LOWER, MIDDLE, TOP = 20.0, 31.25, 42.5
STAIRS = (((6.0, -39.75), LOWER, MIDDLE, "x"), ((25.2, 8.0), LOWER, MIDDLE, "y"), ((2.0, -29.8), MIDDLE, TOP, "x"))
LADDERS = (((37.75, -30.0), MIDDLE, 40.0, 0.0), ((37.75, 19.0), MIDDLE, 41.2, 0.0), ((14.0, -37.8), LOWER, MIDDLE, -90.0))
EDGES = (
    ((0.3, -37.5), (37.5, -37.5), 30.0, 1.25, (0.0, -1.0)),
    ((37.5, -37.5), (37.5, 25.0), 30.0, 1.25, (1.0, 0.0)),
    ((0.3, -27.64), (37.5, -27.64), 41.25, 1.25, (0.0, -1.0)),
    ((37.5, -27.64), (37.5, 27.36), 41.25, 1.25, (1.0, 0.0)),
)
TOP_UNDERSIDE = ((0.3, -27.6), (25.3, -14.2), 41.25)
FLOODS_CALM = ((230000.0, 2400), (160000.0, 4500), (95000.0, 3400))
LIGHT_SPOTS = ((0.0, -17.5), (0.0, 17.5), (-17.5, 0.0), (17.5, 0.0), (-9.0, -9.0), (9.0, 9.0), (9.0, -26.0), (-9.0, 9.0))


def ensure_textures() -> None:
    s.GROWTH_TEX.mkdir(parents=True, exist_ok=True)
    if not (s.GROWTH_TEX / "diff.jpg").exists():
        maps = stx.marine_growth(2048)
        for name, data in maps.items():
            Image.fromarray(data).save(s.GROWTH_TEX / f"{name}.jpg", quality=92)
    if not s.DRIP_TEX.exists():
        Image.fromarray(stx.drip_mask(256, 1024, seed=3)).save(s.DRIP_TEX)


def make_looks(stage) -> dict:
    looks = {}
    for name, recipe in s.RECIPES.items():
        sub, values = pr.shader_inputs(recipe)
        if name == "growth":
            values.update(paint_diff=s.GROWTH_TEX / "diff.jpg", paint_rough=s.GROWTH_TEX / "rough.jpg",
                          paint_normal=s.GROWTH_TEX / "nor_gl.jpg")
        looks[name] = bind_mdl(stage, f"{s.LOOKS}/{name}", sub, values, define=True)
    looks["drip"] = bind_mdl(stage, f"{s.LOOKS}/drip", "rig_decal",
                             {"mask": s.DRIP_TEX, "tint": (0.17, 0.07, 0.025), "oil": 0.0, "roughness": 0.6,
                              "strength": 1.1, "wetness": 0.0, "use_color_tex": False}, define=True)
    lamp = UsdShade.Material.Define(stage, f"{s.LOOKS}/lamp")
    shader = UsdShade.Shader.Define(stage, f"{s.LOOKS}/lamp/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * 25 for c in tf.kelvin_rgb(3600))))
    lamp.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    looks["lamp"] = lamp
    return looks


def write_mesh(stage, path, m, material, double=False):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m["points"], np.float32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m["normals"], np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(m["counts"], np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m["indices"], np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(double)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(np.asarray(m["uvs"], np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def add_light(stage, path, pos, intensity, kelvin, cone, softness=0.5):
    light = UsdLux.SphereLight.Define(stage, path)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    light.CreateRadiusAttr(0.25)
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(*tf.kelvin_rgb(kelvin)))
    shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr(cone)
    shaping.CreateShapingConeSoftnessAttr(softness)
    return light


def calm_floods(stage):
    for n, (intensity, kelvin) in enumerate(FLOODS_CALM):
        prim = stage.OverridePrim(f"/World/Cinematic/Lamps/lamp_{320 + n}/light")
        light = UsdLux.SphereLight(prim)
        light.CreateIntensityAttr(intensity)
        prim.CreateAttribute("spark:baseIntensity", Sdf.ValueTypeNames.Float, custom=True).Set(intensity)
        light.CreateColorAttr(Gf.Vec3f(*tf.kelvin_rgb(kelvin)))
        shaping = UsdLux.ShapingAPI.Apply(prim)
        shaping.CreateShapingConeAngleAttr(72.0)
        shaping.CreateShapingConeSoftnessAttr(0.7)


def light_path(kind, n, live):
    return f"{s.ROOT}/Lights/{kind}_{n:02d}" if live else f"{s.OFFLINE}/Lights/{kind}_{n:02d}"


def zquat(angle_deg):
    h = math.radians(angle_deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


class Instances:
    def __init__(self, stage, looks, root):
        self.stage, self.looks, self.root = stage, looks, root
        self.items: dict[str, dict[str, list]] = {}
        self.meshes: dict[str, dict] = {}

    def add(self, look, proto, position, quat=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        self.items.setdefault(look, {}).setdefault(proto, []).append(
            (tuple(map(float, position)), tuple(map(float, quat)), tuple(map(float, scale))))

    def member(self, look, proto, p, q, size):
        if math.dist(p, q) < 1e-4:
            return
        mid, quat, length = tf.segment_pose(p, q)
        self.add(look, proto, mid, quat, (length, size[0], size[1]))

    def tube(self, look, p, q, r):
        self.member(look, "cyl", p, q, (2 * r, 2 * r))

    def banded(self, looks3, p, q, r):
        for a, b, band in sg.split_by_bands(p, q, (s.GROWTH_TOP, s.SPLASH_TOP)):
            self.tube(looks3[band], a, b, r)

    def write(self):
        total = 0
        for look, buckets in self.items.items():
            path = f"{self.root}/{look}"
            inst = UsdGeom.PointInstancer.Define(self.stage, path)
            protos, idx, pos, ori, scl = [], [], [], [], []
            for proto, entries in buckets.items():
                prim = self._proto(f"{path}/protos/{proto}", proto)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.looks[look])
                for position, quat, scale in entries:
                    idx.append(len(protos))
                    pos.append(Gf.Vec3f(*position))
                    ori.append(Gf.Quath(quat[0], Gf.Vec3h(*quat[1:])))
                    scl.append(Gf.Vec3f(*scale))
                protos.append(prim.GetPath())
            inst.CreatePrototypesRel().SetTargets(protos)
            inst.CreateProtoIndicesAttr(Vt.IntArray(idx))
            inst.CreatePositionsAttr(Vt.Vec3fArray(pos))
            inst.CreateOrientationsAttr(Vt.QuathArray(ori))
            inst.CreateScalesAttr(Vt.Vec3fArray(scl))
            total += len(idx)
        return total

    def _proto(self, path, proto):
        UsdGeom.Scope.Define(self.stage, path.rsplit("/", 1)[0])
        if proto == "cyl":
            prim = UsdGeom.Cylinder.Define(self.stage, path)
            prim.CreateAxisAttr("X")
            prim.CreateRadiusAttr(0.5)
            prim.CreateHeightAttr(1.0)
        elif proto == "box":
            prim = UsdGeom.Cube.Define(self.stage, path)
            prim.CreateSizeAttr(1.0)
        elif proto == "ibeam":
            prim = topside.ibeam_mesh(self.stage, path)
        else:
            prim = write_mesh(self.stage, path, self.meshes[proto], None, double=True)
        return prim.GetPrim()


def wedge_mesh():
    pts = np.array([(0, -0.5, 1), (1, -0.5, 1), (0, -0.5, 0), (0, 0.5, 1), (1, 0.5, 1), (0, 0.5, 0)], float)
    faces = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
    flat_pts, normals, idx, counts = [], [], [], []
    for f in faces:
        p = pts[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        idx += list(range(len(flat_pts), len(flat_pts) + len(f)))
        flat_pts += list(p)
        normals += [n] * len(f)
        counts.append(len(f))
    return {"points": np.array(flat_pts), "normals": np.array(normals), "uvs": np.array(flat_pts)[:, :2],
            "counts": counts, "indices": idx}


def hoop_mesh(radius=0.42, wire=0.025):
    a = np.linspace(-math.pi / 2, math.pi / 2, 17)
    pts = np.stack((radius * np.cos(a), radius * np.sin(a), 0 * a), -1)
    return dg.tube(pts, wire, 6)


def read_legs():
    full = Usd.Stage.Open(str(pr.OUTPUT / "full_rig.usda"), Usd.Stage.LoadNone)
    pts = np.array(UsdGeom.Mesh(full.GetPrimAtPath(LEGS_MESH)).GetPointsAttr().Get())
    return sg.leg_axes(pts)


def build_columns(stage, looks, kit, legs):
    band_looks = (looks["growth"], looks["splash"], looks["column"])
    for i, leg in enumerate(legs):
        bottom, top = leg["bottom"][2], leg["top"][2]
        g = sg.edge_profile(SIDES, s.GROWTH_TOP, 0.55, seed=10 + i)
        sp = sg.edge_profile(SIDES, s.SPLASH_TOP, 0.85, seed=20 + i)
        for band, (lo, hi) in enumerate(((np.full(SIDES + 1, bottom), g), (g, sp), (sp, np.full(SIDES + 1, top)))):
            write_mesh(stage, f"{s.ROOT}/Columns/leg{i}_{s.BAND_NAMES[band]}", sg.column_band(leg, SIDES, lo, hi, rows=1),
                       band_looks[band])
        for z, look in ((3.2, "splash"), (9.4, "column"), (13.6, "column")):
            c = sg.axis_at(leg, z)
            kit.add(look, "stiff", (c[0], c[1], z))
        sm = sg.seams(leg, -1.4, top - 1.0, pitch=2.4)
        for p in sm["rings"]:
            kit.add(("growth", "splash", "seam")[s.band_of(p[2])], "seam", p)
        for p, q in sm["lines"]:
            for a, b, band in sg.split_by_bands(p, q, (s.GROWTH_TOP, s.SPLASH_TOP)):
                kit.tube(("growth", "splash", "seam")[band], a, b, 0.028)
        for pos, angle, size in sg.gussets(leg):
            kit.add("steel", "wedge", pos, zquat(angle), size)
        for pos, angle in sg.anodes(leg, offset=11.0 * i):
            kit.add("anode", "box", pos, zquat(angle + 90.0), (0.3, 0.3, 1.9))
            u = np.array((math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0))
            for dz in (-0.8, 0.8):
                kit.tube("growth", pos + (0, 0, dz) - u * 0.4, pos + (0, 0, dz), 0.04)
        rs = sg.risers(leg, RISERS[i])
        for n, (p, q, r) in enumerate(rs["pipes"]):
            kit.banded(("growth", "splash", "caisson" if r > 0.55 else "riser"), p, q, r)
        for c in rs["clamps"]:
            band = s.band_of(c["center"][2])
            look = ("growth", "splash", "steel")[band]
            kit.member(look, "cyl", c["center"] - (0, 0, 0.18), c["center"] + (0, 0, 0.18),
                       (2 * c["pipe_r"] + 0.16, 2 * c["pipe_r"] + 0.16))
            kit.add(look, "box", c["bracket"], zquat(c["angle"]), (c["length"], 0.12, 0.32))


def build_landing(stage, looks, kit, leg):
    bl = sg.boat_landing(leg, LANDING_ANGLE)
    for p, q, r in bl["tubes"]:
        kit.banded(("growth", "splash", "landing"), p, q, r)
    for p, q, r in bl["rubber"]:
        kit.tube("rubber", p, q, r)
    for p, q, r in bl["rails"] + bl["ladder"] + bl["rungs"]:
        kit.tube("landing", p, q, r)
    for pos, angle in bl["hoops"]:
        kit.add("landing", "hoop", pos, zquat(angle))
    kit.add("grating", "box", bl["deck"], zquat(bl["deck_angle"]), bl["deck_size"])
    u = np.array((math.cos(math.radians(LANDING_ANGLE)), math.sin(math.radians(LANDING_ANGLE)), 0.0))
    t = np.array((-u[1], u[0], 0.0))
    for n, (dz, out, kelvin) in enumerate(((6.4, 2.2, 4000), (13.5, 0.6, 2300))):
        pos = sg.axis_at(leg, dz)
        pos = np.array((pos[0], pos[1], dz)) + u * (leg["radius"] + out) + t * (2.2 if n else -2.6)
        kit.add("steel", "box", pos + (0, 0, 0.2), zquat(LANDING_ANGLE), (0.5, 0.5, 0.3))
        if n == 0:
            kit.tube("landing", pos - (0, 0, dz - 4.8), pos + (0, 0, 0.35), 0.05)
        kit.add("lamp", "box", pos, zquat(LANDING_ANGLE), (0.45, 0.45, 0.04))
        add_light(stage, light_path("landing", n, n == 0), pos + u * 0.6 - (0, 0, 0.3), 200000.0, kelvin, 80.0)
    for side in (-3.6, 0.0, 3.6):
        kit.add("landing", "box", bl["deck"] + t * side - (0, 0, 0.18), zquat(LANDING_ANGLE), (3.0, 0.14, 0.26))


def build_braces(kit, legs):
    br = sg.braces(legs, z=7.4, deck=DECK_UNDERSIDE - 2.3)
    for p, q, r in br["horizontal"] + br["diagonal"]:
        kit.tube("brace", p, q, r)
    for p, q, r in br["cans"]:
        kit.tube("seam", p, q, r)


def build_overhang(stage, looks, kit):
    (x0, y0), (x1, y1), z = OVERHANG
    rect = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    plate = {"points": np.array([(x, y, z - 0.04) for x, y in rect]), "normals": np.tile((0.0, 0.0, -1.0), (4, 1)),
             "uvs": np.array(rect), "counts": [4], "indices": [3, 2, 1, 0]}
    write_mesh(stage, f"{s.ROOT}/Underside/overhang_plate", plate, looks["plate"])
    for y in sg.clear_stations(y0 + 0.4, y1 - 0.4, 3.1):
        kit.member("steel", "ibeam", (x0 + 0.2, y, z - 0.38), (x1 - 0.3, y, z - 0.38), (0.32, 0.76))
    for x in (x0 + 0.4, (x0 + x1) / 2, x1 - 0.3):
        kit.member("steel", "ibeam", (x, y0, z - 0.8), (x, y1, z - 0.8), (0.6, 1.6))
    for y in (y0 + 0.3, y1 - 0.3):
        kit.member("steel", "ibeam", (x0, y, z - 0.8), (x1, y, z - 0.8), (0.6, 1.6))
    for y in (y0 + 1.0, -17.5, 0.0, 17.5):
        kit.member("steel", "ibeam", (x0 + 0.3, y, DECK_UNDERSIDE + 1.6), (x1 - 2.5, y, z - 1.5), (0.4, 0.5))
    for x, r, look in ((29.0, 0.22, "pipe_grey"), (29.7, 0.14, "pipe_yellow"), (33.5, 0.28, "pipe_green")):
        kit.tube(look, (x, y0 + 0.8, z - 1.1), (x, y1 - 0.8, z - 1.1), r)
        for y in np.arange(y0 + 1.5, y1 - 1.0, 4.0):
            kit.tube("galv", (x, y, z - 1.1 + r), (x, y, z - 0.75), 0.025)
    for n, (x, y) in enumerate(((31.25, -30.0), (31.25, -12.0), (31.25, 6.0), (31.25, 20.0))):
        kit.add("steel", "box", (x, y, z - 1.75), scale=(0.7, 0.35, 0.22))
        kit.add("lamp", "box", (x, y, z - 1.87), scale=(0.6, 0.28, 0.03))
        add_light(stage, light_path("overhang", n, n == 1), (x, y, z - 2.2), 130000.0, KELVINS[n % len(KELVINS)], 80.0)


def build_underside(stage, looks, kit, legs):
    z = DECK_UNDERSIDE
    xs = [p[0] for p in OUTLINE]
    ys = [p[1] for p in OUTLINE]
    plate = {"points": np.array([(x, y, z - 0.04) for x, y in OUTLINE]), "normals": np.tile((0.0, 0.0, -1.0), (len(OUTLINE), 1)),
             "uvs": np.array(OUTLINE), "counts": [len(OUTLINE)], "indices": list(range(len(OUTLINE)))[::-1]}
    write_mesh(stage, f"{s.ROOT}/Underside/plate", plate, looks["plate"])
    for x in sg.clear_stations(min(xs) + 0.4, max(xs) - 0.4, 3.1, avoid=[f[0] for f in FLOODS], clear=1.25):
        for a, b in sg.scanline(OUTLINE, x, axis="y"):
            kit.member("steel", "ibeam", (x, a + 0.3, z - 0.38), (x, b - 0.3, z - 0.38), (0.32, 0.76))
    for y in (-17.5, 17.5):
        for a, b in sg.scanline(OUTLINE, y, axis="x"):
            kit.member("steel", "ibeam", (a + 0.4, y, z - 1.1), (b - 0.4, y, z - 1.1), (0.8, 2.2))
    for x in (-17.5, 17.5):
        for a, b in sg.scanline(OUTLINE, x, axis="y"):
            kit.member("steel", "ibeam", (x, a + 0.4, z - 1.1), (x, b - 0.4, z - 1.1), (0.8, 2.2))
    for (x0, y0), (x1, y1) in zip(OUTLINE, OUTLINE[1:] + OUTLINE[:1]):
        kit.member("steel", "ibeam", (x0, y0, z - 0.7), (x1, y1, z - 0.7), (0.5, 1.4))
        d = np.array((x1 - x0, y1 - y0))
        length = float(np.linalg.norm(d))
        u = d / length
        out = np.array((u[1], -u[0])) * 0.06
        angle = math.degrees(math.atan2(u[1], u[0]))
        for f in np.arange(1.0, length - 0.5, 2.2):
            x, y = np.array((x0, y0)) + u * f + out
            kit.add("steel", "box", (x, y, z + 0.62), zquat(angle), (0.1, 0.14, 1.24))
        kit.member("seam", "box", (x0 + out[0], y0 + out[1], z + 1.2), (x1 + out[0], y1 + out[1], z + 1.2), (0.12, 0.1))
    for y in (-3.0, -1.6):
        for a, b in sg.scanline(OUTLINE, y, axis="x"):
            a, b = a + 1.0, b - 1.0
            zt = z - 1.05
            kit.member("galv", "box", (a, y, zt), (b, y, zt), (0.6, 0.02))
            for side in (-0.3, 0.3):
                kit.member("galv", "box", (a, y + side, zt + 0.06), (b, y + side, zt + 0.06), (0.02, 0.12))
            for k, dy in enumerate((-0.15, 0.0, 0.15)):
                kit.tube("cable", (a, y + dy, zt + 0.07), (b, y + dy, zt + 0.07), 0.05 + 0.015 * k)
            for x in np.arange(a + 0.5, b, 3.0):
                for side in (-0.3, 0.3):
                    kit.tube("galv", (x, y + side, zt), (x, y + side, z - 0.75), 0.02)
    for x, r, look in ((-9.2, 0.26, "pipe_grey"), (-8.4, 0.18, "pipe_green"), (-7.7, 0.13, "pipe_yellow"),
                       (6.0, 0.3, "pipe_grey"), (6.8, 0.15, "pipe_red")):
        for a, b in sg.scanline(OUTLINE, x, axis="y"):
            zp = z - 1.45
            kit.tube(look, (x, a + 0.8, zp), (x, b - 0.8, zp), r)
            for y in np.arange(a + 1.5, b - 1.0, 4.0):
                kit.tube("galv", (x, y, zp + r), (x, y, z - 0.75), 0.025)
                kit.member("galv", "box", (x - r - 0.05, y, zp - r - 0.04), (x + r + 0.05, y, zp - r - 0.04), (0.08, 0.05))
    for a, b in (((-11.5, -19.1), (11.5, -19.1)), ((19.1, -11.5), (19.1, 11.5)), ((-19.1, -11.5), (-19.1, 11.5))):
        w = sg.walkway(a, b, z - 3.1, hang_to=z - 0.76)
        centre, angle, size = w["deck"]
        kit.add("grating", "box", centre, zquat(angle), size)
        for p, q in w["rails"] + w["posts"]:
            kit.tube("landing", p, q, 0.025)
        for p, q in w["hangers"]:
            kit.tube("galv", p, q, 0.03)
    for n, (x, y) in enumerate(LIGHT_SPOTS):
        zl = z - 2.35 if (abs(abs(x) - 17.5) < 0.1 or abs(abs(y) - 17.5) < 0.1) else z - 0.9
        kit.add("steel", "box", (x, y, zl + 0.12), scale=(0.7, 0.35, 0.22))
        kit.add("lamp", "box", (x, y, zl), scale=(0.6, 0.28, 0.03))
        add_light(stage, light_path("under", n, n < 4), (x, y, zl - 0.35), 170000.0, KELVINS[n % len(KELVINS)], 80.0)


def add_stair(kit, origin, z0, z1, axis):
    st = tf.stair_tower(origin, z0, z1, axis=axis)
    for centre, size in st["steps"] + st["landings"]:
        kit.add("grating", "box", centre, scale=size)
    for a, b in st["stringers"]:
        kit.member("landing", "box", a, b, (0.02, 0.25))
        kit.member("landing", "cyl", (a[0], a[1], a[2] + 1.0), (b[0], b[1], b[2] + 1.0), (0.05, 0.05))
    f = st["frame"]
    for p, q in f["columns"]:
        kit.member("landing", "ibeam", p, q, (0.16, 0.16))
    for p, q in f["beams"]:
        kit.member("landing", "ibeam", p, q, (0.11, 0.16))
    for p, q in f["braces"]:
        kit.member("landing", "cyl", p, q, (0.08, 0.08))


def add_ladder(kit, base, z0, z1, angle):
    lad = sg.caged_ladder(base, z0, z1, angle)
    for p, q in lad["rails"] + lad["straps"]:
        kit.tube("landing", p, q, 0.03)
    for p, q in lad["rungs"]:
        kit.tube("landing", p, q, 0.02)
    for pos, a in lad["hoops"]:
        kit.add("landing", "hoop", pos, zquat(a))


def build_access(kit):
    for origin, z0, z1, axis in STAIRS:
        add_stair(kit, origin, z0, z1, axis)
    for base, z0, z1, angle in LADDERS:
        add_ladder(kit, base, z0, z1, angle)


def build_edges(kit):
    for p, q, z0, height, out in EDGES:
        e = sg.edge_detail(p, q, z0, height, out)
        for pos, angle in e["ribs"]:
            kit.add("steel", "box", pos, zquat(angle), e["size"])
        kit.member("seam", "box", e["strip"][0], e["strip"][1], (0.12, 0.1))


def build_top_underside(stage, kit):
    (x0, y0), (x1, y1), z = TOP_UNDERSIDE
    for x in sg.clear_stations(x0, x1, 3.1):
        kit.member("steel", "ibeam", (x, y0 + 0.2, z - 0.35), (x, y1 - 0.2, z - 0.35), (0.3, 0.7))
    for y in (y0 + 0.3, y1 - 0.3):
        kit.member("steel", "ibeam", (x0, y, z - 0.6), (x1, y, z - 0.6), (0.5, 1.2))
    for y, r, look in ((-21.0, 0.2, "pipe_grey"), (-20.4, 0.12, "pipe_yellow"), (-18.0, 0.16, "pipe_green")):
        kit.tube(look, (x0 + 0.5, y, z - 1.0), (x1 - 0.5, y, z - 1.0), r)
        for x in np.arange(x0 + 1.5, x1 - 1.0, 4.0):
            kit.tube("galv", (x, y, z - 1.0 + r), (x, y, z - 0.7), 0.025)
    for n, x in enumerate((7.0, 18.0)):
        kit.add("steel", "box", (x, -24.0, z - 0.95), scale=(0.7, 0.35, 0.22))
        kit.add("lamp", "box", (x, -24.0, z - 1.07), scale=(0.6, 0.28, 0.03))
        add_light(stage, light_path("top", n, False), (x, -24.0, z - 1.4), 110000.0, KELVINS[n + 1], 80.0)


def build_drips(stage, looks, legs, seed=7):
    rng = np.random.default_rng(seed)
    count = 0
    for i, leg in enumerate(legs):
        r = leg["radius"] + 0.015
        sources = [3.2 - 0.15, 9.4 - 0.15, 13.6 - 0.15] + list(rng.uniform(s.SPLASH_TOP, leg["top"][2] - 1.0, 6))
        for z0 in sources:
            for _ in range(3):
                a0 = rng.uniform(0, 360)
                width = rng.uniform(0.5, 1.4)
                length = rng.uniform(1.2, 3.5)
                span = math.degrees(width / r)
                a = np.radians(np.linspace(a0 - span / 2, a0 + span / 2, 7))
                zs = np.array((z0, z0 - length))
                pts, normals, uvs = [], [], []
                for zz, v in zip(zs, (1.0, 0.0)):
                    c = sg.axis_at(leg, zz)
                    for k, ang in enumerate(a):
                        pts.append((c[0] + r * math.cos(ang), c[1] + r * math.sin(ang), zz))
                        normals.append((math.cos(ang), math.sin(ang), 0.0))
                        uvs.append((k / 6, v))
                quads = []
                for k in range(6):
                    quads += [k, k + 1, k + 8, k + 7]
                m = {"points": pts, "normals": normals, "uvs": uvs, "counts": [4] * 6, "indices": quads}
                write_mesh(stage, f"{s.OFFLINE}/Drips/d{count:03d}", m, looks["drip"])
                count += 1
    return count


def build() -> Path:
    ensure_textures()
    if s.LAYER.exists():
        s.LAYER.unlink()
    stage = Usd.Stage.CreateNew(str(s.LAYER))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.OverridePrim("/World")
    for group in ("", "/Looks", "/Columns", "/Underside", "/Lights", "/Inst", "/Offline", "/Offline/Drips", "/Offline/Lights"):
        UsdGeom.Scope.Define(stage, f"{s.ROOT}{group}")
    looks = make_looks(stage)
    for name in ("galv", "black", "pipe_grey", "pipe_green", "pipe_yellow", "pipe_red"):
        looks[name] = UsdShade.Material(stage.OverridePrim(f"/World/Looks/topside_{name}"))
    looks["cable"] = looks["black"]
    legs = read_legs()
    kit = Instances(stage, looks, f"{s.ROOT}/Inst")
    kit.meshes.update(stiff=sg.ring_mesh(leg_radius(legs) - 0.05, leg_radius(legs) + 0.38, 0.32, SIDES),
                      seam=sg.ring_mesh(leg_radius(legs) - 0.02, leg_radius(legs) + 0.03, 0.05, SIDES),
                      wedge=wedge_mesh(), hoop=hoop_mesh())
    build_columns(stage, looks, kit, legs)
    landing_leg = max(legs, key=lambda l: l["top"][0] - l["top"][1])
    build_landing(stage, looks, kit, landing_leg)
    build_braces(kit, legs)
    build_underside(stage, looks, kit, legs)
    build_overhang(stage, looks, kit)
    build_access(kit)
    build_edges(kit)
    build_top_underside(stage, kit)
    calm_floods(stage)
    drips = build_drips(stage, looks, legs)
    total = kit.write()
    stage.OverridePrim(LEGS_MESH).SetActive(False)
    for path in SPLASH_BANDS:
        UsdGeom.Imageable(stage.OverridePrim(path)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().Save()
    print(f"STRUCTURE_WRITTEN {s.LAYER} legs={len(legs)} instances={total} drips={drips}")
    return s.LAYER


def leg_radius(legs):
    return float(np.mean([l["radius"] for l in legs]))


if __name__ == "__main__":
    build()

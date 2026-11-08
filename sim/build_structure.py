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
FLOODS_CALM = ((190000.0, 2400), (130000.0, 4500), (75000.0, 3400))
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
    looks["lamp"] = emissive(stage, "lamp", tuple(c * 25 for c in tf.kelvin_rgb(3600)))
    looks["lamp_green"] = emissive(stage, "lamp_green", (0.4, 14.0, 1.5))
    looks["lamp_yellow"] = emissive(stage, "lamp_yellow", (16.0, 11.0, 0.6))
    looks["window"] = emissive(stage, "window", (0.02, 0.025, 0.03))
    looks["ship_window"] = emissive(stage, "ship_window", (3.2, 2.3, 1.3))
    looks["bridge_glass"] = emissive(stage, "bridge_glass", (0.25, 0.4, 0.45))
    looks["nav_red"] = emissive(stage, "nav_red", (40.0, 1.0, 0.6))
    looks["sign_green"] = emissive(stage, "sign_green", (0.3, 6.0, 1.2))
    looks["nav_green"] = emissive(stage, "nav_green", (0.8, 36.0, 4.0))
    looks["nav_white"] = emissive(stage, "nav_white", (40.0, 40.0, 36.0))
    return looks


def emissive(stage, name, color):
    material = UsdShade.Material.Define(stage, f"{s.LOOKS}/{name}")
    shader = UsdShade.Shader.Define(stage, f"{s.LOOKS}/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


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
    if cone is not None:
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(cone)
        shaping.CreateShapingConeSoftnessAttr(softness)
    return light


def sphere_cap(radius, rows=6, seg=12):
    a = np.linspace(0.0, math.pi / 2, rows + 1)
    t = np.linspace(0.0, math.tau, seg + 1)
    points = np.array([(radius * math.cos(v) * math.cos(u), radius * math.cos(v) * math.sin(u), 0.8 * radius * math.sin(v))
                       for v in a for u in t])
    normals = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-9)
    row = seg + 1
    idx = [k for i in range(rows) for j in range(seg) for k in (i * row + j, i * row + j + 1, (i + 1) * row + j + 1, (i + 1) * row + j)]
    return {"points": points, "normals": normals, "uvs": points[:, :2], "counts": [4] * (rows * seg), "indices": idx}


def retire_old(stage):
    hidden = ["/World/Cinematic/Flare/boom_001", "/World/Cinematic/Flare/brace_002", "/World/Topside/lifeboat",
              "/World/Details/helideck_circle", "/World/Details/helideck_h_001", "/World/Details/helideck_h_002",
              "/World/Details/helideck_h_003", "/World/Cinematic/Ship", "/World/Rig/crane_crane",
              "/World/Rig/derrick_derrick"]
    for path in hidden:
        UsdGeom.Imageable(stage.OverridePrim(path)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    cx, cy = HELI_C
    edge = sg.perimeter_points([tuple(v) for v in sg.hexagon(HELI_C, HELI_R - 0.25, HELI_Z + 0.15)], HELI_R * 6 / 16)
    for k in range(16):
        prim = stage.OverridePrim(f"/World/Cinematic/Lamps/lamp_{930 + k}")
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*map(float, edge[k])))
        set_lamp(stage, f"/World/Cinematic/Lamps/lamp_{930 + k}/light", 2500.0)
    for k in range(12):
        a = math.tau * k / 12
        prim = stage.OverridePrim(f"/World/Cinematic/Lamps/lamp_{950 + k}")
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(cx + 6.9 * math.cos(a), cy + 6.9 * math.sin(a), HELI_Z + 0.12))
        set_lamp(stage, f"/World/Cinematic/Lamps/lamp_{950 + k}/light", 600.0)


def set_lamp(stage, path, intensity):
    prim = stage.OverridePrim(path)
    prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(intensity)
    prim.CreateAttribute("spark:baseIntensity", Sdf.ValueTypeNames.Float, custom=True).Set(intensity)


def calm_floods(stage):
    for n, (intensity, kelvin) in enumerate(FLOODS_CALM):
        prim = stage.OverridePrim(f"/World/Cinematic/Lamps/lamp_{320 + n}/light")
        light = UsdLux.SphereLight(prim)
        light.CreateIntensityAttr(intensity)
        prim.CreateAttribute("spark:baseIntensity", Sdf.ValueTypeNames.Float, custom=True).Set(intensity)
        light.CreateColorAttr(Gf.Vec3f(*tf.kelvin_rgb(kelvin)))


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
        elif proto == "sphere":
            prim = UsdGeom.Sphere.Define(self.stage, path)
            prim.CreateRadiusAttr(1.0)
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
        add_light(stage, light_path("landing", n, n == 0), pos + u * 0.6 - (0, 0, 0.3), 170000.0, kelvin, 85.0)
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
        add_light(stage, light_path("overhang", n, n == 1), (x, y, z - 2.2), 110000.0, KELVINS[n % len(KELVINS)], None)


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
        add_light(stage, light_path("under", n, n < 4), (x, y, zl - 0.35), 140000.0, KELVINS[n % len(KELVINS)], None)


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
        add_light(stage, light_path("top", n, False), (x, -24.0, z - 1.4), 90000.0, KELVINS[n + 1], None)


FLARE_BASE, FLARE_END = np.array((-26.0, 8.0, 42.0)), np.array((-57.2, 8.0, 69.4))
FLARE_STRUT = ((-26.0, 8.0, 31.0), (-46.0, 8.0, 59.5))
HELI_C, HELI_R, HELI_Z = (-22.5, -22.5), 14.43, 51.25
LIFEBOATS = ((-19.0, -40.6, 27.4), (-9.0, -40.6, 27.4))


def heat_look(f):
    return "flare_steel" if f < 0.55 else ("heat" if f < 0.8 else "soot")


def build_flare(stage, kit):
    b = sg.lattice_boom(FLARE_BASE, FLARE_END, 3.2, 1.6)
    for p, q, f in b["chords"]:
        kit.tube(heat_look(f), p, q, 0.15)
    for p, q, f in b["struts"]:
        kit.tube(heat_look(f), p, q, 0.08)
    for p, q, f in b["diagonals"]:
        kit.tube(heat_look(f), p, q, 0.065)
    side, up = b["side"], b["up"]
    walk = b["walk"]
    for (p, f0), (q, f1) in zip(walk, walk[1:]):
        if f1 > 0.92:
            break
        mid, quat, length = tf.segment_pose(p, q)
        kit.add("grating", "box", mid, quat, (length, 0.8, 0.05))
        for sgn in (-1, 1):
            for h in (0.55, 1.1):
                kit.tube(heat_look(f0), p + side * sgn * 0.42 + (0, 0, h), q + side * sgn * 0.42 + (0, 0, h), 0.025)
            kit.tube(heat_look(f0), p + side * sgn * 0.42, p + side * sgn * 0.42 + (0, 0, 1.1), 0.025)
    for k, off in enumerate((-0.35, 0.0, 0.35)):
        pts = [FLARE_BASE + side * off - up * 0.4, FLARE_END + side * off - up * 0.2]
        for f0, f1 in ((0.0, 0.55), (0.55, 0.8), (0.8, 1.0)):
            a0 = pts[0] + (pts[1] - pts[0]) * f0
            a1 = pts[0] + (pts[1] - pts[0]) * f1
            kit.tube(heat_look((f0 + f1) / 2), a0, a1, 0.05 - 0.01 * k)
        kit.tube("soot", pts[1], np.array((-58.3 + 0.25 * off, 8.0 + off, 71.0)), 0.04)
    kit.tube("soot", FLARE_END - (0, 0, 0.6), np.array((-58.4, 8.0, 71.3)), 0.6)
    kit.add("soot", "shroud", (-58.4, 8.0, 70.6))
    kit.tube("soot", np.array((-58.4, 8.0, 69.5)), np.array((-58.4, 8.0, 69.9)), 0.95)
    kit.add("flare_steel", "box", FLARE_BASE - (0, 0, 0.4), scale=(2.2, 3.8, 0.8))
    for dy in (-1.3, 1.3):
        kit.add("flare_steel", "box", FLARE_BASE + (0.6, dy, 0.6), scale=(0.9, 0.12, 1.4))
    p, q = np.array(FLARE_STRUT[0]), np.array(FLARE_STRUT[1])
    kit.tube("flare_steel", p, q, 0.38)
    kit.tube("seam", p, p + (q - p) * 0.06, 0.5)
    kit.add("flare_steel", "box", (-24.6, 8.0, 41.2), scale=(1.4, 1.0, 1.6))


def build_helideck(stage, looks, kit):
    z = HELI_Z + 0.04
    hexv = sg.hexagon(HELI_C, HELI_R, z)
    surface = {"points": np.vstack(([HELI_C[0], HELI_C[1], z], hexv)), "normals": np.tile((0.0, 0.0, 1.0), (7, 1)),
               "uvs": np.vstack(([HELI_C], hexv[:, :2])), "counts": [3] * 6,
               "indices": sum(([0, 1 + k, 1 + (k + 1) % 6] for k in range(6)), [])}
    write_mesh(stage, f"{s.ROOT}/Helideck/surface", surface, looks["helideck"])
    write_mesh(stage, f"{s.ROOT}/Helideck/edge_line", sg.hex_ring(HELI_C, HELI_R - 0.3, HELI_R - 1.3, z + 0.01),
               looks["mark_white"])
    write_mesh(stage, f"{s.ROOT}/Helideck/tdpm", sg.disc_ring(HELI_C, 5.4, 6.4, z + 0.01), looks["mark_yellow"])
    write_mesh(stage, f"{s.ROOT}/Helideck/hcircle", sg.disc_ring(HELI_C, 3.3, 3.6, z + 0.01), looks["mark_white"])
    cx, cy = HELI_C
    for dx, dy, sx, sy in ((-1.1, 0.0, 0.75, 4.0), (1.1, 0.0, 0.75, 4.0), (0.0, 0.0, 1.45, 0.75)):
        kit.add("mark_white", "box", (cx + dx, cy + dy, z + 0.012), scale=(sx, sy, 0.02))
    apothem = HELI_R * math.cos(math.pi / 6)
    for k in range(6):
        a0, a1 = hexv[k], hexv[(k + 1) % 6]
        mid = (a0 + a1) / 2
        out = (mid[:2] - np.array(HELI_C)) / np.linalg.norm(mid[:2] - np.array(HELI_C))
        o3 = np.array((out[0], out[1], 0.0))
        angle = math.degrees(math.atan2(a1[1] - a0[1], a1[0] - a0[0]))
        net = {"points": np.array([a0 + o3 * 0.15 - (0, 0, 0.12), a1 + o3 * 0.15 - (0, 0, 0.12),
                                   a1 + o3 * 1.7 - (0, 0, 0.35), a0 + o3 * 1.7 - (0, 0, 0.35)]),
               "normals": np.tile((0.0, 0.0, 1.0), (4, 1)), "counts": [4], "indices": [0, 1, 2, 3]}
        net["uvs"] = net["points"][:, :2] / 1.2
        write_mesh(stage, f"{s.ROOT}/Helideck/net_{k}", net, looks["net"], double=True)
        kit.member("galv", "box", a0 + o3 * 0.12 - (0, 0, 0.25), a1 + o3 * 0.12 - (0, 0, 0.25), (0.3, 0.25))
        kit.member("galv", "cyl", a0 + o3 * 1.72 - (0, 0, 0.35), a1 + o3 * 1.72 - (0, 0, 0.35), (0.06, 0.06))
        for f in np.linspace(0.0, 1.0, 6):
            p = a0 + (a1 - a0) * f
            kit.member("galv", "box", p - (0, 0, 0.15), p + o3 * 1.72 - (0, 0, 0.38), (0.05, 0.05))
        kit.member("steel", "ibeam", np.array((cx, cy, z - 0.6)), (a0[0], a0[1], z - 0.6), (0.4, 0.8))
        kit.member("steel", "ibeam", a0 - (0, 0, 0.6), a1 - (0, 0, 0.6), (0.4, 0.8))
        inner = np.array((cx, cy, 0.0)) + (a0 - np.array((cx, cy, a0[2]))) * 0.45
        kit.tube("steel", a0 * (1, 1, 0) + (0, 0, z - 1.0), inner * (1, 1, 0) + (0, 0, 44.2), 0.22)
    for p in sg.perimeter_points([tuple(v) for v in hexv], 2.4):
        kit.add("lamp_green", "dome", p + (0, 0, 0.02))
    for k in range(6):
        v = hexv[k]
        d = np.array(HELI_C + (z,)) - v
        d[2] = 0
        d /= np.linalg.norm(d)
        pos = v + (0, 0, 0.25)
        ang = math.degrees(math.atan2(d[1], d[0]))
        kit.add("steel", "box", pos + (0, 0, 0.05), zquat(ang), (0.35, 0.5, 0.25))
        kit.add("lamp_yellow", "box", pos + d * 0.18 + (0, 0, 0.05), zquat(ang), (0.02, 0.42, 0.18))


def build_lifeboats(stage, looks, kit):
    hull = sg.lifeboat(8.0, 2.8, 1.0, 1.55)
    for n, (x, y, z) in enumerate(LIFEBOATS):
        m = dict(hull)
        m["points"] = hull["points"] + (x, y, z)
        write_mesh(stage, f"{s.ROOT}/Lifeboats/boat_{n}", m, looks["lifeboat"])
        for dx in np.arange(-2.4, 2.5, 0.8):
            for side in (-1, 1):
                kit.add("window", "box", (x + dx, y + side * 1.12, z + 0.85), scale=(0.5, 0.06, 0.28))
        kit.add("lifeboat_white", "box", (x - 0.4, y, z + 1.56), scale=(2.6, 1.0, 0.1))
        kit.add("lifeboat_white", "box", (x + 2.6, y, z + 1.45), scale=(0.9, 0.8, 0.22))
        kit.tube("rubber", (x - 3.6, y - 1.38, z + 0.05), (x + 3.6, y - 1.38, z + 0.05), 0.08)
        kit.tube("rubber", (x - 3.6, y + 1.38, z + 0.05), (x + 3.6, y + 1.38, z + 0.05), 0.08)
        for dx in (-2.6, 2.6):
            kit.tube("cable", (x + dx, y, z + 3.6), (x + dx * 0.9, y, z + 1.7), 0.03)
            kit.add("landing", "box", (x + dx * 0.9, y, z + 1.75), scale=(0.25, 0.25, 0.3))


SHIP = (-210.0, 330.0)
BOXES = ("box_red", "box_blue", "box_green", "box_white", "box_orange")
CRANE = (31.39, -4.62)
DERRICK = (20.0, 20.0)
DERRICK_FLOOR, DERRICK_TOP_Z = 46.3, 81.3


def shifted(m, offset):
    out = dict(m)
    out["points"] = np.asarray(m["points"], float) + offset
    return out


def build_ship(stage, looks, kit):
    sx, sy = SHIP
    o = np.array((sx, sy, 0.0))
    ring = sg.ship_planform(80.0, 18.0)
    x = ring[:, 0]
    sheer = 5.2 + 2.6 * np.clip((x - 14.0) / 18.0, 0.0, 1.0) ** 1.5
    write_mesh(stage, f"{s.ROOT}/Ship/boot", shifted(sg.extrude_ring(ring, -4.0, 1.0), o), looks["psv_boot"], double=True)
    write_mesh(stage, f"{s.ROOT}/Ship/topsides", shifted(sg.extrude_ring(ring, 1.0, sheer), o), looks["psv_hull"],
               double=True)
    write_mesh(stage, f"{s.ROOT}/Ship/deck", shifted(sg.fan(ring * 0.985, 4.0), o), looks["psv_deck"])
    bow = ring[ring[:, 0] > 30.0]
    write_mesh(stage, f"{s.ROOT}/Ship/foredeck", shifted(sg.fan(np.vstack((bow, [[30.0, 0.0]])) * 0.985, 7.4), o),
               looks["psv_deck"])
    for (cx, cy, cz), size, colour in sg.container_stacks(-38.0, 12.0, 7.4):
        kit.add(BOXES[colour], "box", (sx + cx, sy + cy, 4.0 + cz), scale=size)
    for centre, size in (((24.0, 0.0, 8.5), (18.0, 16.0, 9.0)), ((25.0, 0.0, 14.6), (12.0, 19.0, 3.2)),
                         ((25.5, 0.0, 16.4), (8.0, 9.0, 0.4))):
        kit.add("psv_white", "box", (sx + centre[0], sy + centre[1], centre[2]), scale=size)
    for z in (5.4, 8.4, 11.4):
        for x in np.arange(16.0, 32.0, 2.0):
            for side in (-1, 1):
                kit.add("ship_window", "box", (sx + x, sy + side * 8.02, z), scale=(1.0, 0.05, 0.7))
        for y in np.arange(-6.5, 7.0, 2.0):
            kit.add("ship_window", "box", (sx + 14.98, sy + y, z), scale=(0.05, 1.0, 0.7))
    kit.add("bridge_glass", "box", (sx + 31.02, sy, 14.9), scale=(0.05, 18.0, 1.4))
    for side in (-1, 1):
        kit.add("bridge_glass", "box", (sx + 25.0, sy + side * 9.52, 14.9), scale=(11.0, 0.05, 1.4))
        kit.add("psv_white", "box", (sx + 16.8, sy + side * 4.5, 15.5), scale=(2.4, 2.0, 5.0))
        kit.add("black", "box", (sx + 16.8, sy + side * 4.5, 18.1), scale=(2.5, 2.1, 0.3))
    mast = np.array((sx + 25.5, sy, 16.6))
    kit.tube("psv_white", mast, mast + (0, 0, 10.5), 0.22)
    kit.member("psv_white", "box", mast + (0, -3.2, 7.0), mast + (0, 3.2, 7.0), (0.15, 0.15))
    kit.member("black", "box", mast + (0.6, -2.2, 5.2), mast + (0.6, 2.2, 5.2), (0.25, 0.12))
    kit.member("black", "box", mast + (-0.6, -1.8, 8.0), mast + (-0.6, 1.8, 8.0), (0.2, 0.1))
    kit.add("nav_white", "sphere", mast + (0, 0, 10.7), scale=(0.25, 0.25, 0.25))
    kit.add("nav_red", "sphere", (sx + 25.0, sy + 9.7, 15.6), scale=(0.22, 0.22, 0.22))
    kit.add("nav_green", "sphere", (sx + 25.0, sy - 9.7, 15.6), scale=(0.22, 0.22, 0.22))
    kit.add("nav_white", "sphere", (sx - 39.6, sy, 6.0), scale=(0.2, 0.2, 0.2))
    for n, (lx, ly) in enumerate(((-34.0, -8.4), (-34.0, 8.4), (-6.0, -8.4), (-6.0, 8.4))):
        kit.tube("psv_white", (sx + lx, sy + ly, 4.0), (sx + lx, sy + ly, 11.6), 0.12)
        prim = stage.OverridePrim(f"/World/Cinematic/Lamps/lamp_{910 + n}")
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(sx + lx, sy + ly * 0.95, 12.0))
    for n, (lx, kelvin) in enumerate(((14.6, 4200), (14.6, 3000))):
        add_light(stage, light_path("ship", n, False), (sx + lx, sy + (-4.0 if n else 4.0), 12.6), 120000.0, kelvin, 80.0)
    add_light(stage, light_path("ship_mast", 0, False), mast + (0, 0, 10.7), 4000.0, 6000, None)
    wake = {"points": np.array([(sx - 40.5, sy - 5.0, 0.42), (sx - 40.5, sy + 5.0, 0.42), (sx - 75.0, sy + 12.0, 0.42),
                                (sx - 75.0, sy - 12.0, 0.42)]),
            "normals": np.tile((0.0, 0.0, 1.0), (4, 1)), "counts": [4], "indices": [0, 3, 2, 1]}
    wake["uvs"] = wake["points"][:, :2] / 9.0
    write_mesh(stage, f"{s.ROOT}/Ship/wake", wake, UsdShade.Material(stage.OverridePrim("/World/Looks/cinematic_foam")))


def build_crane(kit):
    cx, cy = CRANE
    kit.tube("crane_yellow", (cx, cy, 42.5), (cx, cy, 47.2), 1.5)
    kit.tube("steel", (cx, cy, 47.2), (cx, cy, 47.55), 1.9)
    kit.add("crane_yellow", "box", (cx + 1.2, cy, 48.95), scale=(4.6, 3.6, 2.8))
    kit.add("black", "box", (cx + 3.8, cy, 48.5), scale=(1.2, 3.4, 1.8))
    kit.add("crane_yellow", "box", (cx - 1.2, cy - 2.3, 48.75), scale=(2.2, 1.6, 2.3))
    kit.add("window", "box", (cx - 2.32, cy - 2.3, 49.1), scale=(0.05, 1.3, 1.1))
    kit.add("window", "box", (cx - 1.2, cy - 3.12, 49.1), scale=(1.8, 0.05, 1.0))
    apex = np.array((cx + 1.6, cy, 56.5))
    for dy in (-1.4, 1.4):
        kit.tube("crane_yellow", (cx + 3.2, cy + dy, 50.35), apex + (0, dy * 0.25, 0), 0.17)
        kit.tube("crane_yellow", (cx + 0.2, cy + dy, 50.35), apex + (0, dy * 0.25, 0), 0.17)
    heel = np.array((cx - 0.6, cy, 49.3))
    ang = math.radians(38.0)
    tip = heel + 26.0 * np.array((math.cos(ang), 0.0, math.sin(ang)))
    boom = sg.rect_lattice(heel, tip, 1.7, 0.8, panel=2.0, depth_ratio=0.8)
    for p, q in boom["chords"]:
        kit.tube("crane_yellow", p, q, 0.1)
    for p, q in boom["girts"] + boom["diagonals"]:
        kit.tube("crane_yellow", p, q, 0.045)
    kit.member("steel", "cyl", tip + (0, -0.5, 0.2), tip + (0, 0.5, 0.2), (0.8, 0.8))
    for dy in (-0.3, 0.3):
        kit.tube("cable", apex + (0, dy, 0), tip + (0, dy, 0.3), 0.025)
    hook = tip - (0, 0, 9.0)
    for dy in (-0.12, 0.12):
        kit.tube("cable", tip + (0, dy, -0.2), hook + (0, dy, 0.6), 0.02)
    kit.add("crane_yellow", "box", hook + (0, 0, 0.3), scale=(0.55, 0.45, 0.9))
    kit.tube("black", hook - (0, 0, 0.15), hook - (0, 0, 0.75), 0.1)
    kit.add("lamp", "box", (cx - 0.1, cy - 2.3, 50.0), scale=(0.3, 0.3, 0.05))


def build_derrick(stage, kit):
    dx, dy = DERRICK
    for x in (15.0, 25.0):
        for y in (15.0, 25.0):
            kit.member("derrick_white", "ibeam", (x, y, 42.5), (x, y, 45.9), (0.4, 0.4))
    for (x0, y0), (x1, y1) in (((15, 15), (25, 15)), ((25, 15), (25, 25)), ((25, 25), (15, 25)), ((15, 25), (15, 15))):
        kit.tube("derrick_white", (x0, y0, 42.6), (x1, y1, 45.8), 0.1)
        kit.tube("derrick_white", (x1, y1, 42.6), (x0, y0, 45.8), 0.1)
    kit.add("steel", "box", (dx, dy, 46.05), scale=(11.0, 11.0, 0.4))
    kit.add("grating", "box", (dx, dy, 46.28), scale=(10.8, 10.8, 0.05))
    kit.add("derrick_white", "box", (dx - 3.4, dy + 4.0, 47.7), scale=(3.4, 2.6, 2.8))
    kit.add("window", "box", (dx - 1.68, dy + 4.0, 47.9), scale=(0.05, 1.6, 0.9))
    kit.member("steel", "box", (dx + 5.3, dy, 46.3), (dx + 11.5, dy, 42.8), (1.2, 0.25))
    tower = sg.rect_lattice((dx, dy, DERRICK_FLOOR), (dx, dy, DERRICK_TOP_Z), 7.0, 2.2, panel=3.2, cross=True)
    for p, q in tower["chords"]:
        kit.tube("derrick_white", p, q, 0.2)
    for p, q in tower["girts"]:
        kit.tube("derrick_white", p, q, 0.08)
    for p, q in tower["diagonals"]:
        kit.tube("derrick_white", p, q, 0.05)

    def half(z):
        return 3.5 + (1.1 - 3.5) * (z - DERRICK_FLOOR) / (DERRICK_TOP_Z - DERRICK_FLOOR)

    zb = 69.0
    hb = half(zb)
    kit.add("grating", "box", (dx + hb + 0.9, dy, zb), scale=(1.8, 2.8, 0.08))
    for k in np.linspace(-1.1, 1.1, 7):
        kit.add("steel", "box", (dx + hb + 1.3, dy + k, zb + 0.15), scale=(1.4, 0.06, 0.12))
    for side in (-1.4, 1.4):
        kit.tube("crane_yellow", (dx + hb, dy + side, zb + 1.1), (dx + hb + 1.8, dy + side, zb + 1.1), 0.03)
        kit.tube("crane_yellow", (dx + hb + 1.8, dy + side, zb), (dx + hb + 1.8, dy + side, zb + 1.1), 0.03)
    for i, x in enumerate(np.linspace(dx + 1.2, dx + 2.6, 4)):
        for y in np.linspace(dy - 1.0, dy + 1.0, 6):
            kit.tube("pipe_grey", (x, y, DERRICK_FLOOR + 0.1), (x + 0.15 * (hb + 1.2 - (x - dx)), y, zb + 1.4), 0.065)
    zc = DERRICK_TOP_Z
    kit.add("grating", "box", (dx, dy, zc + 0.05), scale=(3.6, 3.6, 0.08))
    corners = [(dx - 1.8, dy - 1.8), (dx + 1.8, dy - 1.8), (dx + 1.8, dy + 1.8), (dx - 1.8, dy + 1.8)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        kit.tube("crane_yellow", (x0, y0, zc + 1.1), (x1, y1, zc + 1.1), 0.03)
        kit.tube("crane_yellow", (x0, y0, zc), (x0, y0, zc + 1.1), 0.03)
    kit.add("derrick_white", "box", (dx, dy, zc + 0.75), scale=(2.4, 2.0, 1.2))
    for k in np.linspace(-0.75, 0.75, 4):
        kit.member("steel", "cyl", (dx + k - 0.08, dy, zc + 1.0), (dx + k + 0.08, dy, zc + 1.0), (1.1, 1.1))
    for x, y in corners:
        kit.add("lamp", "box", (x, y, zc + 1.25), scale=(0.25, 0.25, 0.12))
    zt = 60.0
    for k in (-0.3, -0.1, 0.1, 0.3):
        kit.tube("cable", (dx + k, dy, zc + 0.5), (dx + k, dy, zt + 1.2), 0.018)
    kit.add("crane_yellow", "box", (dx, dy, zt), scale=(1.0, 1.3, 2.4))
    kit.add("black", "box", (dx, dy, zt - 2.4), scale=(1.4, 1.2, 2.2))
    kit.tube("pipe_grey", (dx, dy, DERRICK_FLOOR + 0.3), (dx, dy, zt - 3.5), 0.1)
    kit.add("black", "box", (dx, dy, DERRICK_FLOOR + 0.6), scale=(1.4, 1.4, 0.5))
    for n, (lamp, z, corner) in enumerate(((903, 66.0, (-1, -1)), (904, 66.0, (1, 1)), (905, 54.0, (-1, 1)), (906, 54.0, (1, -1)))):
        h = half(z) + 0.35
        prim = stage.OverridePrim(f"/World/Cinematic/Lamps/lamp_{lamp}")
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(dx + corner[0] * h, dy + corner[1] * h, z))


DECK_TOP, RAISED = 42.5, 51.25
TRIM = (
    ((-1.0, 9.0), (11.0, 25.5), 42.6), ((0.5, -1.0), (9.5, 9.0), 42.6), ((9.0, 0.0), (21.0, 8.0), 42.6),
    ((22.0, 1.0), (32.0, 8.0), 42.6), ((11.0, 9.0), (22.0, 16.0), 42.6), ((17.5, 17.5), (22.5, 22.5), 42.6),
    ((-25.5, 4.5), (-4.5, 25.5), 51.4), ((-7.6, -25.1), (0.1, -19.9), 42.6), ((2.4, -25.1), (7.6, -19.9), 42.6),
    ((3.2, -18.6), (7.3, -9.2), 42.6),
)


def trim_stl(stage):
    full = Usd.Stage.Open(str(pr.OUTPUT / "full_rig.usda"), Usd.Stage.LoadNone)
    src = UsdGeom.Mesh(full.GetPrimAtPath("/World/Rig/upper_structure_frame"))
    pts = np.array(src.GetPointsAttr().Get())
    idx = np.array(src.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    normals = np.array(src.GetNormalsAttr().Get()).reshape(-1, 3, 3)
    st = np.array(UsdGeom.PrimvarsAPI(src).GetPrimvar("st").Get()).reshape(-1, 3, 2)
    c = pts[idx].mean(1)
    drop = np.zeros(len(idx), bool)
    for (x0, y0), (x1, y1), z in TRIM:
        drop |= (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z)
    keep = ~drop
    over = UsdGeom.Mesh(stage.OverridePrim("/World/Rig/upper_structure_frame"))
    over.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(keep.sum(), 3, np.int32)))
    over.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx[keep].reshape(-1).astype(np.int32)))
    over.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals[keep].reshape(-1, 3).astype(np.float32)))
    UsdGeom.PrimvarsAPI(over).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        Vt.Vec2fArray.FromNumpy(st[keep].reshape(-1, 2).astype(np.float32)))
    UsdGeom.Imageable(stage.OverridePrim("/World/Rig/pipework_22_pipe")).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return int(drop.sum())


def vessel(stage, looks, name, centre, axis, length, radius, look):
    m = sg.orient(sg.lathe(sg.vessel_profile(length, radius), 32), centre, axis)
    write_mesh(stage, f"{s.ROOT}/Process/{name}", m, looks[look])


def nozzle(kit, base, direction, r, length, look="proc_grey"):
    base, d = np.asarray(base, float), np.asarray(direction, float)
    kit.tube(look, base, base + d * length, r)
    kit.tube(look, base + d * (length - 0.06), base + d * length, r + 0.08)


def rails_around(kit, corners, z, look="crane_yellow"):
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        for h in (0.55, 1.1):
            kit.tube(look, (x0, y0, z + h), (x1, y1, z + h), 0.025)
        kit.tube(look, (x0, y0, z), (x0, y0, z + 1.1), 0.03)


def build_tank(stage, looks, kit, okit):
    cx, cy, r, h = 5.5, 18.5, 4.0, 8.5
    prof = [(0.0, r), (h, r), (h + 0.05, r - 0.05), (h + 0.8, 0.4), (h + 0.85, 0.0)]
    m = sg.orient(sg.lathe(prof, 48), (cx, cy, DECK_TOP), (0, 0, 1))
    write_mesh(stage, f"{s.ROOT}/Process/tank", m, looks["proc_white"])
    for z in (DECK_TOP + 0.15, DECK_TOP + 3.0, DECK_TOP + 6.0, DECK_TOP + h - 0.1):
        kit.add("proc_grey", "tank_ring", (cx, cy, z))
    sp = sg.spiral_stair((cx, cy), r, DECK_TOP, DECK_TOP + h, sweep=0.085)
    for pos, ang in sp["steps"]:
        okit.add("grating", "box", pos, zquat(ang), (0.9, 0.28, 0.04))
    for (pos, ang), (q, _) in zip(sp["steps"][::4], sp["steps"][4::4]):
        kit.tube("crane_yellow", pos, q, 0.06)
    for a, b in zip(sp["rail"], sp["rail"][1:]):
        kit.tube("crane_yellow", a, b, 0.025)
    for k in range(24):
        a0, a1 = math.tau * k / 24, math.tau * (k + 1) / 24
        p = (cx + (r - 0.3) * math.cos(a0), cy + (r - 0.3) * math.sin(a0), DECK_TOP + h + 0.1)
        q = (cx + (r - 0.3) * math.cos(a1), cy + (r - 0.3) * math.sin(a1), DECK_TOP + h + 0.1)
        kit.tube("crane_yellow", np.add(p, (0, 0, 1.1)), np.add(q, (0, 0, 1.1)), 0.025)
        if k % 3 == 0:
            kit.tube("crane_yellow", p, np.add(p, (0, 0, 1.1)), 0.03)
    for ang, z in ((200.0, 43.4), (250.0, 43.0), (290.0, 49.5)):
        u = np.array((math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0.0))
        nozzle(kit, np.array((cx, cy, z)) + u * (r - 0.05), u, 0.3 if z < 44 else 0.15, 0.7)
    kit.tube("pipe_grey", (cx + r * math.cos(4.0), cy + r * math.sin(4.0), DECK_TOP + h + 0.4),
             (cx + r * math.cos(4.0), cy + r * math.sin(4.0), DECK_TOP + h + 1.8), 0.12)


def build_separator(stage, looks, kit, okit):
    cx, cy, cz, length, r = 6.75, 3.6, DECK_TOP + 2.4, 8.5, 1.5
    vessel(stage, looks, "separator", (cx, cy, cz), (1, 0, 0), length, r, "proc_white")
    for x in (cx - 2.8, cx + 2.8):
        kit.add("proc_grey", "box", (x, cy, DECK_TOP + 0.5), scale=(0.45, 2.6, 1.0))
        kit.add("proc_grey", "box", (x, cy, DECK_TOP + 1.05), scale=(0.6, 2.9, 0.12))
    top = cz + r
    for x, nr in ((cx - 3.2, 0.25), (cx - 1.2, 0.18), (cx + 1.6, 0.3), (cx + 3.4, 0.15)):
        nozzle(kit, (x, cy, top - 0.1), (0, 0, 1), nr, 0.6)
    kit.add("crane_yellow", "box", (cx + 0.3, cy + 0.6, top + 0.55), scale=(0.35, 0.35, 0.7))
    kit.tube("pipe_red", (cx + 0.3, cy + 0.6, top + 0.9), (cx + 0.3, cy + 0.6, top + 2.6), 0.07)
    kit.add("crane_yellow", "box", (cx - 0.5, cy + 0.6, top + 0.55), scale=(0.35, 0.35, 0.7))
    kit.tube("pipe_red", (cx - 0.5, cy + 0.6, top + 0.9), (cx - 0.5, cy + 0.6, top + 2.6), 0.07)
    end = cx + length / 2 + r * 0.5
    nozzle(kit, (end - 0.1, cy, cz), (1, 0, 0), 0.4, 0.5)
    kit.add("proc_grey", "box", (end + 0.45, cy, cz), scale=(0.08, 1.0, 1.0))
    for x in (cx - 1.0, cx + 1.0):
        y = cy - r - 0.35
        kit.tube("galv", (x, y, cz - 0.9), (x, y, cz + 0.9), 0.05)
        for z in (cz - 0.9, cz + 0.9):
            kit.tube("galv", (x, y, z), (x, cy - r + 0.05, z), 0.03)
    z = top + 0.35
    kit.add("grating", "box", (cx, cy - 0.2, z), scale=(7.0, 1.4, 0.06))
    for x in np.linspace(cx - 3.4, cx + 3.4, 5):
        kit.tube("proc_grey", (x, cy - 0.85, z), (x, cy - 0.85, cz + 0.4), 0.06)
    rails_around(kit, [(cx - 3.5, cy - 0.9), (cx + 3.5, cy - 0.9), (cx + 3.5, cy + 0.5), (cx - 3.5, cy + 0.5)], z)
    lad = sg.caged_ladder((cx - 3.9, cy - 0.2), DECK_TOP, z, 180.0)
    for p, q in lad["rails"] + lad["straps"]:
        kit.tube("crane_yellow", p, q, 0.03)
    for p, q in lad["rungs"]:
        okit.tube("crane_yellow", p, q, 0.02)
    for pos, a in lad["hoops"]:
        kit.add("crane_yellow", "hoop", pos, zquat(a))
    for x, look, r_ in ((cx - 3.2, "pipe_grey", 0.2), (cx + 1.6, "pipe_green", 0.22), (cx + 3.4, "pipe_yellow", 0.12)):
        pipe_run(stage, looks, f"sep_{look}", [(x, cy, top + 0.5), (x, cy, 47.3), (x, 7.2, 47.3), (x, 7.2, 46.6 + r_),
                                                 (x + 0.8, 7.9, 46.6 + r_)], r_, look)


def build_scrubbers(stage, looks, kit, okit):
    for n, (x, y, r, length, look) in enumerate(((14.0, 4.8, 0.9, 6.0, "proc_white"), (17.0, 4.8, 0.75, 5.0, "proc_green"))):
        base = DECK_TOP + 0.8
        kit.tube("proc_grey", (x, y, DECK_TOP), (x, y, base), r)
        centre = base + r * 0.5 + length / 2
        vessel(stage, looks, f"scrubber_{n}", (x, y, centre), (0, 0, 1), length, r, look)
        topz = base + length + r
        for z, d in ((base + 1.5, (0, -1, 0)), (base + length * 0.7, (-1, 0, 0)), (base + length * 0.5, (1, 0, 0))):
            nozzle(kit, np.array((x, y, z)) + np.array(d) * (r - 0.05), d, 0.15, 0.5)
        nozzle(kit, (x, y, topz - 0.05), (0, 0, 1), 0.15, 0.5)
        pz = base + length * 0.85
        kit.add("grating", "box", (x, y + r + 0.55, pz), scale=(2.2, 1.0, 0.06))
        rails_around(kit, [(x - 1.1, y + r + 0.05), (x + 1.1, y + r + 0.05), (x + 1.1, y + r + 1.05),
                           (x - 1.1, y + r + 1.05)], pz)
        lad = sg.caged_ladder((x + 0.6, y + r + 1.3), DECK_TOP, pz, 90.0)
        for p, q in lad["rails"] + lad["straps"]:
            kit.tube("crane_yellow", p, q, 0.03)
        for p, q in lad["rungs"]:
            okit.tube("crane_yellow", p, q, 0.02)
        for pos, a in lad["hoops"]:
            kit.add("crane_yellow", "hoop", pos, zquat(a))
        zt = 49.4 if n else 50.0
        pipe_run(stage, looks, f"scrub_{n}", [(x, y, topz + 0.45), (x, y, zt), (x, 9.0 + 0.6 * (1 - 2 * n), zt),
                                              (x, 9.0 + 0.6 * (1 - 2 * n), 48.55), (x + 1.2, 9.0 + 0.6 * (1 - 2 * n), 48.55)],
                 0.15, "pipe_grey")
    vessel(stage, looks, "ko_drum", (15.0, 1.7, DECK_TOP + 1.4), (1, 0, 0), 3.2, 0.7, "proc_grey")
    for x in (14.0, 16.0):
        kit.add("proc_grey", "box", (x, 1.7, DECK_TOP + 0.35), scale=(0.3, 1.3, 0.7))
    nozzle(kit, (15.0, 1.7, DECK_TOP + 2.05), (0, 0, 1), 0.15, 0.5)


def build_rack(kit, okit):
    rack = sg.pipe_rack_bents(0.5, 22.0, 9.0, 2.6, (46.2, 48.0), DECK_TOP, bay=5.4)
    for p, q in rack["columns"]:
        kit.member("proc_grey", "ibeam", p, q, (0.3, 0.3))
    for p, q in rack["beams"] + rack["ties"]:
        kit.member("proc_grey", "ibeam", p, q, (0.2, 0.3))
    for x in rack["xs"]:
        kit.tube("proc_grey", (x, 7.7, DECK_TOP + 0.3), (x, 10.3, 45.8), 0.06)
    x0, x1 = rack["xs"][0], rack["xs"][-1]
    for z, pipes in ((46.2, ((-1.0, 0.22, "pipe_grey"), (-0.45, 0.15, "pipe_green"), (0.05, 0.28, "pipe_yellow"),
                              (0.55, 0.12, "galv"), (0.95, 0.18, "pipe_red"))),
                     (48.0, ((-0.9, 0.2, "pipe_grey"), (-0.35, 0.14, "pipe_green"), (0.25, 0.25, "pipe_grey")))):
        for dy, r, look in pipes:
            kit.tube(look, (x0 - 0.3, 9.0 + dy, z + 0.15 + r), (x1 + 0.3, 9.0 + dy, z + 0.15 + r), r)
            for x in rack["xs"]:
                kit.add("proc_grey", "box", (x, 9.0 + dy, z + 0.2), scale=(0.3, 0.12, 0.1))
    zc = 48.0 + 0.9
    kit.member("galv", "box", (x0, 9.0 + 1.0, zc), (x1, 9.0 + 1.0, zc), (0.02, 0.12))
    kit.member("galv", "box", (x0, 9.0 + 0.4, zc), (x1, 9.0 + 0.4, zc), (0.02, 0.12))
    for x in np.arange(x0, x1, 0.3):
        okit.add("galv", "box", (x, 9.7, zc - 0.05), scale=(0.04, 0.6, 0.03))
    for k, dy in enumerate((0.55, 0.7, 0.85)):
        kit.tube("cable", (x0, 9.0 + dy, zc + 0.02), (x1, 9.0 + dy, zc + 0.02), 0.04 + 0.01 * k)
    for x in rack["xs"]:
        kit.tube("galv", (x, 9.4, zc), (x, 9.4, 48.3), 0.025)
        kit.tube("galv", (x, 10.0, zc), (x, 10.0, 48.3), 0.025)


def ehouse(stage, kit, okit, centre, angle, look, n):
    c = np.asarray(centre, float)
    q = zquat(angle)
    u = np.array((math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0))
    v = np.array((-u[1], u[0], 0.0))
    kit.add(look, "box", c + (0, 0, 1.45), q, (6.06, 2.44, 2.9))
    kit.add("proc_grey", "box", c + (0, 0, 0.1), q, (6.2, 2.6, 0.2))
    door = c - v * 1.23 + u * 1.5 + (0, 0, 1.2)
    kit.add("proc_grey", "box", door, q, (0.95, 0.05, 2.0))
    kit.add("lamp", "box", door + (0, 0, 1.25) - v * 0.06, q, (0.35, 0.08, 0.12))
    kit.add("window", "box", c - v * 1.23 - u * 1.2 + (0, 0, 1.8), q, (1.0, 0.04, 0.6))
    ac = c + u * 3.3 + (0, 0, 1.6)
    kit.add("proc_white", "box", ac, q, (0.6, 1.6, 1.2))
    okit.add("black", "box", ac + u * 0.31, q, (0.02, 1.1, 0.8))
    add_light(stage, light_path("ehouse", n, False), door + (0, 0, 1.0) - v * 0.5, 25000.0, 4200, 80.0)


def build_compressor(stage, kit, okit):
    cx, cy, w, d, h = -15.5, 13.0, 11.0, 5.5, 4.5
    z0 = RAISED
    kit.add("proc_grey", "box", (cx, cy, z0 + 0.15), scale=(w + 1.0, d + 1.0, 0.3))
    kit.add("proc_white", "box", (cx, cy, z0 + 0.3 + h / 2), scale=(w, d, h))
    for side in (-1, 1):
        y = cy + side * (d / 2 + 0.08)
        for x0 in (cx - 4.0, cx + 1.5):
            kit.add("proc_grey", "box", (x0 + 1.25, y, z0 + 2.6), scale=(2.7, 0.06, 3.0))
            for z in np.arange(z0 + 1.3, z0 + 4.0, 0.22):
                okit.add("galv", "box", (x0 + 1.25, y + side * 0.06, z), (math.cos(0.26), side * math.sin(0.26), 0.0, 0.0),
                         (2.5, 0.03, 0.2))
    roof = z0 + 0.3 + h
    kit.add("proc_grey", "box", (cx - 3.8, cy, roof + 1.2), scale=(3.0, d - 0.4, 2.4))
    for z in np.arange(roof + 0.3, roof + 2.3, 0.22):
        okit.add("galv", "box", (cx - 5.32, cy, z), (math.cos(0.26), 0.0, math.sin(0.26), 0.0), (0.03, d - 0.6, 0.2))
    for n, x in enumerate((cx + 0.5, cx + 3.2)):
        kit.tube("proc_grey", (x, cy, roof), (x, cy, roof + 6.5), 0.65)
        kit.tube("soot", (x, cy, roof + 6.5), (x, cy, roof + 7.6), 0.68)
        kit.tube("proc_grey", (x, cy, roof + 2.0), (x, cy, roof + 2.4), 0.8)
        kit.tube("proc_grey", (x, cy, roof), (x, cy, roof + 0.4), 0.9)
    ox, oy = cx, cy + d / 2 + 2.3
    kit.add("proc_green", "box", (ox, oy, z0 + 1.2), scale=(8.0, 2.8, 2.0))
    for x in (ox - 2.6, ox, ox + 2.6):
        kit.tube("galv", (x, oy, z0 + 2.2), (x, oy, z0 + 2.45), 1.15)
        okit.add("black", "box", (x, oy, z0 + 2.47), scale=(2.0, 0.08, 0.04))
        okit.add("black", "box", (x, oy, z0 + 2.47), scale=(0.08, 2.0, 0.04))
    for x in (cx - 5.6, cx + 5.6):
        kit.tube("pipe_yellow", (x, cy - 1.5, z0 + 1.0), (x, cy - 1.5, z0 + 3.4), 0.2)
        kit.tube("pipe_yellow", (x, cy - 1.5, z0 + 3.4), (x + (1.2 if x > cx else -1.2), cy - 1.5, z0 + 3.4), 0.2)
    rails_around(kit, [(-24.6, 5.4), (-5.4, 5.4), (-5.4, 24.6), (-24.6, 24.6)], z0)


UTILITIES = ((-24.5, -17.4, 42.6), (-5.5, 1.4, 48.7))
UTILITY_LOOKS = ("machine_green", "vessel_grey", "vessel_white", "black", "pipe_grey", "valve_red")


def hide_topside_utilities(stage):
    full = Usd.Stage.Open(str(pr.OUTPUT / "full_rig.usda"), Usd.Stage.LoadNone)
    (x0, y0, z0), (x1, y1, z1) = UTILITIES
    hidden = 0
    for name in UTILITY_LOOKS:
        pi = UsdGeom.PointInstancer(full.GetPrimAtPath(f"/World/Topside/{name}"))
        if not pi:
            continue
        pos = np.array(pi.GetPositionsAttr().Get())
        sel = np.flatnonzero((pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 1] > y0) & (pos[:, 1] < y1) &
                             (pos[:, 2] > z0) & (pos[:, 2] < z1))
        if len(sel):
            over = UsdGeom.PointInstancer(stage.OverridePrim(f"/World/Topside/{name}"))
            over.CreateInvisibleIdsAttr(Vt.Int64Array([int(i) for i in sel]))
            hidden += len(sel)
    return hidden


def build_gensets(stage, looks, kit, okit):
    y = -14.4
    for n, x in enumerate((-21.9, -17.4, -12.9, -8.4)):
        z = DECK_TOP
        kit.add("steel", "box", (x, y, z + 0.13), scale=(3.9, 2.4, 0.25))
        look = "proc_green" if n % 2 else "proc_grey"
        kit.add(look, "box", (x - 0.3, y, z + 1.3), scale=(2.8, 2.1, 2.1))
        for side in (-1, 1):
            for zz in np.arange(z + 0.6, z + 2.2, 0.2):
                okit.add("galv", "box", (x - 0.3, y + side * 1.08, zz), (math.cos(0.26), side * math.sin(0.26), 0.0, 0.0),
                         (2.4, 0.03, 0.17))
        kit.add("proc_grey", "box", (x + 1.45, y, z + 1.35), scale=(0.5, 2.2, 2.2))
        for zz in np.arange(z + 0.45, z + 2.3, 0.18):
            okit.add("black", "box", (x + 1.71, y, zz), scale=(0.03, 2.0, 0.06))
        kit.member("proc_grey", "cyl", (x - 1.3, y + 0.5, z + 2.75), (x + 0.5, y + 0.5, z + 2.75), (0.55, 0.55))
        kit.tube("proc_grey", (x - 1.0, y + 0.5, z + 2.35), (x - 1.0, y + 0.5, z + 2.75), 0.12)
        kit.tube("proc_grey", (x + 0.5, y + 0.5, z + 2.75), (x + 0.5, y + 0.5, 47.9), 0.18)
        kit.tube("soot", (x + 0.5, y + 0.5, 47.6), (x + 0.5, y + 0.5, 47.95), 0.2)
        kit.add("crane_yellow", "box", (x - 1.85, y - 0.8, z + 1.0), scale=(0.25, 0.4, 0.6))
    cx, cy = -15.0, -1.6
    vessel(stage, looks, "day_tank", (cx, cy, DECK_TOP + 1.5), (1, 0, 0), 7.6, 0.9, "proc_white")
    for x in (cx - 2.5, cx, cx + 2.6):
        nozzle(kit, (x, cy, DECK_TOP + 2.35), (0, 0, 1), 0.12, 0.45)
    kit.tube("galv", (cx + 1.2, cy - 0.98, DECK_TOP + 0.9), (cx + 1.2, cy - 0.98, DECK_TOP + 2.1), 0.04)
    kit.tube("pipe_yellow", (cx - 4.4, cy, DECK_TOP + 1.5), (cx - 4.9, cy, DECK_TOP + 1.5), 0.1)
    kit.tube("pipe_yellow", (cx - 4.9, cy, DECK_TOP + 1.5), (cx - 4.9, -13.0, DECK_TOP + 1.5), 0.1)


def pipe_run(stage, looks, name, points, r, look):
    path = sg.route(points, max(3.0 * r, 0.45))
    write_mesh(stage, f"{s.ROOT}/Process/pipe_{name}", dg.tube(path, r, 12), looks[look])


def basket(kit, okit, centre, size, angle=0.0):
    c = np.asarray(centre, float)
    u = np.array((math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0))
    v = np.array((-u[1], u[0], 0.0))
    lx, ly, h = size
    corners = [c + u * sx * lx / 2 + v * sy * ly / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    kit.add("proc_grey", "box", c + (0, 0, 0.08), zquat(angle), (lx, ly, 0.16))
    for k, p in enumerate(corners):
        q = corners[(k + 1) % 4]
        kit.tube("crane_yellow", p, p + (0, 0, h), 0.06)
        kit.tube("crane_yellow", p + (0, 0, h), q + (0, 0, h), 0.05)
        mid, quat, length = tf.segment_pose(p + (0, 0, h / 2), q + (0, 0, h / 2))
        kit.add("grating", "box", mid, quat, (length, 0.03, h))
        kit.tube("cable", p + (0, 0, h), c + (0, 0, h + 2.2), 0.015)
    kit.add("crane_yellow", "box", c + (0, 0, h + 2.25), scale=(0.18, 0.08, 0.25))
    for k in range(5):
        y = (k - 2) * ly / 6
        kit.tube("pipe_grey", c - u * lx * 0.42 + v * y + (0, 0, 0.35 + 0.18 * (k % 2)),
                 c + u * lx * 0.42 + v * y + (0, 0, 0.35 + 0.18 * (k % 2)), 0.09)
    kit.add("box_white", "box", c + v * ly * 0.25 + (0, 0, 0.6), zquat(angle), (lx * 0.35, ly * 0.3, 0.7))


def container10(kit, okit, centre, angle, look):
    c = np.asarray(centre, float)
    q = zquat(angle)
    u = np.array((math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0))
    kit.add(look, "box", c + (0, 0, 1.295), q, (2.99, 2.44, 2.59))
    kit.add("proc_grey", "box", c + u * 1.5 + (0, 0, 1.295), q, (0.04, 2.3, 2.45))
    for dy in (-0.5, 0.5):
        okit.add("galv", "box", c + u * 1.53 + np.array((-u[1], u[0], 0.0)) * dy + (0, 0, 1.3), q, (0.05, 0.05, 2.3))


def fire_pump_house(stage, kit, okit):
    cx, cy, w, d, h = 4.8, -22.5, 4.4, 5.0, 3.2
    z = DECK_TOP
    kit.add("fire_red", "box", (cx, cy, z + h / 2), scale=(w, d, h))
    kit.add("proc_grey", "box", (cx, cy, z + h + 0.08), scale=(w + 0.4, d + 0.4, 0.16))
    for x in np.arange(cx - w / 2 + 0.4, cx + w / 2, 0.8):
        for side in (-1, 1):
            kit.add("fire_red", "box", (x, cy + side * (d / 2 + 0.03), z + h / 2), scale=(0.08, 0.06, h - 0.2))
    kit.add("proc_grey", "box", (cx + 0.4, cy - d / 2 - 0.06, z + 1.25), scale=(2.2, 0.06, 2.5))
    for zz in np.arange(z + 0.2, z + 2.5, 0.12):
        okit.add("galv", "box", (cx + 0.4, cy - d / 2 - 0.1, zz), scale=(2.15, 0.03, 0.05))
    kit.add("proc_grey", "box", (cx - w / 2 - 0.04, cy + 1.2, z + 1.05), scale=(0.06, 0.95, 2.1))
    kit.add("lamp", "box", (cx - w / 2 - 0.12, cy + 1.2, z + 2.4), scale=(0.12, 0.35, 0.12))
    kit.add("mark_white", "box", (cx - 1.2, cy - d / 2 - 0.05, z + 2.75), scale=(1.4, 0.03, 0.45))
    kit.add("proc_grey", "box", (cx + w / 2 + 0.05, cy, z + 2.0), scale=(0.08, 2.4, 1.4))
    for zz in np.arange(z + 1.4, z + 2.7, 0.18):
        okit.add("galv", "box", (cx + w / 2 + 0.1, cy, zz), (math.cos(0.26), 0.0, math.sin(0.26), 0.0), (0.03, 2.3, 0.15))
    kit.tube("proc_grey", (cx + 1.2, cy + 1.4, z + h), (cx + 1.2, cy + 1.4, z + h + 2.2), 0.18)
    kit.tube("soot", (cx + 1.2, cy + 1.4, z + h + 2.2), (cx + 1.2, cy + 1.4, z + h + 2.5), 0.24)
    for dx in (-1.0, 0.0):
        kit.tube("galv", (cx + dx, cy - 1.0, z + h), (cx + dx, cy - 1.0, z + h + 0.55), 0.3)
    add_light(stage, light_path("pumphouse", 0, False), (cx - w / 2 - 0.6, cy + 1.2, z + 2.5), 20000.0, 4200, 80.0)


def gas_annex_roof(kit, okit):
    z = 40.0
    rails_around(kit, [(25.5, -37.35), (37.35, -37.35), (37.35, -27.9), (25.5, -27.9)], z)
    kit.add("grating", "box", (31.4, -30.2, z + 0.05), scale=(11.4, 0.9, 0.06))
    kit.add("grating", "box", (35.4, -33.7, z + 0.05), scale=(0.9, 6.0, 0.06))
    for x, y in ((27.6, -34.8), (30.8, -34.8), (27.6, -32.0)):
        kit.add("proc_white", "box", (x, y, z + 0.6), scale=(2.4, 1.6, 1.1))
        kit.tube("galv", (x + 0.5, y, z + 1.15), (x + 0.5, y, z + 1.25), 0.55)
        okit.add("black", "box", (x + 0.5, y, z + 1.27), scale=(1.0, 0.05, 0.03))
        okit.add("black", "box", (x + 0.5, y, z + 1.27), scale=(0.05, 1.0, 0.03))
        kit.tube("galv", (x - 1.2, y, z + 0.4), (x - 1.2, -30.6, z + 0.4), 0.12)
    for y in (-35.8, -33.4):
        kit.add("proc_grey", "box", (33.4, y, z + 0.75), scale=(1.2, 1.2, 1.5))
        for zz in np.arange(z + 0.3, z + 1.4, 0.15):
            okit.add("galv", "box", (32.77, y, zz), (math.cos(0.26), 0.0, math.sin(0.26), 0.0), (0.03, 1.1, 0.12))
    mast = sg.rect_lattice((36.4, -36.4, z), (36.4, -36.4, z + 10.0), 0.9, 0.45, panel=1.0, cross=True)
    for p, q in mast["chords"]:
        kit.tube("proc_grey", p, q, 0.05)
    for p, q in mast["girts"] + mast["diagonals"]:
        okit.tube("proc_grey", p, q, 0.02)
    top = np.array((36.4, -36.4, z + 10.0))
    kit.tube("proc_grey", top, top + (0, 0, 2.2), 0.03)
    kit.tube("proc_grey", top + (0.0, 0.0, -1.5), top + (0.9, 0.0, -1.5), 0.04)
    kit.add("mark_white", "dome", top + (1.0, 0.0, -1.5), (0.7071, 0.0, 0.7071, 0.0), (2.5, 2.5, 2.5))
    kit.add("nav_red", "sphere", top + (0, 0, 2.3), scale=(0.15, 0.15, 0.15))
    for k in range(3):
        kit.tube("proc_grey", top + (0, 0, -0.5 - 2.5 * k), top + (-0.6, 0.3, -0.5 - 2.5 * k), 0.03)


def build_laydown(stage, looks, kit, okit):
    container10(kit, okit, (5.3, -17.1, DECK_TOP), 90.0, "box_blue")
    container10(kit, okit, (5.3, -13.8, DECK_TOP), 90.0, "box_orange")
    basket(kit, okit, (5.3, -10.75, DECK_TOP), (2.2, 2.6, 1.1), 0.0)
    for (x, y), look in (((-4.0, -23.55), "box_green"), ((-4.0, -21.1), "box_white")):
        kit.add(look, "box", (x, y, DECK_TOP + 1.295), scale=(6.06, 2.44, 2.59))
        kit.add("proc_grey", "box", (x - 3.05, y, DECK_TOP + 1.295), scale=(0.04, 2.3, 2.45))
    kit.add("box_red", "box", (-4.0, -22.3, DECK_TOP + 2.59 + 1.295), scale=(6.06, 2.44, 2.59))
    basket(kit, okit, (-1.8, -18.9, DECK_TOP), (2.4, 1.8, 1.0), 0.0)
    fire_pump_house(stage, kit, okit)
    gas_annex_roof(kit, okit)
    pipe_run(stage, looks, "firemain", [(2.6, -23.6, DECK_TOP + 0.9), (1.4, -23.6, DECK_TOP + 0.9),
                                               (1.4, -26.2, DECK_TOP + 0.9), (1.4, -26.2, DECK_TOP + 0.05)], 0.16, "pipe_red")
    pipe_run(stage, looks, "tank_out", [(3.89, 14.08, 43.0), (3.89, 12.3, 43.0), (12.5, 12.3, 43.0),
                                               (12.5, 10.4, 43.0), (12.5, 10.4, 46.45), (13.4, 9.8, 46.45)], 0.18,
             "pipe_grey")



PIPE_LOOKS = ("pipe_grey", "pipe_green", "pipe_yellow", "pipe_red", "galv", "pipe_grey")
RUNS = (
    ((1.0, -36.95), (24.2, -36.95), 20.0, 1),
    ((24.8, -36.0), (24.8, 24.2), 20.0, 1),
    ((0.8, -36.95), (25.0, -36.95), 31.25, -1),
    ((37.2, -17.0), (37.2, 23.8), 31.25, -1),
)
MONITORS = ((24.3, -37.0, 20.0), (0.9, -36.9, 20.0), (24.6, -36.9, 31.25), (37.0, 24.4, 31.25))
SHOWERS = ((12.5, -36.6, 20.0, 0.0), (24.4, 6.0, 20.0, 90.0), (37.0, 2.0, 31.25, 90.0), (13.0, -36.6, 31.25, 0.0))
SIGNS = ((6.4, -37.6, 22.3, 0.0), (13.0, -37.6, 22.3, 0.0), (24.9, 9.0, 22.5, 90.0), (2.4, -37.6, 33.6, 0.0),
         (37.55, -4.6, 33.6, 90.0), (37.55, 12.0, 33.6, 90.0), (24.9, -20.0, 22.5, 90.0), (20.0, -37.6, 33.6, 0.0))


def service_run(kit, okit, p0, p1, z0, inward, seed):
    rng = np.random.default_rng(seed)
    st = sg.stations_along(p0, p1, 3.0)
    u, n = np.append(st["dir"], 0.0), np.append(st["normal"] * inward, 0.0)
    q = zquat(st["angle"])
    lanes = [(-0.18, 1.55, 0.11), (0.05, 1.55, 0.07), (0.25, 1.55, 0.09), (-0.15, 2.35, 0.06),
             (0.08, 2.35, 0.12), (0.3, 2.35, 0.05)]
    a, b = np.append(st["points"][0], z0), np.append(st["points"][-1], z0)
    for k, (off, h, r) in enumerate(lanes):
        look = PIPE_LOOKS[k % len(PIPE_LOOKS)]
        kit.tube(look, a + n * off + (0, 0, h + r), b + n * off + (0, 0, h + r), r)
    for i, p in enumerate(st["points"]):
        base = np.append(p, z0)
        kit.member("proc_grey", "ibeam", base, base + (0, 0, 2.5), (0.12, 0.12))
        for h in (1.55, 2.35):
            kit.member("proc_grey", "box", base - n * 0.3 + (0, 0, h - 0.04), base + n * 0.45 + (0, 0, h - 0.04), (0.06, 0.08))
        if i % 2 == 0 and i < len(st["points"]) - 1:
            jb = base + u * 1.5 - n * 0.32 + (0, 0, 1.0)
            kit.add("proc_grey", "box", jb, q, (0.45, 0.18, 0.55))
            okit.add("black", "box", jb - n * 0.1 + (0, 0, -0.2), q, (0.3, 0.04, 0.05))
            kit.tube("galv", jb + (0, 0, 0.27), base + u * 1.5 - n * 0.32 + (0, 0, 2.62), 0.025)
    for k, dz in enumerate((2.65, 2.72, 2.79)):
        kit.tube("galv", a - n * 0.32 + (0, 0, dz), b - n * 0.32 + (0, 0, dz), 0.025)
    kit.member("galv", "box", a + n * 0.1 + (0, 0, 2.9), b + n * 0.1 + (0, 0, 2.9), (0.5, 0.02))
    for side in (-0.15, 0.35):
        kit.member("galv", "box", a + n * side + (0, 0, 2.95), b + n * side + (0, 0, 2.95), (0.02, 0.1))
    for k, dy in enumerate((-0.05, 0.08, 0.2)):
        kit.tube("cable", a + n * dy + (0, 0, 2.96), b + n * dy + (0, 0, 2.96), 0.035)
    for t in np.arange(1.2, st["length"] - 0.5, 2.2):
        k = int(rng.integers(len(lanes)))
        off, h, r = lanes[k]
        c = a + u * t + n * off + (0, 0, h + r)
        okit.add("valve_body", "box", c, q, (r * 3.2, r * 2.6, r * 2.6))
        okit.tube("proc_grey", c, c + (0, 0, r + 0.25), 0.02)
        okit.add("valve_red" if k % 2 else "crane_yellow", "wheel", c + (0, 0, r + 0.27), scale=(1.0, 1.0, 1.0))
        if rng.uniform() > 0.5:
            g = c + u * 0.35 + (0, 0, r + 0.12)
            okit.tube("galv", c + u * 0.35, g, 0.012)
            okit.add("mark_white", "cyl", g + (0, 0, 0.06), (0.7071, 0.0, -0.7071, 0.0), (0.04, 0.12, 0.12))


def fire_monitor(kit, x, y, z):
    kit.tube("fire_red", (x, y, z), (x, y, z + 1.3), 0.08)
    kit.add("fire_red", "box", (x, y, z + 1.4), scale=(0.25, 0.25, 0.2))
    kit.tube("fire_red", (x, y, z + 1.45), (x + 0.45, y, z + 1.7), 0.06)
    kit.tube("black", (x + 0.45, y, z + 1.7), (x + 0.7, y, z + 1.85), 0.07)
    kit.add("crane_yellow", "box", (x - 0.2, y, z + 1.35), scale=(0.08, 0.3, 0.08))
    kit.tube("fire_red", (x, y, z + 0.35), (x, y + 0.5, z + 0.35), 0.06)


def safety_shower(kit, okit, x, y, z, angle):
    q = zquat(angle)
    u = np.array((math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0))
    base = np.array((x, y, z))
    kit.tube("proc_green", base, base + (0, 0, 2.4), 0.05)
    kit.tube("proc_green", base + (0, 0, 2.4), base + (0, 0, 2.4) + u * 0.55, 0.04)
    kit.tube("galv", base + (0, 0, 2.3) + u * 0.55, base + (0, 0, 2.4) + u * 0.55, 0.16)
    kit.tube("proc_green", base + (0, 0, 1.1), base + (0, 0, 1.1) + u * 0.35, 0.03)
    kit.add("galv", "box", base + (0, 0, 1.15) + u * 0.4, q, (0.3, 0.3, 0.05))
    kit.add("sign_green", "box", base + (0, 0, 2.0) - u * 0.05, q, (0.04, 0.35, 0.35))


def stair_lamps(stage, kit):
    for n, (origin, z0, z1, axis) in enumerate(STAIRS):
        st = tf.stair_tower(origin, z0, z1, axis=axis)
        for k, (centre, size) in enumerate(st["landings"]):
            c = np.array(centre) + (0, 0, 2.3)
            kit.add("steel", "box", c + (0, 0, 0.1), scale=(0.35, 0.2, 0.12))
            kit.add("lamp", "box", c, scale=(0.3, 0.16, 0.04))
            add_light(stage, light_path("stair", 10 * n + k, False), c - (0, 0, 0.3), 30000.0, 4000, 85.0)


def build_complexity(stage, looks, kit, okit):
    for n, (p0, p1, z0, inward) in enumerate(RUNS):
        service_run(kit, okit, p0, p1, z0, inward, 40 + n)
    for x, y, z in MONITORS:
        fire_monitor(kit, x, y, z)
    for x, y, z, a in SHOWERS:
        safety_shower(kit, okit, x, y, z, a)
    for x, y, z, a in SIGNS:
        kit.add("sign_green", "box", (x, y, z), zquat(a), (0.6, 0.04, 0.3))
    stair_lamps(stage, kit)


def build_process(stage, looks, kit, okit):
    dropped = trim_stl(stage)
    build_tank(stage, looks, kit, okit)
    build_separator(stage, looks, kit, okit)
    build_scrubbers(stage, looks, kit, okit)
    build_rack(kit, okit)
    build_compressor(stage, kit, okit)
    ehouse(stage, kit, okit, (27.0, 4.7, DECK_TOP), 0.0, "box_white", 0)
    ehouse(stage, kit, okit, (27.0, 1.75, DECK_TOP), 0.0, "box_blue", 1)
    ehouse(stage, kit, okit, (-9.5, 9.0, RAISED), 90.0, "box_white", 2)
    dropped += hide_topside_utilities(stage)
    build_gensets(stage, looks, kit, okit)
    build_laydown(stage, looks, kit, okit)
    build_complexity(stage, looks, kit, okit)
    return dropped


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
    for group in ("", "/Looks", "/Columns", "/Underside", "/Lights", "/Inst", "/Offline", "/Offline/Drips", "/Offline/Lights", "/Helideck", "/Lifeboats", "/Ship", "/Process", "/Offline/Inst"):
        UsdGeom.Scope.Define(stage, f"{s.ROOT}{group}")
    looks = make_looks(stage)
    for name in ("galv", "black", "pipe_grey", "pipe_green", "pipe_yellow", "pipe_red", "valve_red"):
        looks[name] = UsdShade.Material(stage.OverridePrim(f"/World/Looks/topside_{name}"))
    looks["cable"] = looks["black"]
    legs = read_legs()
    kit = Instances(stage, looks, f"{s.ROOT}/Inst")
    kit.meshes.update(stiff=sg.ring_mesh(leg_radius(legs) - 0.05, leg_radius(legs) + 0.38, 0.32, SIDES),
                      seam=sg.ring_mesh(leg_radius(legs) - 0.02, leg_radius(legs) + 0.03, 0.05, SIDES),
                      wedge=wedge_mesh(), hoop=hoop_mesh(), shroud=sg.ring_mesh(0.95, 1.02, 1.6, 32),
                      dome=sphere_cap(0.16), tank_ring=sg.ring_mesh(3.98, 4.12, 0.14, 48),
                      wheel=sg.ring_mesh(0.11, 0.14, 0.025, 16))
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
    build_flare(stage, kit)
    build_helideck(stage, looks, kit)
    build_lifeboats(stage, looks, kit)
    build_ship(stage, looks, kit)
    build_crane(kit)
    build_derrick(stage, kit)
    okit = Instances(stage, looks, f"{s.OFFLINE}/Inst")
    okit.meshes = kit.meshes
    dropped = build_process(stage, looks, kit, okit)
    okit.write()
    retire_old(stage)
    drips = build_drips(stage, looks, legs)
    total = kit.write()
    stage.OverridePrim(LEGS_MESH).SetActive(False)
    for path in SPLASH_BANDS:
        UsdGeom.Imageable(stage.OverridePrim(path)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().GetPrimAtPath("/World").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()
    print(f"STRUCTURE_WRITTEN {s.LAYER} legs={len(legs)} instances={total} drips={drips} trimmed={dropped}")
    return s.LAYER


def leg_radius(legs):
    return float(np.mean([l["radius"] for l in legs]))


if __name__ == "__main__":
    build()

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

import topside_frames as tf
from scene_builder import ASSETS, SceneBuilder, rel

ROOT = "/World/Topside"
GRATING = "MetalWalkway010"
CLAD = "worn_corrugated_iron"
COLORS = {
    "steel": ((0.36, 0.39, 0.41), 0.55, 0.45),
    "steel_white": ((0.68, 0.7, 0.7), 0.5, 0.2),
    "steel_rust": ((0.36, 0.2, 0.12), 0.7, 0.3),
    "yellow": ((0.85, 0.6, 0.05), 0.45, 0.1),
    "galv": ((0.52, 0.54, 0.55), 0.45, 0.8),
    "insulated": ((0.7, 0.72, 0.73), 0.35, 0.9),
    "pipe_grey": ((0.34, 0.36, 0.37), 0.5, 0.4),
    "pipe_red": ((0.55, 0.06, 0.04), 0.45, 0.1),
    "pipe_yellow": ((0.8, 0.6, 0.08), 0.45, 0.1),
    "pipe_green": ((0.14, 0.32, 0.18), 0.5, 0.1),
    "vessel_white": ((0.78, 0.8, 0.8), 0.4, 0.15),
    "vessel_grey": ((0.5, 0.53, 0.55), 0.45, 0.3),
    "machine_blue": ((0.12, 0.25, 0.42), 0.4, 0.2),
    "machine_green": ((0.18, 0.33, 0.25), 0.45, 0.2),
    "valve_red": ((0.6, 0.05, 0.03), 0.4, 0.2),
    "black": ((0.04, 0.04, 0.045), 0.5, 0.3),
}
HIDDEN_PARTS = (
    "module_n76_0", "module_n75_n20", "module_n75_n60", "module_n75_n40",
    "module_n15_n33", "module_n15_n70", "module_4_n33", "module_4_n70", "module_24_n33",
    "module_24_n70", "module_44_n33", "module_44_n70",
)

LOWER, MIDDLE, TOP = 20.0, 31.25, 42.5
LOWER_ROOF, MIDDLE_ROOF = 30.0, 41.2

MODULES = (
    {"name": "pump_skid", "low": (-4.5, -19.0, LOWER), "high": (14.5, -5.0, LOWER_ROOF), "mezz": 25.0,
     "braces": ("-x", "+x"), "steel": "steel", "fill": "pumps"},
    {"name": "scrubbers", "low": (14.5, -36.5, LOWER), "high": (24.5, -20.0, LOWER_ROOF), "mezz": 25.0,
     "braces": ("-y", "+x"), "steel": "steel_white", "fill": "columns", "clad": (("+x", 0.0, 0.5),)},
    {"name": "compression", "low": (18.5, -18.0, LOWER), "high": (24.5, 2.0, LOWER_ROOF), "mezz": None,
     "braces": ("+x",), "steel": "steel", "fill": "compressor", "clad": (("+x", 0.45, 1.0), ("-y", 0.0, 1.0))},
    {"name": "west_lower", "low": (-24.5, -21.5, LOWER), "high": (-19.0, 7.0, LOWER_ROOF), "mezz": 25.0,
     "braces": ("-x",), "steel": "steel_white", "fill": "columns"},
    {"name": "gas_annex", "low": (25.5, -37.0, MIDDLE), "high": (37.0, -18.0, 40.0), "mezz": 35.6,
     "braces": ("+x", "-y"), "steel": "steel", "fill": "compressor", "clad": (("+x", 0.5, 1.0),)},
    {"name": "separation", "low": (26.0, -15.0, MIDDLE), "high": (37.0, 24.0, MIDDLE_ROOF), "mezz": 36.2,
     "braces": ("+x", "+y"), "steel": "steel_white", "fill": "separator"},
    {"name": "process_core", "low": (9.0, -14.0, MIDDLE), "high": (24.5, 11.0, MIDDLE_ROOF), "mezz": 36.2,
     "braces": ("-y",), "steel": "steel", "fill": "separator"},
    {"name": "west_middle", "low": (-24.5, -21.5, MIDDLE), "high": (-11.5, 8.0, MIDDLE_ROOF), "mezz": 36.2,
     "braces": ("-x", "-y"), "steel": "steel_rust", "fill": "columns", "clad": (("-x", 0.55, 1.0),)},
    {"name": "utilities", "low": (-24.5, -17.4, TOP), "high": (-5.5, 1.4, 48.8), "mezz": None,
     "braces": ("-x", "+y"), "steel": "steel_white", "fill": "utilities", "clad": (("-x", 0.0, 0.6),)},
)
RACKS = (
    ((-24.0, 21.5), (18.0, 21.5), LOWER, (24.5, 27.0), 4.0, 6),
    ((-10.5, 21.8), (25.0, 21.8), MIDDLE, (35.0, 37.6), 3.6, 5),
    ((21.5, -18.0), (21.5, 18.0), LOWER, (26.0, 28.2), 3.0, 4),
)
STAIRS = (
    ((17.8, 2.6), LOWER, MIDDLE, "y"),
    ((37.8, -4.0), MIDDLE, TOP, "y"),
    ((-24.3, 9.0), LOWER, MIDDLE, "y"),
    ((-24.3, 9.0), MIDDLE, TOP, "y"),
)
DECK_EDGES = (
    (LOWER, ((0.3, -37.5), (25.0, -37.5), (25.0, 25.0), (-25.0, 25.0), (-25.0, -22.9))),
    (MIDDLE, ((0.3, -37.5), (37.5, -37.5), (37.5, 25.0), (-25.0, 25.0), (-25.0, -22.9))),
)


class Kit:
    def __init__(self, stage, builder: SceneBuilder):
        self.stage = stage
        self.b = builder
        self.items: dict[str, dict[str, list]] = {}

    def add(self, material, proto, position, quat=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        bucket = self.items.setdefault(material, {"ibeam": [], "cyl": [], "box": [], "sphere": []})
        bucket[proto].append((tuple(position), tuple(quat), tuple(scale)))

    def member(self, material, proto, p, q, size):
        if math.dist(p, q) < 1e-4:
            return
        mid, quat, length = tf.segment_pose(p, q)
        self.add(material, proto, mid, quat, (length, size[0], size[1]))

    def box(self, material, center, size):
        self.add(material, "box", center, scale=size)

    def vertical(self, material, x, y, z0, z1, radius):
        self.member(material, "cyl", (x, y, z0), (x, y, z1), (2 * radius, 2 * radius))

    def write(self):
        UsdGeom.Scope.Define(self.stage, ROOT)
        total = 0
        for material_name, buckets in self.items.items():
            path = f"{ROOT}/{material_name}"
            instancer = UsdGeom.PointInstancer.Define(self.stage, path)
            material = self.b.plain(f"topside_{material_name}", *COLORS[material_name])
            protos, indices, positions, orients, scales = [], [], [], [], []
            for proto, entries in buckets.items():
                if not entries:
                    continue
                prim = self._prototype(f"{path}/protos/{proto}", proto, material)
                number = len(protos)
                protos.append(prim.GetPath())
                for position, quat, scale in entries:
                    indices.append(number)
                    positions.append(Gf.Vec3f(*position))
                    orients.append(Gf.Quath(float(quat[0]), Gf.Vec3h(*(float(c) for c in quat[1:]))))
                    scales.append(Gf.Vec3f(*scale))
            instancer.CreatePrototypesRel().SetTargets(protos)
            instancer.CreateProtoIndicesAttr(Vt.IntArray(indices))
            instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
            instancer.CreateOrientationsAttr(Vt.QuathArray(orients))
            instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
            total += len(indices)
        return total

    def _prototype(self, path, proto, material):
        UsdGeom.Scope.Define(self.stage, path.rsplit("/", 1)[0])
        if proto == "cyl":
            prim = UsdGeom.Cylinder.Define(self.stage, path)
            prim.CreateAxisAttr("X")
            prim.CreateRadiusAttr(0.5)
            prim.CreateHeightAttr(1.0)
        elif proto == "sphere":
            prim = UsdGeom.Sphere.Define(self.stage, path)
            prim.CreateRadiusAttr(1.0)
        elif proto == "box":
            prim = UsdGeom.Cube.Define(self.stage, path)
            prim.CreateSizeAttr(1.0)
        else:
            prim = ibeam_mesh(self.stage, path)
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
        return prim


def ibeam_mesh(stage, path):
    t, w = 0.09, 0.1
    outline = [(-0.5, -0.5), (0.5, -0.5), (0.5, -0.5 + t), (w / 2, -0.5 + t), (w / 2, 0.5 - t), (0.5, 0.5 - t),
               (0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5 - t), (-w / 2, 0.5 - t), (-w / 2, -0.5 + t), (-0.5, -0.5 + t)]
    rects = [(0, 1, 2, 11), (10, 3, 4, 9), (8, 5, 6, 7)]
    points, counts, indices = [], [], []
    n = len(outline)
    for x in (-0.5, 0.5):
        points += [(x, y, z) for y, z in outline]
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        indices += [i, j, j + n, i + n]
    for a, b, c, d in rects:
        counts += [4, 4]
        indices += [d, c, b, a, a + n, b + n, c + n, d + n]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def grating_material(stage):
    path = "/World/Looks/topside_grating"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    reader = UsdShade.Shader.Define(stage, f"{path}/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    folder = ASSETS / "textures" / GRATING
    for file, slot, channel, space, kind in (
        ("diff", "diffuseColor", "rgb", "sRGB", Sdf.ValueTypeNames.Color3f),
        ("rough", "roughness", "r", "raw", Sdf.ValueTypeNames.Float),
        ("opacity", "opacity", "r", "raw", Sdf.ValueTypeNames.Float),
    ):
        node = UsdShade.Shader.Define(stage, f"{path}/{file}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(folder / f"{file}.jpg"))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        output_type = Sdf.ValueTypeNames.Float3 if channel == "rgb" else Sdf.ValueTypeNames.Float
        shader.CreateInput(slot, kind).ConnectToSource(node.CreateOutput(channel, output_type))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def floor_panel(stage, path, low, high, z, material):
    (x0, y0), (x1, y1) = low, high
    points = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([Gf.Vec2f(p[0] / 1.2, p[1] / 1.2) for p in points])
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def add_frame(kit, f, steel, column=0.36, beam=0.3, brace=0.2):
    for p, q in f["columns"]:
        kit.member(steel, "ibeam", p, q, (column, column))
    for p, q in f["beams"]:
        kit.member(steel, "ibeam", p, q, (beam * 0.66, beam))
    for p, q in f["braces"]:
        kit.member(steel, "cyl", p, q, (brace, brace))


def add_rail(kit, p, q, z):
    r = tf.rail(p, q, z)
    for a, b in r["rails"]:
        kit.member("yellow", "cyl", a, b, (0.05, 0.05))
    for a, b in r["posts"]:
        kit.member("yellow", "cyl", a, b, (0.05, 0.05))
    for a, b in r["toe"]:
        kit.member("yellow", "box", a, b, (0.012, 0.15))


def add_stair(kit, origin, z0, z1, axis):
    s = tf.stair_tower(origin, z0, z1, axis=axis)
    for center, size in s["steps"]:
        kit.box("galv", center, size)
    for center, size in s["landings"]:
        kit.box("galv", center, size)
    for a, b in s["stringers"]:
        kit.member("yellow", "box", a, b, (0.02, 0.25))
        kit.member("yellow", "cyl", (a[0], a[1], a[2] + 1.0), (b[0], b[1], b[2] + 1.0), (0.045, 0.045))
    add_frame(kit, s["frame"], "yellow", column=0.16, beam=0.16, brace=0.08)


def add_ladder(kit, base, height, facing):
    l = tf.ladder(base, height, facing)
    for a, b in l["rails"]:
        kit.member("yellow", "cyl", a, b, (0.05, 0.05))
    for a, b in l["rungs"]:
        kit.member("yellow", "cyl", a, b, (0.03, 0.03))


def vessel_vertical(kit, x, y, z, height, radius, material, skirt=0.8):
    kit.vertical("steel", x, y, z, z + skirt, radius * 0.95)
    kit.vertical(material, x, y, z + skirt, z + skirt + height, radius)
    for cap in (z + skirt, z + skirt + height):
        kit.add(material, "sphere", (x, y, cap), scale=(radius, radius, radius * 0.5))
    kit.vertical("pipe_grey", x + radius * 0.7, y, z + skirt + height, z + skirt + height + 1.2, 0.09)
    kit.add("steel", "cyl", (x, y, z + skirt + height * 0.6), (0.7071, 0.0, -0.7071, 0.0), (0.12, radius * 2.3, radius * 2.3))


def vessel_horizontal(kit, x, y, z, length, radius, material, axis="x"):
    p = (x - length / 2, y, z + radius + 0.6) if axis == "x" else (x, y - length / 2, z + radius + 0.6)
    q = (x + length / 2, y, z + radius + 0.6) if axis == "x" else (x, y + length / 2, z + radius + 0.6)
    kit.member(material, "cyl", p, q, (2 * radius, 2 * radius))
    for end in (p, q):
        scale = (radius * 0.5, radius, radius) if axis == "x" else (radius, radius * 0.5, radius)
        kit.add(material, "sphere", end, scale=scale)
    for t in (0.22, 0.78):
        c = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, z + 0.45)
        size = (0.35, radius * 1.6, 0.9) if axis == "x" else (radius * 1.6, 0.35, 0.9)
        kit.box("steel", c, size)
    top = z + 2 * radius + 0.6
    kit.vertical("pipe_grey", x, y, top - 0.1, top + 0.9, 0.12)
    kit.add("valve_red", "cyl", (x, y, top + 1.0), (0.7071, 0.0, -0.7071, 0.0), (0.04, 0.5, 0.5))


def pump(kit, x, y, z, axis="x", material="machine_blue"):
    size = (2.4, 0.9, 0.25) if axis == "x" else (0.9, 2.4, 0.25)
    kit.box("steel", (x, y, z + 0.125), size)
    d = (1, 0) if axis == "x" else (0, 1)
    a = (x - d[0] * 1.0, y - d[1] * 1.0, z + 0.7)
    b = (x + d[0] * 0.2, y + d[1] * 0.2, z + 0.7)
    kit.member(material, "cyl", a, b, (0.6, 0.6))
    kit.add("vessel_grey", "sphere", (x + d[0] * 0.65, y + d[1] * 0.65, z + 0.7), scale=(0.38, 0.38, 0.38))
    kit.vertical("pipe_grey", x + d[0] * 0.65, y + d[1] * 0.65, z + 0.7, z + 2.6, 0.1)
    kit.add("valve_red", "cyl", (x + d[0] * 0.65, y + d[1] * 0.65, z + 2.0), (0.7071, 0.0, -0.7071, 0.0), (0.04, 0.42, 0.42))


def compressor(kit, x, y, z, length, width, height):
    kit.box("steel", (x, y, z + 0.2), (length + 0.6, width + 0.6, 0.4))
    kit.box("machine_green", (x - length * 0.15, y, z + 0.4 + height / 2), (length * 0.6, width, height))
    kit.box("vessel_grey", (x + length * 0.33, y, z + 0.4 + height * 0.35), (length * 0.3, width * 0.7, height * 0.7))
    kit.vertical("pipe_grey", x - length * 0.3, y, z + 0.4 + height, z + height + 2.2, 0.25)


def fill(kit, module, rng):
    (x0, y0, z0), (x1, y1, z1) = module["low"], module["high"]
    kind = module["fill"]
    mezz = module["mezz"]
    w, d = x1 - x0, y1 - y0
    if kind == "pumps":
        for i, x in enumerate(np.arange(x0 + 2.0, x1 - 1.5, 3.2)):
            for y in (y0 + 2.2, y1 - 2.5):
                pump(kit, float(x), y, z0, "y", "machine_blue" if i % 2 else "machine_green")
        if mezz:
            vessel_horizontal(kit, x0 + w * 0.3, (y0 + y1) / 2, mezz, w * 0.35, 1.0, "vessel_white")
            vessel_horizontal(kit, x0 + w * 0.75, (y0 + y1) / 2, mezz, w * 0.3, 0.8, "vessel_grey")
    elif kind == "columns":
        count = max(1, int(max(w, d) // 5.5))
        for k in range(count):
            t = (k + 0.5) / count
            x, y = (x0 + w * t, (y0 + y1) / 2) if w > d else ((x0 + x1) / 2, y0 + d * t)
            radius = float(rng.uniform(0.9, min(w, d) * 0.28))
            vessel_vertical(kit, x, y, z0, (z1 - z0) - 2.4, radius, "vessel_white" if k % 2 == 0 else "vessel_grey")
        if mezz:
            for k in range(count):
                t = (k + 0.5) / count
                x, y = (x0 + w * t, y0 + 1.4) if w > d else (x0 + 1.4, y0 + d * t)
                pump(kit, x, y, mezz, "x" if w > d else "y")
    elif kind == "separator":
        axis = "x" if w > d else "y"
        span = max(w, d)
        for k, z in enumerate((z0, mezz) if mezz else (z0,)):
            for j, frac in enumerate((0.3, 0.72)):
                if axis == "x":
                    vessel_horizontal(kit, x0 + w * 0.5, y0 + d * frac, z, span * 0.6, 1.3 - 0.4 * k, "vessel_white", "x")
                else:
                    vessel_horizontal(kit, x0 + w * frac, y0 + d * 0.5, z, span * 0.45, 1.3 - 0.4 * k, "vessel_white" if j else "vessel_grey", "y")
        for k in range(3):
            t = (k + 0.5) / 3
            x, y = (x0 + w * t, y1 - 1.2) if axis == "x" else (x1 - 1.2, y0 + d * t)
            pump(kit, x, y, z0, "x" if axis == "y" else "y")
    elif kind == "compressor":
        length = min(max(w, d) * 0.55, 9.0)
        compressor(kit, (x0 + x1) / 2, (y0 + y1) / 2, z0, length if w > d else min(w, d) * 0.5,
                   min(w, d) * 0.4 if w > d else length, min(z1 - z0 - 2.2, 3.4))
        vessel_vertical(kit, x0 + 1.6, y0 + 1.6, z0, min(z1 - z0 - 2.0, 6.5), 0.8, "vessel_white")
        if mezz:
            for k in range(2):
                pump(kit, x0 + w * (0.3 + 0.4 * k), y1 - 1.4, mezz, "x")
    elif kind == "utilities":
        for k, x in enumerate(np.arange(x0 + 2.5, x1 - 2.0, 4.5)):
            kit.box("machine_green" if k % 2 else "vessel_grey", (float(x), y0 + 3.0, z0 + 1.2), (3.4, 2.2, 2.4))
            kit.vertical("black", float(x) - 0.8, y0 + 3.0, z0 + 2.4, z0 + 5.4, 0.22)
        vessel_horizontal(kit, (x0 + x1) / 2, y1 - 3.0, z0, w * 0.4, 0.9, "vessel_white")


def add_cladding(builder, module, face, lo, hi, material):
    (x0, y0, z0), (x1, y1, z1) = module["low"], module["high"]
    zb, zt = z0 + (z1 - z0) * lo, z0 + (z1 - z0) * hi
    zc, h = (zb + zt) / 2, zt - zb
    t = 0.08
    sizes = {
        "-x": ((x0 - t, (y0 + y1) / 2, zc), (t, y1 - y0, h)),
        "+x": ((x1 + t, (y0 + y1) / 2, zc), (t, y1 - y0, h)),
        "-y": (((x0 + x1) / 2, y0 - t, zc), (x1 - x0, t, h)),
        "+y": (((x0 + x1) / 2, y1 + t, zc), (x1 - x0, t, h)),
    }
    center, size = sizes[face]
    builder.box(f"{ROOT}/Cladding", f"{module['name']}_{face.replace('+', 'p').replace('-', 'n')}", center, size, material, tile=3.0, collide=None)


def add_rack(kit, start, end, z0, tiers, width, count, seed):
    r = tf.pipe_rack(start, end, z0, tiers=tiers, width=width, pipes_per_tier=count, seed=seed)
    for p, q in r["columns"]:
        kit.member("steel", "ibeam", p, q, (0.25, 0.25))
    for p, q in r["beams"]:
        kit.member("steel", "ibeam", p, q, (0.16, 0.22))
    services = ("insulated", "pipe_grey", "pipe_red", "pipe_yellow", "pipe_green", "insulated")
    for k, ((p, q), radius) in enumerate(r["pipes"]):
        kit.member(services[k % len(services)], "cyl", p, q, (2 * radius, 2 * radius))
    for (x, y, z), radius in r["valves"]:
        a = (x - 0.12, y, z) if r["along_x"] else (x, y - 0.12, z)
        b = (x + 0.12, y, z) if r["along_x"] else (x, y + 0.12, z)
        kit.member("pipe_grey", "cyl", a, b, (radius * 2.9, radius * 2.9))
        kit.vertical("steel", x, y, z, z + radius + 0.45, 0.025)
        kit.add("valve_red", "cyl", (x, y, z + radius + 0.45), (0.7071, 0.0, -0.7071, 0.0), (0.035, radius * 3.5, radius * 3.5))
    for p, q in r["trays"]:
        kit.member("galv", "box", p, q, (0.6, 0.02))
        for side in (-0.3, 0.3):
            off = (0, side, 0.05) if r["along_x"] else (side, 0, 0.05)
            kit.member("galv", "box", tuple(a + o for a, o in zip(p, off)), tuple(a + o for a, o in zip(q, off)), (0.02, 0.1))
        kit.member("black", "cyl", (p[0], p[1], p[2] + 0.06), (q[0], q[1], q[2] + 0.06), (0.09, 0.09))


def build(stage, builder: SceneBuilder, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    kit = Kit(stage, builder)
    UsdGeom.Scope.Define(stage, ROOT)
    UsdGeom.Scope.Define(stage, f"{ROOT}/Floors")
    UsdGeom.Scope.Define(stage, f"{ROOT}/Cladding")
    grating = grating_material(stage)
    clad = builder.textured(CLAD)
    floors = 0
    for module in MODULES:
        (x0, y0, z0), (x1, y1, z1) = module["low"], module["high"]
        levels = [z0] + ([module["mezz"]] if module["mezz"] else []) + [z1]
        add_frame(kit, tf.frame(module["low"], module["high"], levels, bay=5.0, brace_faces=module["braces"]), module["steel"])
        if module["mezz"]:
            z = module["mezz"]
            floor_panel(stage, f"{ROOT}/Floors/{module['name']}", (x0, y0), (x1, y1), z + 0.02, grating)
            floors += 1
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
            for p, q in zip(corners, corners[1:]):
                add_rail(kit, p, q, z)
            add_ladder(kit, (x0 + 0.4, y0 + 0.6, z0), z - z0, "y")
        if module["name"] == "utilities":
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
            for p, q in zip(corners, corners[1:]):
                add_rail(kit, p, q, z1)
            floor_panel(stage, f"{ROOT}/Floors/{module['name']}_roof", (x0, y0), (x1, y1), z1 + 0.02, grating)
            floors += 1
        for face, lo, hi in module.get("clad", ()):
            add_cladding(builder, module, face, lo, hi, clad)
        fill(kit, module, rng)
    for k, (start, end, z0, tiers, width, count) in enumerate(RACKS):
        add_rack(kit, start, end, z0, tiers, width, count, seed + k)
    for origin, z0, z1, axis in STAIRS:
        add_stair(kit, origin, z0, z1, axis)
    for z, corners in DECK_EDGES:
        for p, q in zip(corners, corners[1:]):
            add_rail(kit, p, q, z)
    instances = kit.write()
    return {"instances": instances, "floors": floors, "modules": [m["name"] for m in MODULES]}

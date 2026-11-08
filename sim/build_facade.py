from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import facade
import facade_layout as fl
import photoreal as pr
import topside_frames as tf
from photoreal_usd import bind_mdl
from scene_builder import rel

TEX = pr.GENERATED / "facade"
HIDE = (
    "/World/Details/facade_windows",
    "/World/Details/windows_module_n83_40",
    "/World/Details/windows_module_n23_n71",
    "/World/Topside/lifeboat",
)
EXISTING = {"galv": "/World/Looks/topside_galv", "yellow": "/World/Looks/topside_yellow",
            "white_steel": "/World/Looks/topside_steel_white"}
HELIDECK = ((-36.93, -22.5), (-29.7, -35.0), (-15.3, -35.0), (-8.07, -22.5), (-15.3, -10.0), (-29.7, -10.0))
HELI_Z = 51.25
LIFEBOATS = ((-19.0, -40.6, 27.6, 1), (-9.0, -40.6, 27.6, 2))
STAIR = ((0.65, -37.3), 31.25, 42.5)
FLOODS = (
    ((-21.0, -39.3, 41.6), (-21.0, -37.8, 33.0), 4000.0), ((-12.0, -39.3, 41.6), (-12.0, -37.8, 33.0), 2100.0),
    ((-3.0, -39.3, 41.6), (-3.0, -37.8, 33.0), 4000.0), ((-14.0, -39.2, 29.6), (-14.0, -37.8, 22.0), 2100.0),
    ((-4.0, -39.2, 29.6), (-4.0, -37.8, 22.0), 4000.0), ((-19.0, -42.6, 33.0), (-19.0, -40.6, 27.6), 3000.0),
    ((-9.0, -42.6, 33.0), (-9.0, -40.6, 27.6), 3000.0), ((2.1, -26.0, 41.6), (0.62, -26.0, 33.0), 2100.0),
    ((2.9, -33.5, 41.6), (0.62, -33.5, 33.0), 4000.0), ((2.0, -31.0, 29.6), (0.62, -31.0, 22.0), 4000.0),
    ((-19.9, -32.5, 29.6), (-18.42, -32.5, 22.0), 2100.0), ((-26.8, -30.0, 41.6), (-25.32, -30.0, 33.0), 4000.0),
)
FLOOD_POWER = 260000.0
SIGNS = {"muster": 0, "boat1": 1, "boat2": 2, "refuge": 3, "exit": 4, "smoking": 5, "deck": 6, "fire": 7}


class Batch:
    def __init__(self):
        self.parts: dict[str, list] = {}

    def quad(self, mat, pts, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        self.parts.setdefault(mat, []).append((pts, fl.face_normal(pts), uv))

    def box(self, mat, lo, hi):
        for f in fl.box_faces(lo, hi):
            self.quad(mat, f)

    def beam(self, mat, p, q, w, h):
        p, q = np.asarray(p, float), np.asarray(q, float)
        u = q - p
        if np.linalg.norm(u) < 1e-6:
            return
        u /= np.linalg.norm(u)
        side = np.cross(u, (0, 0, 1)) if abs(u[2]) < 0.95 else np.cross(u, (1, 0, 0))
        side /= np.linalg.norm(side)
        up = np.cross(side, u)
        c = [p + side * sx * w / 2 + up * sz * h / 2 for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        d = [x + (q - p) for x in c]
        for a, b, cc, dd in ((c[0], c[1], d[1], d[0]), (c[1], c[2], d[2], d[1]), (c[2], c[3], d[3], d[2]),
                             (c[3], c[0], d[0], d[3]), (c[3], c[2], c[1], c[0]), (d[0], d[1], d[2], d[3])):
            pts = [tuple(a), tuple(b), tuple(cc), tuple(dd)]
            mid = np.mean(pts, axis=0)
            self.quad(mat, fl.orient(pts, mid - (p + q) / 2))

    def write(self, stage, looks):
        for mat, quads in self.parts.items():
            pts = [p for q in quads for p in q[0]]
            nrm = [q[1] for q in quads for _ in range(4)]
            uvs = [uv for q in quads for uv in q[2]]
            mesh = UsdGeom.Mesh.Define(stage, f"{facade.ROOT}/Geo/{mat}")
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(pts)))))
            mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm]))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(False)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in uvs]))
            lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
            mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
            api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
            if mat in looks:
                api.Bind(looks[mat])
            else:
                mesh.GetPrim().CreateRelationship("material:binding").SetTargets([Sdf.Path(EXISTING[mat])])
        return sum(len(q) for q in self.parts.values())


def paint(stage, name, color, chip=0.4, rust=0.15, **extra):
    recipe = {"kind": "paint", "paint_color": color, "chip_amount": chip, "rust_amount": rust, **extra}
    sub, values = pr.shader_inputs(recipe)
    values.update({k: v for k, v in extra.items() if k in ("paint_diff", "paint_rough", "paint_normal", "paint_scale")})
    return bind_mdl(stage, f"{facade.LOOKS}/{name}", sub, values, define=True)


def preview(stage, name, color=(0, 0, 0), roughness=0.5, metallic=0.0, emit=None, diffuse_tex=None, ior=1.5):
    path = f"{facade.LOOKS}/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    st = None
    if emit or diffuse_tex:
        reader = UsdShade.Shader.Define(stage, f"{path}/uv")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(node_name, image):
        node = UsdShade.Shader.Define(stage, f"{path}/{node_name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(image))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
        return node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    if diffuse_tex:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture("diff", diffuse_tex))
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if emit:
        shader.GetInput("emissiveColor").ConnectToSource(texture("emit", emit))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def decal(stage, name, mask, tint, strength=1.0, roughness=0.6):
    values = {"mask": mask, "tint": tint, "strength": strength, "roughness": roughness, "use_color_tex": False}
    return bind_mdl(stage, f"{facade.LOOKS}/{name}", "rig_decal", values, define=True)


def materials(stage) -> dict:
    clad = {"paint_diff": TEX / "clad_diff.png", "paint_rough": TEX / "clad_rough.png",
            "paint_normal": TEX / "clad_nor_gl.png", "paint_scale": 4.0}
    return {
        "clad": paint(stage, "clad", (0.64, 0.65, 0.63), chip=0.3, rust=0.1, streak_amount=0.9, grime_amount=0.5,
                      salt_amount=0.25, **clad),
        "clad_grey": paint(stage, "clad_grey", (0.42, 0.45, 0.47), chip=0.35, rust=0.14, streak_amount=0.9,
                           grime_amount=0.5, **clad),
        "trim": paint(stage, "trim", (0.46, 0.48, 0.49), chip=0.5, rust=0.25),
        "frame": paint(stage, "frame", (0.08, 0.085, 0.09), chip=0.15, rust=0.05, metallic=0.4),
        "door": paint(stage, "door", (0.16, 0.24, 0.3), chip=0.55, rust=0.3),
        "unit": paint(stage, "unit", (0.62, 0.63, 0.61), chip=0.4, rust=0.2),
        "hull": paint(stage, "hull", (0.95, 0.3, 0.015), chip=0.2, rust=0.04, streak_amount=0.2, grime_amount=0.2),
        "fender": paint(stage, "fender", (0.03, 0.03, 0.03), chip=0.0, rust=0.0, streak_amount=0.0),
        "interior": preview(stage, "interior", (0.004, 0.004, 0.005), roughness=0.03, emit=TEX / "interiors.png"),
        "sign": preview(stage, "sign", roughness=0.35, emit=TEX / "signs.png", diffuse_tex=TEX / "signs.png"),
        "louvre": preview(stage, "louvre", roughness=0.6, diffuse_tex=TEX / "louvre.png"),
        "lamp": preview(stage, "lamp", (0.9, 0.9, 0.85), roughness=0.2),
        "heli_green": preview(stage, "heli_green", (0.1, 0.4, 0.1), roughness=0.2),
        "streak": decal(stage, "streak", TEX / "streaks.png", (0.19, 0.085, 0.03), strength=0.9, roughness=0.5),
        "grime": decal(stage, "grime", TEX / "streaks.png", (0.05, 0.05, 0.045), strength=0.6, roughness=0.6),
        "net": decal(stage, "net", TEX / "net.png", (0.06, 0.06, 0.05), strength=1.0, roughness=0.8),
    }


def sign_uv(tile):
    r, c = divmod(tile, 2)
    return ((c / 2, 1 - (r + 1) / 4), ((c + 1) / 2, 1 - (r + 1) / 4), ((c + 1) / 2, 1 - r / 4), (c / 2, 1 - r / 4))


def overlaps(rect, rects, pad=0.15):
    return any(rect[0] < h[2] + pad and h[0] - pad < rect[2] and rect[1] < h[3] + pad and h[1] - pad < rect[3]
               for h in rects)


def add_sign(b, w, along, zc, size, tile, taken):
    rect = (along - size[0] / 2, zc - size[1] / 2, along + size[0] / 2, zc + size[1] / 2)
    if overlaps(rect, taken) or rect[0] < w["along"][0] or rect[2] > w["along"][1]:
        return False
    taken.append(rect)
    b.quad("sign", fl.quad(w, rect[0], rect[1], rect[2], rect[3], fl.CLAD_DEPTH + 0.02), sign_uv(tile))
    return True


def build_wall(b, w, lamps, extra_signs):
    D, G = fl.CLAD_DEPTH, fl.GLASS_DEPTH
    a0, a1 = w["along"]
    z0, z1 = w["z"]
    items = fl.openings(w)
    hs = fl.holes(items)
    clad = "clad_grey" if w["name"].startswith("m71") else "clad"
    for r in fl.subtract((a0 - D, z0, a1 + D, z1), hs):
        b.quad(clad, fl.quad(w, *r, D))
    taken = list(hs)
    h = fl.storey_height(w)
    for s in range(1, w["storeys"]):
        z = z0 + s * h
        b.box("trim", *fl.wall_box(w, a0 - D - 0.02, z - 0.07, a1 + D + 0.02, z + 0.07, D, D + 0.06))
    b.box("trim", *fl.wall_box(w, a0 - D - 0.04, z1 - 0.3, a1 + D + 0.04, z1, D - 0.01, D + 0.09))
    b.box("trim", *fl.wall_box(w, a0 - D, z0, a1 + D, z0 + 0.25, D, D + 0.04))
    for end in (a0 - D, a1 + D - 0.16):
        b.box("trim", *fl.wall_box(w, end, z0, end + 0.16, z1, D, D + 0.05))
    for o in items:
        hole = (o["along"] - o["w"] / 2, o["z0"], o["along"] + o["w"] / 2, o["z0"] + o["h"])
        if o["kind"] == "window":
            for f in fl.reveal(w, hole, D, G):
                b.quad("trim", f)
            u0, v0, u1, v1 = fl.tile_uv(o["tile"])
            b.quad("interior", fl.quad(w, *hole, G), ((u0, v0), (u1, v0), (u1, v1), (u0, v1)))
            t = 0.07
            for lo, hi in (((hole[0], hole[1]), (hole[2], hole[1] + t)), ((hole[0], hole[3] - t), (hole[2], hole[3])),
                           ((hole[0], hole[1]), (hole[0] + t, hole[3])), ((hole[2] - t, hole[1]), (hole[2], hole[3])),
                           ((o["along"] - 0.03, hole[1]), (o["along"] + 0.03, hole[3]))):
                b.box("frame", *fl.wall_box(w, lo[0], lo[1], hi[0], hi[1], G - 0.01, G + 0.06))
            b.box("trim", *fl.wall_box(w, hole[0] - 0.08, hole[1] - 0.07, hole[2] + 0.08, hole[1], D - 0.02, D + 0.09))
            sh = fl.streak_height(o, w)
            k = fl.unit(o["key"] + "v")
            if sh > 0.3 and k < 0.75:
                half = o["w"] * (0.45 + 0.2 * fl.unit(o["key"] + "w"))
                col = int(k * 4 / 0.75) % 4
                top = hole[1] - 0.07
                uv = ((col / 4, 1 - sh / 2.6), ((col + 1) / 4, 1 - sh / 2.6), ((col + 1) / 4, 1), (col / 4, 1))
                mat = "streak" if fl.unit(o["key"] + "m") < 0.65 else "grime"
                b.quad(mat, fl.quad(w, o["along"] - half, top - sh, o["along"] + half, top, D + 0.004), uv)
        elif o["kind"] == "door":
            for f in fl.reveal(w, hole, D, 0.12):
                b.quad("trim", f)
            b.box("door", *fl.wall_box(w, hole[0], hole[1], hole[2], hole[3], 0.08, 0.14))
            b.box("frame", *fl.wall_box(w, hole[2] - 0.2, hole[1] + 0.95, hole[2] - 0.1, hole[1] + 1.0, 0.14, 0.2))
            lz = hole[3] + 0.3
            b.box("frame", *fl.wall_box(w, o["along"] - 0.22, lz + 0.16, o["along"] + 0.22, lz + 0.22, D, D + 0.3))
            b.box("lamp", *fl.wall_box(w, o["along"] - 0.16, lz, o["along"] + 0.16, lz + 0.16, D + 0.02, D + 0.2))
            taken.append((o["along"] - 0.3, lz - 0.05, o["along"] + 0.3, lz + 0.3))
            lamps.append(fl.to_world(w, o["along"], lz - 0.05, D + 0.12))
            tile = SIGNS["exit"] if fl.unit(w["name"] + "d") < 0.5 else SIGNS["muster"]
            for side in (1, -1):
                if add_sign(b, w, o["along"] + side * 1.05, hole[1] + 1.6, (0.8, 0.4), tile, taken):
                    break
        elif fl.unit(o["key"] + "u") < 0.35:
            ua, uz = o["along"], o["z0"] - 0.4
            b.box("unit", *fl.wall_box(w, ua - 0.4, uz, ua + 0.4, uz + 0.55, D, D + 0.4))
            b.quad("louvre", fl.quad(w, ua - 0.34, uz + 0.05, ua + 0.34, uz + 0.5, D + 0.401))
            b.box("frame", *fl.wall_box(w, ua - 0.35, uz - 0.05, ua + 0.35, uz, D, D + 0.45))
            taken.append((ua - 0.4, uz, ua + 0.4, uz + 0.55))
    for along, zc, size, tile in extra_signs.get(w["name"], ()):
        add_sign(b, w, along, zc, size, tile, taken)
    return len(items)


EXTRA_SIGNS = {
    "lq_s_upper": ((-16.0, 31.0, (1.6, 0.8), SIGNS["boat1"]), (-6.0, 31.0, (1.6, 0.8), SIGNS["boat2"]),
                   (-11.0, 40.2, (2.4, 1.0), SIGNS["deck"])),
    "lq_e_upper": ((-26.0, 40.2, (2.6, 1.3), SIGNS["refuge"]),),
    "lq_e_lower": ((-33.0, 21.6, (0.8, 0.4), SIGNS["smoking"]), (-27.0, 21.6, (0.8, 0.4), SIGNS["fire"])),
    "m40_s": ((-17.5, 39.8, (1.6, 0.8), SIGNS["smoking"]),),
}


def lifeboat(b, x, y, z, length=8.4, beam=2.9, keel=1.1, canopy=1.75, n=28, m=20):
    rings = []
    for i in range(n + 1):
        t = -1 + 2 * i / n
        taper = max(1 - abs(t) ** (2.4 if t > 0 else 3.2), 0.0) ** 0.5
        ring = []
        for j in range(m):
            a = 2 * math.pi * j / m
            c, s = math.cos(a), math.sin(a)
            yy = beam / 2 * taper * math.copysign(abs(c) ** (2 / 3.2), c)
            zz = (canopy * (0.75 + 0.25 * taper) if s > 0 else keel * (0.5 + 0.5 * taper)) * math.copysign(abs(s) ** (2 / 3.2), s)
            ring.append((x + t * length / 2, y + yy, z + zz * max(taper, 0.08)))
        rings.append(ring)
    centre = np.array((x, y, z + 0.2))
    for i in range(n):
        for j in range(m):
            pts = [rings[i][j], rings[i + 1][j], rings[i + 1][(j + 1) % m], rings[i][(j + 1) % m]]
            mid = np.mean(pts, axis=0)
            b.quad("hull", fl.orient(pts, mid - centre))
    b.box("fender", (x - length * 0.42, y - beam / 2 - 0.08, z - 0.05), (x + length * 0.42, y + beam / 2 + 0.08, z + 0.12))
    b.box("fender", (x - 1.0, y - 0.55, z + canopy - 0.05), (x + 1.0, y + 0.55, z + canopy + 0.12))
    for k in range(5):
        wx = x - 2.2 + k * 1.1
        b.box("frame", (wx - 0.3, y - beam / 2 * 0.86 - 0.02, z + 0.55), (wx + 0.3, y - beam / 2 * 0.86 + 0.02, z + 0.85))


def helideck(b):
    for x, y in fl.helideck_lights(HELIDECK, 3.0):
        b.box("heli_green", (x - 0.09, y - 0.09, HELI_Z), (x + 0.09, y + 0.09, HELI_Z + 0.14))
    c = np.mean(HELIDECK, axis=0)
    size, pitch = 15.0, 8.0
    pts = [(c[0] - size / 2, c[1] - size / 2, HELI_Z + 0.02), (c[0] + size / 2, c[1] - size / 2, HELI_Z + 0.02),
           (c[0] + size / 2, c[1] + size / 2, HELI_Z + 0.02), (c[0] - size / 2, c[1] + size / 2, HELI_Z + 0.02)]
    k = size / pitch
    b.quad("net", pts, ((0, 0), (k, 0), (k, k), (0, k)))
    for p, q in zip(HELIDECK, HELIDECK[1:] + HELIDECK[:1]):
        e = np.array(q) - np.array(p)
        out = np.array((e[1], -e[0])) / np.linalg.norm(e)
        if np.dot(out, np.array(p) - c) < 0:
            out = -out
        p2, q2 = np.array(p) + out * 1.5, np.array(q) + out * 1.5
        pts = [(p[0], p[1], HELI_Z - 0.1), (q[0], q[1], HELI_Z - 0.1), (q2[0], q2[1], HELI_Z - 0.35), (p2[0], p2[1], HELI_Z - 0.35)]
        u = float(np.linalg.norm(e)) / 1.0
        b.quad("net", fl.orient(pts, (0, 0, 1)), ((0, 0), (u, 0), (u, 1.5), (0, 1.5)))
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(e) / 2.5))):
            a = np.array(p) + e * t
            b.beam("frame", (a[0], a[1], HELI_Z - 0.1), (a[0] + out[0] * 1.5, a[1] + out[1] * 1.5, HELI_Z - 0.35), 0.05, 0.05)


def stair(b):
    s = tf.stair_tower(*STAIR, axis="y")
    for centre, size in s["steps"] + s["landings"]:
        c, h = np.array(centre), np.array(size) / 2
        b.box("galv", tuple(c - h), tuple(c + h))
    for p, q in s["stringers"]:
        b.beam("yellow", p, q, 0.04, 0.25)
        b.beam("yellow", (p[0], p[1], p[2] + 1.0), (q[0], q[1], q[2] + 1.0), 0.05, 0.05)
    for key, size in (("columns", 0.16), ("beams", 0.14), ("braces", 0.08)):
        for p, q in s["frame"][key]:
            b.beam("yellow", p, q, size, size)
    lo, hi = s["low"], s["high"]
    for z in np.arange(STAIR[1], STAIR[2] + 0.1, 3.75):
        for (x0, y0), (x1, y1) in (((hi[0], lo[1]), (hi[0], hi[1])), ((lo[0], lo[1]), (hi[0], lo[1]))):
            r = tf.rail((x0, y0), (x1, y1), z)
            for p, q in r["rails"] + r["posts"]:
                b.beam("yellow", p, q, 0.05, 0.05)


def floods(stage, b):
    UsdGeom.Scope.Define(stage, f"{facade.ROOT}/Lights")
    for i, (pos, target, kelvin) in enumerate(FLOODS):
        d = np.array(target) - np.array(pos)
        d /= np.linalg.norm(d)
        root = UsdGeom.Xform.Define(stage, f"{facade.ROOT}/Lights/flood_{i:02d}")
        ops = UsdGeom.Xformable(root)
        ops.AddTranslateOp().Set(Gf.Vec3d(*(np.array(pos) + d * 0.45)))
        w, x, y, z = tf.aim_from_down(d)
        ops.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
        light = UsdLux.SphereLight.Define(stage, f"{root.GetPath()}/light")
        light.CreateRadiusAttr(0.12)
        light.CreateIntensityAttr(FLOOD_POWER)
        light.CreateColorAttr(Gf.Vec3f(*tf.kelvin_rgb(kelvin)))
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr().Set(55.0)
        shaping.CreateShapingConeSoftnessAttr().Set(0.5)
        p = np.array(pos)
        b.box("frame", tuple(p - (0.25, 0.25, -0.05)), tuple(p + (0.25, 0.25, 0.25)))
        b.box("lamp", tuple(p - (0.2, 0.2, 0.0)), tuple(p + (0.2, 0.2, 0.05)))


def offline_lights(stage, lamps):
    UsdGeom.Scope.Define(stage, facade.OFFLINE)
    for i, p in enumerate(lamps):
        light = UsdLux.SphereLight.Define(stage, f"{facade.OFFLINE}/door_lamp_{i:02d}")
        light.CreateRadiusAttr(0.08)
        light.CreateIntensityAttr(9000.0)
        light.CreateColorAttr(Gf.Vec3f(*facade.COLORS["lamp"]))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*p))


def build() -> Path:
    if facade.LAYER.exists():
        facade.LAYER.unlink()
    stage = Usd.Stage.CreateNew(str(facade.LAYER))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, facade.ROOT)
    UsdGeom.Scope.Define(stage, facade.LOOKS)
    UsdGeom.Scope.Define(stage, f"{facade.ROOT}/Geo")
    looks = materials(stage)
    b = Batch()
    lamps: list = []
    openings = sum(build_wall(b, w, lamps, EXTRA_SIGNS) for w in fl.WALLS)
    for x, y, z, _ in LIFEBOATS:
        lifeboat(b, x, y, z)
    helideck(b)
    stair(b)
    floods(stage, b)
    quads = b.write(stage, looks)
    offline_lights(stage, lamps)
    world = stage.GetRootLayer().GetPrimAtPath("/World")
    world.specifier = Sdf.SpecifierOver
    world.typeName = ""
    for path in HIDE:
        stage.OverridePrim(path).SetActive(False)
    stage.GetRootLayer().Save()
    print(f"FACADE_WRITTEN {facade.LAYER} openings={openings} quads={quads} lamps={len(lamps)}")
    return facade.LAYER


if __name__ == "__main__":
    build()

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dressing_geometry as dg
import greebles as g
import photoreal as pr
import topside_frames as tf
from photoreal_usd import bind_mdl
from scene_builder import ASSETS, rel
from tools.pbr_material import texture_material

LOOKS = f"{g.ROOT}/Looks"
WL = "/World/Looks"
MAT = {
    "steel": f"{WL}/support_steel",
    "galv": f"{WL}/topside_galv",
    "black": f"{WL}/topside_black",
    "red": f"{WL}/topside_valve_red",
    "yellow": f"{WL}/topside_yellow",
    "white": f"{WL}/topside_steel_white",
    "stainless": f"{WL}/topside_insulated",
    "grey": f"{WL}/topside_vessel_grey",
    "gauge": f"{WL}/gauge_body",
    "steel_rust": f"{WL}/topside_steel_rust",
}
X_TO_Z = tuple(tf.segment_pose((0, 0, 0), (0, 0, 1))[1])
X_TO_Y = tuple(tf.segment_pose((0, 0, 0), (0, 1, 0))[1])
IDENT = (1.0, 0.0, 0.0, 0.0)

_stage = []


def S():
    return _stage[0]


LINE_LOOKS = {"main": f"{WL}/topside_pipe_green", "rusty": f"{WL}/topside_steel_rust", "grey": f"{WL}/topside_pipe_grey",
              "orange": f"{WL}/pipe"}


def line_mat(line):
    return LINE_LOOKS[line]


def mesh_data(points, counts, indices, uvs=None, normals=None):
    return {"points": np.asarray(points, float), "counts": np.asarray(counts, int), "indices": np.asarray(indices, int),
            "uvs": None if uvs is None else np.asarray(uvs, float), "normals": None if normals is None else np.asarray(normals, float)}


def merge(*parts):
    pts, counts, idx, offset = [], [], [], 0
    for p in parts:
        pts.append(p["points"])
        counts.append(p["counts"])
        idx.append(p["indices"] + offset)
        offset += len(p["points"])
    return mesh_data(np.vstack(pts), np.concatenate(counts), np.concatenate(idx))


def prism(sides, radius=0.5, length=1.0):
    a = np.linspace(0, 2 * math.pi, sides, endpoint=False) + math.pi / sides
    ring = np.stack([np.cos(a), np.sin(a)], 1) * radius
    pts = [(-length / 2, y, z) for y, z in ring] + [(length / 2, y, z) for y, z in ring]
    counts, idx = [], []
    for k in range(sides):
        j = (k + 1) % sides
        counts.append(4)
        idx += [k, j, sides + j, sides + k]
    counts += [sides, sides]
    idx += list(range(sides - 1, -1, -1)) + list(range(sides, 2 * sides))
    return mesh_data(pts, counts, idx)


def tube_part(path_pts, radius, sides=8):
    m = dg.tube(np.asarray(path_pts, float), radius, sides)
    return mesh_data(m["points"], m["counts"], m["indices"], m["uvs"], m["normals"])


def circle(n, r=1.0, a0=0.0, a1=2 * math.pi):
    a = np.linspace(a0, a1, n)
    return np.stack([np.zeros(n), r * np.cos(a), r * np.sin(a)], 1)


def torus(minor):
    return tube_part(circle(41), minor, 8)


def ubolt():
    legs = np.linspace(-1.35, 0.0, 6)
    left = np.stack([np.zeros(6), -np.ones(6), legs], 1)
    arc = circle(17, 1.0, math.pi, 0.0)[1:-1]
    right = left[::-1] * [1, -1, 1]
    return tube_part(np.vstack([left, arc, right]), 0.07, 6)


def handwheel():
    rim = tube_part(circle(33), 0.08, 6)
    spokes = [tube_part(np.array([(0, 0, 0), (0, math.cos(a) * 0.5, math.sin(a) * 0.5), (0, math.cos(a), math.sin(a))]), 0.05, 5)
              for a in (0.3, 0.3 + 2 * math.pi / 3, 0.3 + 4 * math.pi / 3)]
    hub = prism(10, 0.14, 0.2)
    return merge(rim, *spokes, hub)


def link():
    m = dg.link_mesh(0.072, 0.032, 0.0055)
    return mesh_data(m["points"], m["counts"], m["indices"], m["uvs"], m["normals"])


def ramp():
    pts = [(-0.5, -0.5, 0), (-0.5, 0.5, 0), (-0.5, 0.2, 1), (-0.5, -0.2, 1), (0.5, -0.5, 0), (0.5, 0.5, 0), (0.5, 0.2, 1), (0.5, -0.2, 1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 4, 7, 3), (3, 7, 6, 2), (2, 6, 5, 1), (1, 5, 4, 0)]
    return mesh_data(pts, [4] * 6, [i for q in quads for i in q])


def wedge():
    pts = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, -0.5, 1), (-0.5, 0.5, 0), (0.5, 0.5, 0), (0.5, 0.5, 1)]
    faces = [(0, 2, 1), (3, 4, 5), (0, 3, 5, 2), (1, 2, 5, 4), (0, 1, 4, 3)]
    return mesh_data(pts, [len(f) for f in faces], [i for f in faces for i in f])


def zrot(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, 0.0, math.sin(a))


PROTOS = {"ramp": ramp, "wedge": wedge, "hex": lambda: prism(6), "torus": lambda: torus(0.04), "ubolt": ubolt, "wheel": handwheel, "link": link}


def write_mesh(path, m, material=None, double=False):
    mesh = UsdGeom.Mesh.Define(S(), path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m["points"], np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(m["counts"], np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m["indices"], np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(double)
    if m.get("normals") is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m["normals"], np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if m.get("uvs") is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(m["uvs"], np.float32)))
    pts = np.asarray(m["points"])
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))])
    if material is not None:
        bind(mesh.GetPrim(), material)
    return mesh


def bind(prim, material):
    target = material if isinstance(material, str) else str(material.GetPath())
    UsdShade.MaterialBindingAPI.Apply(prim).GetDirectBindingRel().SetTargets([Sdf.Path(target)])


class Kit:
    def __init__(self, root):
        self.root = root
        self.items = {}

    def add(self, material, proto, pos, quat=IDENT, scale=(1, 1, 1)):
        self.items.setdefault(material, {}).setdefault(proto, []).append((tuple(pos), tuple(quat), tuple(scale)))

    def seg(self, material, proto, p, q, d0, d1=None):
        if math.dist(p, q) < 1e-5:
            return
        mid, quat, length = tf.segment_pose(p, q)
        self.add(material, proto, mid, quat, (length, d0, d0 if d1 is None else d1))

    def box(self, material, centre, size):
        self.add(material, "box", centre, IDENT, size)

    def write(self):
        UsdGeom.Scope.Define(S(), self.root)
        total = 0
        for n, (material, buckets) in enumerate(self.items.items()):
            path = f"{self.root}/set_{n:02d}_{material.rsplit('/', 1)[-1]}"
            inst = UsdGeom.PointInstancer.Define(S(), path)
            protos, ids, pos, ori, sca = [], [], [], [], []
            for proto, entries in buckets.items():
                prim = self.proto(f"{path}/protos/{proto}", proto, material)
                k = len(protos)
                protos.append(prim.GetPath())
                for p, q, s in entries:
                    ids.append(k)
                    pos.append(Gf.Vec3f(*map(float, p)))
                    ori.append(Gf.Quath(float(q[0]), Gf.Vec3h(*(float(c) for c in q[1:]))))
                    sca.append(Gf.Vec3f(*map(float, s)))
            inst.CreatePrototypesRel().SetTargets(protos)
            inst.CreateProtoIndicesAttr(Vt.IntArray(ids))
            inst.CreatePositionsAttr(Vt.Vec3fArray(pos))
            inst.CreateOrientationsAttr(Vt.QuathArray(ori))
            inst.CreateScalesAttr(Vt.Vec3fArray(sca))
            total += len(ids)
        return total

    def proto(self, path, proto, material):
        UsdGeom.Scope.Define(S(), path.rsplit("/", 1)[0])
        if proto == "cyl":
            prim = UsdGeom.Cylinder.Define(S(), path)
            prim.CreateAxisAttr("X")
            prim.CreateRadiusAttr(0.5)
            prim.CreateHeightAttr(1.0)
        elif proto == "box":
            prim = UsdGeom.Cube.Define(S(), path)
            prim.CreateSizeAttr(1.0)
        elif proto == "sphere":
            prim = UsdGeom.Sphere.Define(S(), path)
            prim.CreateRadiusAttr(1.0)
        else:
            prim = write_mesh(path, PROTOS[proto]())
        bind(prim.GetPrim(), material)
        return prim


def paint_look(name, color, chip=0.1, rust=0.0, **extra):
    sub, values = pr.shader_inputs({"kind": "paint", "paint_color": color, "chip_amount": chip, "rust_amount": rust,
                                    "streak_amount": 0.0, "salt_amount": 0.0, **extra})
    values.update(paint_scale=extra.get("scale", 0.6))
    return bind_mdl(S(), f"{LOOKS}/{name}", sub, values, define=True)


def decal_look(name, color_tex, mask, roughness=0.5, oil=0.0, tint=(1.0, 1.0, 1.0), strength=1.0):
    values = {"mask": mask, "use_color_tex": color_tex is not None, "tint": tint, "roughness": roughness,
              "strength": strength, "oil": oil, "wetness": 0.0}
    if color_tex is not None:
        values["color_tex"] = color_tex
    return bind_mdl(S(), f"{LOOKS}/{name}", "rig_decal", values, define=True)


_atlas = []


def atlas_look():
    if not _atlas:
        _atlas.append(decal_look("signs_atlas", g.TEXTURES / "atlas.jpg", g.TEXTURES / "atlas_mask.png", roughness=0.42))
    return _atlas[0]


def atlased(m, name):
    m["uvs"] = np.array([g.atlas_uv(name, u, v) for u, v in m["uvs"]])
    return m


def quad_mesh(centre, u, v, hu, hv, uv=((0, 0), (1, 1))):
    c, u, v = (np.asarray(a, float) for a in (centre, u, v))
    pts = [c - u * hu - v * hv, c + u * hu - v * hv, c + u * hu + v * hv, c - u * hu + v * hv]
    (u0, v0), (u1, v1) = uv
    n = np.cross(u, v)
    return mesh_data(pts, [4], [0, 1, 2, 3], [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], [n] * 4)


def disc_mesh(centre, normal, radius, n=32):
    normal = np.asarray(normal, float)
    u = np.cross((0, 0, 1), normal)
    v = np.array((0.0, 0.0, 1.0))
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = [np.asarray(centre) + radius * (math.cos(t) * u + math.sin(t) * v) for t in a]
    uvs = [(0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t)) for t in a]
    return mesh_data([centre] + ring, [3] * n, [i for k in range(n) for i in (0, 1 + k, 1 + (k + 1) % n)],
                     [(0.5, 0.5)] + uvs, [normal] * (n + 1))


def sleeve_mesh(x0, x1, y, z, radius, rows=4, sides=36):
    pts, uvs, normals = [], [], []
    for i in range(rows + 1):
        s = i / rows
        for j in range(sides + 1):
            t = j / sides
            a = 2 * math.pi * (t - 0.25)
            n = (0.0, math.cos(a), math.sin(a))
            pts.append((x1 + (x0 - x1) * s, y + radius * n[1], z + radius * n[2]))
            uvs.append((s, t))
            normals.append(n)
    idx = []
    for i in range(rows):
        for j in range(sides):
            a = i * (sides + 1) + j
            idx += [a, a + sides + 1, a + sides + 2, a + 1]
    return mesh_data(pts, [4] * (rows * sides), idx, uvs, normals)


def flange_joint(kit, bolts, centre, axis, r, material, new=True):
    c, ax = np.asarray(centre, float), np.asarray(axis, float)
    e1 = np.cross(ax, (0.0, 0.0, 1.0))
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array((1.0, 0.0, 0.0))
    e2 = np.cross(ax, e1)
    f = g.flange_dims(r)
    at = lambda t, off=0.0: tuple(c + ax * t + off)
    hw = f["half_width"]
    if new:
        for side in (-1, 1):
            kit.seg(material, "cyl", at(side * 0.004), at(side * 0.049), 2 * f["outer"])
            kit.seg(material, "cyl", at(side * 0.048), at(side * 0.11), 2 * r * 1.12)
        kit.seg(MAT["black"], "cyl", at(-0.004), at(0.004), 2 * f["outer"] - 0.006)
        hw = 0.049
    else:
        kit.seg(MAT["black"], "cyl", at(-0.004), at(0.004), 2 * f["outer"] + 0.004)
    for du, dv in g.bolt_ring(f["pcd"], f["count"]):
        off = e1 * du + e2 * dv
        bolts.seg(MAT["steel"], "cyl", at(-hw - 0.03, off), at(hw + 0.03, off), 2 * f["bolt_r"])
        for side in (-1, 1):
            m = side * (hw + 0.011)
            bolts.seg(MAT["steel"], "hex", at(m - 0.011, off), at(m + 0.011, off), 2 * f["nut_r"])


def flanges(kit):
    for new, table in ((False, g.EXISTING_FLANGES), (True, g.NEW_FLANGES)):
        for line, xs in table.items():
            y, z, r, _ = g.LINES[line]
            for x in xs:
                flange_joint(kit, kit, (x, y, z), (1.0, 0.0, 0.0), r, line_mat(line), new)


RACK_SERVICES = ("insulated", "pipe_grey", "pipe_red", "pipe_yellow", "pipe_green", "insulated")


def topside_racks(live, off):
    import topside as ts

    for k, (start, end, z0, tiers, width, count) in enumerate(ts.RACKS):
        r = tf.pipe_rack(start, end, z0, tiers=tiers, width=width, pipes_per_tier=count, seed=11 + k)
        length = math.dist(start, end)
        portals = tf.stations(0.0, length, 6.0)
        mids = [(p + q) / 2 for p, q in zip(portals, portals[1:])]
        yaw = IDENT if r["along_x"] else (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        for n, ((a, b), radius) in enumerate(r["pipes"]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            u = (b - a) / np.linalg.norm(b - a)
            material = f"{WL}/topside_{RACK_SERVICES[n % len(RACK_SERVICES)]}"
            for t in portals:
                p = a + u * t
                shoe = (0.3, 2 * radius + 0.08, 0.1) if r["along_x"] else (2 * radius + 0.08, 0.3, 0.1)
                live.box(MAT["galv"], (p[0], p[1], p[2] - radius - 0.05), shoe)
                s = radius + 0.006
                off.add(MAT["galv"], "ubolt", tuple(p), yaw, (s, s, s))
            for j, t in enumerate(mids):
                if (j + n) % 2 == 0:
                    flange_joint(live, off, a + u * (t + 0.7 * (n % 3 - 1)), u, radius, material)


def welds(kit):
    for line, xs in g.WELDS.items():
        y, z, r, _ = g.LINES[line]
        for x in xs:
            kit.add(line_mat(line), "torus", (x, y, z), IDENT, (r + 0.0015,) * 3)


def supports(kit):
    y, z, r, _ = g.LINES["main"]
    for x in g.MAIN_SUPPORTS:
        h = z - r - 0.016
        kit.box(MAT["steel"], (x, y, h / 2), (0.14, 0.36, h))
        kit.box(MAT["steel"], (x, y, 0.015), (0.34, 0.5, 0.03))
        kit.box(MAT["steel"], (x - 0.075, y, h / 2), (0.012, 0.3, h * 0.9))
        for dx in (-0.13, 0.13):
            for dy in (-0.19, 0.19):
                kit.seg(MAT["galv"], "hex", (x + dx, y + dy, 0.03), (x + dx, y + dy, 0.05), 0.035)
    for line, xs in g.SUPPORTS.items():
        y, z, r, _ = g.LINES[line]
        s = r + 0.006
        for x in xs + (g.MAIN_SUPPORTS if line == "main" else ()):
            kit.add(MAT["galv"], "ubolt", (x, y, z), IDENT, (s, s, s))
            kit.box(MAT["steel"], (x, y, z - r - 0.008), (0.16, 2 * r + 0.12, 0.016))
            kit.box(MAT["steel"], (x, y, z - r + 0.012), (0.12, r * 0.9, 0.024))
            for side in (-1, 1):
                kit.seg(MAT["galv"], "hex", (x, y + side * s, z - r - 0.016), (x, y + side * s, z - r + 0.004), 0.03)


def labels():
    root = f"{g.ZONE_ROOT}/Labels"
    UsdGeom.Scope.Define(S(), root)
    for k, (line, x, name, direction) in enumerate(g.LABELS):
        y, z, r, _ = g.LINES[line]
        key = g.label_key(name, direction)
        length = 0.22 + 2.2 * r
        write_mesh(f"{root}/label_{k:02d}", atlased(sleeve_mesh(x - length / 2, x + length / 2, y, z, r + 0.0025), key), atlas_look())


def gauges(kit):
    root = f"{g.ZONE_ROOT}/Dials"
    UsdGeom.Scope.Define(S(), root)
    for k, (line, x, dial, side) in enumerate(g.GAUGES):
        y, z, r, _ = g.LINES[line]
        top = z + r
        kit.seg(MAT["galv"], "cyl", (x, y, top - 0.02), (x, y, top + 0.1), 0.026)
        kit.seg(MAT["steel"], "hex", (x, y, top + 0.02), (x, y, top + 0.045), 0.05)
        kit.box(MAT["galv"], (x, y, top + 0.075), (0.05, 0.045, 0.05))
        kit.seg(MAT["red"], "cyl", (x, y, top + 0.075), (x + 0.07, y, top + 0.075), 0.012)
        kit.seg(MAT["galv"], "cyl", (x, y, top + 0.1), (x, y, top + 0.15), 0.02)
        c = (x, y, top + 0.205)
        kit.seg(MAT["black"], "cyl", (x, y - 0.02 * side, c[2]), (x, y + 0.02 * side, c[2]), 0.13)
        kit.add(MAT["galv"], "torus", (x, y + 0.021 * side, c[2]), X_TO_Y, (0.06, 0.06, 0.06))
        write_mesh(f"{root}/dial_{k:02d}", atlased(disc_mesh((x, y + 0.0215 * side, c[2]), (0, side, 0), 0.056), dial), atlas_look())


def handwheels(kit):
    for line, x, d in g.HANDWHEELS:
        y, z, r, _ = g.LINES[line]
        kit.seg(MAT["grey"], "cyl", (x - 0.11, y, z), (x + 0.11, y, z), r * 2.7)
        kit.seg(MAT["grey"], "cyl", (x - 0.15, y, z), (x - 0.11, y, z), r * 2.9)
        kit.seg(MAT["grey"], "cyl", (x + 0.11, y, z), (x + 0.15, y, z), r * 2.9)
        for xx in (x - 0.13, x + 0.13):
            for dy, dz in g.bolt_ring(r * 1.3, 8):
                kit.seg(MAT["steel"], "hex", (xx - 0.03, y + dy, z + dz), (xx + 0.03, y + dy, z + dz), r * 0.22)
        bonnet = z + r * 1.35
        kit.seg(MAT["grey"], "cyl", (x, y, z), (x, y, bonnet + 0.12), r * 1.4)
        kit.seg(MAT["grey"], "hex", (x, y, bonnet + 0.12), (x, y, bonnet + 0.16), r * 2.0)
        kit.seg(MAT["galv"], "cyl", (x, y, bonnet + 0.16), (x, y, bonnet + 0.32), 0.022)
        kit.add(MAT["red"], "wheel", (x, y, bonnet + 0.28), X_TO_Z, (d / 2, d / 2, d / 2))


def transmitters(kit):
    root = f"{g.ZONE_ROOT}/Tubing"
    UsdGeom.Scope.Define(S(), root)
    for k, (x, y, line, tap) in enumerate(g.TRANSMITTERS):
        py, pz, pr_, _ = g.LINES[line]
        top = 1.28
        kit.box(MAT["galv"], (x, y, 0.006), (0.22, 0.22, 0.012))
        kit.seg(MAT["galv"], "cyl", (x, y, 0.0), (x, y, top), 0.06)
        kit.seg(MAT["galv"], "cyl", (x, y, top), (x, y, top + 0.01), 0.07)
        side = 1 if y > py else -1
        bx = (x, y + 0.08 * side, top - 0.12)
        kit.box(MAT["galv"], (x, y + 0.04 * side, top - 0.12), (0.06, 0.08, 0.1))
        kit.box(MAT["white"], bx, (0.1, 0.08, 0.12))
        kit.seg(MAT["white"], "cyl", (x, bx[1], bx[2] + 0.06), (x, bx[1], bx[2] + 0.13), 0.1)
        kit.seg(MAT["galv"], "cyl", (x, bx[1] + 0.05 * side, bx[2] + 0.095), (x, bx[1] + 0.052 * side, bx[2] + 0.095), 0.07)
        kit.box(MAT["grey"], (x, bx[1], bx[2] - 0.09), (0.09, 0.06, 0.06))
        kit.seg(MAT["galv"], "cyl", (tap, py, pz + pr_ - 0.01), (tap, py, pz + pr_ + 0.07), 0.03)
        kit.box(MAT["galv"], (tap, py, pz + pr_ + 0.09), (0.045, 0.04, 0.04))
        ctrl = [(tap, py, pz + pr_ + 0.11), (tap, py, pz + pr_ + 0.22), (tap + (x - tap) * 0.5, (py + y) / 2, pz + pr_ + 0.26),
                (x, bx[1], bx[2] + 0.02), (x, bx[1], bx[2] - 0.12)]
        tube = dg.spline(np.array(ctrl), 10)
        write_mesh(f"{root}/impulse_{k}", tube_part(tube, 0.0065, 8), MAT["stainless"])
        cable = dg.spline(np.array([(x, bx[1], bx[2] - 0.12), (x, bx[1] + 0.05 * side, 0.4), (x + 0.1, y + 0.14 * side, 0.03),
                                    (x + 0.6, y + 0.3 * side, 0.012)]), 10)
        write_mesh(f"{root}/signal_{k}", tube_part(cable, 0.008, 6), f"{LOOKS}/cable_black")


def tubing(kit):
    root = f"{g.ZONE_ROOT}/Tubing"
    UsdGeom.Scope.Define(S(), root)
    x0, x1 = g.TUBING_X
    for k, y in enumerate(g.TUBING_Y):
        z = 0.07 + 0.0 * k
        ctrl = [(x0 - 0.2, y - 0.25, 0.75), (x0 - 0.05, y - 0.05, 0.25), (x0 + 0.3, y, z), (0.0, y, z), (x1 - 0.3, y, z),
                (x1 + 0.05, y - 0.05, 0.25), (x1 + 0.2, y - 0.25, 0.75)]
        write_mesh(f"{root}/bundle_{k}", tube_part(dg.spline(np.array(ctrl), 12), 0.0065, 8), MAT["stainless"])
    for x in g.stations(x0 + 0.4, x1 - 0.4, 1.2):
        yc = sum(g.TUBING_Y) / 3
        kit.box(MAT["galv"], (x, yc, 0.035), (0.03, 0.16, 0.012))
        kit.box(MAT["galv"], (x, yc, 0.018), (0.012, 0.06, 0.036))
        kit.box(MAT["galv"], (x, yc, 0.075), (0.03, 0.15, 0.008))


def column_kit(kit):
    root = f"{g.ZONE_ROOT}/Signs"
    UsdGeom.Scope.Define(S(), root)
    for k, (col, face, z, kind) in enumerate(g.JBOXES):
        nx, ny = g.facing(face)
        depth = {"jb": 0.12, "esd": 0.1, "socket": 0.11, "switch": 0.06}[kind]
        w, h = {"jb": (0.24, 0.3), "esd": (0.14, 0.18), "socket": (0.16, 0.22), "switch": (0.09, 0.12)}[kind]
        cx, cy = g.on_column(col, face, 0.125 + depth / 2 + 0.01)
        size = (depth, w, h) if nx else (w, depth, h)
        body = {"jb": MAT["grey"], "esd": MAT["yellow"], "socket": MAT["grey"], "switch": MAT["white"]}[kind]
        kit.box(body, (cx, cy, z), size)
        bx, by = g.on_column(col, face, 0.125 + 0.005)
        kit.box(MAT["galv"], (bx, by, z), (0.01 if nx else w + 0.06, 0.01 if ny else w + 0.06, h + 0.06))
        fx, fy = g.on_column(col, face, 0.125 + depth + 0.01)
        if kind == "esd":
            kit.seg(MAT["red"], "cyl", (fx, fy, z + 0.02), (fx + nx * 0.03, fy + ny * 0.03, z + 0.02), 0.035)
            kit.seg(MAT["red"], "cyl", (fx + nx * 0.03, fy + ny * 0.03, z + 0.02), (fx + nx * 0.045, fy + ny * 0.045, z + 0.02), 0.06)
        if kind == "socket":
            kit.seg(MAT["yellow"], "cyl", (fx, fy, z - 0.02), (fx + nx * 0.05, fy + ny * 0.05, z - 0.02), 0.08)
        if kind == "switch":
            kit.box(MAT["black"], (fx + nx * 0.006, fy + ny * 0.006, z), (0.012 if nx else 0.02, 0.012 if ny else 0.02, 0.04))
        if kind == "jb":
            for dx in (-0.06, 0.0, 0.06):
                ox, oy = (0.0, dx) if nx else (dx, 0.0)
                kit.seg(MAT["galv"], "hex", (cx + ox, cy + oy, z - h / 2 - 0.025), (cx + ox, cy + oy, z - h / 2), 0.03)
        lx, ly = (0.0, 0.07) if nx else (0.07, 0.0)
        cx2, cy2 = g.on_column(col, face, 0.125 + 0.03)
        p0 = (cx2 + lx, cy2 + ly, z + h / 2)
        p1 = (cx2 + lx, cy2 + ly, g.TRAY_Z - 0.03)
        kit.seg(MAT["galv"], "cyl", (p0[0], p0[1], p0[2] - 0.01), p1, 0.032)
        for zz in g.stations(z + h / 2 + 0.25, g.TRAY_Z - 0.2, 0.6):
            kit.box(MAT["galv"], (p0[0] - nx * 0.012, p0[1] - ny * 0.012, zz), (0.03 if nx else 0.07, 0.07 if nx else 0.03, 0.025))
        kit.seg(MAT["galv"], "cyl", (p0[0], p0[1], p0[2] - 0.01), (p0[0], p0[1], p0[2] + 0.04), 0.045)
    for k, (col, face, z, name, w, h) in enumerate(g.SIGNS):
        nx, ny = g.facing(face)
        n = np.array((nx, ny, 0.0))
        u = np.cross((0, 0, 1), n)
        cx, cy = g.on_column(col, face, 0.125 + 0.016)
        write_mesh(f"{root}/sign_{k:02d}", atlased(quad_mesh((cx, cy, z), u, (0, 0, 1), w / 2, h / 2), name), atlas_look())
        bx, by = g.on_column(col, face, 0.125 + 0.008)
        if not name.startswith("hazard"):
            kit.box(MAT["galv"], (bx, by, z), (0.012 if nx else w + 0.02, 0.012 if ny else w + 0.02, h + 0.02))
        for dz in (-h / 2 + 0.06, h / 2 - 0.06):
            kit.box(MAT["steel"], (col[0], col[1], z + dz), (0.27, 0.27, 0.025))


def stripes(stripe_look):
    root = f"{g.ZONE_ROOT}/Stripes"
    UsdGeom.Scope.Define(S(), root)
    for k, (cx, cy) in enumerate(g.STRIPES):
        for j, face in enumerate(("+x", "-x", "+y", "-y")):
            nx, ny = g.facing(face)
            n = np.array((nx, ny, 0.0))
            u = np.cross((0, 0, 1), n)
            c = (cx + nx * 0.1265, cy + ny * 0.1265, 0.3)
            write_mesh(f"{root}/col_{k}_{j}", quad_mesh(c, u, (0, 0, 1), 0.1265, 0.3, ((0, 0), (0.42, 1.0))), stripe_look)


def tray(kit, root):
    z = g.TRAY_Z
    for n, (y, x0, x1) in enumerate(g.TRAYS):
        mid = (x0 + x1) / 2
        for side in (-0.2, 0.2):
            kit.box(MAT["galv"], (mid, y + side, z + 0.04), (x1 - x0, 0.012, 0.08))
            kit.box(MAT["galv"], (mid, y + side * 0.96, z + 0.003), (x1 - x0, 0.02, 0.006))
        for x in g.stations(x0, x1, 0.3):
            kit.box(MAT["galv"], (x, y, z), (0.025, 0.4, 0.01))
        for x in g.stations(x0 + 0.3, x1 - 0.3, 1.5):
            kit.seg(MAT["galv"], "cyl", (x, y, z - 0.03), (x, y, 3.95), 0.016)
            kit.box(MAT["galv"], (x, y, z - 0.025), (0.04, 0.5, 0.03))
        rng = np.random.default_rng(n)
        for c, (dy, r, look) in enumerate(((-0.13, 0.016, "cable_black"), (-0.08, 0.012, "cable_black"), (-0.03, 0.02, "cable_black"),
                                           (0.03, 0.011, "cable_blue"), (0.08, 0.014, "cable_black"), (0.13, 0.013, "cable_orange"))):
            xs = np.linspace(x0, x1, 30)
            ys = y + dy + 0.012 * np.sin(xs * rng.uniform(0.6, 1.4) + rng.uniform(0, 6))
            zs = np.full_like(xs, z + 0.006 + r)
            write_mesh(f"{root}/tray_cable_{n}_{c}", tube_part(np.stack([xs, ys, zs], 1), r, 6), f"{LOOKS}/{look}")


def drains(kit):
    for x, y in g.DRAINS:
        kit.box(MAT["galv"], (x, y, 0.006), (0.32, 0.32, 0.012))
        kit.box(MAT["black"], (x, y, 0.004), (0.26, 0.26, 0.008))
        for k in range(9):
            kit.box(MAT["steel"], (x - 0.12 + k * 0.03, y, 0.011), (0.008, 0.26, 0.01))


def rag_mesh(cx, cy, rot, seed, size=(0.34, 0.26)):
    rng = np.random.default_rng(seed)
    res = 14
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, res), np.linspace(-0.5, 0.5, res), indexing="ij")
    r = np.hypot(u, v)
    lift = 0.006 + 0.03 * np.clip(1 - r * 1.9, 0, 1) * (0.5 + 0.5 * np.sin(u * 17 + rng.uniform(0, 6)) * np.cos(v * 13))
    lift += 0.008 * rng.random(u.shape)
    a = math.radians(rot)
    x = cx + (u * size[0]) * math.cos(a) - (v * size[1]) * math.sin(a)
    y = cy + (u * size[0]) * math.sin(a) + (v * size[1]) * math.cos(a)
    pts = np.stack([x, y, lift], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(res - 1), np.arange(res - 1), indexing="ij")
    q = i * res + j
    idx = np.stack([q, q + res, q + res + 1, q + 1], -1).reshape(-1)
    uvs = np.stack([u + 0.5, v + 0.5], -1).reshape(-1, 2)
    return mesh_data(pts, np.full(len(idx) // 4, 4), idx, uvs)


def bucket_mesh(cx, cy, r0=0.12, r1=0.145, h=0.3, n=24):
    a = np.linspace(0, 2 * math.pi, n + 1)
    pts, idx, counts = [], [], []
    for r, z in ((r0, 0.0), (r1, h), (r1 - 0.006, h), (r0 - 0.006, 0.012)):
        pts += [(cx + r * math.cos(t), cy + r * math.sin(t), z) for t in a]
    row = n + 1
    for ring in range(3):
        for j in range(n):
            s = ring * row + j
            idx += [s, s + 1, s + row + 1, s + row]
            counts.append(4)
    idx += list(range(0, n)) [::-1]
    counts.append(n)
    idx += list(range(3 * row, 3 * row + n))
    counts.append(n)
    return mesh_data(pts, counts, idx)


def debris(kit):
    root = f"{g.ZONE_ROOT}/Debris"
    UsdGeom.Scope.Define(S(), root)
    rag = paint_look("rag", (0.32, 0.27, 0.2), chip=0.0, grime_amount=0.9, scale=0.3)
    bucket = paint_look("bucket", (0.62, 0.24, 0.04), chip=0.05, grime_amount=0.6, scale=0.5)
    hose = f"{LOOKS}/hose_yellow"
    glove = paint_look("glove", (0.6, 0.45, 0.12), chip=0.0, grime_amount=0.8, scale=0.3)
    oil = decal_look("oil", None, ASSETS / "decals" / "AsphaltDamageSet002" / "opacity.jpg", roughness=0.1, oil=1.0,
                     tint=(0.018, 0.016, 0.014), strength=0.9)
    for k, (kind, (x, y), rot) in enumerate(g.DEBRIS):
        if kind == "rag":
            write_mesh(f"{root}/rag_{k}", rag_mesh(x, y, rot, k), rag, double=True)
        elif kind == "bucket":
            write_mesh(f"{root}/bucket_{k}", bucket_mesh(x, y), bucket, double=True)
            handle = circle(17, 0.145, 0.0, math.pi)
            handle = np.stack([x + handle[:, 1], y + handle[:, 0], 0.3 + handle[:, 2] * 0.8], 1)
            write_mesh(f"{root}/bucket_handle_{k}", tube_part(handle, 0.004, 5), MAT["galv"])
            write_mesh(f"{root}/bucket_rag_{k}", rag_mesh(x + 0.02, y, 40, 99, (0.2, 0.16)), rag, double=True)
        elif kind in ("cable_coil", "hose_coil"):
            look, r = (f"{LOOKS}/cable_black", 0.011) if kind == "cable_coil" else (hose, 0.018)
            for layer in range(2):
                pts = dg.coil((x, y, 0.0), 0.16 + layer * r, 0.3 - layer * r * 2, 5.0 - layer * 1.5, r * (1 + 1.9 * layer))
                write_mesh(f"{root}/{kind}_{k}_{layer}", tube_part(pts, r, 7), look)
            tail = dg.spline(np.array([(x + 0.3, y, r), (x + 0.6, y + 0.15, r), (x + 0.9, y + 0.05, r)]), 8)
            write_mesh(f"{root}/{kind}_{k}_tail", tube_part(tail, r, 7), look)
        elif kind == "puddle":
            a = math.radians(rot)
            write_mesh(f"{root}/puddle_{k}", quad_mesh((x, y, 0.004 + 0.0005 * k), (math.cos(a), math.sin(a), 0),
                                                      (-math.sin(a), math.cos(a), 0), 0.45, 0.45, ((0, 0.5), (0.5, 1))), oil)
        elif kind == "chain":
            t = np.linspace(0, 1, 60)
            path = np.stack([x + 0.5 * t + 0.08 * np.sin(t * 15), y + 0.25 * np.sin(t * 6), np.full_like(t, 0.012)], 1)
            for c, b in dg.chain_links(path, 0.05):
                q = Gf.Matrix3d(*b.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
                kit.add(MAT["steel"], "link", c, (q.GetReal(), *q.GetImaginary()), (1, 1, 1))
        elif kind == "nuts":
            rng = np.random.default_rng(k)
            for j in range(9):
                px, py = x + rng.uniform(-0.25, 0.25), y + rng.uniform(-0.07, 0.07)
                yaw = rng.uniform(0, math.pi)
                if j % 3:
                    kit.seg(MAT["steel"], "hex", (px, py, 0.0), (px, py, 0.016), 0.032)
                else:
                    kit.add(MAT["galv"], "torus", (px, py, 0.003), X_TO_Z, (0.018, 0.018, 0.018))
                    kit.seg(MAT["steel"], "cyl", (px, py, 0.012), (px + 0.09 * math.cos(yaw), py + 0.09 * math.sin(yaw), 0.012), 0.022)
        elif kind == "spanner":
            a = math.radians(rot)
            d = np.array((math.cos(a), math.sin(a), 0.0))
            c = np.array((x, y, 0.006))
            kit.seg(MAT["galv"], "box", tuple(c - d * 0.12), tuple(c + d * 0.12), 0.026, 0.009)
            for end in (-1, 1):
                kit.seg(MAT["galv"], "cyl", tuple(c + d * 0.135 * end + (0, 0, -0.0045)), tuple(c + d * 0.135 * end + (0, 0, 0.0045)), 0.05)
        elif kind == "gloves":
            for j, (dx, dy, rr) in enumerate(((0.0, 0.0, rot), (0.12, 0.05, rot + 35))):
                a = math.radians(rr)
                q = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
                kit.add(f"{LOOKS}/glove", "sphere", (x + dx, y + dy, 0.012), q, (0.11, 0.05, 0.014))
                kit.add(f"{LOOKS}/glove", "sphere", (x + dx + 0.1 * math.cos(a), y + dy + 0.1 * math.sin(a), 0.012), q, (0.04, 0.045, 0.012))


def emissive_look(name, diffuse, emit, gain):
    material = UsdShade.Material.Define(S(), f"{LOOKS}/{name}")
    shader = UsdShade.Shader.Define(S(), f"{LOOKS}/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    reader = UsdShade.Shader.Define(S(), f"{LOOKS}/{name}/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for key, path, slot, scale in (("diff", diffuse, "diffuseColor", 1.0), ("emit", emit, "emissiveColor", gain)):
        tex = UsdShade.Shader.Define(S(), f"{LOOKS}/{name}/{key}")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(path))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(scale, scale, scale, 1.0))
        shader.CreateInput(slot, Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def fascia(kit):
    root = f"{g.ROOT}/Topside/Fascia"
    UsdGeom.Scope.Define(S(), root)
    looks = [decal_look(f"fascia_streak_{tex}", None, ASSETS / "decals" / tex / "opacity.jpg", roughness=0.6,
                        tint=(0.2, 0.085, 0.03), strength=0.95) for tex in ("Leaking008", "Leaking009", "Leaking001")]
    n = 0
    for p, q, z0, height, out in g.FASCIA:
        p2, q2 = np.array(p, float), np.array(q, float)
        u = (q2 - p2) / np.linalg.norm(q2 - p2)
        o = np.array(out, float)
        normal = np.array((o[0], o[1], 0.0))
        across = np.cross((0.0, 0.0, 1.0), normal)
        top = z0 + height - 0.12
        for k, f in enumerate(g.scuppers(p, q)):
            c = p2 + u * f
            kit.box(MAT["black"], (c[0] + o[0] * 0.02, c[1] + o[1] * 0.02, top - 0.05), (0.06 if o[0] else 0.28, 0.06 if o[1] else 0.28, 0.12))
            kit.box(MAT["steel"], (c[0] + o[0] * 0.12, c[1] + o[1] * 0.12, top - 0.11), (0.22 if o[0] else 0.1, 0.1 if o[0] else 0.22, 0.03))
            length = 1.4 + 0.9 * ((k * 7) % 5) / 4
            centre = (c[0] + o[0] * 0.015, c[1] + o[1] * 0.015, top - 0.08 - length / 2)
            write_mesh(f"{root}/streak_{n:03d}", quad_mesh(centre, across, (0, 0, 1), 0.2 + 0.05 * (k % 3), length / 2), looks[n % 3])
            n += 1
    return n


def id_boards(kit):
    root = f"{g.ROOT}/Topside/Boards"
    UsdGeom.Scope.Define(S(), root)
    look = emissive_look("id_board", g.TEXTURES / "id_board.jpg", g.TEXTURES / "id_board_emit.jpg", 2.5)
    for k, ((x, y, z), face, w, h) in enumerate(g.ID_BOARDS):
        nx, ny = g.facing(face)
        n = np.array((nx, ny, 0.0))
        u = np.cross((0.0, 0.0, 1.0), n)
        write_mesh(f"{root}/board_{k}", quad_mesh((x + nx * 0.03, y + ny * 0.03, z), u, (0, 0, 1), w / 2, h / 2), look)
        kit.box(MAT["galv"], (x, y, z), (0.05 if nx else w + 0.2, 0.05 if ny else w + 0.2, h + 0.2))
        for side in (-1, 1):
            px, py = x + u[0] * side * w * 0.4, y + u[1] * side * w * 0.4
            kit.seg(MAT["galv"], "box", (px - nx * 0.1, py - ny * 0.1, z - h / 2 - 0.1), (px - nx * 0.1, py - ny * 0.1, z + h / 2 + 0.1), 0.1, 0.1)
            kit.seg(MAT["black"], "cyl", (px + nx * 0.4, py + ny * 0.4, z + h / 2 + 0.35), (px + nx * 0.1, py + ny * 0.1, z + h / 2 + 0.1), 0.05)


def oil_quad(path, x, y, size, rot, look, q=0, z=0.004):
    quarters = (((0, 0.5), (0.5, 1)), ((0.5, 0.5), (1, 1)), ((0, 0), (0.5, 0.5)), ((0.5, 0), (1, 0.5)))
    a = math.radians(rot)
    write_mesh(path, quad_mesh((x, y, z), (math.cos(a), math.sin(a), 0), (-math.sin(a), math.cos(a), 0), size / 2, size / 2,
                               quarters[q % 4]), look)


def cables_and_protectors(kit, root):
    for k, (look, r, ctrl) in enumerate(g.CABLES):
        pts = dg.spline(np.array([(x, y, r) for x, y in ctrl]), 12)
        write_mesh(f"{root}/deck_cable_{k}", tube_part(pts, r, 8), f"{LOOKS}/{look}")
    for (x, y), rot in g.PROTECTORS:
        for seg in (-0.45, 0.45):
            a = math.radians(rot)
            c = (x + math.cos(a) * seg, y + math.sin(a) * seg, 0.0)
            kit.add(MAT["yellow"], "ramp", c, zrot(rot), (0.88, 0.52, 0.055))
            kit.add(MAT["black"], "box", (c[0], c[1], 0.056), zrot(rot), (0.86, 0.17, 0.006))


def drip_trays(kit, root, oil):
    for k, ((x, y), rot) in enumerate(g.DRIP_TRAYS):
        kit.box(MAT["galv"], (x, y, 0.005), (0.72, 0.5, 0.01))
        for dx, dy, sx, sy in ((0.36, 0, 0.012, 0.5), (-0.36, 0, 0.012, 0.5), (0, 0.25, 0.72, 0.012), (0, -0.25, 0.72, 0.012)):
            kit.box(MAT["galv"], (x + dx, y + dy, 0.035), (sx, sy, 0.06))
        oil_quad(f"{root}/tray_oil_{k}", x, y, 0.62, 15 * k, oil, k, 0.012)
        oil_quad(f"{root}/tray_spill_{k}", x + 0.5, y - 0.15, 0.55, 40 + 25 * k, oil, k + 1)


def gas_rack(kit, root):
    x, y = g.GAS_RACK
    kit.box(MAT["galv"], (x, y, 0.01), (1.5, 0.45, 0.02))
    for dx in (-0.7, 0.7):
        kit.box(MAT["yellow"], (x + dx, y - 0.15, 0.7), (0.06, 0.06, 1.4))
        kit.box(MAT["yellow"], (x + dx, y + 0.18, 0.55), (0.06, 0.06, 1.1))
    for z in (0.45, 1.05):
        kit.box(MAT["yellow"], (x, y - 0.15, z), (1.46, 0.05, 0.05))
    kit.box(MAT["yellow"], (x, y + 0.18, 1.08), (1.46, 0.05, 0.04))
    looks = [f"{LOOKS}/gas_{c}" for c in ("maroon", "black", "grey", "blue", "maroon")]
    for k, dx in enumerate((-0.52, -0.26, 0.0, 0.26, 0.52)):
        c = (x + dx, y, 0.0)
        kit.seg(looks[k], "cyl", c, (c[0], c[1], 1.28), 0.23)
        kit.add(looks[k], "sphere", (c[0], c[1], 1.28), IDENT, (0.115, 0.115, 0.09))
        kit.seg(MAT["galv"], "cyl", (c[0], c[1], 1.34), (c[0], c[1], 1.47), 0.09)
        kit.seg(MAT["black"], "cyl", (c[0], c[1], 1.0), (c[0], c[1], 1.05), 0.235)
    for z in (0.75, 1.15):
        pts = dg.sag((x - 0.68, y + 0.13, z), (x + 0.68, y + 0.13, z), 0.04, 30)
        for cc, b in dg.chain_links(pts, 0.05):
            q = Gf.Matrix3d(*b.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
            kit.add(MAT["steel"], "link", cc, (q.GetReal(), *q.GetImaginary()), (0.8, 0.8, 0.8))


def pallet_and_tarp(kit, root, tarp, wood):
    x, y = g.PALLET
    for dy in (-0.45, 0.0, 0.45):
        kit.box(wood, (x, y + dy, 0.05), (1.2, 0.1, 0.1))
    for dx in np.linspace(-0.54, 0.54, 7):
        kit.box(wood, (x + dx, y, 0.12), (0.11, 1.0, 0.022))
    boxes = [((x - 0.55, y - 0.48), (x + 0.1, y + 0.48), 0.75), ((x + 0.12, y - 0.4), (x + 0.58, y + 0.3), 0.55)]
    for (bx0, by0), (bx1, by1), top in boxes:
        kit.box(f"{LOOKS}/cardboard", ((bx0 + bx1) / 2, (by0 + by1) / 2, (0.13 + top) / 2), (bx1 - bx0, by1 - by0, top - 0.13))
    res = 30
    xs, ys, zs = dg.drape(boxes, (x - 0.85, y - 0.7, x + 0.85, y + 0.7), 0.0, res, seed=5, fall=0.3)
    pts = np.stack([xs, ys, zs], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(res - 1), np.arange(res - 1), indexing="ij")
    a = i * res + j
    quads = np.stack([a, a + res, a + res + 1, a + 1], -1).reshape(-1)
    write_mesh(f"{root}/tarp", mesh_data(pts, np.full(len(quads) // 4, 4), quads, np.stack([xs, ys], -1).reshape(-1, 2)), tarp,
               double=True)
    for n, dy in enumerate((-0.3, 0.3)):
        strap = np.array([(x - 0.9, y + dy, 0.13), (x - 0.6, y + dy, 0.78), (x + 0.1, y + dy, 0.79), (x + 0.6, y + dy, 0.6),
                          (x + 0.9, y + dy, 0.13)])
        write_mesh(f"{root}/strap_{n}", tube_part(dg.spline(strap, 6), 0.008, 5), f"{LOOKS}/cable_orange")


def scaffold(kit, wood):
    x, y = g.SCAFFOLD
    for dx in (-0.9, 0.0, 0.9):
        kit.box(wood, (x + dx, y, 0.04), (0.1, 0.62, 0.08))
    for layer in range(3):
        for col in (-0.13, 0.13):
            jitter = 0.03 * ((layer * 3 + int(col * 10)) % 3 - 1)
            kit.box(wood, (x + jitter, y + col, 0.1 + layer * 0.04), (2.4, 0.225, 0.038))
    for k in range(7):
        dz = 0.03 + 0.048 * (k // 4)
        kit.seg(MAT["galv"], "cyl", (x - 1.5, y + 0.42 + 0.05 * (k % 4), dz), (x + 1.5 + 0.1 * (k % 3), y + 0.42 + 0.05 * (k % 4), dz),
                0.048)
    for k in range(4):
        kit.box(MAT["steel"], (x + 1.2 + 0.12 * k, y + 0.75, 0.03), (0.09, 0.07, 0.06))


def barrier(kit, root, plastic):
    (x0, y0), (x1, y1) = g.BARRIER
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for cx, cy in corners:
        kit.seg(MAT["black"], "cyl", (cx, cy, 0.0), (cx, cy, 0.04), 0.3)
        kit.seg(MAT["red"], "cyl", (cx, cy, 0.04), (cx, cy, 0.95), 0.045)
        kit.seg(MAT["white"], "cyl", (cx, cy, 0.55), (cx, cy, 0.7), 0.047)
    for p, q in zip(corners, corners[1:] + corners[:1]):
        pts = dg.sag((p[0], p[1], 0.9), (q[0], q[1], 0.9), 0.12, 40)
        for k, (cc, b) in enumerate(dg.chain_links(pts, 0.07)):
            qq = Gf.Matrix3d(*b.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
            kit.add(plastic[k % 2], "link", cc, (qq.GetReal(), *qq.GetImaginary()), (1.4, 1.4, 1.4))
    cx = (x0 + x1) / 2
    write_mesh(f"{root}/tag", atlased(quad_mesh((cx, y0 - 0.01, 0.7), (-1, 0, 0), (0, 0, 1), 0.075, 0.075), "tag"), atlas_look())
    write_mesh(f"{root}/tag_back", atlased(quad_mesh((cx, y0 + 0.01, 0.7), (1, 0, 0), (0, 0, 1), 0.075, 0.075), "tag"), atlas_look())
    kit.seg(MAT["black"], "cyl", (cx, y0, 0.775), (cx, y0, 0.84), 0.004)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    kit.box(MAT["black"], (mx, my, 0.002), (0.6, 0.6, 0.004))
    kit.box(MAT["galv"], (mx, my, 0.008), (0.7, 0.7, 0.004))
    kit.box(MAT["black"], (mx, my, 0.011), (0.58, 0.58, 0.004))
    kit.add(MAT["galv"], "box", (mx + 0.38, my, 0.28), (math.cos(math.radians(-35)), 0.0, math.sin(math.radians(-35)), 0.0),
            (0.62, 0.62, 0.012))


def small_debris(kit, root, rope, oil):
    for k, (x, y) in enumerate(g.ROPES):
        for layer in range(2):
            pts = dg.coil((x, y, 0.0), 0.12 + layer * 0.016, 0.3 - layer * 0.032, 5.0 - layer * 1.5, 0.016 * (1 + 1.9 * layer))
            write_mesh(f"{root}/rope_{k}_{layer}", tube_part(pts, 0.016, 7), rope)
        tail = dg.spline(np.array([(x + 0.3, y, 0.016), (x + 0.7, y - 0.25, 0.016), (x + 1.0, y - 0.1, 0.016)]), 8)
        write_mesh(f"{root}/rope_{k}_tail", tube_part(tail, 0.016, 7), rope)
    for x, y, rot in g.CHOCKS:
        kit.add(f"{LOOKS}/wood", "wedge", (x, y, 0.0), zrot(rot), (0.25, 0.14, 0.11))
    for k, ((x, y), radius, seed) in enumerate(g.SCATTER):
        rng = np.random.default_rng(100 + seed)
        for j in range(14):
            px, py = x + rng.uniform(-radius, radius), y + rng.uniform(-radius, radius)
            yaw = rng.uniform(0, 180)
            kind = j % 5
            if kind == 0:
                kit.seg(MAT["steel"], "hex", (px, py, 0.0), (px, py, 0.014), 0.03)
            elif kind == 1:
                a = math.radians(yaw)
                kit.seg(MAT["steel"], "cyl", (px, py, 0.008), (px + 0.07 * math.cos(a), py + 0.07 * math.sin(a), 0.008), 0.014)
            elif kind == 2:
                kit.add(MAT["white"] if j % 2 else MAT["black"], "box", (px, py, 0.002), zrot(yaw), (0.12, 0.006, 0.002))
            elif kind == 3:
                kit.add(MAT["galv"], "torus", (px, py, 0.002), X_TO_Z, (0.016, 0.016, 0.016))
            else:
                kit.add(MAT["steel_rust"], "box", (px, py, 0.003), zrot(yaw), (0.05, 0.03, 0.004))
        oil_quad(f"{root}/scatter_stain_{k}", x, y, 0.7, 30 * k, oil, k + 2, 0.003)
    for x, y in g.TINS:
        kit.seg(MAT["red"], "cyl", (x, y, 0.0), (x, y, 0.06), 0.09)
        kit.seg(MAT["galv"], "cyl", (x, y, 0.06), (x, y, 0.068), 0.095)
        for j in range(3):
            a = 2.1 * j
            kit.seg(f"{LOOKS}/cardboard", "cyl", (x + 0.07 * math.cos(a), y + 0.07 * math.sin(a), 0.006),
                    (x + 0.1 * math.cos(a + 0.4), y + 0.1 * math.sin(a + 0.4), 0.006), 0.011)
    for x, y, rot in g.GLOVE_SPOTS:
        a = math.radians(rot)
        kit.add(f"{LOOKS}/glove", "sphere", (x, y, 0.012), zrot(rot), (0.11, 0.05, 0.014))
        kit.add(f"{LOOKS}/glove", "sphere", (x + 0.1 * math.cos(a), y + 0.1 * math.sin(a), 0.012), zrot(rot), (0.04, 0.045, 0.012))


def grit(kit, root, grit_look):
    spots = [(x, y, 0.15) for x, y in g.COLUMNS] + [(x, y, 0.25) for x, y in g.GRIT_SUPPORTS]
    spots += [(x, g.LINES[line][0], 0.1) for line in ("grey", "orange", "rusty") for x in g.SUPPORTS[line]]
    for k, (x, y, half) in enumerate(spots):
        oil_quad(f"{root}/grit_{k:02d}", x, y, 2 * half + 0.55, 47 * k, grit_look, k, 0.0035 + 0.0002 * k)
        rng = np.random.default_rng(200 + k)
        for j in range(10):
            a = rng.uniform(0, 2 * math.pi)
            d = half + rng.uniform(0.0, 0.25)
            kit.add(MAT["steel_rust"], "box", (x + d * math.cos(a), y + d * math.sin(a), 0.002), zrot(rng.uniform(0, 180)),
                    (rng.uniform(0.015, 0.04), rng.uniform(0.01, 0.025), 0.003))


def clutter(kit):
    root = f"{g.ZONE_ROOT}/Clutter"
    UsdGeom.Scope.Define(S(), root)
    oil = f"{LOOKS}/oil"
    tarp = paint_look("tarp_green", (0.06, 0.14, 0.08), chip=0.0, grime_amount=0.6, scale=1.2)
    wood = str(paint_look("wood", (0.33, 0.23, 0.13), chip=0.0, grime_amount=0.75, scale=0.4).GetPath())
    paint_look("cardboard", (0.42, 0.3, 0.17), chip=0.0, grime_amount=0.6, scale=0.5)
    for name, color in (("maroon", (0.3, 0.04, 0.04)), ("black", (0.03, 0.03, 0.03)), ("grey", (0.38, 0.4, 0.41)),
                        ("blue", (0.05, 0.13, 0.32))):
        paint_look(f"gas_{name}", color, chip=0.35, rust=0.1, scale=0.5)
    plastic = [paint_look("chain_red", (0.55, 0.05, 0.03), chip=0.0, scale=0.3), paint_look("chain_white", (0.7, 0.7, 0.68), chip=0.0, scale=0.3)]
    grit_look = decal_look("grit", None, ASSETS / "decals" / "AsphaltDamageSet002" / "opacity.jpg", roughness=0.8,
                           tint=(0.17, 0.08, 0.035), strength=0.85)
    rope = texture_material(S(), f"{LOOKS}/rope", ASSETS / "textures" / "Rope002", tile=1.0, to_asset_path=rel)
    cables_and_protectors(kit, root)
    drip_trays(kit, root, oil)
    gas_rack(kit, root)
    pallet_and_tarp(kit, root, tarp, wood)
    scaffold(kit, wood)
    barrier(kit, root, [str(m.GetPath()) for m in plastic])
    small_debris(kit, root, rope, oil)
    grit(kit, root, grit_look)


def vents(kit):
    root = f"{g.ROOT}/Topside/Vents"
    UsdGeom.Scope.Define(S(), root)
    for (x, y, z), base in g.VENT_BASES.items():
        top = z - 0.35
        kit.seg(MAT["grey"], "cyl", (x, y, base), (x, y, top), 0.4)
        kit.seg(MAT["grey"], "cyl", (x, y, base), (x, y, base + 0.04), 0.62)
        if top - base > 3:
            for zz in g.stations(base + 1.5, top - 0.6, 2.5):
                kit.seg(MAT["grey"], "cyl", (x, y, zz), (x, y, zz + 0.05), 0.5)
            for a in (0.4, 2.5, 4.6):
                kit.seg(MAT["galv"], "cyl", (x, y, top - 1.2), (x + 2.2 * math.cos(a), y + 2.2 * math.sin(a), base), 0.03)
        kit.box(MAT["galv"], (x, y, top + 0.15), (0.62, 0.62, 0.3))
        for face in ("+x", "-x", "+y", "-y"):
            nx, ny = g.facing(face)
            for k in range(4):
                zz = top + 0.05 + 0.065 * k
                kit.add(MAT["black"], "box", (x + nx * 0.315, y + ny * 0.315, zz), IDENT,
                        (0.012 if nx else 0.5, 0.012 if ny else 0.5, 0.022))
        kit.box(MAT["galv"], (x, y, top + 0.32), (0.78, 0.78, 0.025))
    ex, ey, ez, h = 32.0, 22.0, 42.5, 10.0
    for zz in (ez + 2.5, ez + 5.0, ez + 7.5):
        kit.seg(MAT["steel_rust"], "cyl", (ex, ey, zz), (ex, ey, zz + 0.08), 1.95)
    kit.seg(MAT["black"], "cyl", (ex, ey, ez + h - 0.1), (ex, ey, ez + h + 0.25), 2.05)
    for a in (0.3, 2.4, 4.5):
        kit.seg(MAT["galv"], "cyl", (ex + 0.9 * math.cos(a), ey + 0.9 * math.sin(a), ez + h - 1.5),
                (ex + 5.0 * math.cos(a), ey + 5.0 * math.sin(a), ez), 0.025)


def build_looks():
    UsdGeom.Scope.Define(S(), LOOKS)
    paint_look("cable_black", (0.025, 0.025, 0.028), chip=0.0, scale=0.3)
    paint_look("cable_blue", (0.04, 0.12, 0.3), chip=0.0, scale=0.3)
    paint_look("cable_orange", (0.6, 0.22, 0.04), chip=0.0, scale=0.3)
    paint_look("hose_yellow", (0.62, 0.42, 0.03), chip=0.05, scale=0.4)
    return texture_material(S(), f"{LOOKS}/hazard_stripes", ASSETS / "textures" / "PaintedMetal016", tile=1.0, to_asset_path=rel)


def build(path=g.LAYER) -> dict:
    if Path(path).exists():
        Path(path).unlink()
    stage = Usd.Stage.CreateNew(str(path))
    _stage[:] = [stage]
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, g.ROOT)
    stripe_look = build_looks()
    zone = UsdGeom.Xform.Define(stage, g.ZONE_ROOT)
    zone.AddTranslateOp().Set(Gf.Vec3d(*g.ZONE_T))
    kit = Kit(f"{g.ZONE_ROOT}/Kit")
    flanges(kit)
    welds(kit)
    supports(kit)
    labels()
    gauges(kit)
    handwheels(kit)
    transmitters(kit)
    tubing(kit)
    column_kit(kit)
    stripes(stripe_look)
    drains(kit)
    debris(kit)
    clutter(kit)
    UsdGeom.Xform.Define(stage, g.OFFLINE)
    offzone = UsdGeom.Xform.Define(stage, f"{g.OFFLINE}/Zone")
    offzone.AddTranslateOp().Set(Gf.Vec3d(*g.ZONE_T))
    trays = Kit(f"{g.ZONE_ROOT}/TrayKit")
    UsdGeom.Scope.Define(stage, f"{g.ZONE_ROOT}/Cables")
    tray(trays, f"{g.ZONE_ROOT}/Cables")
    top_live, top_off = Kit(f"{g.ROOT}/Topside/Kit"), Kit(f"{g.OFFLINE}/Topside/Kit")
    UsdGeom.Xform.Define(stage, f"{g.ROOT}/Topside")
    UsdGeom.Xform.Define(stage, f"{g.OFFLINE}/Topside")
    topside_racks(top_live, top_off)
    fascia(top_live)
    id_boards(top_live)
    vents(top_live)
    counts = {"zone_instances": kit.write(), "tray_instances": trays.write(), "topside_instances": top_live.write(),
              "topside_offline": top_off.write()}
    counts["prims"] = sum(1 for _ in Usd.PrimRange(stage.GetPrimAtPath(g.ROOT)))
    stage.GetRootLayer().GetPrimAtPath("/World").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()
    print(f"GREEBLES_WRITTEN {path} {counts}")
    return counts


if __name__ == "__main__":
    build()

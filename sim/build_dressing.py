from __future__ import annotations

import json
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import dressing as d
import dressing_geometry as dg
import photoreal as pr
from photoreal_usd import bind_mdl
from scene_builder import ASSETS, SceneBuilder, rel
from tools.pbr_material import texture_material

ROOT = "/World/Photoreal"
LOOKS = f"{ROOT}/Looks"
NV_BOUNDS = json.loads((pr.SIM_DIR / "assets" / "nv_bounds.json").read_text())
MANIFEST = {a["id"]: a for a in json.loads((pr.SIM_DIR / "assets" / "manifest.json").read_text())["assets"]}
HOSE_COLORS = {"hose_yellow": (0.62, 0.42, 0.03), "hose_red": (0.42, 0.05, 0.03), "hose_black": (0.03, 0.03, 0.03),
               "cable_black": (0.025, 0.025, 0.028), "hose_coil": (0.62, 0.42, 0.03)}
DECAL_TINTS = {"rust": (0.2, 0.085, 0.03), "salt": (0.6, 0.58, 0.53), "oil": (0.018, 0.016, 0.014),
               "scuff": (0.05, 0.05, 0.05)}


def paint_look(name: str, color, chip=0.15, rust=0.0, **extra):
    recipe = {"kind": "paint", "paint_color": color, "chip_amount": chip, "rust_amount": rust, "streak_amount": 0.0,
              "salt_amount": 0.0, **extra}
    sub, values = pr.shader_inputs(recipe)
    values.update(paint_scale=extra.get("scale", 0.6))
    return bind_mdl(d_stage(), f"{LOOKS}/{name}", sub, values, define=True)


_stage = []


def d_stage():
    return _stage[0]


def decal_look(name: str, mask, tint, oil=0.0, roughness=0.55, strength=1.0):
    path = f"{LOOKS}/{name}"
    return bind_mdl(d_stage(), path, "rig_decal", {"mask": mask, "tint": tint, "oil": oil, "roughness": roughness,
                                                   "strength": strength, "wetness": 0.0, "use_color_tex": False},
                    define=True)


def write_mesh(path, points, normals, uvs, counts, indices, material, double=True):
    mesh = UsdGeom.Mesh.Define(d_stage(), path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(double)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(np.asarray(uvs, np.float32)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def tube_mesh(path, pts, radius, material, sides=10):
    m = dg.tube(np.asarray(pts, float), radius, sides)
    return write_mesh(path, m["points"], m["normals"], m["uvs"], m["counts"], m["indices"], material)


def quad(path, centre, u_axis, v_axis, half_u, half_v, material, uv=((0, 0), (1, 1)), lift=(0, 0, 0)):
    c = np.asarray(centre, float) + np.asarray(lift, float)
    u, v = np.asarray(u_axis, float), np.asarray(v_axis, float)
    pts = [c - u * half_u - v * half_v, c + u * half_u - v * half_v, c + u * half_u + v * half_v, c - u * half_u + v * half_v]
    n = np.cross(u, v)
    (u0, v0), (u1, v1) = uv
    write_mesh(path, pts, [n] * 4, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], [4], [0, 1, 2, 3], material, double=False)


def nv_prop(parent: str, asset: str, x, y, z, rot, index: int):
    holder = UsdGeom.Xform.Define(d_stage(), f"{parent}/{asset}_{index:02d}")
    ops = UsdGeom.Xformable(holder)
    ops.AddTranslateOp().Set(Gf.Vec3d(x, y, z - NV_BOUNDS[asset]["min"][2]))
    ops.AddRotateZOp().Set(float(rot))
    model = d_stage().DefinePrim(f"{holder.GetPath()}/model", "Xform")
    model.GetReferences().AddReference(MANIFEST[asset]["url"])
    model.GetVariantSets().AddVariantSet("PhysX").SetVariantSelection("Disabled")
    for body in NV_BOUNDS[asset].get("bodies", []):
        d_stage().OverridePrim(f"{model.GetPath()}/{body}").CreateAttribute(
            "physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    return z + NV_BOUNDS[asset]["max"][2] - NV_BOUNDS[asset]["min"][2]


def local_prop(builder: SceneBuilder, parent: str, asset: str, x, y, z, rot):
    holder = builder.asset(parent, asset, (x, y, z), rotate_z=rot)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(holder.GetPrim()).ComputeAlignedRange().GetMax()[2]


def add_props(builder, parent, props):
    UsdGeom.Scope.Define(d_stage(), parent)
    top = d.DECK
    for index, (asset, x, y, rot, lift) in enumerate(props):
        z = top if lift is None else d.DECK + lift
        if asset.startswith("nv_"):
            top = nv_prop(parent, asset, x, y, z, rot, index)
        else:
            top = local_prop(builder, parent, asset, x, y, z, rot)


def add_hoses():
    UsdGeom.Scope.Define(d_stage(), f"{ROOT}/Hoses")
    for name, radius, ctrl in d.HOSES:
        look = paint_look(name, HOSE_COLORS[name], chip=0.05, scale=0.4)
        pts = dg.spline(np.array([(x, y, d.DECK + z) for x, y, z in ctrl]), 10)
        tube_mesh(f"{ROOT}/Hoses/{name}", pts, radius, look)
    rope = texture_material(d_stage(), f"{LOOKS}/rope", ASSETS / "textures" / "Rope002", tile=1.0, to_asset_path=rel)
    for name, (x, y), inner, outer, turns, radius in d.COILS:
        look = rope if name.startswith("rope") else paint_look(name, HOSE_COLORS[name], chip=0.05, scale=0.4)
        for layer in range(2):
            pts = dg.coil((x, y, d.DECK), inner + layer * radius, outer - layer * radius * 2, turns - layer * 1.5,
                          radius * (1 + 1.9 * layer))
            tube_mesh(f"{ROOT}/Hoses/{name}_{layer}", pts, radius, look, sides=8)
    cable = paint_look("cable_overhead", (0.03, 0.03, 0.032), chip=0.0, scale=0.3)
    for index, (name, p0, p1, depth, count, radius) in enumerate(d.OVERHEAD):
        for k in range(count):
            off = np.array([0.0, (k - count / 2) * radius * 2.6, -k * 0.012])
            pts = dg.sag(np.asarray(p0) + off, np.asarray(p1) + off, depth + 0.04 * k, 40)
            tube_mesh(f"{ROOT}/Hoses/{name}_{index}_{k}", pts, radius, cable, sides=6)


def add_chains():
    UsdGeom.Scope.Define(d_stage(), f"{ROOT}/Chains")
    steel = bind_mdl(d_stage(), f"{LOOKS}/chain_steel", *pr.shader_inputs(
        {"kind": "textured", "texture": pr.RUST, "rust_amount": 0.0, "metallic": 0.35, "scale": 0.25,
         "streak_amount": 0.0}), define=True)
    for name, p0, p1, depth, pitch in d.CHAINS:
        inst = UsdGeom.PointInstancer.Define(d_stage(), f"{ROOT}/Chains/{name}")
        proto = f"{inst.GetPath()}/link"
        m = dg.link_mesh(pitch * 1.45, pitch * 0.62, pitch * 0.11)
        write_mesh(proto, m["points"], m["normals"], m["uvs"], m["counts"], m["indices"], steel)
        inst.CreatePrototypesRel().SetTargets([Sdf.Path(proto)])
        links = dg.chain_links(dg.sag(p0, p1, depth, 160), pitch)
        inst.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(*c) for c, _ in links]))
        quats = []
        for _, basis in links:
            rot = Gf.Matrix3d(*basis.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
            quats.append(Gf.Quath(rot.GetReal(), *rot.GetImaginary()))
        inst.CreateOrientationsAttr(Vt.QuathArray(quats))
        inst.CreateProtoIndicesAttr(Vt.IntArray([0] * len(links)))


def add_tarps():
    UsdGeom.Scope.Define(d_stage(), f"{ROOT}/Tarps")
    look = paint_look("tarp_blue", (0.035, 0.11, 0.24), chip=0.0, rust=0.0, scale=1.2, grime_amount=0.55)
    for name, boxes, extent in d.TARPS:
        res = 36
        xs, ys, zs = dg.drape(boxes, extent, 0.0, res, seed=3)
        zs = zs + d.DECK
        pts = np.stack([xs, ys, zs], -1).reshape(-1, 3)
        gx = np.gradient(zs, xs[:, 0], axis=0)
        gy = np.gradient(zs, ys[0, :], axis=1)
        normals = np.stack([-gx, -gy, np.ones_like(zs)], -1).reshape(-1, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        i, j = np.meshgrid(np.arange(res - 1), np.arange(res - 1), indexing="ij")
        a = i * res + j
        quads = np.stack([a, a + res, a + res + 1, a + 1], -1).reshape(-1)
        uvs = np.stack([xs, ys], -1).reshape(-1, 2)
        write_mesh(f"{ROOT}/Tarps/{name}", pts, normals, uvs, np.full(len(quads) // 4, 4), quads, look)


def add_decals():
    UsdGeom.Scope.Define(d_stage(), f"{ROOT}/Decals")
    atlas = ASSETS / "decals" / "AsphaltDamageSet002" / "opacity.jpg"
    oil = decal_look("decal_oil", atlas, DECAL_TINTS["oil"], oil=1.0, roughness=0.12, strength=0.9)
    scuff = decal_look("decal_scuff", atlas, DECAL_TINTS["scuff"], roughness=0.75, strength=0.55)
    quarters = (((0, 0.5), (0.5, 1)), ((0.5, 0.5), (1, 1)), ((0, 0), (0.5, 0.5)), ((0.5, 0), (1, 0.5)))
    for kind, items, look in (("oil", d.OIL, oil), ("scuff", d.SCUFFS, scuff)):
        for k, ((x, y), size, q) in enumerate(items):
            a = math.radians(37 * k + 11)
            u, v = (math.cos(a), math.sin(a), 0), (-math.sin(a), math.cos(a), 0)
            quad(f"{ROOT}/Decals/{kind}_{k:02d}", (x, y, d.DECK + 0.004 + 0.001 * k), u, v, size / 2, size / 2, look, quarters[q])
    looks = {}
    for k, ((cx, cy), face, tex, tint) in enumerate(d.COLUMN_STREAKS):
        key = (tex, tint)
        if key not in looks:
            looks[key] = decal_look(f"streak_{tint}_{tex}", ASSETS / "decals" / tex / "opacity.jpg", DECAL_TINTS[tint],
                                    roughness=0.6, strength=0.95 if tint == "rust" else 0.7)
        sign = 1.0 if face[0] == "+" else -1.0
        normal = np.array((sign, 0, 0) if face[1] == "x" else (0, sign, 0))
        u_axis = np.cross((0, 0, 1), normal)
        top = d.DECK + 3.92
        height = 2.4 if tint == "rust" else 1.5
        centre = np.array((cx, cy, top - height / 2)) + normal * 0.123
        quad(f"{ROOT}/Decals/column_{k:02d}", centre, u_axis, (0, 0, 1), 0.118, height / 2, looks[key])
    stripes = texture_material(d_stage(), f"{LOOKS}/hazard_stripes", ASSETS / "textures" / "PaintedMetal016", tile=1.0,
                               to_asset_path=rel)
    for k, ((x0, y0), (x1, y1)) in enumerate(d.HAZARD_STRIPS):
        length = math.hypot(x1 - x0, y1 - y0)
        u = np.array(((x1 - x0) / length, (y1 - y0) / length, 0))
        v = np.cross((0, 0, 1), u)
        quad(f"{ROOT}/Decals/hazard_{k}", ((x0 + x1) / 2, (y0 + y1) / 2, d.DECK + 0.003), u, v, length / 2, 0.075,
             stripes, ((0, 0), (length / 0.6, 0.25)))


def upgrade_hero_textures(stage):
    for name in ("green_metal_rust", "rust_coarse_01"):
        for file in ("diff", "rough", "nor_gl"):
            prim = stage.OverridePrim(f"/World/RepairZone/Looks/{name}/{file}")
            prim.CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(rel(ASSETS / "textures" / f"{name}_4k" / f"{file}.jpg")))


def build(stage) -> dict:
    _stage[:] = [stage]
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Scope.Define(stage, LOOKS)
    builder = SceneBuilder(stage)
    add_props(builder, f"{ROOT}/Props", d.PROPS)
    add_props(builder, pr.OFFLINE_SCOPE, d.OFFLINE_PROPS)
    add_hoses()
    add_chains()
    add_tarps()
    add_decals()
    upgrade_hero_textures(stage)
    counts = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        counts[prim.GetTypeName()] = counts.get(prim.GetTypeName(), 0) + 1
    return counts

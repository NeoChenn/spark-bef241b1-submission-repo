"""Dress the full rig: weathered PBR paint, deck lighting, aviation and helideck lights, props and signs.

    /isaac-sim/python.sh sim/dress_rig.py        (PYTHONPATH = repo root)

Writes sim/output/full_rig_dressed.usda, a layer over full_rig.usda (sublayer), so prim paths, cameras and
collisions of the rig stay as they are. Generated textures go to sim/assets/generated/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--src", default="full_rig.usda", help="rig layer in sim/output to dress")
parser.add_argument("--out", default="full_rig_dressed.usda")
parser.add_argument("--no-props", action="store_true")
parser.add_argument("--emissive", type=float, default=1.0, help="scale for all emissive surfaces")
parser.add_argument("--lights", type=float, default=1.0, help="scale for all light intensities")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

from sim import dress_plan as dp  # noqa: E402
from sim.ocean_legs import cluster_circles, waterline_points  # noqa: E402
from sim.tools.pbr_material import texture_material  # noqa: E402

SIM = Path(__file__).resolve().parent
OUTPUT = SIM / "output"
ASSETS = SIM / "assets"
TEXTURES = ASSETS / "third_party" / "textures"
GENERATED = ASSETS / "generated"
ROOT = "/World/Dress"
F3 = Sdf.ValueTypeNames.Float3
C3 = Sdf.ValueTypeNames.Color3f
FL = Sdf.ValueTypeNames.Float


def rel(path: Path) -> str:
    return Path(os.path.relpath(path, OUTPUT)).as_posix()


def remote_sources() -> dict[str, str]:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    return {a["id"]: a["url"] for a in manifest["assets"] if a.get("remote")}


class Looks:
    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.cache: dict[tuple, UsdShade.Material] = {}

    def textured(self, key: str, tile: float) -> UsdShade.Material:
        if (key, tile) not in self.cache:
            mat = dp.MATERIALS[key]
            path = f"{ROOT}/Looks/{key}_t{str(tile).replace('.', 'p')}"
            material = texture_material(self.stage, path, TEXTURES / mat.texture, tile=dp.uv_scale(tile, mat.metres),
                                        tint=mat.tint, metallic=mat.metallic, to_asset_path=rel)
            if key == "helideck":
                self._helideck_marks(path, tile)
            self.cache[(key, tile)] = material
        return self.cache[(key, tile)]

    def _helideck_marks(self, path: str, tile: float) -> None:
        scale, offset = dp.helideck_uv_transform(tile)
        xform = UsdShade.Shader.Define(self.stage, f"{path}/marks_uv")
        xform.CreateIdAttr("UsdTransform2d")
        xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            UsdShade.Shader.Get(self.stage, f"{path}/uv").GetOutput("result"))
        xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
        xform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*offset))
        diff = UsdShade.Shader.Get(self.stage, f"{path}/diff")
        diff.GetInput("file").Set(rel(GENERATED / "helideck_markings.png"))
        diff.GetInput("st").ConnectToSource(xform.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        diff.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        diff.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

    def plain(self, key: str, color, roughness=0.5, metallic=0.0, emissive=None, opacity_file=None) -> UsdShade.Material:
        if key in self.cache:
            return self.cache[key]
        path = f"{ROOT}/Looks/{key}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", C3).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", FL).Set(roughness)
        shader.CreateInput("metallic", FL).Set(metallic)
        if emissive is not None:
            shader.CreateInput("emissiveColor", C3).Set(Gf.Vec3f(*(c * args.emissive for c in emissive)))
        if opacity_file is not None:
            reader = UsdShade.Shader.Define(self.stage, f"{path}/uv")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(self.stage, f"{path}/mask")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(opacity_file))
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            channel = "a" if opacity_file.suffix == ".png" and "iso7010" in opacity_file.name else "r"
            if channel == "a":
                shader.GetInput("diffuseColor").ConnectToSource(tex.CreateOutput("rgb", F3))
                tex.GetInput("sourceColorSpace").Set("sRGB")
            shader.CreateInput("opacity", FL).ConnectToSource(tex.CreateOutput(channel, FL))
            shader.CreateInput("opacityThreshold", FL).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        self.cache[key] = material
        return material


def bind(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def mesh(stage, path, points, counts, indices, uvs=None, material=None, collide=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(list(counts)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in indices]))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    if uvs is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(uvs, np.float32)))
    if material is not None:
        bind(m.GetPrim(), material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    return m


def box_mesh(stage, path, lo, hi, material=None, uv_metres=1.0, collide=False):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts, uvs, idx = [], [], []
    for axis in range(3):
        for side in (0, 1):
            u_ax, v_ax = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = lo.copy() if side == 0 else hi.copy()
                p[u_ax] = lo[u_ax] if du == 0 else hi[u_ax]
                p[v_ax] = lo[v_ax] if dv == 0 else hi[v_ax]
                corners.append(p)
            base = len(pts)
            for p in corners:
                pts.append(p)
                uvs.append((p[u_ax] / uv_metres, p[v_ax] / uv_metres))
            idx += [base, base + 1, base + 2, base + 3]
    return mesh(stage, path, pts, [4] * 6, idx, uvs, material, collide)


def cylinder_mesh(stage, path, cx, cy, r, z0, z1, material, u_metres=4.0, seg=48):
    a = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    pts = np.concatenate([np.c_[ring, np.full(seg + 1, z0)], np.c_[ring, np.full(seg + 1, z1)]])
    u = a * r / u_metres
    uvs = np.concatenate([np.c_[u, np.zeros(seg + 1)], np.c_[u, np.ones(seg + 1)]])
    j = np.arange(seg)
    quads = np.stack([j, j + 1, j + seg + 2, j + seg + 1], 1).ravel()
    return mesh(stage, path, pts, [4] * seg, quads, uvs, material)


def xform_at(stage, path, matrix=None, translate=None, rot_z=0.0):
    xf = UsdGeom.Xform.Define(stage, path)
    if matrix is not None:
        xf.AddTransformOp().Set(Gf.Matrix4d(matrix))
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
        if rot_z:
            xf.AddRotateZOp().Set(rot_z)
    return xf


def rig_meshes(stage):
    for prim in stage.GetPrimAtPath("/World/Rig").GetChildren():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
            yield prim, pts[idx]


def dress_materials(stage, looks) -> np.ndarray:
    all_tris = []
    for prim, tris in rig_meshes(stage):
        name = prim.GetName()
        tile = dp.mesh_tile(name)
        labels = dp.classify_faces(name, tris)
        default = dp.mesh_material(name)
        bind(prim, looks.textured(default, tile))
        api = UsdShade.MaterialBindingAPI(prim)
        for key in sorted(set(labels) - {default}):
            faces = np.flatnonzero(labels == key)
            subset = api.CreateMaterialBindSubset(f"dress_{key}", Vt.IntArray([int(f) for f in faces]), "face")
            bind(subset.GetPrim(), looks.textured(key, tile))
        all_tris.append(tris)
        print(f"DRESSED {name} {dict(zip(*np.unique(labels, return_counts=True)))}", flush=True)
    return np.concatenate(all_tris)


def leg_streaks(stage, looks, tris) -> None:
    looks.plain("rust_run", (0.22, 0.1, 0.04), 0.85, opacity_file=GENERATED / "rust_streaks.png")
    material = looks.cache["rust_run"]
    pts = tris.reshape(-1, 3)
    counts = np.full(len(tris), 3)
    idx = np.arange(len(pts))
    for i, (x, y, r) in enumerate(cluster_circles(waterline_points(pts, counts, idx, 10.0))):
        cylinder_mesh(stage, f"{ROOT}/Weathering/leg_streaks_{i}", x, y, r + 0.03, 5.5, dp.DECK_TOPS[0] - dp.SLAB - 0.02, material)


def add_lights(stage, looks) -> None:
    housing = looks.plain("fixture_housing", (0.05, 0.05, 0.055), 0.45, 0.6)
    pole_mat = looks.textured("steel_grey", 3.0)
    for light in dp.lights():
        path = f"{ROOT}/Lights/{light.name}"
        target = light.target or (light.pos[0], light.pos[1], light.pos[2] - 1.0)
        xform_at(stage, path, matrix=dp.aim_matrix(light.pos, target))
        glow = looks.plain(f"glow_{light.fixture}_{'%.2f_%.2f_%.2f' % light.color}".replace(".", "p"), (0, 0, 0), 0.3,
                           emissive=tuple(c * 2000.0 for c in light.color))
        if light.kind == "sphere":
            lux = UsdLux.SphereLight.Define(stage, f"{path}/light")
            lux.CreateRadiusAttr(light.size[0])
            sphere = UsdGeom.Sphere.Define(stage, f"{path}/lamp")
            sphere.CreateRadiusAttr(light.size[0] * 0.95)
            bind(sphere.GetPrim(), glow)
        else:
            lux = UsdLux.RectLight.Define(stage, f"{path}/light")
            lux.CreateWidthAttr(light.size[0])
            lux.CreateHeightAttr(light.size[1])
            w, h = light.size[0] / 2, light.size[1] / 2
            box_mesh(stage, f"{path}/housing", (-w - 0.05, -h - 0.05, 0.01), (w + 0.05, h + 0.05, 0.25), housing)
            mesh(stage, f"{path}/lens", [(-w, -h, 0.005), (w, -h, 0.005), (w, h, 0.005), (-w, h, 0.005)], [4],
                 [3, 2, 1, 0], material=glow)
            if light.cone:
                shaping = UsdLux.ShapingAPI.Apply(lux.GetPrim())
                shaping.CreateShapingConeAngleAttr(light.cone)
                shaping.CreateShapingConeSoftnessAttr(0.3)
        hide_from_camera(lux)
        lux.CreateIntensityAttr(light.intensity * args.lights)
        lux.CreateColorAttr(Gf.Vec3f(*light.color))
        if light.fixture == "flood":
            deck = max([d for d in (*dp.DECK_TOPS, dp.HELIDECK_TOP) if d < light.pos[2]])
            x, y = light.pos[0], light.pos[1]
            box_mesh(stage, f"{ROOT}/Lights/{light.name}_pole", (x - 0.08, y - 0.08, deck),
                     (x + 0.08, y + 0.08, light.pos[2] - 0.3), pole_mat, collide=True)


def hide_from_camera(light) -> None:
    light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)


def add_flare(stage, looks) -> None:
    steel = looks.textured("derrick_white", 3.0)
    members, flame = dp.flare_boom()
    for i, (a, b, r) in enumerate(members):
        a, b = np.asarray(a), np.asarray(b)
        xform_at(stage, f"{ROOT}/Flare/member_{i}", matrix=dp.aim_matrix((a + b) / 2, b))
        tube = UsdGeom.Cylinder.Define(stage, f"{ROOT}/Flare/member_{i}/tube")
        tube.CreateRadiusAttr(r)
        tube.CreateHeightAttr(float(np.linalg.norm(b - a)))
        tube.CreateAxisAttr("Z")
        bind(tube.GetPrim(), steel)
    fire = looks.plain("flame", (0, 0, 0), 0.9, emissive=(1400.0, 420.0, 60.0))
    core = looks.plain("flame_core", (0, 0, 0), 0.9, emissive=(2600.0, 1500.0, 450.0))
    for name, scale, material, lift in (("flame", (1.3, 1.3, 3.6), fire, 2.2), ("flame_core", (0.6, 0.6, 1.6), core, 0.8)):
        xf = xform_at(stage, f"{ROOT}/Flare/{name}", translate=(flame[0], flame[1], flame[2] + lift))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        ball = UsdGeom.Sphere.Define(stage, f"{xf.GetPath()}/shape")
        ball.CreateRadiusAttr(1.0)
        bind(ball.GetPrim(), material)
        for flag in ("doNotCastShadows", "invisibleToSecondaryRays"):
            UsdGeom.PrimvarsAPI(ball).CreatePrimvar(flag, Sdf.ValueTypeNames.Bool).Set(True)
    light = UsdLux.SphereLight.Define(stage, f"{ROOT}/Flare/light")
    light.AddTranslateOp().Set(Gf.Vec3d(flame[0], flame[1], flame[2] + 2.0))
    light.CreateRadiusAttr(1.2)
    hide_from_camera(light)
    light.CreateIntensityAttr(1.5e5 * args.lights)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.5, 0.18))


def add_helideck_lamps(stage, looks) -> None:
    for i, (pos, color) in enumerate(dp.helideck_lamps()):
        key = "lamp_green" if color[1] > 0.9 else "lamp_yellow"
        material = looks.plain(key, (0, 0, 0), 0.3, emissive=tuple(c * 3000.0 for c in color))
        lamp = UsdGeom.Cylinder.Define(stage, f"{ROOT}/Helideck/lamp_{i}")
        lamp.CreateRadiusAttr(0.1)
        lamp.CreateHeightAttr(0.08)
        lamp.CreateAxisAttr("Z")
        lamp.AddTranslateOp().Set(Gf.Vec3d(*pos))
        bind(lamp.GetPrim(), material)


def add_windows(stage, looks) -> None:
    lit = looks.plain("window_lit", (0.02, 0.02, 0.02), 0.1, emissive=(1100.0, 700.0, 340.0))
    dark = looks.plain("window_dark", (0.015, 0.02, 0.025), 0.08, 0.2)
    frame = looks.plain("window_frame", (0.55, 0.57, 0.58), 0.5, 0.3)
    for i, w in enumerate(dp.windows()):
        x, y, z = w.pos
        nx, ny = w.normal
        tx, ty = -ny, nx
        def quad(half_w, half_h, push):
            return [(x + tx * sx * half_w + nx * push, y + ty * sx * half_w + ny * push, z + sz * half_h)
                    for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        mesh(stage, f"{ROOT}/Windows/frame_{i}", quad(0.8, 0.55, 0.0), [4], [0, 1, 2, 3], material=frame)
        mesh(stage, f"{ROOT}/Windows/glass_{i}", quad(0.7, 0.45, 0.01), [4], [0, 1, 2, 3], material=lit if w.lit else dark)


def add_signs(stage, looks) -> None:
    for code, (x, y, z), (nx, ny) in dp.SIGNS:
        material = looks.plain(f"sign_{code}", (1, 1, 1), 0.5, opacity_file=ASSETS / "third_party" / "decals" / f"iso7010_{code}.png")
        tx, ty, h = -ny, nx, 0.22
        pts = [(x + tx * sx * h + nx * 0.02, y + ty * sx * h + ny * 0.02, z + sz * h) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        mesh(stage, f"{ROOT}/Signs/{code}", pts, [4], [0, 1, 2, 3], [(0, 0), (1, 0), (1, 1), (0, 1)], material)


def localise_textures(prim, folder: Path) -> None:
    local = {f.stem: f for f in folder.rglob("*.*")}
    for p in Usd.PrimRange(prim):
        for attr in p.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if value and value.path.startswith("/") and Path(value.path).stem in local:
                attr.Set(Sdf.AssetPath(rel(local[Path(value.path).stem])))


def place_model(stage, prop, source: str, folder: Path | None):
    holder = xform_at(stage, f"{ROOT}/Props/{prop.name}", translate=prop.pos, rot_z=prop.rot_z)
    fit = UsdGeom.Xform.Define(stage, f"{holder.GetPath()}/fit")
    model = UsdGeom.Xform.Define(stage, f"{fit.GetPath()}/model")
    model.GetPrim().GetReferences().AddReference(source if folder is None else rel(Path(source)))
    layer = Usd.Stage.Open(source)
    move = fit.AddTranslateOp()
    fit.AddScaleOp().Set(Gf.Vec3f(UsdGeom.GetStageMetersPerUnit(layer) if layer else 1.0))
    if layer is not None and UsdGeom.GetStageUpAxis(layer) == UsdGeom.Tokens.y:
        fit.AddRotateXOp().Set(90.0)
    if folder is not None:
        localise_textures(model.GetPrim(), folder)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeRelativeBound(fit.GetPrim(), holder.GetPrim()).ComputeAlignedRange()
    if box.IsEmpty():
        print(f"PROP_EMPTY {prop.name} {source}", flush=True)
        holder.GetPrim().SetActive(False)
        return
    mid = box.GetMidpoint()
    move.Set(Gf.Vec3d(-mid[0], -mid[1], -box.GetMin()[2]))
    for p in Usd.PrimRange(model.GetPrim()):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")


def add_props(stage, looks, cloud) -> None:
    remote = remote_sources()
    containers = [looks.textured("container_green", 1.0), looks.textured("container_blue", 1.0)]
    for i, prop in enumerate(dp.props()):
        lo, hi = prop.box()
        if prop.on_floor and dp.box_blocked(cloud, (lo[0] + 0.05, lo[1] + 0.05, lo[2] + 0.05), hi):
            print(f"PROP_BLOCKED {prop.name} {prop.pos}", flush=True)
            continue
        if prop.asset == "container":
            sx, sy, sz = prop.size
            holder = xform_at(stage, f"{ROOT}/Props/{prop.name}", translate=prop.pos, rot_z=prop.rot_z)
            box_mesh(stage, f"{holder.GetPath()}/body", (-sx / 2, -sy / 2, 0), (sx / 2, sy / 2, sz),
                     containers[i % 2], uv_metres=2.44, collide=True)
        elif prop.asset in remote:
            place_model(stage, prop, remote[prop.asset], None)
        else:
            folder = ASSETS / "third_party" / "models" / prop.asset
            place_model(stage, prop, str(sorted(folder.glob("*.usd*"))[0]), folder)


def add_cameras(stage) -> None:
    shots = {
        "dress_mid": ((48.0, -46.0, 58.0), (8.0, 0.0, 44.0), 24.0),
        "dress_walkway": ((12.0, 0.0, 21.7), (-25.0, -7.0, 20.8), 18.0),
        "dress_legs": ((-48.0, -44.0, 3.0), (-15.0, -15.0, 8.0), 24.0),
    }
    for name, (eye, target, focal) in shots.items():
        cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 40000))
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        cam.AddTransformOp().Set(view.GetInverse())


def write_textures() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    dp.helideck_markings(2048).save(GENERATED / "helideck_markings.png")
    dp.streak_mask(512, 1024, seed=7).save(GENERATED / "rust_streaks.png")


def main() -> None:
    write_textures()
    out = OUTPUT / args.out
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    stage.GetRootLayer().subLayerPaths.append(args.src)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.Scope.Define(stage, ROOT)
    looks = Looks(stage)
    tris = dress_materials(stage, looks)
    leg_streaks(stage, looks, tris)
    add_lights(stage, looks)
    add_helideck_lamps(stage, looks)
    add_flare(stage, looks)
    add_windows(stage, looks)
    add_signs(stage, looks)
    if not args.no_props:
        add_props(stage, looks, dp.surface_points(tris, 0.15))
    add_cameras(stage)
    stage.GetRootLayer().Save()
    print(f"DRESSED_WRITTEN {out}", flush=True)


main()
os._exit(0)

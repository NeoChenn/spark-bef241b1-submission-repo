"""Import a 3D model (glTF/GLB, FBX, OBJ, STL, PLY, USD/USDZ, STEP/IGES) into a clean Isaac-ready USD.

    /isaac-sim/python.sh sim/tools/import_model.py model.glb --name rig
    scripts/import_model.sh model.blend --name rig      (adds .blend/.dae/.3ds/.x3d via Blender, on watson)

Writes sim/assets/rig_model/<name>/<name>.usd with textures/ beside it and report.json. The result is in metres,
Z up, main deck at z = 0 and the topside centred on the origin. Prim /Rig is the default prim; /Rig/Model carries
the unit, axis and origin fix. Walkable meshes get UsdPhysics collision (triangle mesh, static).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Import a 3D model as clean USD.")
parser.add_argument("input", type=Path)
parser.add_argument("--name", help="output folder and file name (default: input stem)")
parser.add_argument("--out-root", type=Path, default=SIM_DIR / "assets" / "rig_model")
parser.add_argument("--units", default="auto", choices=["auto", "m", "cm", "mm", "in", "ft"])
parser.add_argument("--up", default="auto", choices=["auto", "X", "Y", "Z"])
parser.add_argument("--scale", type=float, help="extra uniform scale after unit fix, e.g. for a scaled-down model")
parser.add_argument("--deck-z", type=float, help="main deck height in metres after unit/axis fix (default: detect)")
parser.add_argument("--no-center", action="store_true", help="keep the source XY origin")
parser.add_argument("--drop", action="append", default=[], help="regex on prim names to delete (sea, boats, ...)")
parser.add_argument("--collision", default="auto", choices=["auto", "names", "all", "none"])
parser.add_argument("--keep-lights", action="store_true")
parser.add_argument("--source-url", default="")
parser.add_argument("--licence", default="")
parser.add_argument("--source-name", help="original file name when the input is an intermediate (Blender GLB)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_fix import TextureMap, converter_for, deck_level, guess_scale, guess_up_axis, is_walkable  # noqa: E402

AXES = "XYZ"
UP_ROTATION = {"Z": ("X", 0.0), "Y": ("X", 90.0), "X": ("Y", -90.0)}


def enable(extension: str) -> None:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(extension, True)


def convert(src: Path, raw_dir: Path) -> Path:
    kind = converter_for(src.name)
    if kind == "usd":
        return src
    if kind == "usdz":
        with zipfile.ZipFile(src) as archive:
            archive.extractall(raw_dir)
            first = next(n for n in archive.namelist() if n.lower().endswith((".usd", ".usda", ".usdc")))
        return raw_dir / first
    dst = raw_dir / f"{src.stem}.usd"
    if kind == "assimp":
        enable("omni.kit.asset_converter")
        import omni.kit.asset_converter as converter

        ctx = converter.AssetConverterContext()
        ctx.export_preview_surface = True
        ctx.embed_textures = False
        ctx.single_mesh = True
        ctx.ignore_camera = ctx.ignore_light = ctx.ignore_animations = True
        task = converter.get_instance().create_converter_task(str(src), str(dst), None, ctx)
        if not asyncio.get_event_loop().run_until_complete(task.wait_until_finished()):
            raise SystemExit(f"asset converter failed: {task.get_error_message()}")
    elif kind == "hoops":
        enable("omni.kit.converter.hoops_core")
        import omni.kit.converter.hoops_core as hoops

        _, status = hoops.get_instance().create_converter_task(str(src), str(dst), {})
        if status.error_code:
            raise SystemExit(f"CAD converter failed: {status.error_msg}")
    else:
        raise SystemExit(f"{src.suffix} needs Blender: run scripts/import_model.sh on watson")
    return dst


def expand_instances(stage: Usd.Stage) -> None:
    while instances := [p for p in stage.Traverse() if p.IsInstance()]:
        with Sdf.ChangeBlock():
            for prim in instances:
                prim.SetInstanceable(False)


def copy_into_rig(flat: Sdf.Layer, out: Sdf.Layer) -> list[tuple[str, str]]:
    roots = [spec for spec in flat.rootPrims if spec.specifier == Sdf.SpecifierDef]
    for path, kind in (("/Rig", "Xform"), ("/Rig/Model", "Xform")):
        spec = Sdf.CreatePrimInLayer(out, path)
        spec.specifier, spec.typeName = Sdf.SpecifierDef, kind
    moves = [(spec.path, Sdf.Path(f"/Rig/Model/{spec.name}")) for spec in roots]
    for src, dst in moves:
        Sdf.CopySpec(flat, src, out, dst)
    return [(str(a), str(b)) for a, b in moves]


def remap_targets(stage: Usd.Stage, moves: list[tuple[str, str]]) -> None:
    def fix(path: Sdf.Path) -> Sdf.Path:
        for old, new in moves:
            if path.HasPrefix(Sdf.Path(old)) and not path.HasPrefix(Sdf.Path("/Rig")):
                return path.ReplacePrefix(Sdf.Path(old), Sdf.Path(new))
        return path

    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if any(fix(t) != t for t in targets):
                rel.SetTargets([fix(t) for t in targets])
        for attr in prim.GetAttributes():
            sources = attr.GetConnections()
            if any(fix(s) != s for s in sources):
                attr.SetConnections([fix(s) for s in sources])


def triangles(mesh: UsdGeom.Mesh, xform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space triangle area vectors and centroids, plus world points."""
    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
    if len(points) == 0 or len(counts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    world = points @ xform[:3, :3] + xform[3, :3]
    per_face = np.clip(counts - 2, 0, None)
    starts = np.cumsum(counts) - counts
    face = np.repeat(np.arange(len(counts)), per_face)
    k = np.arange(per_face.sum()) - np.repeat(np.cumsum(per_face) - per_face, per_face)
    a, b, c = (world[indices[starts[face] + o]] for o in (0, k + 1, k + 2))
    area = np.cross(b - a, c - a) * 0.5
    flip = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    if flip != (np.linalg.det(xform[:3, :3]) < 0):
        area = -area
    return area, (a + b + c) / 3.0, world


def fix_matrix(scale: float, up: str) -> np.ndarray:
    axis, angle = UP_ROTATION[up]
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    if axis == "X":
        r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return r * scale


def gather(stage: Usd.Stage, root: Usd.Prim) -> list[dict]:
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        xform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        area, centre, world = triangles(UsdGeom.Mesh(prim), xform)
        meshes.append({"prim": prim, "area": area, "centre": centre, "points": world})
    return meshes


def flat_area(meshes: list[dict]) -> dict[str, float]:
    totals = dict.fromkeys(AXES, 0.0)
    for m in meshes:
        size = np.linalg.norm(m["area"], axis=1) + 1e-12
        for i, axis in enumerate(AXES):
            facing = np.abs(m["area"][:, i]) / size > 0.95
            totals[axis] += float(size[facing].sum())
    return totals


def deck_samples(meshes: list[dict], rot: np.ndarray) -> list[tuple[float, float]]:
    z = np.concatenate([m["centre"] @ rot.T for m in meshes])[:, 2]
    area = np.concatenate([m["area"] @ rot.T for m in meshes])
    size = np.linalg.norm(area, axis=1) + 1e-12
    up = area[:, 2] / size > 0.9
    if area[up, 2].sum() < 0.25 * np.abs(area[:, 2]).sum():
        up = np.abs(area[:, 2]) / size > 0.9
    if not up.any():
        return []
    bins = np.floor(z[up] / 0.1).astype(np.int64)
    _, inverse = np.unique(bins, return_inverse=True)
    weight = np.bincount(inverse, weights=size[up])
    zmean = np.bincount(inverse, weights=size[up] * z[up]) / weight
    return list(zip(zmean.tolist(), weight.tolist()))


def apply_fix(model: Usd.Prim, scale: float, up: str, offset: np.ndarray) -> None:
    xf = UsdGeom.Xformable(model)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "importFix").Set(Gf.Vec3d(*offset.tolist()))
    axis, angle = UP_ROTATION[up]
    if angle:
        getattr(xf, f"AddRotate{axis}Op")(UsdGeom.XformOp.PrecisionDouble, "importFix").Set(angle)
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "importFix").Set(Gf.Vec3d(scale, scale, scale))


def add_collision(meshes: list[dict], rot: np.ndarray, mode: str) -> list[str]:
    chosen = []
    for m in meshes:
        prim = m["prim"]
        area = m["area"] @ rot.T
        size = np.linalg.norm(area, axis=1)
        total = float(size.sum()) or 1e-12
        up_area = float(size[area[:, 2] > 0.9 * size].sum())
        name = f"{prim.GetParent().GetName()}/{prim.GetName()}"
        if mode == "all" or (mode == "names" and is_walkable(name, 0.0)) or (
            mode == "auto" and is_walkable(name, up_area / total, up_area=up_area)
        ):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
            chosen.append(str(prim.GetPath()))
    return chosen


def remap_textures(stage: Usd.Stage, textures: TextureMap, layer_dir: Path) -> int:
    count = 0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                new = textures.map(value.path, layer_dir)
                if new is not None and new != value.path:
                    attr.Set(Sdf.AssetPath(new))
                    count += 1
    return count


def bind_defaults(stage: Usd.Stage, meshes: list[dict]) -> int:
    unbound = [m["prim"] for m in meshes if not UsdShade.MaterialBindingAPI(m["prim"]).ComputeBoundMaterial()[0]]
    if not unbound:
        return 0
    material = UsdShade.Material.Define(stage, "/Rig/Model/ImportLooks/default_steel")
    shader = UsdShade.Shader.Define(stage, "/Rig/Model/ImportLooks/default_steel/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.45, 0.43))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for prim in unbound:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return len(unbound)


def main() -> None:
    src = args.input.resolve()
    name = args.name or src.stem
    out_dir = (args.out_root / name).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    raw_dir = Path(tempfile.mkdtemp(prefix=f"import_{name}_"))

    raw_path = convert(src, raw_dir)
    raw = Usd.Stage.Open(str(raw_path))
    raw_up, raw_mpu = UsdGeom.GetStageUpAxis(raw), UsdGeom.GetStageMetersPerUnit(raw)
    expand_instances(raw)
    drop = [re.compile(p, re.IGNORECASE) for p in args.drop]
    remove = [p for p in raw.Traverse() if any(r.search(p.GetName()) for r in drop)]
    if not args.keep_lights:
        remove += [p for p in raw.Traverse() if p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdGeom.Camera)]
    for prim in remove:
        if prim.IsValid():
            prim.SetActive(False)

    out_layer = Sdf.Layer.CreateAnonymous(".usda")
    moves = copy_into_rig(raw.Flatten(), out_layer)
    stage = Usd.Stage.Open(out_layer)
    remap_targets(stage, moves)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Rig"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    model = stage.GetPrimAtPath("/Rig/Model")

    meshes = gather(stage, model)
    if not meshes:
        raise SystemExit("no visible meshes found")
    points = np.concatenate([m["points"] for m in meshes])
    extent = float((points.max(0) - points.min(0)).max())
    up = guess_up_axis(raw_up, flat_area(meshes), forced=args.up)
    scale = guess_scale(extent, raw_mpu, args.units) * (args.scale or 1.0)
    rot = fix_matrix(scale, up)
    fixed = points @ rot.T
    deck = args.deck_z if args.deck_z is not None else deck_level(deck_samples(meshes, rot))
    deck = 0.0 if deck is None else deck
    topside = fixed[fixed[:, 2] >= deck - 0.5]
    topside = topside if len(topside) else fixed
    centre = (topside.min(0) + topside.max(0)) / 2.0
    offset = np.array([0.0 if args.no_center else -centre[0], 0.0 if args.no_center else -centre[1], -deck])
    apply_fix(model, scale, up, offset)

    collision = [] if args.collision == "none" else add_collision(meshes, rot * scale, args.collision)
    textures = TextureMap(out_dir, search_dirs=[raw_dir, src.parent])
    remapped = remap_textures(stage, textures, raw_path.parent)
    defaults = bind_defaults(stage, meshes)
    stage.GetPrimAtPath("/Rig").SetCustomDataByKey("import", {"source": args.source_name or src.name})

    out_file = out_dir / f"{name}.usd"
    stage.GetRootLayer().Export(str(out_file))
    shutil.rmtree(raw_dir, ignore_errors=True)

    final = (fixed + offset).astype(np.float64)
    report = {
        "name": name,
        "source": args.source_name or src.name,
        "source_url": args.source_url,
        "licence": args.licence,
        "converter": converter_for(src.name),
        "source_up_axis": raw_up,
        "source_meters_per_unit": raw_mpu,
        "detected_up_axis": up,
        "scale_to_metres": scale,
        "deck_z_before_shift_m": round(float(deck), 3),
        "bounds_min_m": np.round(final.min(0), 3).tolist(),
        "bounds_max_m": np.round(final.max(0), 3).tolist(),
        "size_m": np.round(final.max(0) - final.min(0), 3).tolist(),
        "meshes": len(meshes),
        "triangles": int(sum(len(m["area"]) for m in meshes)),
        "materials": sum(1 for p in stage.Traverse() if p.IsA(UsdShade.Material)),
        "meshes_given_default_material": defaults,
        "textures_copied": len(textures.copied),
        "texture_paths_rewritten": remapped,
        "missing_textures": textures.missing,
        "collision_meshes": len(collision),
        "usd_mb": round(out_file.stat().st_size / 1e6, 2),
        "total_mb": round(sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6, 2),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMPORT_REPORT " + json.dumps(report), flush=True)


code = 0
try:
    main()
except BaseException as error:
    import traceback

    traceback.print_exc()
    print(f"IMPORT_FAILED {error}", flush=True)
    code = 1
sys.stdout.flush()
os._exit(code)

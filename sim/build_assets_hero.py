from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assets_hero as ah
from scene_builder import ASSETS, rel

BOUNDS = {**json.loads((ah.SIM_DIR / "assets" / "nv_bounds.json").read_text()),
          **json.loads((ah.SIM_DIR / "assets" / "hero_bounds.json").read_text())}
MANIFEST = {a["id"]: a for a in json.loads((ah.SIM_DIR / "assets" / "manifest.json").read_text())["assets"]}
FRONT = {"fuel_junction_box": 90.0}


def source(asset: str) -> tuple[str, tuple, tuple]:
    if asset.startswith("nv_"):
        b = BOUNDS[asset]
        return b.get("url") or MANIFEST[asset]["url"], tuple(b["min"]), tuple(b["max"])
    file = sorted((ASSETS / "models" / asset).glob("*.usdc"))[0]
    stage = Usd.Stage.Open(str(file))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(
        stage.GetPseudoRoot()).ComputeAlignedRange()
    return rel(file), tuple(box.GetMin()), tuple(box.GetMax())


def relink_textures(stage, model, asset: str):
    local = {f.stem: f for f in (ASSETS / "models" / asset).rglob("*.*")}
    for prim in Usd.PrimRange(model):
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if value and value.path.startswith("/") and Path(value.path).stem in local:
                attr.Set(Sdf.AssetPath(rel(local[Path(value.path).stem])))


def place(stage, parent: str, name: str, asset: str, pos, rot: float, scale: float, centre_z=False):
    url, lo, hi = source(asset)
    holder = UsdGeom.Xform.Define(stage, f"{parent}/{name}")
    ops = UsdGeom.Xformable(holder)
    ops.AddTranslateOp().Set(Gf.Vec3d(*pos))
    ops.AddRotateZOp().Set(float(rot))
    ops.AddScaleOp().Set(Gf.Vec3f(scale))
    model = stage.DefinePrim(f"{holder.GetPath()}/model", "Xform")
    model.GetReferences().AddReference(url)
    if asset.startswith("nv_"):
        model.GetVariantSets().AddVariantSet("PhysX").SetVariantSelection("Disabled")
    cz = -(lo[2] + hi[2]) / 2 if centre_z else -lo[2]
    UsdGeom.Xformable(model).AddTranslateOp().Set(Gf.Vec3d(-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, cz))
    if asset.startswith("fuel_"):
        model.SetInstanceable(True)
    elif not asset.startswith("nv_"):
        relink_textures(stage, model, asset)
    return lo, hi


def wall_item(stage, parent, index, asset, x, y, face, height):
    _, lo, hi = source(asset)
    rot = ah.FACING[face] + FRONT.get(asset, 0.0)
    a = math.radians(FRONT.get(asset, 0.0))
    depth = abs((hi[0] - lo[0]) * math.cos(a)) + abs((hi[1] - lo[1]) * math.sin(a))
    sx, sy, _ = ah.wall_pose(x, y, face)
    nx, ny = math.cos(math.radians(ah.FACING[face])), math.sin(math.radians(ah.FACING[face]))
    pos = (sx + nx * depth / 2, sy + ny * depth / 2, ah.DECK + height)
    place(stage, parent, f"{asset}_{index:02d}", asset, pos, rot, 1.0, centre_z=True)


def build() -> Path:
    if ah.LAYER.exists():
        ah.LAYER.unlink()
    stage = Usd.Stage.CreateNew(str(ah.LAYER))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, ah.ROOT)
    UsdGeom.Scope.Define(stage, f"{ah.ROOT}/Equipment")
    UsdGeom.Scope.Define(stage, f"{ah.ROOT}/Wall")
    UsdGeom.Scope.Define(stage, f"{ah.ROOT}/Hung")
    UsdGeom.Scope.Define(stage, ah.OFFLINE)
    for index, (asset, x, y, rot, scale) in enumerate(ah.PLACE):
        place(stage, f"{ah.ROOT}/Equipment", f"{asset}_{index:02d}", asset, (x, y, ah.DECK), rot, scale)
    for index, (asset, x, y, face, height) in enumerate(ah.WALL):
        wall_item(stage, f"{ah.ROOT}/Wall", index, asset, x, y, face, height)
    for index, (asset, x, y, rot, height) in enumerate(ah.HUNG):
        place(stage, f"{ah.ROOT}/Hung", f"{asset}_{index:02d}", asset, (x, y, ah.DECK + height), rot, 1.0, centre_z=True)
    for path in ah.HIDE:
        stage.OverridePrim(path).SetActive(False)
    stage.GetRootLayer().GetPrimAtPath("/World").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()
    print(f"ASSETS_HERO_WRITTEN {ah.LAYER} items={len(ah.PLACE) + len(ah.WALL) + len(ah.HUNG)} hidden={len(ah.HIDE)}")
    return ah.LAYER


if __name__ == "__main__":
    build()

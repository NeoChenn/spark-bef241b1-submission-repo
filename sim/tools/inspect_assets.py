from __future__ import annotations

import json
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

SIM_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = SIM_DIR / "assets" / "third_party" / "models"


def bbox_size(cache: UsdGeom.BBoxCache, prim: Usd.Prim, scale: float) -> list[float]:
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return [0.0, 0.0, 0.0]
    return [round(value * scale, 3) for value in box.GetSize()]


report = {}
for usd_file in sorted(MODELS_DIR.glob("*/*.usd*")):
    stage = Usd.Stage.Open(str(usd_file))
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    parts = {
        child.GetName(): bbox_size(cache, child, meters_per_unit)
        for child in root.GetChildren()
        if child.IsA(UsdGeom.Imageable)
    }
    report[usd_file.parent.name] = {
        "file": usd_file.name,
        "default_prim": str(root.GetPath()),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": meters_per_unit,
        "size_m": bbox_size(cache, root, meters_per_unit),
        "meshes": len(meshes),
        "triangles_approx": sum(len(mesh.GetAttribute("faceVertexIndices").Get() or []) // 3 for mesh in meshes),
        "parts": parts,
    }

out = SIM_DIR / "output" / "asset_report.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"ASSET_REPORT_WRITTEN {out}")

simulation_app.close()

from __future__ import annotations

import sys
from pathlib import Path

from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dressing
import photoreal as pr
from photoreal_usd import as_overs, bind_mdl


def override_materials(stage) -> int:
    count = 0
    for name, recipe in pr.RECIPES.items():
        sub, values = pr.shader_inputs(recipe)
        for path in pr.material_paths(name):
            bind_mdl(stage, path, sub, values)
            count += 1
    return count


def build() -> Path:
    if pr.LAYER.exists():
        pr.LAYER.unlink()
    stage = Usd.Stage.CreateNew(str(pr.LAYER))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    count = override_materials(stage)
    dressing = build_dressing.build(stage)
    as_overs(stage.GetRootLayer(), keep=(build_dressing.ROOT,))
    stage.GetRootLayer().Save()
    print(f"PHOTOREAL_WRITTEN {pr.LAYER} materials={count} dressing={dressing}")
    return pr.LAYER


if __name__ == "__main__":
    build()

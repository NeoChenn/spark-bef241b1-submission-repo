from __future__ import annotations

import os
from pathlib import Path

import photoreal as pr

LAYER = pr.OUTPUT / "facade.usda"
ROOT = "/World/Facade"
LOOKS = f"{ROOT}/Looks"
OFFLINE = f"{ROOT}/Offline"
GLOW = {"interior": 1.0, "sign": 0.04, "lamp": 6.0}
COLORS = {"lamp": (1.0, 0.8, 0.55)}


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_FACADE", "1") == "0":
        return False
    root = stage.GetRootLayer()
    if any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        return True
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    root.subLayerPaths.insert(0, target)
    return True


def glow_scale(name: str, windows: float) -> float:
    return GLOW[name] * (windows if name in ("interior", "sign") else min(1.0, windows * 1.5))


def apply(stage, windows: float = 1.0, quality: str | None = None) -> bool:
    from pxr import Gf, UsdShade

    if not attach(stage):
        return False
    for name in GLOW:
        g = glow_scale(name, windows)
        if name in COLORS:
            shader = UsdShade.Shader.Get(stage, f"{LOOKS}/{name}/surface")
            if shader:
                shader.GetInput("emissiveColor").Set(Gf.Vec3f(*(c * g for c in COLORS[name])))
            continue
        shader = UsdShade.Shader.Get(stage, f"{LOOKS}/{name}/emit")
        if shader:
            shader.GetInput("scale").Set(Gf.Vec4f(g, g, g, 1.0))
    offline = stage.GetPrimAtPath(OFFLINE)
    if offline.IsValid():
        offline.SetActive((quality or pr.current_quality()) == "offline")
    return True

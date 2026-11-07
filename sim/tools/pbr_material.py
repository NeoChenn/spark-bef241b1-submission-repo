"""UsdPreviewSurface material from a texture folder in sim/assets/third_party/textures/<id>/.

Folders hold diff, nor_gl (OpenGL, +Y up), rough, metal, ao, disp and opacity maps as
.jpg or .png. Missing maps are skipped. Needs pxr (Isaac Sim python.sh).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pxr import Gf, Sdf, Usd, UsdShade

SLOTS = (
    ("diff", "diffuseColor", "rgb"),
    ("rough", "roughness", "r"),
    ("metal", "metallic", "r"),
    ("ao", "occlusion", "r"),
    ("opacity", "opacity", "r"),
    ("nor_gl", "normal", "normal"),
)


def find_map(folder: Path, name: str) -> Path | None:
    for ext in (".jpg", ".png"):
        if (folder / f"{name}{ext}").exists():
            return folder / f"{name}{ext}"
    return None


def texture_material(
    stage: Usd.Stage,
    path: str,
    folder: Path,
    tile: float = 1.0,
    tint: tuple[float, float, float] | None = None,
    metallic: float | None = None,
    to_asset_path: Callable[[Path], str] = lambda p: p.as_posix(),
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))

    reader = UsdShade.Shader.Define(stage, f"{path}/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    if tile != 1.0:
        xform = UsdShade.Shader.Define(stage, f"{path}/tile")
        xform.CreateIdAttr("UsdTransform2d")
        xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(tile, tile))
        st = xform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    for name, slot, kind in SLOTS:
        image = find_map(folder, name)
        if image is None:
            continue
        if slot == "metallic" and metallic is not None:
            continue
        node = UsdShade.Shader.Define(stage, f"{path}/{name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(to_asset_path(image))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if kind == "rgb" else "raw")
        if kind == "r":
            out = node.CreateOutput("r", Sdf.ValueTypeNames.Float)
            shader.CreateInput(slot, Sdf.ValueTypeNames.Float).ConnectToSource(out)
            if slot == "opacity":
                shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
            continue
        if kind == "normal":
            node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            out = node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput(slot, Sdf.ValueTypeNames.Normal3f).ConnectToSource(out)
            continue
        if tint is not None:
            node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1))
        out = node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput(slot, Sdf.ValueTypeNames.Color3f).ConnectToSource(out)

    if metallic is not None or find_map(folder, "metal") is None:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic or 0.0)
    return material

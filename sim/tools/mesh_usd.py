from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

MAPS = ("albedo", "rough", "metal", "normal")


def baked(scene):
    import trimesh

    out = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        mesh.apply_transform(matrix)
        out.append(("m_" + re.sub(r"\W", "_", node), mesh))
    return out


def up_matrix(up: str) -> np.ndarray:
    m = np.eye(4)
    if up == "Y":
        m[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    return m


def embedded(meshes) -> dict:
    mat = getattr(meshes[0][1].visual, "material", None)
    out = {}
    if getattr(mat, "baseColorTexture", None) is not None:
        out["albedo"] = mat.baseColorTexture
    if getattr(mat, "normalTexture", None) is not None:
        out["normal"] = mat.normalTexture
    mr = getattr(mat, "metallicRoughnessTexture", None)
    if mr is not None:
        bands = mr.convert("RGB").split()
        out["rough"], out["metal"] = bands[1], bands[2]
    return out


def save_texture(src, dst: Path, size: int, mode: str, tint=None) -> Path:
    from PIL import Image

    im = (src if isinstance(src, Image.Image) else Image.open(src)).convert(mode)
    if tint:
        im = Image.merge("RGB", [band.point(lambda v, k=k: v * k) for band, k in zip(im.split(), tint)])
    if max(im.size) > size:
        im.thumbnail((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)
    return dst


def tiled(path: Path, size, tiles: int, mode: str):
    from PIL import Image

    im = Image.open(path).convert(mode).resize((size[0] // tiles, size[1] // tiles))
    out = Image.new(mode, size)
    for i in range(tiles):
        for j in range(tiles):
            out.paste(im, (i * im.size[0], j * im.size[1]))
    return np.asarray(out, np.float32) / 255.0


def weather(albedo_path: Path, rough_path, grime: Path, rust: Path, noise: Path, amount: float):
    from PIL import Image

    albedo = np.asarray(Image.open(albedo_path).convert("RGB"), np.float32) / 255.0
    size = (albedo.shape[1], albedo.shape[0])
    dirt = tiled(grime, size, 2, "L")[..., None]
    rust_rgb = tiled(rust, size, 3, "RGB")
    patch = tiled(noise, size, 1, "L")
    mask = np.clip((patch - (1.0 - 0.35 * amount)) * 6.0, 0.0, 1.0)[..., None] * (0.4 + 0.6 * dirt)
    out = albedo * (1.0 - 0.45 * amount * dirt) + np.array([0.05, 0.035, 0.02]) * amount * dirt
    out = out * (1.0 - mask) + rust_rgb * 0.8 * mask
    rough = None
    if rough_path is not None:
        r = np.asarray(Image.open(rough_path).convert("L").resize(size), np.float32) / 255.0
        rough = Image.fromarray((np.clip(r + 0.25 * amount * dirt[..., 0] + 0.4 * mask[..., 0], 0, 1) * 255).astype(np.uint8))
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)), rough


def write(meshes, out: Path, textures: dict, color, roughness: float, metallic: float):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    material = UsdShade.Material.Define(stage, "/root/_materials/surface")
    shader = UsdShade.Shader.Define(stage, "/root/_materials/surface/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, "/root/_materials/surface/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    wiring = {"albedo": ("diffuseColor", "rgb", Sdf.ValueTypeNames.Color3f, "sRGB"),
              "rough": ("roughness", "r", Sdf.ValueTypeNames.Float, "raw"),
              "metal": ("metallic", "r", Sdf.ValueTypeNames.Float, "raw"),
              "normal": ("normal", "rgb", Sdf.ValueTypeNames.Normal3f, "raw")}
    for key, path in textures.items():
        target, channel, kind, space = wiring[key]
        tex = UsdShade.Shader.Define(stage, f"/root/_materials/surface/{key}")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./{path.relative_to(out.parent).as_posix()}"))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        if key == "normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        tex.CreateOutput(channel, Sdf.ValueTypeNames.Float3 if len(channel) == 3 else Sdf.ValueTypeNames.Float)
        shader.CreateInput(target, kind).ConnectToSource(tex.ConnectableAPI(), channel)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for name, mesh in meshes:
        prim = UsdGeom.Mesh.Define(stage, f"/root/{name}")
        pts = np.asarray(mesh.vertices, np.float32)
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
        prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, np.int32).reshape(-1)))
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, np.float32)))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        prim.CreateSubdivisionSchemeAttr("none")
        prim.CreateDoubleSidedAttr(True)
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None and len(uv) == len(pts):
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return lo, hi


def main():
    import trimesh

    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--up", default="Z", choices=["Y", "Z"])
    ap.add_argument("--color", type=float, nargs=3, default=(0.6, 0.6, 0.6))
    ap.add_argument("--roughness", type=float, default=0.5)
    ap.add_argument("--metallic", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--tint", type=float, nargs=3)
    ap.add_argument("--weather", type=float, default=0.0)
    for key in MAPS:
        ap.add_argument(f"--{key}", type=Path)
    args = ap.parse_args()
    scene = trimesh.load(args.src, force="scene")
    meshes = baked(scene)
    fix = up_matrix(args.up) @ np.diag([args.scale] * 3 + [1])
    for _, mesh in meshes:
        mesh.apply_transform(fix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    textures = {}
    inside = embedded(meshes)
    if args.weather > 0 and args.albedo:
        lib = Path(__file__).resolve().parents[1] / "assets"
        tex = lib / "third_party" / "textures"
        albedo, rough = weather(args.albedo, args.rough, tex / "SurfaceImperfections011" / "opacity.jpg",
                                tex / "rust_coarse_01" / "diff.jpg", lib / "generated" / "hero_noise.png", args.weather)
        inside.update(albedo=albedo, rough=rough if rough is not None else inside.get("rough"))
        args.albedo = None
        if rough is not None:
            args.rough = None
    for key in MAPS:
        src = getattr(args, key) or inside.get(key)
        if src is not None:
            textures[key] = save_texture(src, args.out.parent / "textures" / f"{args.out.stem}_{key}.jpg", args.size,
                                         "RGB" if key in ("albedo", "normal") else "L",
                                         args.tint if key == "albedo" else None)
    lo, hi = write(meshes, args.out, textures, args.color, args.roughness, args.metallic)
    print("BOUNDS", lo.round(3).tolist(), hi.round(3).tolist(), "faces", sum(len(m.faces) for _, m in meshes))


if __name__ == "__main__":
    main()

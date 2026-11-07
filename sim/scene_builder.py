from __future__ import annotations

import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

SIM_DIR = Path(__file__).resolve().parent
ASSETS = SIM_DIR / "assets" / "third_party"
GENERATED = SIM_DIR / "assets" / "generated"
OUTPUT = SIM_DIR / "output"


def rel(path: Path) -> str:
    return Path(os.path.relpath(path, OUTPUT)).as_posix()


class SceneBuilder:
    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.count = 0
        self.materials: dict[str, UsdShade.Material] = {}

    def unique(self, parent: str, name: str) -> str:
        self.count += 1
        return f"{parent}/{name}_{self.count:03d}"

    def plain(self, name: str, color, roughness=0.5, metallic=0.0) -> UsdShade.Material:
        if name in self.materials:
            return self.materials[name]
        material, shader = self._surface(name)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        return material

    def textured(self, texture: str) -> UsdShade.Material:
        if texture in self.materials:
            return self.materials[texture]
        material, shader = self._surface(texture)
        path = str(material.GetPath())
        reader = UsdShade.Shader.Define(self.stage, f"{path}/uv")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        folder = ASSETS / "textures" / texture
        for file, slot, kind in (
            ("diff", "diffuseColor", "rgb"),
            ("rough", "roughness", "r"),
            ("metal", "metallic", "r"),
            ("nor_gl", "normal", "normal"),
        ):
            image = folder / f"{file}.jpg"
            if not image.exists():
                if slot == "metallic":
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
                continue
            node = UsdShade.Shader.Define(self.stage, f"{path}/{file}")
            node.CreateIdAttr("UsdUVTexture")
            node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(image))
            node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
            node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            color_space = "sRGB" if kind == "rgb" else "raw"
            node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
            if kind == "r":
                output = node.CreateOutput("r", Sdf.ValueTypeNames.Float)
                shader.CreateInput(slot, Sdf.ValueTypeNames.Float).ConnectToSource(output)
            else:
                if kind == "normal":
                    node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                output = node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                value_type = Sdf.ValueTypeNames.Normal3f if kind == "normal" else Sdf.ValueTypeNames.Color3f
                shader.CreateInput(slot, value_type).ConnectToSource(output)
        return material

    def _surface(self, name: str):
        path = f"/World/Looks/{name}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        self.materials[name] = material
        return material, shader

    def _mesh(self, path, points, normals, uvs, counts, indices, material, collide):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if collide:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(collide)
        return mesh

    def box(self, parent, name, center, size, material, tile=1.0, collide="convexHull"):
        sx, sy, sz = (value / 2 for value in size)
        cx, cy, cz = center
        faces = (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
            ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
            ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        )
        half = (sx, sy, sz)
        points, normals, uvs, indices = [], [], [], []
        for number, (normal, u_axis, v_axis) in enumerate(faces):
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                offset = [normal[i] * half[i] + du * u_axis[i] * half[i] + dv * v_axis[i] * half[i] for i in range(3)]
                points.append((cx + offset[0], cy + offset[1], cz + offset[2]))
                normals.append(normal)
                uvs.append(
                    (
                        sum(offset[i] * abs(u_axis[i]) for i in range(3)) / tile + cx / tile,
                        sum(offset[i] * abs(v_axis[i]) for i in range(3)) / tile + cy / tile,
                    )
                )
            base = number * 4
            indices.extend((base, base + 1, base + 2, base + 3))
        return self._mesh(
            self.unique(parent, name), points, normals, uvs, [4] * 6, indices, material, collide
        )

    def tube(self, parent, name, start, end, radius, material, tile=1.0, segments=32, collide=None):
        start_v, end_v = Gf.Vec3d(*start), Gf.Vec3d(*end)
        axis = end_v - start_v
        length = axis.GetLength()
        rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), axis.GetNormalized())
        circumference = 2 * math.pi * radius
        points, normals, uvs, indices = [], [], [], []
        rings = 2
        for ring in range(rings):
            x = length * ring / (rings - 1)
            for step in range(segments + 1):
                angle = 2 * math.pi * step / segments
                ny, nz = math.cos(angle), math.sin(angle)
                points.append((x, ny * radius, nz * radius))
                normals.append((0, ny, nz))
                uvs.append((x / tile, step / segments * circumference / tile))
        for step in range(segments):
            a, b = step, step + 1
            c, d = a + segments + 1, b + segments + 1
            indices.extend((a, c, d, b))
        path = self.unique(parent, name)
        mesh = self._mesh(path, points, normals, uvs, [4] * segments, indices, material, collide)
        xform = UsdGeom.Xformable(mesh)
        xform.AddTranslateOp().Set(start_v)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rotation.GetQuat()))
        return mesh

    def disc(self, parent, name, center, axis_x, radius, material):
        points, normals, uvs, indices = [(center[0], center[1], center[2])], [(axis_x, 0, 0)], [(0.5, 0.5)], []
        segments = 32
        for step in range(segments):
            angle = 2 * math.pi * step / segments
            y, z = math.cos(angle), math.sin(angle)
            points.append((center[0], center[1] + y * radius, center[2] + z * radius))
            normals.append((axis_x, 0, 0))
            uvs.append((0.5 + y / 2, 0.5 + z / 2))
            indices.extend((0, step + 1, (step + 1) % segments + 1))
        return self._mesh(
            self.unique(parent, name), points, normals, uvs, [3] * segments, indices, material, None
        )

    def flange(self, parent, x, y, z, pipe_radius, material):
        radius = pipe_radius * 1.45
        width = 0.05
        self.tube(parent, "flange", (x - width, y, z), (x + width, y, z), radius, material)
        self.disc(parent, "flange_face", (x - width, y, z), -1, radius, material)
        self.disc(parent, "flange_face", (x + width, y, z), 1, radius, material)

    def asset(self, parent, asset_id, position, rotate_z=0.0, rotate_y=0.0, part=None, scale=1.0, on_deck=True):
        files = sorted((ASSETS / "models" / asset_id).glob("*.usdc"))
        if not files:
            raise FileNotFoundError(f"{asset_id} is missing. Run sim/tools/fetch_assets.py first.")
        name = part or asset_id
        holder_path = self.unique(parent, name)
        holder = UsdGeom.Xform.Define(self.stage, holder_path)
        holder_ops = UsdGeom.Xformable(holder)
        holder_ops.AddTranslateOp().Set(Gf.Vec3d(*position))
        holder_ops.AddRotateZOp().Set(rotate_z)
        holder_ops.AddRotateYOp().Set(rotate_y)
        holder_ops.AddScaleOp().Set(Gf.Vec3f(scale))

        inner = UsdGeom.Xform.Define(self.stage, f"{holder_path}/model")
        inner.GetPrim().GetReferences().AddReference(rel(files[0]))
        if part:
            names = {child.GetName() for child in inner.GetPrim().GetChildren()}
            if part not in names:
                raise KeyError(f"{asset_id} has no part {part!r}")
            for child in inner.GetPrim().GetChildren():
                if child.GetName() not in (part, "_materials"):
                    child.SetActive(False)

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeLocalBound(inner.GetPrim()).ComputeAlignedRange()
        middle = box.GetMidpoint()
        offset_z = -box.GetMin()[2] if on_deck else -middle[2]
        UsdGeom.Xformable(inner).AddTranslateOp().Set(Gf.Vec3d(-middle[0], -middle[1], offset_z))

        local_textures = {file.stem: file for file in (ASSETS / "models" / asset_id).rglob("*.*")}
        for prim in Usd.PrimRange(inner.GetPrim()):
            for attribute in prim.GetAttributes():
                value = attribute.Get() if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset else None
                if value and value.path.startswith("/"):
                    texture = local_textures.get(Path(value.path).stem)
                    if texture:
                        attribute.Set(Sdf.AssetPath(rel(texture)))

        for prim in Usd.PrimRange(inner.GetPrim()):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        return holder

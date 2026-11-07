"""Build the offshore rig deck scene and render preview images.

    C:\\isaacsim\\python.bat sim\\build_rig_deck.py --seed 7
    ./python.sh sim/build_rig_deck.py --seed 7        (Linux and Docker)

Outputs, all in sim/output/:
    rig_deck.usda         the scene; open it in Isaac Sim with File > Open
    rig_deck_meta.json    damage location, work zone and camera data for other teams
    preview_*.png         rendered views

The scene uses metres with Z up, and the deck surface is at z = 0. Run
sim/tools/fetch_assets.py first. Changing --seed moves the pipe damage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

parser = argparse.ArgumentParser(description="Build the rig deck scene.")
parser.add_argument("--seed", type=int, default=7, help="controls the damage position and shape")
parser.add_argument("--hdri", default="aristea_wreck", help="sky name from sim/assets/manifest.json")
parser.add_argument("--robots", action="store_true", help="add two stand-in robots (needs internet)")
parser.add_argument("--no-render", action="store_true", help="write the scene without previews")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

SIM_DIR = Path(__file__).resolve().parent
ASSETS = SIM_DIR / "assets" / "third_party"
OUTPUT = SIM_DIR / "output"
SCENE_PATH = OUTPUT / "rig_deck.usda"

DECK_X = 8.0  # half length
DECK_Y = 5.0  # half width
MAIN_PIPE_Z = 1.1
MAIN_PIPE_RADIUS = 0.15
# Kept free of props so robots can reach both sides of the main pipe.
WORK_ZONE = {"x": [-2.5, 2.5], "y": [-2.2, 2.2]}


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

    # Materials ----------------------------------------------------------
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

    # Geometry -----------------------------------------------------------
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
        """Flat ring face closing the end of a tube along +X or -X."""
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
        """Reference a downloaded model, optionally keeping one part of a modular kit.

        Without on_deck, ``position`` sets the model's center, which suits pipe
        fittings. With on_deck, the model's base sits at position.z.
        """
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

        # Poly Haven's USD files keep absolute paths from their build server.
        # Point those at the textures downloaded alongside each model.
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


def damage_patch(builder, rng, material):
    """Irregular corrosion patch wrapped around the top half of the main pipe."""
    center_x = rng.uniform(-1.4, 1.4)
    center_angle = math.radians(rng.uniform(35, 145))  # measured from +Y toward +Z
    base = rng.uniform(0.13, 0.2)
    lobes = [(rng.randint(2, 7), rng.uniform(0.08, 0.28), rng.uniform(0, math.tau)) for _ in range(4)]
    surface = MAIN_PIPE_RADIUS + 0.003

    rings, segments = 8, 48
    points, normals, uvs, indices = [], [], [], []

    def add_vertex(dx, arc):
        angle = center_angle + arc / MAIN_PIPE_RADIUS
        ny, nz = math.cos(angle), math.sin(angle)
        points.append((center_x + dx, ny * surface, MAIN_PIPE_Z + nz * surface))
        normals.append((0, ny, nz))
        uvs.append((dx / 0.5 + 0.5, arc / 0.5 + 0.5))

    add_vertex(0, 0)
    max_extent = 0.0
    for ring in range(1, rings + 1):
        fraction = ring / rings
        for step in range(segments):
            angle = math.tau * step / segments
            wobble = 1 + sum(size * math.sin(count * angle + phase) for count, size, phase in lobes)
            distance = base * wobble * fraction
            max_extent = max(max_extent, distance)
            add_vertex(distance * math.cos(angle), distance * math.sin(angle))
    counts = []
    for step in range(segments):
        indices.extend((0, 1 + step, 1 + (step + 1) % segments))
        counts.append(3)
    for ring in range(rings - 1):
        for step in range(segments):
            a = 1 + ring * segments + step
            b = 1 + ring * segments + (step + 1) % segments
            indices.extend((a, a + segments, b + segments, b))
            counts.append(4)
    builder._mesh("/World/Pipes/damage_patch", points, normals, uvs, counts, indices, material, None)
    world = (center_x, math.cos(center_angle) * surface, MAIN_PIPE_Z + math.sin(center_angle) * surface)
    normal = (0.0, math.cos(center_angle), math.sin(center_angle))
    return {
        "position_m": [round(v, 4) for v in world],
        "surface_normal": [round(v, 4) for v in normal],
        "approx_radius_m": round(max_extent, 3),
        "pipe_radius_m": MAIN_PIPE_RADIUS,
    }


def build_structure(b):
    grate = b.textured("metal_grate_rusty")
    steel = b.textured("rusty_metal_02")
    plate = b.textured("metal_plate")
    paint = b.textured("rusty_painted_metal")
    yellow = b.plain("safety_yellow", (0.85, 0.6, 0.05), 0.45, 0.2)

    b.box("/World/Deck", "deck_grating", (0, 0, -0.03), (DECK_X * 2, DECK_Y * 2, 0.06), grate, tile=1.5)
    b.box("/World/Deck", "subdeck", (0, 0, -0.5), (DECK_X * 2, DECK_Y * 2, 0.04), plate, tile=3.0, collide=None)
    for x in range(-8, 9, 2):
        b.box("/World/Deck", "beam", (x, 0, -0.25), (0.2, DECK_Y * 2, 0.4), steel, tile=2.0)
    for y in (-5, 0, 5):
        b.box("/World/Deck", "beam", (0, y, -0.25), (DECK_X * 2, 0.2, 0.4), steel, tile=2.0)

    for x in (-6, 6):
        for y in (-3.5, 3.5):
            b.tube("/World/Deck", "leg", (x, y, -0.5), (x, y, -30), 0.8, paint, tile=3.0)

    # Overhead frame, which carries the upper pipe rack.
    for x in (-7, 0, 7):
        for y in (-4.4, 4.4):
            if (x, y) == (0, -4.4):
                continue
            b.box("/World/Structure", "column", (x, y, 2.0), (0.25, 0.25, 4.0), paint, tile=2.0)
        b.box("/World/Structure", "cross_beam", (x, 0, 4.1), (0.25, 9.05, 0.3), paint, tile=2.0)
    for y in (-4.4, 4.4):
        b.box("/World/Structure", "long_beam", (0, y, 4.1), (14.25, 0.25, 0.3), paint, tile=2.0)

    # Handrail around the deck, with a gap at the stairs.
    def rail(start, end):
        for height in (0.6, 1.1):
            b.tube(
                "/World/Railing", "rail",
                (start[0], start[1], height), (end[0], end[1], height), 0.025, yellow, segments=10,
            )
        length = math.dist(start, end)
        posts = max(1, round(length / 1.5))
        for index in range(posts + 1):
            t = index / posts
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            b.tube("/World/Railing", "post", (x, y, 0), (x, y, 1.1), 0.025, yellow, segments=10)

    edge_x, edge_y = DECK_X - 0.05, DECK_Y - 0.05
    rail((-edge_x, -edge_y), (edge_x, -edge_y))
    rail((-edge_x, edge_y), (2.0, edge_y))
    rail((-edge_x, -edge_y), (-edge_x, edge_y))
    rail((edge_x, -edge_y), (edge_x, edge_y))


def build_pipes(b, rng):
    painted = b.textured("green_metal_rust")
    rusty = b.textured("rusty_painted_metal")
    plate = b.textured("blue_metal_plate")
    steel = b.textured("rusty_metal_02")
    grey = b.plain("pipe_grey", (0.33, 0.35, 0.36), 0.55, 0.6)
    orange = b.plain("pipe_orange", (0.75, 0.3, 0.06), 0.5, 0.2)

    # The repair target: a large line through the middle of the work zone.
    b.tube("/World/Pipes", "main_pipe", (-7.6, 0, MAIN_PIPE_Z), (7.6, 0, MAIN_PIPE_Z), MAIN_PIPE_RADIUS,
           painted, tile=1.2, segments=64, collide="none")
    for x in (-7.6, -3.0, 3.0, 7.6):
        b.flange("/World/Pipes", x, 0, MAIN_PIPE_Z, MAIN_PIPE_RADIUS, painted)
    for x in (-6.0, -3.4, 3.4, 6.0):
        height = MAIN_PIPE_Z - MAIN_PIPE_RADIUS
        b.box("/World/Pipes", "support", (x, 0, height / 2), (0.14, 0.5, height), steel)
        b.box("/World/Pipes", "support_base", (x, 0, 0.02), (0.4, 0.7, 0.04), steel)
    # Scaled from the 10 cm kit so its bore lines up with the 30 cm main line.
    b.asset("/World/Pipes", "modular_pipes", (-5.0, 0, MAIN_PIPE_Z), rotate_y=90,
            part="pipe_valve_large_metal", scale=3.0, on_deck=False)

    # Neighbouring lines, which add clutter around the work zone.
    for y, z, radius, material, tile in (
        (-3.0, 0.7, 0.1, grey, 1.0),
        (-3.5, 0.7, 0.07, orange, 1.0),
        (3.2, 0.8, 0.12, rusty, 1.0),
    ):
        b.tube("/World/Pipes", "side_line", (-7.6, y, z), (7.6, y, z), radius, material,
               tile=tile, collide="none")
        for x in (-6.5, -2.5, 2.5, 6.5):
            b.box("/World/Pipes", "side_support", (x, y, (z - radius) / 2), (0.1, 0.25, z - radius), steel)
        for x in (-7.6, 7.6):
            b.flange("/World/Pipes", x, y, z, radius, material)
    for x, y in ((-1.0, -3.0), (4.5, -3.0), (0.0, 3.2)):
        b.asset("/World/Pipes", "modular_pipes", (x, y, 0.75), rotate_y=90,
                part="pipe_valve_small_1_metal", scale=2.0, on_deck=False)

    # Overhead rack hung from the frame.
    for y, z, radius in ((-1.6, 3.6, 0.09), (-1.2, 3.6, 0.06), (1.3, 3.65, 0.1), (1.8, 3.65, 0.06)):
        b.tube("/World/Pipes", "overhead", (-7.5, y, z), (7.5, y, z), radius, grey if y < 0 else plate,
               collide="none")
    for x in (-7, 0, 7):
        b.box("/World/Pipes", "rack_bar", (x, 0, 3.5), (0.08, 4.4, 0.08), steel)
        for y in (-2.2, 2.2):
            b.box("/World/Pipes", "rack_hanger", (x, y, 3.8), (0.06, 0.06, 0.55), steel)

    # A vertical riser made from the Poly Haven pipe kit, joining the overhead rack.
    for z in (1.2, 2.4):
        b.asset("/World/Pipes", "modular_pipes", (6.9, -2.2, z), part="pipe_140cm_metal", scale=1.0, on_deck=False)
    b.asset("/World/Pipes", "modular_industrial_pipes_01", (-6.8, 3.0, 0.0), rotate_z=180)

    return damage_patch(b, rng, b.textured("rust_coarse_01"))


def build_props(b):
    parent = "/World/Props"
    for asset, position, rotate in (
        ("portable_welding_cart", (3.3, -2.0, 0), 200),
        ("portable_generator", (5.7, 2.0, 0), 80),
        ("utility_box_02", (-3.8, 4.5, 0), 180),
        ("utility_box_01", (-2.9, 4.5, 0), 180),
        ("power_box_01", (-4.7, -1.9, 0), 30),
        ("propane_tank", (3.2, 2.2, 0), 0),
        ("propane_tank", (3.6, 2.4, 0), 40),
        ("korean_fire_extinguisher_01", (-3.2, -2.1, 0), 200),
        ("metal_toolbox", (2.9, -2.6, 0), 15),
        ("barrel_03", (6.9, 4.3, 0), 0),
        ("barrel_03", (6.2, 4.4, 0), 70),
        ("worn_metal_rack", (-7.2, -4.4, 0), 0),
    ):
        b.asset(parent, asset, position, rotate_z=rotate)

    # Stairs leading up off the deck, through the handrail gap.
    b.asset(parent, "modular_fire_escape", (4.2, 5.75, 0), rotate_z=0, part="modular_fire_escape_stairs")

    for x in (-3.5, 3.5):
        for y in (-2.5, 2.5):
            b.asset(parent, "caged_hanging_light", (x, y, 3.4), on_deck=False)


def add_environment(stage, b):
    sea = b.plain("sea", (0.03, 0.11, 0.15), 0.12, 0.0)
    b.box("/World/Environment", "sea", (0, 0, -22.0), (4000, 4000, 0.1), sea, tile=100.0, collide=None)

    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/sky")
    dome.CreateTextureFileAttr().Set(rel(ASSETS / "hdris" / f"{args.hdri}.hdr"))
    dome.CreateTextureFormatAttr().Set("latlong")
    dome.CreateIntensityAttr(1100.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Environment/sun")
    sun.CreateIntensityAttr(1800.0)
    sun.CreateAngleAttr(1.5)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(52, 18, 30))

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)


def add_camera(stage, name, eye, target, focal=24.0):
    camera = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
    camera.CreateFocalLengthAttr(focal)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 8000))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    return str(camera.GetPath())


def add_robots(stage):
    """Reference NVIDIA's Ridgeback Franka to check scale and GPU load."""
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Could not reach the Isaac Sim asset server.")
    url = f"{root}/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    for name, position, yaw in (("fitter", (0.2, -1.3, 0.0), 90), ("welder", (-0.2, 1.3, 0.0), -90)):
        prim = UsdGeom.Xform.Define(stage, f"/World/Robots/{name}")
        prim.GetPrim().GetReferences().AddReference(url)
        ops = UsdGeom.Xformable(prim)
        ops.ClearXformOpOrder()
        ops.AddTranslateOp().Set(Gf.Vec3d(*position))
        ops.AddRotateZOp().Set(yaw)


def build_scene() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if SCENE_PATH.exists():
        SCENE_PATH.unlink()
    stage = Usd.Stage.CreateNew(str(SCENE_PATH))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    builder = SceneBuilder(stage)
    for group in ("Deck", "Structure", "Railing", "Pipes", "Props", "Environment", "Cameras"):
        UsdGeom.Scope.Define(stage, f"/World/{group}")

    build_structure(builder)
    damage = build_pipes(builder, rng)
    build_props(builder)
    add_environment(stage, builder)
    if args.robots:
        add_robots(stage)

    dx, dy, dz = damage["position_m"]
    cameras = {
        "hero": add_camera(stage, "hero", (9.5, -8.5, 4.6), (0.4, 0, 0.9), 22),
        "deck_end": add_camera(stage, "deck_end", (-7.4, -3.9, 2.0), (2.5, 0.6, 1.1), 20),
        "repair_closeup": add_camera(stage, "repair_closeup", (dx + 0.7, dy - 1.2, dz + 0.7), (dx, dy, dz), 35),
        "overview": add_camera(stage, "overview", (0, -0.01, 21), (0, 0, 0), 24),
    }
    stage.GetRootLayer().Save()

    metadata = {
        "seed": args.seed,
        "units": "metres, Z up, deck surface at z=0",
        "deck_half_extents_m": [DECK_X, DECK_Y],
        "work_zone_m": WORK_ZONE,
        "main_pipe": {"z_m": MAIN_PIPE_Z, "y_m": 0.0, "radius_m": MAIN_PIPE_RADIUS, "x_range_m": [-7.6, 7.6]},
        "damage": damage,
        "cameras": cameras,
    }
    (OUTPUT / "rig_deck_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"SCENE_WRITTEN {SCENE_PATH}")
    return metadata


def render_previews(cameras: dict) -> None:
    import numpy as np
    import omni.kit.app
    import omni.usd
    from PIL import Image

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep

    context = omni.usd.get_context()
    context.open_stage(str(SCENE_PATH))
    for _ in range(600):
        simulation_app.update()
        if context.get_stage_loading_status()[2] == 0:
            break
    for _ in range(60):
        simulation_app.update()

    for name, path in cameras.items():
        product = rep.create.render_product(path, (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=24, delta_time=0.0)
        data = np.asarray(annotator.get_data())
        out = OUTPUT / f"preview_{name}.png"
        Image.fromarray(data[:, :, :3].astype("uint8")).save(out)
        print(f"PREVIEW_WRITTEN {out} mean={data[:, :, :3].mean():.1f}")
        annotator.detach(product)
        product.destroy()
    rep.orchestrator.wait_until_complete()


metadata = build_scene()
if not args.no_render:
    render_previews(metadata["cameras"])
simulation_app.close()

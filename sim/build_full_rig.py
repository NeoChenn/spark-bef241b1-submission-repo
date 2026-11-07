from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Build the full rig scene.")
parser.add_argument("--scale", type=float, default=0.25, help="metres per STL unit")
parser.add_argument("--sea-level", type=float, default=-75.0, help="sea level in STL units")
parser.add_argument("--seed", type=int, default=7, help="controls the gauge readings")
parser.add_argument("--no-render", action="store_true", help="write the scene without previews")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

import rig_details
from scene_builder import ASSETS, GENERATED, OUTPUT, SceneBuilder, rel

STL_PATH = ASSETS / "oil_rig_ansyslearn" / "OilRig2.stl"
SCENE_PATH = OUTPUT / "full_rig.usda"
SMOOTH_ANGLE_COS = np.cos(np.radians(35.0))

PLATE = "metal_plate"
CLADDING = "box_profile_metal_sheet"
CONCRETE = "concrete_floor_damaged_01"
DENTS = "blue_metal_plate"
PAINTS = {
    "deck": ((0.33, 0.37, 0.38), 0.6, 0.2, PLATE),
    "concrete": ((0.56, 0.55, 0.52), 0.8, 0.0, CONCRETE),
    "frame": ((0.34, 0.38, 0.42), 0.5, 0.3, DENTS),
    "rail": ((0.85, 0.6, 0.05), 0.45, 0.1, None),
    "crane": ((0.88, 0.55, 0.04), 0.4, 0.1, None),
    "derrick": ((0.8, 0.82, 0.83), 0.45, 0.2, None),
    "helideck": ((0.07, 0.25, 0.17), 0.65, 0.0, None),
    "pipe": ((0.72, 0.27, 0.06), 0.45, 0.2, None),
    "tank": ((0.86, 0.88, 0.88), 0.35, 0.1, DENTS),
    "vessel": ((0.13, 0.28, 0.42), 0.4, 0.2, DENTS),
    "module_white": ((0.8, 0.82, 0.82), 0.5, 0.0, CLADDING),
    "module_blue": ((0.16, 0.33, 0.48), 0.5, 0.0, CLADDING),
    "module_grey": ((0.46, 0.49, 0.51), 0.55, 0.1, CLADDING),
}
TILE_METRES = {"deck": 2.5, "concrete": 5.0, "module_white": 4.0, "module_blue": 4.0, "module_grey": 4.0}
FULL_TEXTURES = {
    "module_white": "worn_corrugated_iron",
    "module_blue": "worn_corrugated_iron",
    "module_grey": "worn_corrugated_iron",
    "concrete": "Concrete042C",
}
SEA_TILE_METRES = 70.0
SKY_ROTATION_DEG = 120.0
HAZE_SETTINGS = {
    "rtx:fog:enabled": True,
    "rtx:fog:fogZup:enabled": True,
    "rtx:fog:fogColor": Gf.Vec3f(0.66, 0.7, 0.75),
    "rtx:fog:fogColorIntensity": 1.0,
    "rtx:fog:fogStartHeight": 0.0,
    "rtx:fog:fogHeightDensity": 0.006,
    "rtx:fog:fogHeightFalloff": 0.12,
    "rtx:fog:fogStartDist": 150.0,
    "rtx:fog:fogEndDist": 2600.0,
    "rtx:fog:fogDistanceDensity": 0.0,
}
FACADE_MIN_AREA = 300.0
MODULE_PAINTS = ("module_white", "module_blue", "module_grey")


def load_stl(path: Path):
    raw = path.read_bytes()
    count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(raw, dtype=record, count=count, offset=84)["vertices"].astype(np.float64)
    points, inverse = np.unique(np.round(triangles.reshape(-1, 3), 3), axis=0, return_inverse=True)
    return points, inverse.reshape(-1, 3)


def connected_parts(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    parent = list(range(vertex_count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for a, b, c in faces.tolist():
        root = find(a)
        parent[find(b)] = root
        parent[find(c)] = root
    roots = np.array([find(index) for index in range(vertex_count)])
    _, labels = np.unique(roots[faces[:, 0]], return_inverse=True)
    return labels


def classify(points: np.ndarray, faces: np.ndarray, part_faces: np.ndarray, areas, normals) -> tuple[str, np.ndarray]:
    corners = points[faces[part_faces]]
    low = corners.reshape(-1, 3).min(0)
    size = corners.reshape(-1, 3).max(0) - low
    count = len(part_faces)
    labels = np.full(count, "frame", dtype=object)
    middle_z = corners[:, :, 2].mean(1)

    if count > 30000:
        labels[:] = "rail"
        labels[(np.abs(normals[part_faces, 2]) > 0.9) & (areas[part_faces] > 300)] = "deck"
        return "top_deck", labels
    if count > 10000:
        labels[(np.abs(normals[part_faces, 2]) > 0.9) & (areas[part_faces] > 300)] = "deck"
        labels[(np.abs(normals[part_faces, 2]) < 0.1) & (areas[part_faces] > FACADE_MIN_AREA)] = "module_white"
        return "upper_structure", labels
    if 1500 < count < 2000:
        labels[:] = "crane"
        return "crane", labels
    if size[2] > 300:
        labels[middle_z < -3] = "concrete"
        labels[(middle_z >= -3) & (normals[part_faces, 2] > 0.9) & (areas[part_faces] > 300)] = "deck"
        walls = (middle_z >= -3) & (np.abs(normals[part_faces, 2]) < 0.1) & (areas[part_faces] > FACADE_MIN_AREA)
        labels[walls] = "module_white"
        return "legs_and_lower_deck", labels
    if 1200 < count < 1500 and size[2] > 150:
        labels[:] = "derrick"
        return "derrick", labels
    if 800 < count < 1000 or 500 < count < 550:
        labels[:] = "pipe"
        return f"pipework_{int(low[0])}", labels
    if 300 < count < 350:
        labels[:] = "frame"
        labels[np.abs(normals[part_faces, 2]) > 0.9] = "helideck"
        return "helideck", labels
    if count == 68:
        name = "vessel" if max(size[0], size[1]) < 10 else "tank"
        labels[:] = name
        return f"{name}_{int(low[0])}_{int(low[1])}", labels
    paint = MODULE_PAINTS[(int(abs(low[0])) // 5 + int(abs(low[1])) // 5) % len(MODULE_PAINTS)]
    labels[:] = paint
    return f"module_{int(low[0])}_{int(low[1])}", labels


def corner_normals(faces: np.ndarray, face_normals: np.ndarray, areas: np.ndarray) -> np.ndarray:
    neighbours: dict[int, list[int]] = {}
    for face, vertices in enumerate(faces.tolist()):
        for vertex in vertices:
            neighbours.setdefault(vertex, []).append(face)
    result = np.zeros((len(faces), 3, 3))
    weighted = face_normals * areas[:, None]
    for face, vertices in enumerate(faces.tolist()):
        for corner, vertex in enumerate(vertices):
            nearby = np.array(neighbours[vertex])
            keep = nearby[face_normals[nearby] @ face_normals[face] > SMOOTH_ANGLE_COS]
            total = weighted[keep].sum(0)
            result[face, corner] = total / max(np.linalg.norm(total), 1e-12)
    return result


def box_uvs(corners: np.ndarray, face_normals: np.ndarray, tile: float) -> np.ndarray:
    dominant = np.abs(face_normals).argmax(1)
    u_axis = np.where(dominant == 0, 1, 0)
    v_axis = np.where(dominant == 2, 1, 2)
    rows = np.arange(len(corners))[:, None]
    columns = np.arange(3)[None, :]
    return np.stack((corners[rows, columns, u_axis[:, None]], corners[rows, columns, v_axis[:, None]]), axis=2) / tile


class Materials:
    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.cache: dict[str, UsdShade.Material] = {}

    def _base(self, name: str):
        path = f"/World/Looks/{name}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        reader = UsdShade.Shader.Define(self.stage, f"{path}/uv")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        return material, shader, reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def _texture(self, material, st, image: Path, file_id: str, color_space: str):
        node = UsdShade.Shader.Define(self.stage, f"{material.GetPath()}/{file_id}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(image))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        return node

    def _normal_map(self, material, shader, st, folder: Path):
        node = self._texture(material, st, folder / "nor_gl.jpg", "nor_gl", "raw")
        node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )

    def get(self, name: str) -> UsdShade.Material:
        if name not in self.cache:
            material, shader, st = self._base(name)
            color, roughness, metallic, relief = PAINTS[name]
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            if name in FULL_TEXTURES:
                folder = ASSETS / "textures" / FULL_TEXTURES[name]
                diffuse = self._texture(material, st, folder / "diff.jpg", "diff", "sRGB")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    diffuse.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                )
                rough = self._texture(material, st, folder / "rough.jpg", "rough", "raw")
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                    rough.CreateOutput("r", Sdf.ValueTypeNames.Float)
                )
                self._normal_map(material, shader, st, folder)
            else:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
                if relief:
                    self._normal_map(material, shader, st, ASSETS / "textures" / relief)
            self.cache[name] = material
        return self.cache[name]


def write_mesh(stage, materials, path, world, faces, face_normals, normals, material_name):
    used, remapped = np.unique(faces, return_inverse=True)
    remapped = remapped.reshape(-1, 3)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(world[used].astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remapped.reshape(-1).astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    uvs = box_uvs(world[faces], face_normals, TILE_METRES.get(material_name, 3.0))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set(Vt.Vec2fArray.FromNumpy(uvs.reshape(-1, 2).astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials.get(material_name))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def add_camera(stage, name, eye, target, focal):
    camera = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
    camera.CreateFocalLengthAttr(focal)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 40000))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    return str(camera.GetPath())


def add_environment(stage):
    sea = UsdGeom.Mesh.Define(stage, "/World/Environment/sea")
    extent = 12000.0
    sea.CreatePointsAttr(
        [(-extent, -extent, 0), (extent, -extent, 0), (extent, extent, 0), (-extent, extent, 0)]
    )
    sea.CreateFaceVertexCountsAttr([4])
    sea.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    sea.CreateNormalsAttr([(0, 0, 1)] * 4)
    sea.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    sea.CreateSubdivisionSchemeAttr("none")
    tiles = extent / SEA_TILE_METRES
    UsdGeom.PrimvarsAPI(sea).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(-tiles, -tiles), (tiles, -tiles), (tiles, tiles), (-tiles, tiles)])

    material = UsdShade.Material.Define(stage, "/World/Looks/sea")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/sea/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.012, 0.06, 0.09))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.13)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
    reader = UsdShade.Shader.Define(stage, "/World/Looks/sea/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    normal = UsdShade.Shader.Define(stage, "/World/Looks/sea/ripples")
    normal.CreateIdAttr("UsdUVTexture")
    normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(GENERATED / "sea_normal.png"))
    normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    normal.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    normal.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(sea.GetPrim()).Bind(material)

    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/sky")
    dome.CreateTextureFileAttr().Set(rel(GENERATED / "ocean_sky.hdr"))
    dome.CreateTextureFormatAttr().Set("latlong")
    dome.CreateIntensityAttr(480.0)
    UsdGeom.Xformable(dome).AddRotateZOp().Set(SKY_ROTATION_DEG)
    sun = UsdLux.DistantLight.Define(stage, "/World/Environment/sun")
    sun.CreateIntensityAttr(1100.0)
    sun.CreateAngleAttr(2.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(55, 15, 35))

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)


def build_scene() -> dict:
    if not STL_PATH.exists():
        raise FileNotFoundError(f"Missing {STL_PATH}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points, faces = load_stl(STL_PATH)
    parts = connected_parts(faces, len(points))

    cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    lengths = np.maximum(np.linalg.norm(cross, axis=1), 1e-12)
    areas = lengths / 2
    face_normals = cross / lengths[:, None]
    normals = corner_normals(faces, face_normals, areas)

    world = (points - np.array([0.0, 0.0, args.sea_level])) * args.scale

    if SCENE_PATH.exists():
        SCENE_PATH.unlink()
    stage = Usd.Stage.CreateNew(str(SCENE_PATH))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    for group in (
        "Rig", "Details", "Details/Gauges", "Details/Pipes", "Details/Props", "Details/Walkway",
        "Environment", "Cameras",
    ):
        UsdGeom.Scope.Define(stage, f"/World/{group}")

    materials = Materials(stage)
    builder = SceneBuilder(stage)
    glass = builder.plain("window_glass", (0.03, 0.07, 0.1), 0.08, 0.6)
    helideck_center = None
    facade_faces = []
    leg_faces = None
    inventory = {}
    used_names: set[str] = set()
    for part in range(parts.max() + 1):
        part_faces = np.flatnonzero(parts == part)
        name, labels = classify(points, faces, part_faces, areas, face_normals)
        name = name.replace("-", "n")
        while name in used_names:
            name += "_b"
        used_names.add(name)
        inventory[name] = {"triangles": len(part_faces)}
        corners = world[faces[part_faces]].reshape(-1, 3)
        part_low, part_high = corners.min(0), corners.max(0)
        if name.startswith("module_") and part_high[2] - part_low[2] > 5 and max(part_high[:2] - part_low[:2]) > 8:
            rig_details.add_windows(builder, part_low, part_high, name, glass)
        if name == "helideck":
            tops = part_faces[(labels == "helideck") & (face_normals[part_faces, 2] > 0.9)]
            top_z = world[faces[tops]][:, :, 2].max()
            helideck_faces = tops[world[faces[tops]][:, :, 2].mean(1) > top_z - 0.05]
            top_points = world[faces[helideck_faces]].reshape(-1, 3)
            helideck_center = (top_points.min(0) + top_points.max(0)) / 2
        if name == "legs_and_lower_deck":
            leg_faces = part_faces[labels == "concrete"]
        if name in ("upper_structure", "legs_and_lower_deck"):
            facade_faces.append(part_faces[labels == "module_white"])
        for material_name in sorted(set(labels)):
            chosen = part_faces[labels == material_name]
            write_mesh(
                stage, materials, f"/World/Rig/{name}_{material_name}", world, faces[chosen],
                face_normals[chosen], normals[chosen], material_name,
            )
    add_environment(stage)
    rig_details.add_helideck_markings(builder, helideck_center)
    facade = np.concatenate(facade_faces)
    window_count = rig_details.add_facade_windows(builder, world[faces[facade]], face_normals[facade], glass)
    inspection_points = rig_details.add_process_area(builder, stage, args.seed)
    rig_details.add_props(builder)
    rig_details.add_splash_zone(builder, world[faces[leg_faces]])
    rig_details.add_walkway_lines(builder, rig_details.PATROL_ROUTE)
    blocked = rig_details.route_blockages(world[faces], rig_details.PATROL_ROUTE)

    low, high = world.min(0), world.max(0)
    center = (low + high) / 2
    cameras = {
        "wide": add_camera(stage, "wide", (center[0] + 150, center[1] - 165, 62), (center[0], center[1], 30), 28),
        "sea_level": add_camera(stage, "sea_level", (center[0] - 95, center[1] - 120, 6), (center[0], center[1], 42), 24),
        "topside": add_camera(stage, "topside", (center[0] + 60, center[1] - 70, 85), (center[0], center[1], 45), 26),
        "on_deck": add_camera(stage, "on_deck", (34.0, -24.0, 46.5), (0.0, 8.0, 49.0), 20),
        "process_area": add_camera(stage, "process_area", (-11.0, 13.5, 44.6), (-15.0, 23.6, 43.7), 24),
        "gauge_closeup": add_camera(stage, "gauge_closeup", (-12.0, 22.0, 43.85), (-12.0, 23.32, 43.8), 40),
        "helideck": add_camera(stage, "helideck", (center[0] - 50, center[1] - 60, 72), (-22.0, -22.0, 51.0), 30),
    }
    layer = stage.GetRootLayer()
    layer_data = dict(layer.customLayerData)
    layer_data["renderSettings"] = HAZE_SETTINGS
    layer.customLayerData = layer_data
    layer.Save()

    deck_heights = {
        name: round((raw - args.sea_level) * args.scale, 2)
        for name, raw in (("lower_deck", 5.0), ("middle_deck", 50.0), ("top_deck", 95.0))
    }
    metadata = {
        "source": "Oil Rig by AnsysLearn, Sketchfab, CC BY 4.0",
        "scale_m_per_stl_unit": args.scale,
        "scale_note": "Estimated from proportions. The source gives no real-world size.",
        "units": "metres, Z up, sea level at z=0",
        "bounds_m": {"min": low.round(2).tolist(), "max": high.round(2).tolist()},
        "deck_heights_m": deck_heights,
        "patrol_route_m": [[x, y, rig_details.TOP_DECK_Z] for x, y in rig_details.PATROL_ROUTE],
        "patrol_route_blocked_samples": blocked,
        "inspection_points": inspection_points,
        "parts": inventory,
        "cameras": cameras,
    }
    (OUTPUT / "full_rig_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"SCENE_WRITTEN {SCENE_PATH} parts={len(inventory)} size={(high - low).round(1)}")
    print(f"ROUTE_BLOCKED {len(blocked)} {blocked[:12]}")
    print(f"FACADE_WINDOWS {window_count}")
    return metadata


def render_previews(cameras: dict) -> None:
    import carb.settings
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

    settings = carb.settings.get_settings()
    for key, value in HAZE_SETTINGS.items():
        path = "/" + key.replace(":", "/")
        print(f"HAZE_LOADED {path} = {settings.get(path)}")
        if isinstance(value, bool):
            settings.set_bool(path, value)
        elif isinstance(value, float):
            settings.set_float(path, value)
        else:
            settings.set_float_array(path, list(value))
    for _ in range(30):
        simulation_app.update()

    for name, path in cameras.items():
        product = rep.create.render_product(path, (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=24, delta_time=0.0)
        data = np.asarray(annotator.get_data())
        out = OUTPUT / f"full_rig_{name}.png"
        Image.fromarray(data[:, :, :3].astype("uint8")).save(out)
        print(f"PREVIEW_WRITTEN {out} mean={data[:, :, :3].mean():.1f}")
        annotator.detach(product)
        product.destroy()
    rep.orchestrator.wait_until_complete()


metadata = build_scene()
if not args.no_render:
    render_previews(metadata["cameras"])
simulation_app.close()

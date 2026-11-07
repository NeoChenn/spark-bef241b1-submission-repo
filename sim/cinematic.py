"""Cinematic layer for the full rig: skies, ocean, rig lighting and presets.

This layer sits on top of sim/output/full_rig.usda, so the engineering scene and
its data stay unchanged. Import after SimulationApp has started.
"""

from __future__ import annotations

import math

import carb.settings
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

import ocean
from scene_builder import ASSETS, GENERATED, OUTPUT, SceneBuilder, rel

SCENE_PATH = OUTPUT / "cinematic.usda"
SEA_NORMAL_TILE_M = 26.0
FOAM_TILE_M = 14.0
LAMP_COLOR = (1.0, 0.56, 0.17)
WINDOW_GLOW = Gf.Vec3f(1.1, 0.62, 0.26)
DERRICK_TOP = (20.0, 20.0, 81.6)

GLOW_SETTINGS = {
    "rtx:post:lensFlares:enabled": True,
    "rtx:post:lensFlares:physicalSettings": False,
    "rtx:post:lensFlares:flareScale": 0.08,
}

PRESETS = {
    "night": {
        "sky": GENERATED / "sky_night.hdr",
        "dome": 400.0,
        "sky_rotation": 40.0,
        "sun": 0.0,
        "sun_rotation": (55.0, 15.0, 35.0),
        "sun_color": (0.7, 0.85, 1.0),
        "fog_color": (0.05, 0.16, 0.18),
        "fog_height": 0.002,
        "wave_height": 1.5,
        "lights": 1.0,
        "windows": 1.0,
        "sea_color": (0.002, 0.012, 0.016),
    },
    "storm": {
        "sky": GENERATED / "sky_storm.hdr",
        "dome": 900.0,
        "sky_rotation": 140.0,
        "sun": 90.0,
        "sun_rotation": (50.0, 0.0, 300.0),
        "sun_color": (0.8, 0.88, 1.0),
        "fog_color": (0.2, 0.24, 0.27),
        "fog_height": 0.03,
        "wave_height": 3.2,
        "lights": 0.8,
        "windows": 0.8,
        "sea_color": (0.008, 0.02, 0.024),
    },
    "golden": {
        "sky": ASSETS / "hdris" / "evening_road_01_puresky.hdr",
        "dome": 150.0,
        "sky_rotation": 205.0,
        "sun": 500.0,
        "sun_rotation": (82.0, 0.0, 60.0),
        "sun_color": (1.0, 0.66, 0.36),
        "fog_color": (0.72, 0.56, 0.4),
        "fog_height": 0.004,
        "wave_height": 1.0,
        "lights": 0.0,
        "windows": 0.15,
        "sea_color": (0.01, 0.03, 0.04),
    },
}


def render_settings(name: str) -> dict:
    """Settings saved in the scene file, so opening it gives the same look."""
    preset = PRESETS[name]
    return {
        **GLOW_SETTINGS,
        "rtx:fog:enabled": True,
        "rtx:fog:fogZup:enabled": True,
        "rtx:fog:fogColor": Gf.Vec3f(*preset["fog_color"]),
        "rtx:fog:fogColorIntensity": 1.0,
        "rtx:fog:fogStartHeight": 0.0,
        "rtx:fog:fogHeightDensity": preset["fog_height"],
        "rtx:fog:fogHeightFalloff": 0.12,
        "rtx:fog:fogDistanceDensity": 0.0,
    }


def emissive_material(stage, name: str, color) -> UsdShade.Material:
    path = f"/World/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def add_lamp(stage, material, index: int, position, color, radius: float, intensity: float, sphere: float) -> None:
    path = f"/World/Cinematic/Lamps/lamp_{index:03d}"
    root = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*position))
    if sphere > 0:
        bulb = UsdGeom.Sphere.Define(stage, f"{path}/bulb")
        bulb.CreateRadiusAttr(sphere)
        UsdShade.MaterialBindingAPI.Apply(bulb.GetPrim()).Bind(material)
    light = UsdLux.SphereLight.Define(stage, f"{path}/light")
    light.CreateRadiusAttr(radius)
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(*color))


def lamp_positions() -> list[tuple]:
    """Lamp posts along the deck edges of the three levels, chosen from the deck outline."""
    points = []
    for x in np.arange(-25, 37, 9.0):
        points.append((float(x), 26.6, 46.0))
    for y in np.arange(-11, 26, 9.0):
        points.append((36.6, float(y), 46.0))
    for x in np.arange(11, 37, 9.0):
        points.append((float(x), -12.6, 46.0))
    for x in np.arange(-25, 2, 9.0):
        points.append((float(x), -27.6, 46.0))
    for y in np.arange(-27, 26, 9.0):
        points.append((-26.8, float(y), 46.0))
    for x in np.arange(-16, 30, 11.0):
        points.append((float(x), -14.0, 36.0))
        points.append((float(x), -24.0, 24.5))
    return points


def build_ocean(stage):
    rest, distance, counts, indices = ocean.make_grid()
    points, normals = ocean.surface(rest, distance, 0.0, 1.5)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Cinematic/ocean")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy((rest / SEA_NORMAL_TILE_M).astype(np.float32))
    )

    path = "/World/Looks/cinematic_sea"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.002, 0.012, 0.016))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
    reader = UsdShade.Shader.Define(stage, f"{path}/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    scroll = UsdShade.Shader.Define(stage, f"{path}/scroll")
    scroll.CreateIdAttr("UsdTransform2d")
    scroll.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    scroll.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0, 0))
    ripples = UsdShade.Shader.Define(stage, f"{path}/ripples")
    ripples.CreateIdAttr("UsdUVTexture")
    ripples.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(GENERATED / "sea_normal.png"))
    ripples.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        scroll.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    ripples.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    ripples.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    ripples.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    ripples.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
    ripples.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        ripples.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return rest, distance


def leg_centers(stage) -> list[tuple]:
    centers = []
    for prim in stage.Traverse():
        if "splash_band" not in prim.GetName():
            continue
        translate = prim.GetAttribute("xformOp:translate").Get()
        first = prim.GetAttribute("points").Get()[0]
        centers.append((translate[0], translate[1], math.hypot(first[1], first[2])))
    return centers


def build_foam(stage) -> None:
    path = "/World/Looks/cinematic_foam"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.62, 0.62))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    reader = UsdShade.Shader.Define(stage, f"{path}/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    image = UsdShade.Shader.Define(stage, f"{path}/mask")
    image.CreateIdAttr("UsdUVTexture")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(GENERATED / "foam.png"))
    image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
        image.CreateOutput("a", Sdf.ValueTypeNames.Float)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))

    for number, (x, y, radius) in enumerate(leg_centers(stage)):
        rings, segments = 5, 96
        widths = np.linspace(radius - 0.2, radius + 7.5, rings)
        angles = np.arange(segments + 1) / segments * math.tau
        grid_r, grid_a = np.meshgrid(widths, angles, indexing="ij")
        points = np.stack((x + grid_r * np.cos(grid_a), y + grid_r * np.sin(grid_a), np.full(grid_r.shape, 0.35)), axis=-1)
        ring, step = np.meshgrid(np.arange(rings - 1), np.arange(segments), indexing="ij")
        a = ring * (segments + 1) + step
        quads = np.stack((a, a + segments + 1, a + segments + 2, a + 1), axis=-1).reshape(-1).astype(np.int32)
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Cinematic/foam_{number}")
        flat = points.reshape(-1, 3).astype(np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(flat))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(quads) // 4, 4, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.tile(np.array((0, 0, 1), np.float32), (len(flat), 1))))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy((flat[:, :2] / FOAM_TILE_M).astype(np.float32))
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def build_rig_lights(stage) -> None:
    amber = emissive_material(stage, "lamp_amber", tuple(c * 7 for c in LAMP_COLOR))
    red = emissive_material(stage, "lamp_red", (26.0, 0.9, 0.4))
    for index, position in enumerate(lamp_positions()):
        add_lamp(stage, amber, index, position, LAMP_COLOR, 0.25, 240000.0, 0.34)
    add_lamp(stage, red, 900, DERRICK_TOP, (1.0, 0.05, 0.03), 1.0, 1200000.0, 1.3)
    add_lamp(stage, red, 901, (-36.0, -32.0, 52.5), (1.0, 0.05, 0.03), 0.3, 60000.0, 0.4)
    add_lamp(stage, red, 902, (37.2, 26.8, 47.5), (1.0, 0.05, 0.03), 0.3, 60000.0, 0.4)


def build_floodlights(stage) -> None:
    """Unseen lights that make warm pools on the decks and on the sea below."""
    deck_floods = (
        (-15, 10, 50), (10, -6, 50), (28, 10, 50), (-5, -20, 38), (20, -20, 38), (-10, -20, 27), (15, -20, 27),
    )
    for index, position in enumerate(deck_floods):
        add_lamp(stage, None, 300 + index, position, LAMP_COLOR, 1.0, 190000.0, 0.0)
    for index, position in enumerate(((0, -36, 17), (28, -22, 17), (-22, -36, 17), (40, 6, 17))):
        add_lamp(stage, None, 320 + index, position, LAMP_COLOR, 1.5, 380000.0, 0.0)


def build_flare(stage) -> None:
    """Flare boom and flame. The flame is emissive geometry shaped for bloom, not a fluid sim."""
    b = SceneBuilder(stage)
    steel = b.plain("flare_steel", (0.1, 0.11, 0.12), 0.5, 0.5)
    base = (-26.0, 8.0, 42.0)
    tip = (-58.0, 8.0, 70.0)
    b.tube("/World/Cinematic/Flare", "boom", base, tip, 0.55, steel, segments=16)
    b.tube("/World/Cinematic/Flare", "brace", (-26.0, 8.0, 31.0), (-46.0, 8.0, 59.5), 0.32, steel, segments=12)
    fire = emissive_material(stage, "flare_flame", (34.0, 11.0, 1.4))
    core = emissive_material(stage, "flare_core", (70.0, 40.0, 10.0))
    for index, (dx, dz, radius, stretch, material) in enumerate(
        (
            (-0.6, 2.2, 1.0, 2.4, core),
            (-1.4, 4.6, 1.5, 2.2, fire),
            (-2.8, 8.0, 1.2, 2.5, fire),
            (-0.4, 5.6, 0.9, 2.0, fire),
            (-4.6, 11.0, 0.7, 2.2, fire),
        )
    ):
        blob = UsdGeom.Sphere.Define(stage, f"/World/Cinematic/Flare/flame_{index}")
        blob.CreateRadiusAttr(radius)
        ops = UsdGeom.Xformable(blob)
        ops.AddTranslateOp().Set(Gf.Vec3d(tip[0] + dx, tip[1], tip[2] + dz))
        ops.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, stretch))
        UsdShade.MaterialBindingAPI.Apply(blob.GetPrim()).Bind(material)
    add_lamp(stage, None, 400, (tip[0] - 1.5, tip[1], tip[2] + 7.0), (1.0, 0.42, 0.1), 3.0, 3000000.0, 0.0)


def add_ship(stage) -> None:
    """Distant supply vessel, kept simple because it reads as a silhouette."""
    b = SceneBuilder(stage)
    hull = b.plain("ship_hull", (0.05, 0.07, 0.09), 0.6, 0.2)
    deck = b.plain("ship_deck", (0.3, 0.32, 0.34), 0.6, 0.1)
    x, y = -210.0, 330.0
    b.box("/World/Cinematic/Ship", "hull", (x, y, 2.5), (72.0, 15.0, 6.0), hull, collide=None)
    b.box("/World/Cinematic/Ship", "house", (x + 22.0, y, 9.5), (18.0, 12.0, 9.0), deck, collide=None)
    b.box("/World/Cinematic/Ship", "bridge", (x + 25.0, y, 16.0), (9.0, 10.0, 4.0), deck, collide=None)
    amber = emissive_material(stage, "ship_lamp", tuple(c * 22 for c in LAMP_COLOR))
    for index, offset in enumerate((-30.0, -12.0, 8.0, 25.0)):
        add_lamp(stage, amber, 910 + index, (x + offset, y - 8.0, 8.0), LAMP_COLOR, 0.3, 90000.0, 0.45)


def build_stage() -> Usd.Stage:
    if SCENE_PATH.exists():
        SCENE_PATH.unlink()
    stage = Usd.Stage.CreateNew(str(SCENE_PATH))
    stage.GetRootLayer().subLayerPaths.append(rel(OUTPUT / "full_rig.usda"))
    leak = OUTPUT / "leak_flanges.usda"          # gas-leak job's bolted flanges (sim/build_leak_flanges.py)
    if leak.exists():
        stage.GetRootLayer().subLayerPaths.insert(0, rel(leak))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetPrimAtPath("/World/Environment/sea").SetActive(False)
    for group in ("Cinematic", "Cinematic/Lamps", "Cinematic/Ship", "Cinematic/Flare"):
        UsdGeom.Scope.Define(stage, f"/World/{group}")

    build_ocean(stage)
    build_foam(stage)
    build_rig_lights(stage)
    build_floodlights(stage)
    build_flare(stage)
    add_ship(stage)

    window = UsdShade.Shader.Get(stage, "/World/Looks/window_glass/surface")
    window.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(WINDOW_GLOW)
    return stage


def apply_preset(stage, name: str) -> None:
    preset = PRESETS[name]
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Environment/sky"))
    dome.GetTextureFileAttr().Set(rel(preset["sky"]))
    dome.GetIntensityAttr().Set(preset["dome"])
    UsdGeom.XformCommonAPI(dome.GetPrim())
    dome.GetPrim().GetAttribute("xformOp:rotateZ").Set(preset["sky_rotation"])

    sun_prim = stage.GetPrimAtPath("/World/Environment/sun")
    sun = UsdLux.DistantLight(sun_prim)
    sun.GetIntensityAttr().Set(preset["sun"])
    sun.GetColorAttr().Set(Gf.Vec3f(*preset["sun_color"]))
    sun_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(*preset["sun_rotation"]))

    sea = UsdShade.Shader.Get(stage, "/World/Looks/cinematic_sea/surface")
    sea.GetInput("diffuseColor").Set(Gf.Vec3f(*preset["sea_color"]))
    window = UsdShade.Shader.Get(stage, "/World/Looks/window_glass/surface")
    window.GetInput("emissiveColor").Set(WINDOW_GLOW * preset["windows"])

    scale = preset["lights"]
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Cinematic")):
        if prim.IsA(UsdLux.SphereLight):
            base = prim.GetAttribute("spark:baseIntensity")
            if not base.IsValid():
                base = prim.CreateAttribute("spark:baseIntensity", Sdf.ValueTypeNames.Float, custom=True)
                base.Set(prim.GetAttribute("inputs:intensity").Get())
            prim.GetAttribute("inputs:intensity").Set(base.Get() * scale)

    settings = carb.settings.get_settings()
    settings.set_float_array("/rtx/fog/fogColor", list(preset["fog_color"]))
    settings.set_float("/rtx/fog/fogHeightDensity", preset["fog_height"])
    for key, value in GLOW_SETTINGS.items():
        path = "/" + key.replace(":", "/")
        if isinstance(value, bool):
            settings.set_bool(path, value)
        else:
            settings.set_float(path, value)


CAMERA_KEYS = (
    (0.0, (420.0, -520.0, 190.0), (4.0, -4.0, 30.0), 28.0),
    (6.0, (170.0, -240.0, 38.0), (4.0, -6.0, 40.0), 28.0),
    (12.0, (-190.0, -170.0, 22.0), (0.0, -6.0, 42.0), 28.0),
    (16.0, (-250.0, 70.0, 34.0), (0.0, -6.0, 42.0), 28.0),
    (20.0, (-34.0, -66.0, 47.0), (-4.0, -12.0, 38.0), 22.0),
    (24.0, (330.0, 110.0, 70.0), (4.0, -4.0, 34.0), 26.0),
)
SEQUENCE_SECONDS = CAMERA_KEYS[-1][0]


def camera_at(time: float):
    """Return eye, target and focal length, smoothly interpolated through CAMERA_KEYS."""
    times = [key[0] for key in CAMERA_KEYS]
    time = min(max(time, 0.0), times[-1] - 1e-6)
    index = int(np.searchsorted(times, time, side="right")) - 1
    span = times[index + 1] - times[index]
    u = (time - times[index]) / span

    def spline(field: int):
        p0 = np.array(CAMERA_KEYS[max(index - 1, 0)][field], dtype=float)
        p1 = np.array(CAMERA_KEYS[index][field], dtype=float)
        p2 = np.array(CAMERA_KEYS[index + 1][field], dtype=float)
        p3 = np.array(CAMERA_KEYS[min(index + 2, len(CAMERA_KEYS) - 1)][field], dtype=float)
        return 0.5 * (
            2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2 + (3 * p1 - p0 - 3 * p2 + p3) * u**3
        )

    return spline(1), spline(2), float(spline(3))


def set_camera(stage, path: str, time: float) -> None:
    eye, target, focal = camera_at(time)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
    camera.GetFocalLengthAttr().Set(focal)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    camera.GetPrim().GetAttribute("xformOp:transform").Set(view.GetInverse())


def animate_effects(stage, time: float) -> None:
    """Flame flicker and a slow pulse on the derrick beacon."""
    for index in range(5):
        prim = stage.GetPrimAtPath(f"/World/Cinematic/Flare/flame_{index}")
        wave = math.sin(time * (7.0 + index * 1.7) + index * 2.1) * 0.5 + math.sin(time * 13.0 + index) * 0.5
        stretch = 2.2 + 0.32 * wave
        prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(1.0 + 0.12 * wave, 1.0 + 0.12 * wave, stretch))
    beacon = stage.GetPrimAtPath("/World/Cinematic/Lamps/lamp_900/light")
    base = beacon.GetAttribute("spark:baseIntensity")
    if base.IsValid():
        pulse = 0.55 + 0.45 * max(math.sin(time * 2.2), 0.0)
        beacon.GetAttribute("inputs:intensity").Set(base.Get() * pulse)


def animate_ocean(stage, rest, distance, time: float, height: float) -> None:
    points, normals = ocean.surface(rest, distance, time, height)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Cinematic/ocean"))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    scroll = UsdShade.Shader.Get(stage, "/World/Looks/cinematic_sea/scroll")
    scroll.GetInput("translation").Set(Gf.Vec2f(time * 0.035, time * 0.02))

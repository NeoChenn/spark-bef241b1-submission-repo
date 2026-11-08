from __future__ import annotations

import math

import carb.settings
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

import assets_hero
import facade
import flare_fx
import weather
import greebles
import look
import ocean
import hero_mat
import ocean_fx
import photoreal
import structure
from scene_builder import ASSETS, GENERATED, OUTPUT, SceneBuilder, rel
from topside_frames import aim_from_down, kelvin_rgb

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
        "fog_color": (0.03, 0.08, 0.095),
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
        "wave_height": 2.5,
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
    "dusk": {
        "sky": ASSETS / "hdris" / "belfast_sunset_puresky.hdr",
        "dome": 180.0,
        "sky_rotation": 333.0,
        "sun": 1200.0,
        "sun_rotation": (85.0, 0.0, 205.0),
        "sun_color": (1.0, 0.52, 0.22),
        "fog_color": (0.8, 0.45, 0.22),
        "fog_height": 0.004,
        "wave_height": 1.0,
        "lights": 0.7,
        "windows": 0.8,
        "sea_color": (0.012, 0.024, 0.03),
        "sun_disc": 1.0,
    },
}


def render_settings(name: str) -> dict:
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


SODIUM, NEUTRAL, LED = 2000.0, 4000.0, 6000.0
DOWN = (0.0, 0.0, -1.0)
AREA_LIGHTS = (
    ("rect", (18.75, -11.0, 53.0), NEUTRAL, 110000.0, (0.0, 0.7, -1.0), (3.0, 1.2)),
    ("rect", (-18.0, 16.4, 50.5), NEUTRAL, 70000.0, (0.0, 0.5, -1.0), (2.2, 0.8)),
    ("rect", (-5.0, 16.4, 50.5), NEUTRAL, 60000.0, (0.0, 0.5, -1.0), (2.2, 0.8)),
    ("spot", (16.0, 16.0, 60.0), LED, 1350000.0, (0.25, 0.25, -1.0), 38.0),
    ("spot", (24.0, 24.0, 60.0), LED, 1050000.0, (-0.25, -0.25, -1.0), 38.0),
    ("spot", (-20.0, -8.0, 48.3), LED, 675000.0, DOWN, 55.0),
    ("spot", (-10.0, -8.0, 48.3), LED, 525000.0, DOWN, 55.0),
    ("spot", (31.5, -10.0, 40.6), SODIUM, 975000.0, DOWN, 50.0),
    ("spot", (31.5, 3.0, 40.6), SODIUM, 750000.0, DOWN, 50.0),
    ("spot", (31.5, 16.0, 40.6), SODIUM, 975000.0, DOWN, 50.0),
    ("spot", (29.0, -4.0, 35.9), SODIUM, 450000.0, DOWN, 60.0),
    ("spot", (34.0, 9.0, 35.9), SODIUM, 450000.0, DOWN, 60.0),
    ("spot", (14.0, -6.0, 40.6), NEUTRAL, 675000.0, DOWN, 50.0),
    ("spot", (20.0, 5.0, 40.6), NEUTRAL, 525000.0, DOWN, 50.0),
    ("rect", (31.0, -27.0, 39.5), LED, 90000.0, DOWN, (2.6, 1.0)),
    ("spot", (31.0, -31.0, 35.3), LED, 600000.0, DOWN, 60.0),
    ("spot", (-18.0, -8.0, 40.6), SODIUM, 240000.0, DOWN, 45.0),
    ("spot", (0.0, -12.0, 24.7), LED, 450000.0, DOWN, 60.0),
    ("spot", (8.0, -12.0, 24.7), LED, 450000.0, DOWN, 60.0),
    ("spot", (4.0, -12.0, 29.6), LED, 675000.0, DOWN, 55.0),
    ("spot", (12.0, -12.0, 29.6), LED, 675000.0, DOWN, 55.0),
    ("spot", (19.5, -28.0, 29.6), SODIUM, 750000.0, DOWN, 50.0),
    ("spot", (21.5, -8.0, 29.6), NEUTRAL, 525000.0, DOWN, 50.0),
)
BULBS = (
    ((-1.0, 4.0, 48.0), SODIUM, 260000.0),
    ((8.0, -12.5, 47.0), SODIUM, 220000.0),
    ((34.0, 24.0, 46.0), SODIUM, 90000.0),
    ((-26.8, -5.0, 46.0), SODIUM, 200000.0),
    ((36.6, 6.0, 46.0), LED, 220000.0),
    ((8.0, 26.6, 46.0), SODIUM, 200000.0),
    ((-12.0, 26.6, 46.0), NEUTRAL, 160000.0),
    ((8.0, -37.2, 33.5), SODIUM, 160000.0),
    ((20.0, -37.2, 33.5), SODIUM, 160000.0),
    ((-5.0, 24.7, 23.0), SODIUM, 80000.0),
    ((-24.7, -15.0, 23.0), SODIUM, 60000.0),
    ((24.7, -10.0, 23.0), LED, 140000.0),
    ((14.0, -37.2, 23.0), SODIUM, 160000.0),
    ((24.7, 15.0, 23.0), SODIUM, 120000.0),
    ((40.2, -1.0, 34.7), SODIUM, 70000.0),
    ((40.2, 2.5, 38.4), SODIUM, 70000.0),
)
SEA_FLOODS = (
    ((2.0, -31.0, 17.0), SODIUM, 330000.0),
    ((14.0, 8.0, 17.0), LED, 220000.0),
    ((-16.0, 2.0, 17.0), SODIUM, 120000.0),
)
POLES = ((18.75, -11.6, 53.0), (-18.0, 15.8, 50.5), (-5.0, 15.8, 50.5))


def add_fixture(stage, index: int, kind: str, position, kelvin, intensity, aim, shape) -> None:
    path = f"/World/Cinematic/Lamps/area_{index:03d}"
    root = UsdGeom.Xform.Define(stage, path)
    ops = UsdGeom.Xformable(root)
    ops.AddTranslateOp().Set(Gf.Vec3d(*position))
    w, x, y, z = aim_from_down(aim)
    ops.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    color = kelvin_rgb(kelvin)
    glow = emissive_material(stage, f"fixture_{int(kelvin)}", tuple(c * 20 for c in color))
    dark = SceneBuilder(stage).plain("fixture_housing", (0.12, 0.13, 0.14), 0.5, 0.6)
    width, depth = (shape[0], shape[1]) if kind == "rect" else (0.45, 0.3)
    for name, z, size, material in (
        ("housing", 0.12, (width + 0.12, depth + 0.12, 0.18), dark),
        ("lens", 0.02, (width, depth, 0.02), glow),
    ):
        part = UsdGeom.Cube.Define(stage, f"{path}/{name}")
        part.CreateSizeAttr(1.0)
        part_ops = UsdGeom.Xformable(part)
        part_ops.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        part_ops.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(material)
    if kind == "rect":
        light = UsdLux.RectLight.Define(stage, f"{path}/light")
        light.CreateWidthAttr(shape[0])
        light.CreateHeightAttr(shape[1])
    else:
        light = UsdLux.SphereLight.Define(stage, f"{path}/light")
        light.CreateRadiusAttr(0.18)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.2))
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(shape)
        shaping.CreateShapingConeSoftnessAttr(0.35)
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(*color))


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


RED = (1.0, 0.05, 0.03)
HELIDECK = (-22.5, -22.5, 51.4)
RED_BEACONS = (
    (900, DERRICK_TOP, 1.0, 1200000.0, 1.3),
    (901, (-36.0, -32.0, 52.5), 0.3, 60000.0, 0.4),
    (902, (37.2, 26.8, 47.5), 0.3, 60000.0, 0.4),
    (903, (15.0, 15.0, 66.0), 0.3, 90000.0, 0.45),
    (904, (25.0, 25.0, 66.0), 0.3, 90000.0, 0.45),
    (905, (15.0, 25.0, 54.0), 0.3, 90000.0, 0.45),
    (906, (25.0, 15.0, 54.0), 0.3, 90000.0, 0.45),
    (907, (34.7, -4.5, 51.6), 0.3, 90000.0, 0.45),
    (908, (-55.0, 8.0, 66.5), 0.4, 150000.0, 0.5),
    (909, (-11.6, 20.8, 41.6), 0.3, 60000.0, 0.4),
)
STROBES = ((970, (-27.2, 27.0, 49.2)), (971, (37.2, -27.2, 49.2)), (972, (-27.2, -27.4, 49.2)), (973, (37.2, 27.2, 49.6)))
STATUS = ((980, (1.1, -5.8, 23.0), (0.1, 0.45, 1.0)), (981, (11.1, -5.8, 23.0), (0.1, 0.45, 1.0)),
          (982, (-3.9, -5.8, 23.0), (0.1, 1.0, 0.3)), (983, (6.1, -5.8, 23.0), (1.0, 0.75, 0.05)))


def ring(center, radius: float, count: int) -> list[tuple]:
    return [(center[0] + radius * math.cos(2 * math.pi * k / count), center[1] + radius * math.sin(2 * math.pi * k / count),
             center[2]) for k in range(count)]


def build_rig_lights(stage) -> None:
    red = emissive_material(stage, "lamp_red", (26.0, 0.9, 0.4))
    green = emissive_material(stage, "lamp_green", (0.6, 22.0, 2.0))
    yellow = emissive_material(stage, "lamp_yellow", (20.0, 14.0, 0.5))
    white = emissive_material(stage, "lamp_white", (30.0, 30.0, 30.0))
    bulbs = {k: emissive_material(stage, f"lamp_{int(k)}", tuple(c * 8 for c in kelvin_rgb(k))) for k in (SODIUM, NEUTRAL, LED)}
    for index, (position, kelvin, intensity) in enumerate(BULBS):
        add_lamp(stage, bulbs[kelvin], index, position, kelvin_rgb(kelvin), 0.25, intensity, 0.3)
    for index, (kind, position, kelvin, intensity, aim, shape) in enumerate(AREA_LIGHTS):
        add_fixture(stage, 500 + index, kind, position, kelvin, intensity, aim, shape)
    for index, position, radius, intensity, sphere in RED_BEACONS:
        add_lamp(stage, red, index, position, RED, radius, intensity, sphere)
    for k, position in enumerate(ring(HELIDECK, 10.5, 16)):
        add_lamp(stage, green, 930 + k, position, (0.05, 1.0, 0.15), 0.15, 20000.0, 0.22)
    for k, position in enumerate(ring(HELIDECK, 8.2, 12)):
        add_lamp(stage, yellow, 950 + k, position, (1.0, 0.75, 0.05), 0.12, 12000.0, 0.16)
    for index, position in STROBES:
        add_lamp(stage, white, index, position, (1.0, 1.0, 1.0), 0.3, 200000.0, 0.35)
    for index, position, color in STATUS:
        add_lamp(stage, emissive_material(stage, f"lamp_status_{index}", tuple(c * 18 for c in color)), index, position,
                 color, 0.12, 15000.0, 0.15)


def build_floodlights(stage) -> None:
    for index, (position, kelvin, intensity) in enumerate(SEA_FLOODS):
        add_lamp(stage, None, 320 + index, position, kelvin_rgb(kelvin), 1.0, intensity, 0.0)
    b = SceneBuilder(stage)
    steel = b.plain("lamp_pole", (0.3, 0.32, 0.33), 0.5, 0.6)
    for x, y, top in POLES:
        b.tube("/World/Cinematic/Lamps", "pole", (x, y, 42.5), (x, y, top + 0.3), 0.09, steel, segments=12)


def build_flare(stage) -> None:
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
    leak = OUTPUT / "leak_flanges.usda"
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


def _relight(prim, f: dict) -> None:
    light = prim.GetChild("light") if prim else None
    if not light:
        return
    color = kelvin_rgb(f["kelvin"])
    light.GetAttribute("inputs:color").Set(Gf.Vec3f(*color))
    light.GetAttribute("inputs:intensity").Set(f["intensity"])
    base = light.GetAttribute("spark:baseIntensity")
    if base.IsValid():
        base.Set(f["intensity"])
    if f.get("cone") is not None:
        shaping = UsdLux.ShapingAPI.Apply(light)
        shaping.CreateShapingConeAngleAttr().Set(f["cone"])
        shaping.CreateShapingConeSoftnessAttr().Set(f["softness"])
        shaping.CreateShapingFocusAttr().Set(f["focus"])
    lens = prim.GetChild("lens")
    if lens:
        glow = emissive_material(prim.GetStage(), f"fixture_glow_{int(f['kelvin'])}", tuple(c * 18 for c in color))
        UsdShade.MaterialBindingAPI.Apply(lens).Bind(glow)


BEACON_GAIN = 0.35
FILL = {"center": (18.75, -5.75, 46.3), "size": (9.0, 6.0), "kelvin": 4200.0, "intensity": 220.0}


def tame_beacons(stage) -> None:
    red = UsdShade.Shader.Get(stage, "/World/Looks/lamp_red/surface")
    if red:
        red.GetInput("emissiveColor").Set(Gf.Vec3f(9.0, 0.3, 0.14))
    for index, _, _, intensity, _ in RED_BEACONS:
        light = stage.GetPrimAtPath(f"/World/Cinematic/Lamps/lamp_{index:03d}/light")
        if light:
            light.GetAttribute("inputs:intensity").Set(intensity * BEACON_GAIN)
            base = light.GetAttribute("spark:baseIntensity")
            if base.IsValid():
                base.Set(intensity * BEACON_GAIN)


def add_fill(stage) -> None:
    path = "/World/Cinematic/Lamps/zone_fill"
    if stage.GetPrimAtPath(path):
        return
    light = UsdLux.RectLight.Define(stage, path)
    light.CreateWidthAttr(FILL["size"][0])
    light.CreateHeightAttr(FILL["size"][1])
    light.CreateIntensityAttr(FILL["intensity"])
    light.CreateColorAttr(Gf.Vec3f(*kelvin_rgb(FILL["kelvin"])))
    light.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*FILL["center"]))


def place_sun_disc(stage, preset: dict) -> None:
    path = "/World/Cinematic/SunDisc"
    gain = preset.get("sun_disc", 0.0)
    prim = stage.GetPrimAtPath(path)
    if not prim and gain <= 0:
        return
    if not prim:
        disc = UsdGeom.Sphere.Define(stage, path)
        disc.CreateRadiusAttr(look.SUN_DISC["radius"])
        disc.AddTranslateOp()
        for flag in ("doNotCastShadows", "invisibleToSecondaryRays"):
            disc.GetPrim().CreateAttribute(f"primvars:{flag}", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(disc.GetPrim()).Bind(emissive_material(stage, "sun_disc", (0.0, 0.0, 0.0)))
        prim = disc.GetPrim()
    d = np.asarray(look.sun_direction(preset["sun_rotation"])) * look.SUN_DISC["distance"]
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*(d + np.array(WIDE_CENTER))))
    UsdShade.Shader.Get(stage, "/World/Looks/sun_disc/surface").GetInput("emissiveColor").Set(
        Gf.Vec3f(*(c * look.SUN_DISC["gain"] * gain for c in look.SUN_DISC["color"])))
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if gain > 0 else "invisible")


WIDE_CENTER = (4.0, -6.0, 30.0)


def relight(stage) -> None:
    tame_beacons(stage)
    add_fill(stage)
    lamps = "/World/Cinematic/Lamps"
    for index, (kind, _, kelvin, intensity, _, shape) in enumerate(AREA_LIGHTS):
        _relight(stage.GetPrimAtPath(f"{lamps}/area_{500 + index:03d}"),
                 look.fixture(kind, kelvin, intensity, shape if kind == "spot" else None))
    for index, (_, kelvin, intensity) in enumerate(BULBS):
        _relight(stage.GetPrimAtPath(f"{lamps}/lamp_{index:03d}"), look.fixture("bulb", kelvin, intensity))


def add_work_light(stage) -> None:
    meta = OUTPUT / "rig_merged_meta.json"
    if not stage.GetPrimAtPath("/World/Pipes/damage_patch") or not meta.exists():
        return
    import json

    k = look.WORK_LIGHT
    pos, aim = look.work_light(json.loads(meta.read_text())["world"]["damage"]["position_m"])
    root = UsdGeom.Xform.Define(stage, "/World/Cinematic/Lamps/work_light")
    w, x, y, z = aim_from_down(aim)
    m = Gf.Matrix4d().SetRotate(Gf.Quatd(float(w), float(x), float(y), float(z)))
    m.SetTranslateOnly(Gf.Vec3d(*pos))
    root.MakeMatrixXform().Set(m)
    light = UsdLux.SphereLight.Define(stage, "/World/Cinematic/Lamps/work_light/light")
    light.CreateRadiusAttr().Set(k["radius"])
    light.CreateIntensityAttr().Set(k["intensity"])
    light.CreateColorAttr().Set(Gf.Vec3f(*kelvin_rgb(k["kelvin"])))
    shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr().Set(k["cone"])
    shaping.CreateShapingConeSoftnessAttr().Set(k["softness"])
    base = light.GetPrim().GetAttribute("spark:baseIntensity")
    if base.IsValid():
        base.Set(k["intensity"])


DECK_FLOODS = (
    ((12.0, -2.0, 46.8), (14.5, -5.5, 42.5), SODIUM, 3.0e6, 28.0),
    ((26.5, -11.0, 46.8), (23.5, -10.5, 42.5), SODIUM, 3.0e6, 28.0),
    ((12.0, -11.5, 46.8), (15.0, -10.5, 42.5), SODIUM, 2.0e6, 28.0),
    ((27.0, -1.5, 46.8), (24.0, -6.5, 42.5), SODIUM, 2.4e6, 28.0),
)


DECK_Z = 42.5
ZONE_SPOT = ((20.2, -3.6, 44.8), (18.26, -5.25, 43.1), LED, 9.0e7, 12.0)


def add_zone_spot(stage) -> None:
    pos, target, kelvin, intensity, cone = ZONE_SPOT
    path = "/World/Cinematic/Lamps/area_610"
    if not stage.GetPrimAtPath(path):
        add_fixture(stage, 610, "spot", pos, kelvin, intensity, tuple(np.subtract(target, pos)), cone)
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.4))
        b = SceneBuilder(stage)
        b.tube("/World/Cinematic/Lamps", "zone_stand", (pos[0], pos[1], DECK_Z), (pos[0], pos[1], pos[2] - 0.05), 0.03,
               b.plain("lamp_pole", (0.3, 0.32, 0.33), 0.5, 0.6), segments=12)
    f = look.fixture("spot", kelvin, intensity, cone)
    _relight(stage.GetPrimAtPath(path), {**f, "softness": 0.25})


def add_deck_floods(stage, gain: float = 1.0) -> None:
    b = SceneBuilder(stage)
    steel = b.plain("lamp_pole", (0.3, 0.32, 0.33), 0.5, 0.6)
    for k, (pos, target, kelvin, intensity, cone) in enumerate(DECK_FLOODS):
        path = f"/World/Cinematic/Lamps/area_{600 + k:03d}"
        if not stage.GetPrimAtPath(path):
            add_fixture(stage, 600 + k, "spot", pos, kelvin, intensity, tuple(np.subtract(target, pos)), cone)
            b.tube("/World/Cinematic/Lamps", f"deck_pole_{k}", (pos[0], pos[1], 42.5), (pos[0], pos[1], pos[2] + 0.3), 0.08,
                   steel, segments=12)
        _relight(stage.GetPrimAtPath(path), look.fixture("spot", kelvin, intensity * gain, cone))


CURRENT = {"time": "night"}


def apply_preset(stage, name: str) -> None:
    assets_hero.attach(stage)
    photoreal.apply(stage, name)
    greebles.apply(stage)
    structure.apply(stage, name)
    hero_mat.apply(stage, name)
    flare_fx.attach(stage)
    relight(stage)
    add_deck_floods(stage)
    add_zone_spot(stage)
    add_work_light(stage)
    set_time(stage, name)


def set_time(stage, name: str, atm=None) -> None:
    preset = PRESETS[name]
    CURRENT["time"] = name
    facade.apply(stage, preset["windows"])
    ocean_fx.apply(stage, name, preset["wave_height"])
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

    place_sun_disc(stage, preset)
    sea = UsdShade.Shader.Get(stage, "/World/Looks/cinematic_sea/surface")
    sea.GetInput("diffuseColor").Set(Gf.Vec3f(*preset["sea_color"]))
    window = UsdShade.Shader.Get(stage, "/World/Looks/window_glass/surface")
    window.GetInput("emissiveColor").Set(WINDOW_GLOW * preset["windows"])

    scale = preset["lights"]
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Cinematic")):
        if prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.RectLight):
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
    if atm is not None:
        atm.preset = name
        atm.set_weather(atm.w)


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
    for index in range(5):
        prim = stage.GetPrimAtPath(f"/World/Cinematic/Flare/flame_{index}")
        wave = math.sin(time * (7.0 + index * 1.7) + index * 2.1) * 0.5 + math.sin(time * 13.0 + index) * 0.5
        stretch = 2.2 + 0.32 * wave
        prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(1.0 + 0.12 * wave, 1.0 + 0.12 * wave, stretch))
    flare_fx.update(stage, time, weather.flare_wind(flare_fx.WIND))
    pulse = 0.55 + 0.45 * max(math.sin(time * 2.2), 0.0)
    strobe = 1.0 if (time % 1.6) < 0.12 else 0.02
    for index, level in [(b[0], pulse) for b in RED_BEACONS] + [(i, strobe) for i, _ in STROBES]:
        light = stage.GetPrimAtPath(f"/World/Cinematic/Lamps/lamp_{index:03d}/light")
        base = light.GetAttribute("spark:baseIntensity")
        if base.IsValid():
            light.GetAttribute("inputs:intensity").Set(base.Get() * level)


def animate_ocean(stage, rest, distance, time: float, height: float) -> None:
    points, normals = ocean.surface(rest, distance, time, height)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Cinematic/ocean"))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    scroll = UsdShade.Shader.Get(stage, "/World/Looks/cinematic_sea/scroll")
    scroll.GetInput("translation").Set(Gf.Vec2f(time * 0.035, time * 0.02))
    ocean_fx.animate(stage, time)

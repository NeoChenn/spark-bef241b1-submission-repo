from __future__ import annotations

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKY = os.path.join(ROOT, "sim/assets/generated/sky_night.hdr")
AMBER = (1.0, 0.56, 0.17)
LED = (0.78, 0.88, 1.0)
SAFETY = {"red": (1.0, 0.05, 0.03), "green": (0.05, 1.0, 0.2), "amber": (1.0, 0.6, 0.05), "blue": (0.1, 0.45, 1.0)}
ROBOT_COLORS = {"graphite": (0.07, 0.072, 0.078), "white": (0.62, 0.62, 0.6), "orange": (0.8, 0.22, 0.03)}
CORAL_LIN = (0.693, 0.184, 0.105)


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def aim_matrix(pos, target):
    p, t = np.asarray(pos, float), np.asarray(target, float)
    z = p - t
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, z, p
    return m


def grade(rgb: np.ndarray, vignette: float = 0.35, lift=(2, 7, 9)) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    v = 1.0 - vignette * np.clip(r2 / 2.0, 0, 1) ** 1.2
    f = rgb.astype(np.float32) * v[..., None]
    f = f + np.asarray(lift, np.float32) * (1.0 - f / 255.0)
    return np.clip(f, 0, 255).astype(np.uint8)


def lamps(work):
    wx, wy, wz = work
    return [
        dict(name="flood_back", pos=(wx - 2.5, -3.6, 3.9), color=AMBER, radius=0.35, intensity=9.0e4),
        dict(name="flood_key", pos=(wx + 3.5, 3.4, 3.9), color=AMBER, radius=0.45, intensity=5.5e4),
        dict(name="led_rim", pos=(wx - 1.2, -1.6, 2.4), color=LED, radius=0.15, intensity=2.2e4),
        dict(name="led_fill", pos=(wx + 5.5, 1.0, 2.0), color=LED, radius=0.6, intensity=6.0e3),
    ]


def safety_lights():
    return [
        ("red", (-7.95, -4.95, 1.18)), ("red", (7.95, -4.95, 1.18)), ("red", (-7.95, 4.95, 1.18)),
        ("green", (-4.5, -4.95, 1.18)), ("green", (-1.5, -4.95, 1.18)), ("blue", (3.0, -4.95, 1.18)),
        ("amber", (-7.0, -4.4, 4.33)), ("amber", (7.0, -4.4, 4.33)), ("amber", (-7.0, 4.4, 4.33)),
    ]


def _material(stage, path, color, rough, metal=0.0, clearcoat=0.0, emissive=None):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.08)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def _lamp(stage, path, pos, color, radius, intensity, bulb=0.0, emissive=None, aim=None, cone=None):
    from pxr import Gf, UsdGeom, UsdLux, UsdShade

    root = UsdGeom.Xform.Define(stage, path)
    if aim is None:
        UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*pos))
    else:
        UsdGeom.Xformable(root).AddTransformOp().Set(Gf.Matrix4d(aim_matrix(pos, aim).tolist()))
    light = UsdLux.SphereLight.Define(stage, path + "/light")
    light.CreateRadiusAttr(radius)
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(*color))
    if cone is not None:
        sh = UsdLux.ShapingAPI.Apply(light.GetPrim())
        sh.CreateShapingConeAngleAttr(cone)
        sh.CreateShapingConeSoftnessAttr(0.4)
    if bulb > 0:
        b = UsdGeom.Sphere.Define(stage, path + "/bulb")
        b.CreateRadiusAttr(bulb)
        UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(emissive)


def _scale_input(shader, name, k, kind):
    from pxr import Gf, Sdf, UsdShade

    inp = shader.GetInput(name)
    if not inp:
        return
    if inp.HasConnectedSource():
        src = UsdShade.Shader(inp.GetConnectedSource()[0].GetPrim())
        s = src.GetInput("scale")
        cur = s.Get() if s else None
        cur = cur if cur is not None else Gf.Vec4f(1, 1, 1, 1)
        src.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(cur[0] * k, cur[1] * k, cur[2] * k, cur[3]))
    else:
        v = inp.Get()
        if v is None:
            return
        inp.Set(Gf.Vec3f(*(c * k for c in v)) if kind == "color" else float(v) * k)


def wet_down(stage, root, albedo=0.5, rough=0.45, deck_rough=0.3, sea_path=None):
    from pxr import Usd, UsdShade

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdShade.Shader) or (sea_path and str(prim.GetPath()).startswith(sea_path)):
            continue
        sh = UsdShade.Shader(prim)
        if sh.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        deck = "grate" in str(prim.GetPath()) or "metal_plate" in str(prim.GetPath())
        _scale_input(sh, "diffuseColor", albedo, "color")
        _scale_input(sh, "roughness", deck_rough if deck else rough, "float")


def paint_robot(stage, robot_root, color="graphite"):
    from pxr import Usd, UsdGeom, UsdShade

    body = _material(stage, "/World/Looks/robot_body", ROBOT_COLORS[color], 0.42, 0.3, 0.6)
    pad = _material(stage, "/World/Looks/robot_brush", CORAL_LIN, 0.55, 0.0, 0.2, emissive=tuple(c * 0.06 for c in CORAL_LIN))
    root = stage.GetPrimAtPath(robot_root)
    n = {"body": 0, "brush": 0, "instanced": 0}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstanceProxy():
            n["instanced"] += 1
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        name = str(bound.GetPath()).lower() if bound else ""
        is_pad = "brush" in name or "orange" in name
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        api.Bind(pad if is_pad else body, UsdShade.Tokens.strongerThanDescendants)
        n["brush" if is_pad else "body"] += 1
    return n


def apply(stage, deck_root, work, dome=220.0, moon=25.0, lamp_gain=1.0, robot_root=None, robot_color="graphite"):
    import carb.settings
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    env = deck_root + "/Environment"
    sky = UsdLux.DomeLight(stage.GetPrimAtPath(env + "/sky"))
    sky.GetTextureFileAttr().Set(Sdf.AssetPath(SKY))
    sky.GetIntensityAttr().Set(dome)
    UsdGeom.XformCommonAPI(sky.GetPrim()).SetRotate(Gf.Vec3f(0, 0, 40.0))
    sun = UsdLux.DistantLight(stage.GetPrimAtPath(env + "/sun"))
    sun.GetIntensityAttr().Set(moon)
    sun.GetColorAttr().Set(Gf.Vec3f(*LED))
    sea = UsdShade.Shader.Get(stage, deck_root + "/Looks/sea/surface")
    sea.GetInput("diffuseColor").Set(Gf.Vec3f(0.001, 0.006, 0.009))
    sea.GetInput("roughness").Set(0.07)
    wet_down(stage, deck_root, sea_path=deck_root + "/Looks/sea")

    base = deck_root + "/NightLook"
    UsdGeom.Scope.Define(stage, base)
    for lamp in lamps(work):
        _lamp(stage, f"{base}/{lamp['name']}", lamp["pos"], lamp["color"], lamp["radius"], lamp["intensity"] * lamp_gain)
    wx, wy, wz = work
    _lamp(stage, f"{base}/work_spot", (wx + 1.0, 1.6, 2.6), LED, 0.08, 1.2e5 * lamp_gain, aim=work, cone=22.0)
    bulb_amber = _material(stage, "/World/Looks/night_bulb_amber", (0.02, 0.02, 0.02), 0.5,
                           emissive=tuple(c * 40 for c in AMBER))
    for k, (x, y) in enumerate(((-3.5, -2.5), (-3.5, 2.5), (3.5, -2.5), (3.5, 2.5))):
        _lamp(stage, f"{base}/cage_{k}", (x, y, 3.42), AMBER, 0.06, 2.5e4 * lamp_gain, bulb=0.045, emissive=bulb_amber)
    mats = {c: _material(stage, f"/World/Looks/night_safety_{c}", (0.02, 0.02, 0.02), 0.5,
                         emissive=tuple(v * 60 for v in rgb)) for c, rgb in SAFETY.items()}
    for k, (c, pos) in enumerate(safety_lights()):
        _lamp(stage, f"{base}/safety_{k}", pos, SAFETY[c], 0.04, 1.5e3 * lamp_gain, bulb=0.04, emissive=mats[c])
    ship = _material(stage, "/World/Looks/night_ship", (0.02, 0.02, 0.02), 0.5, emissive=tuple(c * 30 for c in AMBER))
    for k, (x, y) in enumerate(((-260, -420), (-235, -425), (-210, -418), (180, -520), (-520, 120))):
        b = UsdGeom.Sphere.Define(stage, f"{base}/far_{k}")
        b.CreateRadiusAttr(1.1)
        UsdGeom.Xformable(b).AddTranslateOp().Set(Gf.Vec3d(x, y, -19.0))
        UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(ship)

    s = carb.settings.get_settings()
    s.set_bool("/rtx/fog/enabled", True)
    s.set_bool("/rtx/fog/fogZup/enabled", True)
    s.set_float_array("/rtx/fog/fogColor", [0.01, 0.03, 0.04])
    s.set_float("/rtx/fog/fogColorIntensity", 1.0)
    s.set_float("/rtx/fog/fogStartDist", 12.0)
    s.set_float("/rtx/fog/fogEndDist", 900.0)
    s.set_float("/rtx/fog/fogDistanceDensity", 0.9)
    s.set_float("/rtx/fog/fogHeightDensity", 0.0)
    s.set_bool("/rtx/post/lensFlares/enabled", True)
    s.set_bool("/rtx/post/lensFlares/physicalSettings", False)
    s.set_float("/rtx/post/lensFlares/flareScale", 0.05)
    if robot_root:
        return paint_robot(stage, robot_root, robot_color)

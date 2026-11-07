import math
from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from rig.sim import CAM_H, CAM_W

CAMERA_PATH = "/World/Camera"
CAMERA_EYE = Gf.Vec3d(6.0, -6.0, 4.0)
CAMERA_TARGET = Gf.Vec3d(0.0, 0.0, 0.5)
CAMERA_HFOV = 60.0
APERTURE = 20.955
SPIN_DEG_PER_S = 45.0


@dataclass
class Scene:
    spinner_rotate: UsdGeom.XformOp


def _material(stage: Usd.Stage, name: str, rgb: tuple[float, float, float], roughness: float = 0.5) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _paint(stage: Usd.Stage, prim: UsdGeom.Gprim, name: str, rgb: tuple[float, float, float]) -> None:
    prim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(_material(stage, name, rgb))


def _place(prim: UsdGeom.Xformable, pos: tuple[float, float, float], scale: tuple[float, float, float] | None = None) -> None:
    prim.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if scale:
        prim.AddScaleOp().Set(Gf.Vec3f(*scale))


def _camera(stage: Usd.Stage) -> UsdGeom.Camera:
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.CreateFocalLengthAttr(APERTURE / 2 / math.tan(math.radians(CAMERA_HFOV / 2)))
    cam.CreateHorizontalApertureAttr(APERTURE)
    cam.CreateVerticalApertureAttr(APERTURE * CAM_H / CAM_W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    view = Gf.Matrix4d().SetLookAt(CAMERA_EYE, CAMERA_TARGET, Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    return cam


def build_scene(stage: Usd.Stage) -> Scene:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    _place(ground, (0, 0, -0.05), (20, 20, 0.1))
    _paint(stage, ground, "Ground", (0.45, 0.47, 0.5))

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(500.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1600.0)
    sun.CreateAngleAttr(0.53)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(45.0, 0.0, 30.0))

    spinner = UsdGeom.Cube.Define(stage, "/World/Spinner")
    spinner.CreateSizeAttr(1.0)
    spinner.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
    rotate = spinner.AddRotateZOp()
    rotate.Set(0.0)
    _paint(stage, spinner, "Red", (0.8, 0.1, 0.1))

    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    sphere.CreateRadiusAttr(0.5)
    _place(sphere, (2, 1, 0.5))
    _paint(stage, sphere, "Blue", (0.1, 0.3, 0.85))

    cylinder = UsdGeom.Cylinder.Define(stage, "/World/Cylinder")
    cylinder.CreateRadiusAttr(0.4)
    cylinder.CreateHeightAttr(1.5)
    _place(cylinder, (-2, 1, 0.75))
    _paint(stage, cylinder, "Green", (0.1, 0.7, 0.2))

    cone = UsdGeom.Cone.Define(stage, "/World/Cone")
    cone.CreateRadiusAttr(0.5)
    cone.CreateHeightAttr(1.0)
    _place(cone, (1, -2, 0.5))
    _paint(stage, cone, "Yellow", (0.9, 0.8, 0.1))

    _camera(stage)
    return Scene(spinner_rotate=rotate)


def spin(scene: Scene, t: float) -> None:
    scene.spinner_rotate.Set((SPIN_DEG_PER_S * t) % 360.0)

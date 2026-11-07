"""USD side of the sim bridge: adds the robot, rust overlay, cameras and demo props to the rig deck.

Everything here needs pxr (run inside Isaac Sim). The orchestrator-facing logic is in isaac_bridge.py.

Rust, wet remover and primer are shown as three thin sleeves around the pipe segment (radius 0.1522,
0.153 and 0.1536 m over the 0.15 m pipe). Each sleeve is a fixed grid of quads; `SurfaceLayer.update`
keeps only the faces whose cell is rusty / sprayed / painted, so the overlay always matches the
RustState grid exactly (rows = z along the axis, cols = theta, as in RustState.surface_to_uv).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from integration.kinematics import RobotModel
from policy.interfaces import PipeGeometry

REPO = Path(__file__).resolve().parent.parent
DECK_SCENE = REPO / "sim/output/rig_deck.usda"
ROBOT_USD = REPO / "sim/output/robot/go2_ur3_lift/go2_ur3_lift.usda"
RUST_TEXTURE = REPO / "sim/assets/third_party/textures/rust_coarse_01/diff.jpg"

ROBOT_PATH = "/World/Robots/scrubber_dog"
PIPE_SCOPE = "/World/Pipes"
LOOKS = "/World/Looks/Repair"
INSPECTION_CAMERA = "/World/Cameras/repair_closeup"
HERO_CAMERA = "/World/Cameras/repair_hero"
WORKER_PATH = "/World/People/worker"
SEGMENT_LENGTH = 0.8


# ----------------------------------------------------------------------------------------- pipe
def find_main_pipe(stage) -> Usd.Prim:
    for prim in stage.GetPrimAtPath(PIPE_SCOPE).GetChildren():
        if prim.GetName().startswith("main_pipe"):
            return prim
    raise RuntimeError("no main_pipe prim under /World/Pipes")


def write_pipe_attrs(prim, origin, axis=(1.0, 0.0, 0.0), ref=(0.0, 1.0, 0.0), radius=0.15,
                     length=SEGMENT_LENGTH) -> None:
    """INTERFACE.md section 2: rig:* attrs so the bridge reads the geometry instead of hard-coding it."""
    for name, kind, value in (
        ("rig:axis", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*axis)),
        ("rig:ref", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*ref)),
        ("rig:radius", Sdf.ValueTypeNames.Float, float(radius)),
        ("rig:segmentOrigin", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*origin)),
        ("rig:segmentLength", Sdf.ValueTypeNames.Float, float(length)),
    ):
        prim.CreateAttribute(name, kind, custom=True).Set(value)


def read_pipe(prim, n_z=40, n_theta=64) -> PipeGeometry:
    get = lambda name: prim.GetAttribute(name).Get()  # noqa: E731
    return PipeGeometry(origin=list(get("rig:segmentOrigin")), axis=list(get("rig:axis")),
                        ref=list(get("rig:ref")), radius=float(get("rig:radius")),
                        length=float(get("rig:segmentLength")), n_z=n_z, n_theta=n_theta)


# ------------------------------------------------------------------------------------ materials
def preview_material(stage, name, color, roughness, metallic=0.0, texture: Path | None = None, tile=(1.0, 1.0)):
    material = UsdShade.Material.Define(stage, f"{LOOKS}/{name}")
    shader = UsdShade.Shader.Define(stage, f"{LOOKS}/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if texture is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    else:
        reader = UsdShade.Shader.Define(stage, f"{LOOKS}/{name}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        transform = UsdShade.Shader.Define(stage, f"{LOOKS}/{name}/tile")
        transform.CreateIdAttr("UsdTransform2d")
        transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*tile))
        image = UsdShade.Shader.Define(stage, f"{LOOKS}/{name}/diffuse")
        image.CreateIdAttr("UsdUVTexture")
        image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture).replace("\\", "/"))
        image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        image.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color, 1.0))
        image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
        image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


# ------------------------------------------------------------------------------- surface layers
class SurfaceLayer:
    """A sleeve over the pipe segment whose quads are switched on per RustState cell."""

    def __init__(self, stage, path: str, pipe: PipeGeometry, radius: float, material, sub: int = 2):
        self.pipe, self.sub = pipe, sub
        self.n_z, self.n_t = pipe.n_z * sub, pipe.n_theta * sub
        theta = np.linspace(0.0, 2 * math.pi, self.n_t + 1)
        z = np.linspace(-pipe.length / 2, pipe.length / 2, self.n_z + 1)
        tt, zz = np.meshgrid(theta, z)                      # (n_z+1, n_t+1)
        normal = np.cos(tt)[..., None] * pipe.ref + np.sin(tt)[..., None] * pipe.binormal
        points = pipe.origin + zz[..., None] * pipe.axis + radius * normal
        st = np.stack([tt * radius / 0.3, (zz + pipe.length / 2) / 0.3], -1)
        idx = np.arange((self.n_z + 1) * (self.n_t + 1)).reshape(self.n_z + 1, self.n_t + 1)
        quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1)   # (n_z, n_t, 4)
        self.quads = quads.reshape(-1, 4)
        self.st = st.reshape(-1, 2)
        self.mesh = UsdGeom.Mesh.Define(stage, path)
        self.mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.reshape(-1, 3).astype(np.float32)))
        self.mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normal.reshape(-1, 3).astype(np.float32)))
        self.mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self.mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.mesh.CreateDoubleSidedAttr(False)
        self.st_primvar = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        bind(self.mesh.GetPrim(), material)
        self._last = None
        self.update(np.zeros(pipe.shape, bool))

    def _fine(self, field: np.ndarray) -> np.ndarray:
        """Cell grid (n_z, n_theta) -> sub-sampled face grid with bilinear smoothing (theta periodic)."""
        f = np.asarray(field, np.float32)
        if self.sub == 1:
            return f
        zc = (np.arange(self.n_z) + 0.5) / self.sub - 0.5
        tc = (np.arange(self.n_t) + 0.5) / self.sub - 0.5
        z0 = np.clip(np.floor(zc).astype(int), 0, f.shape[0] - 1)
        z1 = np.clip(z0 + 1, 0, f.shape[0] - 1)
        wz = np.clip(zc - np.floor(zc), 0, 1)[:, None]
        t0 = np.floor(tc).astype(int) % f.shape[1]
        t1 = (t0 + 1) % f.shape[1]
        wt = (tc - np.floor(tc))[None, :]
        top = f[z0][:, t0] * (1 - wt) + f[z0][:, t1] * wt
        bottom = f[z1][:, t0] * (1 - wt) + f[z1][:, t1] * wt
        return top * (1 - wz) + bottom * wz

    def update(self, field: np.ndarray, threshold: float = 0.5) -> None:
        on = (self._fine(field) > threshold).reshape(-1)
        key = on.tobytes()
        if key == self._last:
            return
        self._last = key
        quads = self.quads[on]
        if not len(quads):
            UsdGeom.Imageable(self.mesh).MakeInvisible()
            return
        UsdGeom.Imageable(self.mesh).MakeVisible()
        self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4] * len(quads)))
        self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(quads.reshape(-1).astype(np.int32)))
        self.st_primvar.Set(Vt.Vec2fArray.FromNumpy(self.st[quads.reshape(-1)].astype(np.float32)))


class RustOverlay:
    """Wet remover, rust and primer sleeves driven from a RustState-like world (rust/sprayed/painted)."""

    def __init__(self, stage, pipe: PipeGeometry):
        wet = preview_material(stage, "wet_remover", (0.05, 0.07, 0.09), 0.08, 0.0)
        rust = preview_material(stage, "rust_overlay", (1.0, 1.0, 1.0), 0.85, 0.0, RUST_TEXTURE, (1.0, 1.0))
        primer = preview_material(stage, "primer", (0.42, 0.47, 0.42), 0.55, 0.0)
        self.wet = SurfaceLayer(stage, f"{PIPE_SCOPE}/rust_overlay_wet", pipe, 0.1522, wet)
        self.rust = SurfaceLayer(stage, f"{PIPE_SCOPE}/rust_overlay", pipe, 0.1530, rust)
        self.primer = SurfaceLayer(stage, f"{PIPE_SCOPE}/rust_overlay_primer", pipe, 0.1536, primer)

    def update(self, world) -> None:
        rust = world.rust[0].numpy()
        painted = world.painted[0].numpy() if hasattr(world, "painted") else np.zeros_like(rust, bool)
        sprayed = world.sprayed[0].numpy() & ~painted
        self.wet.update(sprayed.astype(np.float32))
        self.rust.update(np.clip(rust / 0.1, 0, 1) * ~painted, threshold=0.5)
        self.primer.update(painted.astype(np.float32))


# ---------------------------------------------------------------------------------------- robot
class RobotPoser:
    """References the imported robot (physics variant off) and writes link transforms from FK."""

    def __init__(self, stage, model: RobotModel, path: str = ROBOT_PATH):
        self.model = model
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(str(ROBOT_USD).replace("\\", "/"))
        prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
        self.root = UsdGeom.Xformable(prim)
        self.root.ClearXformOpOrder()
        self.root_op = self.root.AddTransformOp()
        self.ops = {}
        by_child = {j.child: j for j in model.joints.values()}
        # Link prims only: the root is the prim named after the root link directly under Geometry, and a
        # child link must sit under its URDF parent link. Visual meshes reuse names like "base" (the trunk
        # mesh has its own authored rotation), so matching by name alone would clobber them.
        link_prims = {}
        for p in Usd.PrimRange(prim):
            name, parent = p.GetName(), p.GetParent().GetName()
            if name == model.root and parent == "Geometry" and name not in link_prims:
                link_prims[name] = p
            elif name in by_child and by_child[name].parent == parent and name not in link_prims:
                link_prims[name] = p
        for name, p in link_prims.items():
            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            self.ops[name] = xf.AddTransformOp()
        self.ops[model.root].Set(Gf.Matrix4d(1.0))
        self.by_child = by_child
        missing = set(by_child) - set(self.ops)
        if missing:
            raise RuntimeError(f"robot USD is missing link prims: {sorted(missing)[:5]}")
        # Steel wire brush instead of the URDF's orange head, which a colour-based detector reads as rust.
        brush = preview_material(stage, "wire_brush", (0.36, 0.38, 0.42), 0.35, 0.9)
        for p in Usd.PrimRange(prim):
            if p.GetName() == "scrubber_link":
                for child in Usd.PrimRange(p):
                    if child.IsA(UsdGeom.Gprim):
                        UsdShade.MaterialBindingAPI.Apply(child).Bind(
                            brush, UsdShade.Tokens.strongerThanDescendants)

    def set(self, base: np.ndarray, q: dict) -> None:
        self.root_op.Set(Gf.Matrix4d(*np.asarray(base, float).T.reshape(-1)))
        for name, joint in self.by_child.items():
            local = joint.local(q.get(joint.name, 0.0))
            self.ops[name].Set(Gf.Matrix4d(*local.T.reshape(-1)))


# -------------------------------------------------------------------------------------- cameras
def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)) -> Gf.Matrix4d:
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(*map(float, up)))
    return view.GetInverse()


class Camera:
    def __init__(self, stage, path, focal=35.0, aperture=20.955, clip=(0.02, 2000.0)):
        self.path = path
        self.camera = UsdGeom.Camera.Define(stage, path)
        self.camera.CreateFocalLengthAttr(float(focal))
        self.camera.CreateHorizontalApertureAttr(float(aperture))
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(*clip))
        xf = UsdGeom.Xformable(self.camera)
        xf.ClearXformOpOrder()
        self.op = xf.AddTransformOp()

    def look(self, eye, target) -> None:
        self.op.Set(look_at_matrix(eye, target))

    def world_matrix(self) -> np.ndarray:
        return np.array(UsdGeom.Xformable(self.camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    @property
    def focal(self) -> float:
        return float(self.camera.GetFocalLengthAttr().Get())

    @property
    def aperture(self) -> float:
        return float(self.camera.GetHorizontalApertureAttr().Get())


def inspection_views(pipe: PipeGeometry, theta: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """INTERFACE.md: nominal eye on the damage-normal side, plus +-0.35 rad / +-0.12 m alternatives."""
    surf, n = pipe.surface_point(theta, 0.0)
    n2 = pipe.surface_point(theta + 0.35, 0.0)[1]
    n3 = pipe.surface_point(theta - 0.35, 0.0)[1]
    return [
        (surf + 0.85 * n + 0.15 * pipe.axis, surf),
        (surf + 0.7 * n2 + 0.12 * pipe.axis, surf + 0.03 * pipe.axis),
        (surf + 0.7 * n3 - 0.12 * pipe.axis, surf),
    ]


# ---------------------------------------------------------------------------------------- props
def add_worker(stage, path: str = WORKER_PATH):
    """A simple high-vis figure for the person-proximity hook (hidden until moved in)."""
    root = UsdGeom.Xform.Define(stage, path)
    vest = preview_material(stage, "hi_vis", (1.0, 0.42, 0.02), 0.6)
    dark = preview_material(stage, "overalls", (0.1, 0.12, 0.16), 0.8)
    skin = preview_material(stage, "skin", (0.62, 0.45, 0.35), 0.7)
    helmet = preview_material(stage, "helmet", (0.95, 0.95, 0.9), 0.4)
    for name, kind, size, pos, material in (
        ("legs", "box", (0.32, 0.22, 0.85), (0, 0, 0.43), dark),
        ("torso", "box", (0.42, 0.26, 0.62), (0, 0, 1.17), vest),
        ("head", "sphere", 0.11, (0, 0, 1.6), skin),
        ("hard_hat", "sphere", 0.125, (0, 0, 1.66), helmet),
    ):
        if kind == "box":
            prim = UsdGeom.Cube.Define(stage, f"{path}/{name}")
            prim.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
        else:
            prim = UsdGeom.Sphere.Define(stage, f"{path}/{name}")
            prim.CreateRadiusAttr(size)
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
        bind(prim.GetPrim(), material)
    xf = UsdGeom.Xformable(root)
    op = xf.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, -50))
    yaw = xf.AddRotateZOp()
    return op, yaw


def add_weld_bead(stage, path, pipe: PipeGeometry, theta: float, z: float, height: float, radius: float):
    surf, n = pipe.surface_point(theta, z)
    bead = UsdGeom.Sphere.Define(stage, path)
    bead.CreateRadiusAttr(1.0)
    xf = UsdGeom.Xformable(bead)
    # A flattened sphere centred on the surface, its short axis along the normal.
    xaxis = np.cross(n, pipe.axis)          # right-handed frame [axis, n x axis, n]
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = pipe.axis * radius, xaxis * radius, n * height * 1.3, surf
    xf.AddTransformOp().Set(Gf.Matrix4d(*M.T.reshape(-1)))
    bind(bead.GetPrim(), preview_material(stage, "weld_bead", (0.32, 0.3, 0.28), 0.45, 0.8))
    return bead


def add_spray_cone(stage, path=f"{PIPE_SCOPE}/spray_mist"):
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateHeightAttr(1.0)
    cone.CreateRadiusAttr(0.5)
    cone.CreateAxisAttr("Z")
    bind(cone.GetPrim(), preview_material(stage, "mist", (0.75, 0.85, 0.95), 0.2))
    UsdGeom.Imageable(cone).MakeInvisible()
    xf = UsdGeom.Xformable(cone)
    op = xf.AddTransformOp()
    return cone, op


def set_spray(cone, op, tip: np.ndarray, direction: np.ndarray, length: float, on: bool) -> None:
    if not on:
        UsdGeom.Imageable(cone).MakeInvisible()
        return
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    zaxis = -d                                   # cone apex is at +Z/2: point the apex back at the nozzle
    xaxis = np.cross([0.0, 0.0, 1.0] if abs(zaxis[2]) < 0.9 else [1.0, 0.0, 0.0], zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    centre = np.asarray(tip, float) + d * length / 2
    width = length * 0.55
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = xaxis * width, yaxis * width, zaxis * length, centre
    op.Set(Gf.Matrix4d(*M.T.reshape(-1)))
    UsdGeom.Imageable(cone).MakeVisible()


def add_lens_mist(stage, camera_path: str):
    """Spray mist on the inspection lens (occlusion hook): a grey card 3 cm in front of the camera."""
    card = UsdGeom.Cube.Define(stage, f"{camera_path}/lens_mist")
    card.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(card)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.03))
    xf.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 0.001))
    bind(card.GetPrim(), preview_material(stage, "lens_mist", (0.03, 0.03, 0.03), 1.0))
    UsdGeom.Imageable(card).MakeInvisible()
    return card


def add_glare_light(stage, eye, target, path="/World/Cameras/glare_light"):
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateRadiusAttr(0.05)
    light.CreateIntensityAttr(0.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.9))
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*map(float, eye)))
    return light

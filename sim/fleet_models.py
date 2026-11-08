"""USD models for the fleet: Q quadruped (Go2 + lift + UR3), A quadrotor, C magnetic crawler, R ROV, and the
fixed F&G detectors (post, yellow head, alarm beacon).

Import inside Isaac Sim (needs pxr). Every model is posed kinematically from a fleet_replay.Track each frame:
nothing here is simulated by PhysX. Q reuses the imported robot USD (Physics variant 'none') through
integration.repair_scene.RobotPoser, with the trot animation from integration/isaac_bridge.py `_walk`;
the drone tilts into its velocity and spins its props, the crawler turns its wheels by distance travelled.
Each robot has a coloured ID plate and a status light whose colour follows its state.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from fleet_replay import ROBOT_COLORS, Track, deck_height, state_color
from integration import repair_scene as rs
from integration.kinematics import ARM_START, ARM_STOWED, TRUNK_HEIGHT, RobotModel, axis_angle, pose

LOOKS = "/World/Looks/Fleet"
ID_COLORS = [(0.95, 0.75, 0.1), (0.1, 0.75, 0.95), (0.95, 0.35, 0.75), (0.4, 0.95, 0.35), (1.0, 0.5, 0.15),
             (0.65, 0.5, 1.0)]
GLOW = 12.0
_materials: dict = {}


# ------------------------------------------------------------------------------------- materials
def material(stage, name: str, color, roughness: float = 0.5, metallic: float = 0.0, emissive=None,
             opacity: Optional[float] = None):
    """UsdPreviewSurface; returns (material, shader) so inputs can be animated."""
    key = (str(stage.GetRootLayer().identifier), name)
    if key in _materials and _materials[key][0].GetPrim().IsValid():
        return _materials[key]
    path = f"{LOOKS}/{name}"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    if opacity is not None:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _materials[key] = (mat, shader)
    return mat, shader


def translucent(stage, name: str, color, emissive=(0.0, 0.0, 0.0), emissive_nits: float = 0.0,
                opacity: float = 0.3):
    """OmniPBR with fractional opacity (UsdPreviewSurface opacity renders as a hard cutout in RTX).
    Needs /rtx/raytracing/fractionalCutoutOpacity on, which FleetScene.open sets."""
    key = (str(stage.GetRootLayer().identifier), name)
    if key in _materials and _materials[key][0].GetPrim().IsValid():
        return _materials[key]
    path = f"{LOOKS}/{name}"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(emissive_nits > 0)
    shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(emissive_nits))
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    _materials[key] = (mat, shader)
    return mat, shader


def set_opacity(shader, value: float) -> None:
    name = "opacity_constant" if shader.GetInput("opacity_constant") else "opacity"
    shader.GetInput(name).Set(float(value))


def set_emissive(shader, color) -> None:
    name = "emissive_color" if shader.GetInput("emissive_color") else "emissiveColor"
    shader.GetInput(name).Set(Gf.Vec3f(*map(float, color)))


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat[0] if isinstance(mat, tuple) else mat)


def box(stage, path, center, size, mat, rotate_z: float = 0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    if rotate_z:
        xf.AddRotateZOp().Set(float(rotate_z))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    bind(cube.GetPrim(), mat)
    return cube


def sphere(stage, path, center, radius, mat, scale=None):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(float(radius))
    xf = UsdGeom.Xformable(s)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    bind(s.GetPrim(), mat)
    return s


def cylinder(stage, path, center, radius, height, mat, axis: str = "Z"):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(radius))
    c.CreateHeightAttr(float(height))
    c.CreateAxisAttr(axis)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    bind(c.GetPrim(), mat)
    return c


def matrix(m: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(*np.asarray(m, float).T.reshape(-1))


def _state_since(track: Track, t: float) -> float:
    now = track.latest(t)
    since = now.t
    for s in reversed(track.samples):
        if s.t > t:
            continue
        if s.state != now.state:
            break
        since = s.t
    return since


# ----------------------------------------------------------------------------------------- base
class FleetRobot:
    tag_height = 0.8

    def __init__(self, stage, robot_id: str, index: int, root: str):
        self.stage, self.id, self.path = stage, robot_id, f"{root}/{robot_id}"
        own = ROBOT_COLORS.get(robot_id)
        self.color = tuple(c / 255.0 for c in own) if own else ID_COLORS[index % len(ID_COLORS)]
        self.xform = UsdGeom.Xform.Define(stage, self.path)
        self.op = self.xform.AddTransformOp()
        self.op.Set(Gf.Matrix4d(1.0))
        self.id_mat = material(stage, f"id_{robot_id}", self.color, 0.4, emissive=tuple(c * 1.5 for c in self.color))
        self.light_mat = material(stage, f"status_{robot_id}", (0.05, 0.05, 0.05), 0.3, emissive=(0, 0, 0))
        self._state = None
        self.last_position = np.zeros(3)

    def set_status(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        c = np.array(state_color(state), float) / 255.0
        self.light_mat[1].GetInput("emissiveColor").Set(Gf.Vec3f(*(c * GLOW)))
        self.light_mat[1].GetInput("diffuseColor").Set(Gf.Vec3f(*c))

    def update(self, track: Track, t: float) -> None:
        raise NotImplementedError

    def tag_point(self) -> np.ndarray:
        return self.last_position + np.array([0.0, 0.0, self.tag_height])


# ---------------------------------------------------------------------------------------- Q: Go2
class Quadruped(FleetRobot):
    """Go2 + lift + UR3, the same USD and FK as the repair demo. Arm stowed while walking, unfolded to the
    work pose while INSPECTING/WORKING. Legs trot with the _walk gait (thigh/calf swing on a diagonal pair)."""
    tag_height = 0.95
    _model: Optional[RobotModel] = None

    def __init__(self, stage, robot_id, index, root):
        super().__init__(stage, robot_id, index, root)
        if Quadruped._model is None:
            Quadruped._model = RobotModel.from_urdf()
        self.model = Quadruped._model
        self.poser = rs.RobotPoser(stage, self.model, path=f"{self.path}/robot")
        self.badge = UsdGeom.Xform.Define(stage, f"{self.path}_badge")
        self.badge_op = self.badge.AddTransformOp()
        b = self.badge.GetPath().pathString
        sphere(stage, f"{b}/status", (-0.22, 0.0, 0.13), 0.035, self.light_mat)
        box(stage, f"{b}/id_left", (0.0, 0.105, 0.03), (0.16, 0.006, 0.06), self.id_mat)
        box(stage, f"{b}/id_right", (0.0, -0.105, 0.03), (0.16, 0.006, 0.06), self.id_mat)
        cylinder(stage, f"{b}/antenna", (-0.25, 0.06, 0.2), 0.004, 0.22, material(stage, "antenna", (0.05, 0.05, 0.05)))
        self.stand = {k: v for k, v in self.model.default_q().items() if "hip" in k or "thigh" in k or "calf" in k}

    def update(self, track: Track, t: float) -> None:
        p, yaw, speed = track.position(t), track.yaw_at(t), track.speed(t)
        state = track.state(t).upper()
        q = self.model.default_q()
        q.update(ARM_STOWED)
        if state in ("INSPECTING", "WORKING", "RUNNING"):
            s = float(np.clip((t - _state_since(track, t)) / 1.5, 0.0, 1.0))
            s = s * s * (3 - 2 * s)
            for name in ARM_STOWED:
                q[name] = ARM_STOWED[name] + (ARM_START[name] - ARM_STOWED[name]) * s
            if state == "WORKING":     # turning a valve handwheel: wrist roll
                q["ur3_wrist_3_joint"] = ARM_START["ur3_wrist_3_joint"] + 1.2 * math.sin(t * 2.0) * s
        phase = track.distance(t) / 0.18 * math.pi
        swing = 0.25 if speed > 0.08 else 0.0
        for leg, sign in (("FL", 1), ("RR", 1), ("FR", -1), ("RL", -1)):
            lift = max(0.0, sign * math.sin(phase))
            q[f"{leg}_thigh_joint"] = self.stand[f"{leg}_thigh_joint"] - swing * lift * 0.6
            q[f"{leg}_calf_joint"] = self.stand[f"{leg}_calf_joint"] - swing * lift * 0.9
        z = deck_height(p[2]) + TRUNK_HEIGHT + 0.01 * abs(math.sin(phase)) * swing
        base = pose(p[0], p[1], z, yaw)
        self.poser.set(base, q)
        self.badge_op.Set(matrix(base))
        self.set_status(state)
        self.last_position = np.array([p[0], p[1], z])


# -------------------------------------------------------------------------------------- A: drone
class Drone(FleetRobot):
    """Inspection quadrotor, ~1.1 m motor to motor: body, 4 arms, spinning props, nav lights, thermal gimbal."""
    tag_height = 0.7
    ARM = 0.42

    def __init__(self, stage, robot_id, index, root):
        super().__init__(stage, robot_id, index, root)
        P = self.path
        shell = material(stage, "drone_shell", (0.16, 0.17, 0.19), 0.35, 0.2)
        carbon = material(stage, "drone_carbon", (0.03, 0.03, 0.035), 0.5)
        motor = material(stage, "drone_motor", (0.55, 0.56, 0.6), 0.3, 0.9)
        blade = material(stage, "drone_blade", (0.08, 0.08, 0.08), 0.4)
        blur = translucent(stage, "drone_blur", (0.3, 0.3, 0.32), opacity=0.18)
        red = material(stage, "nav_red", (1, 0, 0), emissive=(GLOW, 0.0, 0.0))
        green = material(stage, "nav_green", (0, 1, 0), emissive=(0.0, GLOW, 0.1))
        white = material(stage, "nav_white", (1, 1, 1), emissive=(GLOW, GLOW, GLOW))
        lens = material(stage, "thermal_lens", (0.02, 0.02, 0.03), 0.05, 0.8)
        box(stage, f"{P}/body", (0, 0, 0), (0.34, 0.24, 0.11), shell)
        sphere(stage, f"{P}/canopy", (0.02, 0, 0.055), 0.13, shell, scale=(1.2, 0.85, 0.45))
        box(stage, f"{P}/battery", (-0.02, 0, -0.075), (0.24, 0.13, 0.05), carbon)
        self.props = []
        for k, a in enumerate((45.0, 135.0, 225.0, 315.0)):
            r = math.radians(a)
            tip = np.array([self.ARM * math.cos(r), self.ARM * math.sin(r), 0.0])
            box(stage, f"{P}/arm_{k}", tip / 2, (self.ARM, 0.035, 0.025), carbon, rotate_z=a)
            cylinder(stage, f"{P}/motor_{k}", tip + [0, 0, 0.03], 0.032, 0.05, motor)
            prop = UsdGeom.Xform.Define(stage, f"{P}/prop_{k}")
            xf = UsdGeom.Xformable(prop)
            xf.AddTranslateOp().Set(Gf.Vec3d(*(tip + [0, 0, 0.065])))
            spin = xf.AddRotateZOp()
            box(stage, f"{P}/prop_{k}/blade", (0, 0, 0), (0.34, 0.032, 0.006), blade)
            cylinder(stage, f"{P}/prop_{k}/disc", (0, 0, 0), 0.17, 0.002, blur)
            self.props.append((spin, 1 if k % 2 else -1))
            # landing legs under the motors
            cylinder(stage, f"{P}/leg_{k}", tip * 0.55 + [0, 0, -0.12], 0.008, 0.2, carbon)
        # aviation convention: red to port (+y when facing +x), green to starboard, white tail strobe
        sphere(stage, f"{P}/nav_port", (self.ARM * 0.71, self.ARM * 0.71, 0.0), 0.022, red)
        sphere(stage, f"{P}/nav_starboard", (self.ARM * 0.71, -self.ARM * 0.71, 0.0), 0.022, green)
        self.strobe = sphere(stage, f"{P}/nav_tail", (-0.19, 0, 0.02), 0.02, white)
        # thermal / RGB gimbal under the nose
        box(stage, f"{P}/gimbal_yoke", (0.12, 0, -0.085), (0.03, 0.12, 0.05), carbon)
        sphere(stage, f"{P}/gimbal_head", (0.13, 0, -0.13), 0.055, shell)
        cylinder(stage, f"{P}/gimbal_lens", (0.18, 0, -0.13), 0.028, 0.02, lens, axis="X")
        sphere(stage, f"{P}/status", (-0.1, 0.0, 0.07), 0.025, self.light_mat)
        box(stage, f"{P}/id_left", (0.0, 0.122, 0.0), (0.14, 0.004, 0.05), self.id_mat)
        box(stage, f"{P}/id_right", (0.0, -0.122, 0.0), (0.14, 0.004, 0.05), self.id_mat)

    def update(self, track: Track, t: float) -> None:
        p, yaw, v = track.position(t), track.yaw_at(t), track.velocity(t)
        state = track.state(t).upper()
        landed = state in ("IDLE", "DOCKED", "CHARGING", "LANDED") and track.speed(t) < 0.05
        hover = 0.0 if landed else 0.06 * math.sin(t * 1.7 + len(self.id))
        horizontal = np.array([v[0], v[1], 0.0])
        m = pose(p[0], p[1], p[2] + 0.12 + hover, yaw)          # legs reach 0.22 m below the body centre
        speed = float(np.linalg.norm(horizontal))
        if speed > 0.05:
            axis = np.cross([0.0, 0.0, 1.0], horizontal / speed)
            tilt = axis_angle(axis, min(0.22, 0.045 * speed))
            m[:3, :3] = tilt[:3, :3] @ m[:3, :3] if tilt.shape == (4, 4) else tilt @ m[:3, :3]
        self.op.Set(matrix(m))
        rate = 0.0 if landed else 2900.0       # deg/s; reads as a blur at 24 fps, the disc does the rest
        for spin, sign in self.props:
            spin.Set(float((sign * rate * t + 37 * sign) % 360.0))
        vis = UsdGeom.Imageable(self.strobe)
        vis.MakeVisible() if (t * 1.2) % 1.0 < 0.12 else vis.MakeInvisible()
        self.set_status(state)
        self.last_position = m[:3, 3].copy()


# ------------------------------------------------------------------------------------ C: crawler
class Crawler(FleetRobot):
    """Magnetic-wheel crawler: low chassis, 4 magnet wheels, sensor mast with camera head and a UT probe."""
    tag_height = 0.55
    WHEEL_R = 0.07

    def __init__(self, stage, robot_id, index, root):
        super().__init__(stage, robot_id, index, root)
        P = self.path
        body = material(stage, "crawler_body", (0.85, 0.42, 0.05), 0.45, 0.1)
        dark = material(stage, "crawler_dark", (0.05, 0.05, 0.055), 0.6)
        steel = material(stage, "crawler_steel", (0.6, 0.62, 0.65), 0.3, 0.9)
        glass = material(stage, "crawler_glass", (0.02, 0.03, 0.05), 0.05, 0.6)
        r = self.WHEEL_R
        box(stage, f"{P}/chassis", (0, 0, r + 0.03), (0.5, 0.26, 0.09), body)
        box(stage, f"{P}/deck", (-0.04, 0, r + 0.09), (0.34, 0.2, 0.03), dark)
        self.wheels = []
        for k, (x, y) in enumerate(((0.17, 0.17), (0.17, -0.17), (-0.17, 0.17), (-0.17, -0.17))):
            w = UsdGeom.Xform.Define(stage, f"{P}/wheel_{k}")
            xf = UsdGeom.Xformable(w)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, r))
            rot = xf.AddRotateYOp()
            cylinder(stage, f"{P}/wheel_{k}/tyre", (0, 0, 0), r, 0.05, dark, axis="Y")
            cylinder(stage, f"{P}/wheel_{k}/hub", (0, 0.027 if y > 0 else -0.027, 0), r * 0.55, 0.008, steel, axis="Y")
            box(stage, f"{P}/wheel_{k}/spoke", (0, 0.032 if y > 0 else -0.032, 0), (r * 1.4, 0.004, 0.012), body)
            self.wheels.append(rot)
        cylinder(stage, f"{P}/mast", (0.12, 0, r + 0.2), 0.012, 0.22, steel)
        self.head = UsdGeom.Xform.Define(stage, f"{P}/head")
        hx = UsdGeom.Xformable(self.head)
        hx.AddTranslateOp().Set(Gf.Vec3d(0.12, 0, r + 0.33))
        self.pan = hx.AddRotateZOp()
        box(stage, f"{P}/head/housing", (0, 0, 0), (0.09, 0.12, 0.06), dark)
        cylinder(stage, f"{P}/head/lens", (0.05, 0, 0), 0.02, 0.012, glass, axis="X")
        lamp = material(stage, "crawler_lamp", (1, 1, 1), emissive=(6, 6, 5.5))
        box(stage, f"{P}/head/lamp_l", (0.046, 0.045, 0.0), (0.006, 0.018, 0.018), lamp)
        box(stage, f"{P}/head/lamp_r", (0.046, -0.045, 0.0), (0.006, 0.018, 0.018), lamp)
        # UT probe on a short boom at the front, touching the surface
        box(stage, f"{P}/probe_boom", (0.29, 0, r + 0.02), (0.1, 0.025, 0.02), steel)
        cylinder(stage, f"{P}/probe", (0.33, 0, r * 0.55), 0.018, r * 0.9, body)
        sphere(stage, f"{P}/status", (-0.18, 0, r + 0.12), 0.025, self.light_mat)
        box(stage, f"{P}/id_left", (-0.04, 0.132, r + 0.03), (0.16, 0.004, 0.06), self.id_mat)
        box(stage, f"{P}/id_right", (-0.04, -0.132, r + 0.03), (0.16, 0.004, 0.06), self.id_mat)

    def update(self, track: Track, t: float) -> None:
        p, yaw = track.position(t), track.yaw_at(t)
        state = track.state(t).upper()
        m = pose(p[0], p[1], deck_height(p[2]), yaw)
        self.op.Set(matrix(m))
        roll = math.degrees(track.distance(t) / self.WHEEL_R)
        for rot in self.wheels:
            rot.Set(float(roll % 360.0))
        scanning = state in ("INSPECTING", "WORKING", "RUNNING", "FINDING")
        self.pan.Set(float(35.0 * math.sin(t * 1.3)) if scanning else 0.0)
        self.set_status(state)
        self.last_position = m[:3, 3].copy()


# ----------------------------------------------------------------------------------------- R: ROV
class ROV(FleetRobot):
    """Work-class ROV block: yellow buoyancy top, open frame, thrusters, lights. Optional, simple."""
    tag_height = 1.0

    def __init__(self, stage, robot_id, index, root):
        super().__init__(stage, robot_id, index, root)
        P = self.path
        foam = material(stage, "rov_foam", (0.95, 0.75, 0.05), 0.6)
        frame = material(stage, "rov_frame", (0.12, 0.12, 0.13), 0.5, 0.6)
        lamp = material(stage, "rov_lamp", (1, 1, 1), emissive=(8, 8, 7))
        box(stage, f"{P}/foam", (0, 0, 0.32), (1.1, 0.75, 0.25), foam)
        for k, (x, y) in enumerate(((0.5, 0.33), (0.5, -0.33), (-0.5, 0.33), (-0.5, -0.33))):
            box(stage, f"{P}/post_{k}", (x, y, 0.0), (0.05, 0.05, 0.45), frame)
            cylinder(stage, f"{P}/thruster_{k}", (x * 1.1, y * 1.2, 0.05), 0.08, 0.12, frame, axis="X")
        box(stage, f"{P}/skid", (0, 0, -0.22), (1.1, 0.75, 0.04), frame)
        sphere(stage, f"{P}/lamp_l", (0.56, 0.2, 0.1), 0.04, lamp)
        sphere(stage, f"{P}/lamp_r", (0.56, -0.2, 0.1), 0.04, lamp)
        sphere(stage, f"{P}/status", (-0.4, 0, 0.47), 0.04, self.light_mat)
        box(stage, f"{P}/id_left", (0.0, 0.38, 0.32), (0.3, 0.004, 0.12), self.id_mat)

    def update(self, track: Track, t: float) -> None:
        p, yaw = track.position(t), track.yaw_at(t)
        m = pose(p[0], p[1], p[2] + 0.03 * math.sin(t * 0.9), yaw)
        self.op.Set(matrix(m))
        self.set_status(track.state(t).upper())
        self.last_position = m[:3, 3].copy()


MODELS = {"Q": Quadruped, "A": Drone, "C": Crawler, "R": ROV}

BEACON_OFF = (0.08, 0.05, 0.0)
BEACON_ALARM = (1.0, 0.22, 0.02)      # red-amber
BEACON_GLOW = 60.0


class FixedDetector:
    """Fixed F&G detector: a grey post from the deck, a yellow detector head with a dark sensor cup, and a beacon
    on top that flashes red-amber while the detector is in alarm. Static; only the beacon animates."""

    def __init__(self, stage, det_id: str, head_xyz, deck_z: float, kind: str, root: str = "/World/Fleet"):
        safe = det_id.replace("-", "_")
        self.path = f"{root}/Detectors/{safe}"
        UsdGeom.Xform.Define(stage, self.path)
        x, y, z = map(float, head_xyz)
        post = material(stage, "detector_post", (0.45, 0.47, 0.5), 0.6, 0.6)
        yellow = material(stage, "detector_head", (0.98, 0.78, 0.05), 0.45)
        dark = material(stage, "detector_cup", (0.05, 0.05, 0.06), 0.7)
        h = max(z - deck_z - 0.12, 0.2)
        cylinder(stage, f"{self.path}/post", (x, y, deck_z + h / 2), 0.035, h, post)
        cylinder(stage, f"{self.path}/foot", (x, y, deck_z + 0.01), 0.12, 0.02, post)
        if kind == "point_gas":
            box(stage, f"{self.path}/head", (x, y, z), (0.16, 0.12, 0.2), yellow)
            cylinder(stage, f"{self.path}/cup", (x, y, z - 0.16), 0.045, 0.12, dark)
        else:                                       # optical: a yellow box with a lens on the long side
            box(stage, f"{self.path}/head", (x, y, z), (0.26, 0.16, 0.16), yellow)
            cylinder(stage, f"{self.path}/lens", (x + 0.15, y, z), 0.05, 0.05, dark, axis="X")
        self.mat = material(stage, f"detector_beacon_{safe}", (0.25, 0.12, 0.05), 0.3, emissive=BEACON_OFF)
        self.beacon = sphere(stage, f"{self.path}/beacon", (x, y, z + 0.17), 0.055, self.mat)
        self.halo_mat = translucent(stage, f"detector_halo_{safe}", (1.0, 0.3, 0.05), BEACON_ALARM, 0.0, 0.0)
        self.halo = sphere(stage, f"{self.path}/halo", (x, y, z + 0.17), 0.13, self.halo_mat)
        self.state: Optional[str] = "init"
        self.set_state(None, 0.0)

    def set_state(self, state: Optional[str], t: float) -> None:
        if state == "alarm":
            on = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.5) > 0.35          # 1.5 Hz flash
            set_emissive(self.mat[1], np.array(BEACON_ALARM) * (GLOW * 2 if on else 2.0))
            self.halo_mat[1].GetInput("emissive_intensity").Set(BEACON_GLOW * 20 if on else 0.0)
            self.halo_mat[1].GetInput("enable_emission").Set(bool(on))
            set_opacity(self.halo_mat[1], 0.22 if on else 0.0)
        elif self.state != state:
            set_emissive(self.mat[1], BEACON_OFF)
            self.halo_mat[1].GetInput("enable_emission").Set(False)
            set_opacity(self.halo_mat[1], 0.0)
        self.state = state


def build_robot(stage, robot_id: str, kind: str, index: int, root: str = "/World/Fleet") -> FleetRobot:
    return MODELS.get(kind, Crawler)(stage, robot_id, index, root)

"""What the Isaac Sim bridge needs for the deck sweep (see integration/INTERFACE.md, section 7).

The deck half of the team's future Isaac bridge, like integration/leak_bridge.py is the leak half:

    class RigIsaacBridge(LeakBridgeMixin, DeckBridgeMixin, <their SimBridge implementation>):
        ...  # see integration/rig_bridge_example.py for what the host must provide

Same methods and meaning as the CPU mock (policy/deck/mock.py):
    deck_frame() -> (rgb, CameraModel)     deck_reference() -> rgb        request_deck_view(i) -> bool
    pick_debris(u, v) -> bool              body_frame() -> (rgb, CameraModel)
    body_range() -> (H, W) metres of ground distance per pixel

The maths (USD camera -> CameraModel, depth -> ground distance, pixel -> deck point, world -> robot base) are plain
numpy and pxr, tested on CPU with usd-core (tests/test_deck_bridge.py). Only ``grab_camera`` touches Isaac
(omni.replicator), and it imports it inside the function. Nothing here has been run inside Isaac Sim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from policy.interfaces import CameraModel, camera_from_usd

RGB_ANNOTATOR = "rgb"
DEPTH_ANNOTATOR = "distance_to_image_plane"     # metres along the camera's forward axis; inf where nothing is hit


# ------------------------------------------------------------------ pure maths (CPU)
def usd_camera_model(stage, path: str, width: int, height: int, time=None) -> CameraModel:
    """CameraModel (OpenCV: x right, y down, z forward) of the USD camera at ``path``.

    Reads the prim's world transform (USD camera: looks down -Z, +Y up) and its focal length and horizontal
    aperture, then uses policy.interfaces.camera_from_usd. Square pixels, principal point at the image centre."""
    from pxr import Usd, UsdGeom
    cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
    if not cam:
        raise KeyError(f"no USD camera at {path}")
    t = Usd.TimeCode.Default() if time is None else time
    M = np.array(cam.ComputeLocalToWorldTransform(t))
    focal = float(cam.GetFocalLengthAttr().Get(t))
    aperture = float(cam.GetHorizontalApertureAttr().Get(t))
    return camera_from_usd(M, focal, aperture, width, height)


def _pixel_rays(camera: CameraModel, H: int, W: int) -> np.ndarray:
    """(H, W, 3) camera-frame rays with z = 1 for every pixel (integer pixel coords, as detection.rig_vision)."""
    K = np.asarray(camera.K, np.float64)
    v, u = np.mgrid[:H, :W].astype(np.float64)
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)


def depth_to_ground_range(depth: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Depth image (distance to the image plane, metres) -> ground distance per pixel (metres, float32).

    Ground distance = horizontal distance (world x, y; world z up) from the camera to the 3D point the pixel sees,
    the same quantity as detection.rig_vision.floor_range gives for the bare deck. Pixels with no hit (inf, NaN,
    zero or negative depth) are inf. Works at any deck height, because it uses the measured depth."""
    d = np.asarray(depth, np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    H, W = d.shape
    R = np.asarray(camera.T_world_cam, np.float64)[:3, :3]
    valid = np.isfinite(d) & (d > 0)
    off = (np.where(valid, d, 0.0)[..., None] * _pixel_rays(camera, H, W)) @ R.T     # world offset from the eye
    return np.where(valid, np.hypot(off[..., 0], off[..., 1]), np.inf).astype(np.float32)


def pixel_ray(camera: CameraModel, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """(eye, unit direction) of the world ray through pixel (u, v)."""
    K = np.asarray(camera.K, np.float64)
    T = np.asarray(camera.T_world_cam, np.float64)
    d = T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    return T[:3, 3].copy(), d / np.linalg.norm(d)


def pixel_to_deck(camera: CameraModel, u: float, v: float, deck_z: float = 0.0) -> Optional[np.ndarray]:
    """World point where the ray through pixel (u, v) meets the deck plane z = deck_z; None if it never does
    (the pixel is at or above the horizon, or the camera is below the deck)."""
    eye, d = pixel_ray(camera, u, v)
    if d[2] > -1e-9:
        return None
    t = (deck_z - eye[2]) / d[2]
    return eye + t * d if t > 0 else None


def world_to_base(p_world, base_xyz, base_yaw: float) -> np.ndarray:
    """World point -> robot base frame (origin at the base, x forward, z up; only the yaw of the base counts,
    as in policy/envs/pick_state.py where the deck is level)."""
    d = np.asarray(p_world, np.float64) - np.asarray(base_xyz, np.float64)
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])


def deck_z_from_meta(meta_path: str, deck: str = "top_deck") -> float:
    """Deck height from a rig meta file: ``deck_heights_m[deck]`` (sim/output/full_rig_meta.json; top deck
    42.5 m), else 0.0 (sim/output/rig_deck_meta.json and INTERFACE.md put the deck at z = 0)."""
    meta = json.loads(Path(meta_path).read_text())
    return float(meta.get("deck_heights_m", {}).get(deck, 0.0))


# ------------------------------------------------------------------ test debris (pxr only)
def spawn_debris_can(stage, path: str, position, yaw: float = 0.0, radius: float = 0.033, height: float = 0.12,
                     lying: bool = True, color=(0.8, 0.08, 0.08), physics: bool = False, mass: float = 0.02) -> str:
    """Add a drink can (USD cylinder, red material) on the deck for demos and videos. Returns the prim path.

    ``position``: world (x, y, deck_z): the point on the deck under the can's centre. ``lying`` (the default, as
    in policy/envs/pick_state.py): the can's axis is horizontal, along ``yaw`` (radians from world +x).
    ``physics``: also make it a rigid body with a collider and a mass, so it rests on the deck in Isaac."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    x, y, z = (float(c) for c in position)
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z + (radius if lying else height / 2)))
    xf.AddRotateZOp().Set(math.degrees(yaw))
    cyl = UsdGeom.Cylinder.Define(stage, f"{path}/body")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(UsdGeom.Tokens.x if lying else UsdGeom.Tokens.z)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mat_path = f"{path}/Looks/CanRed"
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.4)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)
    prim = xf.GetPrim()
    prim.CreateAttribute("rig:debris", Sdf.ValueTypeNames.Bool).Set(True)
    if physics:
        from pxr import UsdPhysics
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return path


def hide_prim(stage, path: str) -> None:
    """Make a picked object disappear: invisible, and no longer colliding if it has a collider."""
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    UsdGeom.Imageable(prim).MakeInvisible()
    for p in Usd.PrimRange(prim):
        a = p.GetAttribute("physics:collisionEnabled")
        if a and a.IsValid():
            a.Set(False)


# ------------------------------------------------------------------ the mixin
def _no_pick_skill(target_xy_base, yaw):
    raise NotImplementedError(
        "DeckBridgeMixin.pick_skill is not set. Plug in the pick policy: bridge.pick_skill = "
        "fn(target_xy_base (2,), yaw) -> bool, with the target in the robot base frame (x forward, metres).")


class DeckBridgeMixin:
    """Deck-sweep methods for the Isaac bridge. The host class must provide:
        stage (Usd.Stage), robot_position() -> (3,), robot_yaw() -> rad,
        move_base_to(xyz, facing) -> bool (one control step toward it; True when there), advance(dt),
        and, optionally, patrol_stop (int, the current patrol stop; set it in go_to_patrol_pose) and
        pick_skill(target_xy_base, yaw) -> bool (the arm's pick; the RL pick policy plugs in here).
    Cameras: ``deck_camera_path`` looks down the deck ahead, ``body_camera_path`` looks ahead (may be the same).
    """

    deck_camera_path: str = "/World/Robots/scrubber_dog/base_link/body_camera"
    body_camera_path: str = "/World/Robots/scrubber_dog/base_link/body_camera"
    deck_resolution: tuple[int, int] = (640, 360)          # (W, H), like the wrist camera
    deck_z: float = 0.0                                     # deck height (see deck_z_from_meta)
    deck_view_offsets: tuple = ((0.0, 0.0), (0.0, 0.35), (0.35, 0.0))   # (forward, left) m in the base frame
    pick_standoff: float = 0.26        # walk until the object is this far ahead (pick_state.py obj_x 0.20-0.32 m)
    pick_reach: tuple = (0.20, 0.32)   # x range in the base frame the pick skill was trained on (Go2 crouched)
    pick_lateral: float = 0.10         # and |y| up to this (pick_state.py obj_y)
    pick_max_walk: float = 3.0         # do not walk further than this to pick something up
    debris_match_m: float = 0.15       # a registered debris prim this close to the pick point is the one picked
    pick_skill: Callable = staticmethod(_no_pick_skill)

    # -- state (created on first use, so the host's __init__ does not have to call anything)
    def _deck(self) -> dict:
        if not hasattr(self, "_deck_state"):
            self._deck_state = {"refs": {}, "view": 0, "home": None, "last": {}, "stop": None}
        return self._deck_state

    @property
    def debris_prims(self) -> dict:
        """Registry of spawned debris: prim path -> world position on the deck."""
        if not hasattr(self, "_debris_prims"):
            self._debris_prims = {}
        return self._debris_prims

    def deck_stop(self) -> int:
        return int(getattr(self, "patrol_stop", 0))

    # -- cameras (the one Isaac-only part) -------------------------------------
    def grab_camera(self, path: str, depth: bool = False) -> dict:
        """One rendered frame of the camera at ``path``: {"rgb": HxWx3 uint8, "depth": HxW float32 or absent}.

        Isaac: a replicator render product per camera, with the "rgb" annotator (as rig/sim/main.py) and, when
        asked, "distance_to_image_plane". Created on first use and kept. The host's ``advance`` must have rendered
        a frame since the scene changed (or override this method). Tests override it with a fake."""
        import omni.replicator.core as rep
        cache = self.__dict__.setdefault("_rp_cache", {})
        if path not in cache:
            rp = rep.create.render_product(path, tuple(self.deck_resolution))
            anns = {}
            for name in (RGB_ANNOTATOR, DEPTH_ANNOTATOR):
                anns[name] = rep.AnnotatorRegistry.get_annotator(name)
                anns[name].attach(rp)
            cache[path] = anns
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        anns = cache[path]
        out = {"rgb": np.ascontiguousarray(np.asarray(anns[RGB_ANNOTATOR].get_data())[..., :3], dtype=np.uint8)}
        if depth:
            out["depth"] = np.asarray(anns[DEPTH_ANNOTATOR].get_data(), np.float32)
        return out

    def camera_model(self, path: str) -> CameraModel:
        W, H = self.deck_resolution
        return usd_camera_model(self.stage, path, W, H)

    # -- deck camera -----------------------------------------------------------
    def deck_frame(self):
        st = self._deck()
        if st["stop"] != self.deck_stop():           # a new patrol stop: this pose is view 0
            st.update(stop=self.deck_stop(), view=0, home=None)
        if st["view"] == 0:
            st["home"] = (np.asarray(self.robot_position(), float).copy(), float(self.robot_yaw()))
        img = self.grab_camera(self.deck_camera_path)["rgb"]
        st["last"]["deck"] = ((self.deck_stop(), st["view"]), img)
        return img, self.camera_model(self.deck_camera_path)

    def capture_deck_reference(self) -> np.ndarray:
        """Store the current deck view as the clean picture of this (patrol stop, view). Call it when the deck
        is known clean (commissioning run, before any debris is spawned)."""
        img = self.grab_camera(self.deck_camera_path)["rgb"]
        self._deck()["refs"][(self.deck_stop(), self._deck()["view"])] = img
        return img

    def deck_reference(self) -> np.ndarray:
        """The clean picture of this patrol stop and view.

        If none is stored yet, the current frame is stored and returned: the first visit is taken as clean (the
        same rule as the mock's commissioning picture). So on a first visit nothing counts as a change; debris
        put down after that is found. For the demo, run one clean patrol (or call capture_deck_reference at each
        stop) before spawning debris, or load references saved from such a run with load_references()."""
        st = self._deck()
        key = (self.deck_stop(), st["view"])
        if key not in st["refs"]:
            last_key, img = st["last"].get("deck", (None, None))
            st["refs"][key] = img if last_key == key else self.grab_camera(self.deck_camera_path)["rgb"]
        return st["refs"][key]

    def save_references(self, directory: str) -> Path:
        """Write every stored reference as PNG plus index.json (stop, view, file, camera)."""
        from PIL import Image
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        index = []
        for (stop, view), img in sorted(self._deck()["refs"].items()):
            name = f"ref_stop{stop:02d}_view{view}.png"
            Image.fromarray(np.asarray(img, np.uint8)).save(d / name)
            index.append({"stop": stop, "view": view, "file": name})
        (d / "index.json").write_text(json.dumps({"camera": self.deck_camera_path, "references": index}, indent=1))
        return d / "index.json"

    def load_references(self, directory: str) -> int:
        """Load references written by save_references. Returns how many."""
        from PIL import Image
        d = Path(directory)
        index = json.loads((d / "index.json").read_text())
        refs = self._deck()["refs"]
        for r in index["references"]:
            refs[(int(r["stop"]), int(r["view"]))] = np.asarray(Image.open(d / r["file"]).convert("RGB"))
        return len(index["references"])

    def _walk_to(self, target, facing, max_steps: int = 400, dt: float = 0.1) -> bool:
        for _ in range(max_steps):
            if self.move_base_to(np.asarray(target, float), facing):
                return True
            self.advance(dt)
        return False

    def request_deck_view(self, index: int) -> bool:
        """Step the body to view ``index``: (forward, left) offset ``deck_view_offsets[index]`` from the pose of
        view 0 at this stop, same heading. Index 0 goes back. False if there is no such view or it is unreachable."""
        st = self._deck()
        if index < 0 or index >= len(self.deck_view_offsets):
            return False
        if st["home"] is None:
            st["home"] = (np.asarray(self.robot_position(), float).copy(), float(self.robot_yaw()))
        p0, yaw = st["home"]
        fwd, left = self.deck_view_offsets[index]
        c, s = math.cos(yaw), math.sin(yaw)
        target = p0 + np.array([c * fwd - s * left, s * fwd + c * left, 0.0])
        if not self._walk_to(target, np.array([c, s, 0.0])):
            return False
        st["view"] = index
        return True

    # -- the arm -----------------------------------------------------------------
    def pick_yaw(self, u: float, v: float, point_world: np.ndarray) -> float:
        """Object orientation for the pick skill, in the world frame (rad). Default 0: the detector gives only a
        pick point. Override with an estimate (e.g. the changed-pixel blob's long axis) when the pick needs it."""
        return 0.0

    def pick_debris(self, u: float, v: float) -> bool:
        """Pick up the object at pixel (u, v) of the last deck_frame().

        Projects the pixel onto the deck plane (z = deck_z) with the deck camera model, walks so the point is
        ``pick_standoff`` ahead if it is out of the pick skill's reach, converts it to the base frame and calls
        ``pick_skill(target_xy_base, yaw_base)``. On success, hides the matching registered debris prim (if any).
        False if the pixel is not on the deck, too far to walk, or the pick fails."""
        cam = self.camera_model(self.deck_camera_path)
        p = pixel_to_deck(cam, u, v, self.deck_z)
        if p is None:
            return False
        base = np.asarray(self.robot_position(), float)
        yaw = float(self.robot_yaw())
        local = world_to_base(p, base, yaw)
        if not (self.pick_reach[0] <= local[0] <= self.pick_reach[1] and abs(local[1]) <= self.pick_lateral):
            d = p[:2] - base[:2]
            dist = float(np.linalg.norm(d))
            if dist - self.pick_standoff > self.pick_max_walk or dist < 1e-6:
                return False
            heading = d / dist
            stand = p[:2] - self.pick_standoff * heading
            if not self._walk_to([stand[0], stand[1], base[2]], np.array([heading[0], heading[1], 0.0])):
                return False
            base, yaw = np.asarray(self.robot_position(), float), float(self.robot_yaw())
            local = world_to_base(p, base, yaw)
        obj_yaw = self.pick_yaw(u, v, p) - yaw
        ok = bool(self.pick_skill(local[:2].copy(), float(math.atan2(math.sin(obj_yaw), math.cos(obj_yaw)))))
        if ok:
            self._remove_picked(p)
        return ok

    def spawn_debris(self, name: str, position, yaw: float = 0.0, **kw) -> str:
        """spawn_debris_can under /World/Debris/<name>, registered so a successful pick hides it."""
        pos = np.asarray(position, float)
        if pos.shape[0] == 2:
            pos = np.array([pos[0], pos[1], self.deck_z])
        path = spawn_debris_can(self.stage, f"/World/Debris/{name}", pos, yaw, **kw)
        self.debris_prims[path] = pos
        return path

    def _remove_picked(self, p_world: np.ndarray) -> Optional[str]:
        if not self.debris_prims:
            return None
        path, pos = min(self.debris_prims.items(), key=lambda kv: float(np.linalg.norm(kv[1][:2] - p_world[:2])))
        if float(np.linalg.norm(pos[:2] - p_world[:2])) > self.debris_match_m:
            return None
        hide_prim(self.stage, path)
        del self.debris_prims[path]
        return path

    # -- body camera -------------------------------------------------------------
    def body_frame(self):
        """RGB and camera model of the body camera; also keeps the depth of the same frame for body_range()."""
        frame = self.grab_camera(self.body_camera_path, depth=True)
        cam = self.camera_model(self.body_camera_path)
        self._deck()["last"]["body"] = (frame.get("depth"), cam)
        return frame["rgb"], cam

    def body_range(self) -> np.ndarray:
        """Ground distance (m) per pixel, aligned with the last body_frame() (grabs a new frame if none)."""
        depth, cam = self._deck()["last"].pop("body", (None, None))
        if depth is None:
            depth = self.grab_camera(self.body_camera_path, depth=True)["depth"]
            cam = self.camera_model(self.body_camera_path)
        return depth_to_ground_range(depth, cam)

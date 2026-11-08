"""Run integration.isaac_bridge.IsaacRigBridge on CPU without Isaac Sim.

The USD scene module (integration.repair_scene, which needs pxr) is replaced by stand-ins, and the
inspection "render" is the CPU mock's ray-cast image of the RustState world. Everything else, including
IK, contact force, parking, scrubbing and the orchestrator, is the real code.
"""
from __future__ import annotations

import sys
import types

import numpy as np

from policy.interfaces import PipeGeometry


def install_stub_scene(pipe: PipeGeometry):
    rs = types.ModuleType("integration.repair_scene")

    class Attr:
        def __init__(self, v=0.0):
            self.v = v

        def Set(self, v):
            self.v = v

        def Get(self):
            return self.v

    class Anything:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: Attr()

    class Camera:
        def __init__(self, stage, path, focal=35.0, aperture=20.955, clip=None):
            self.focal, self.aperture, self.M = focal, aperture, np.eye(4)

        def look(self, eye, target):
            eye, target = np.asarray(eye, float), np.asarray(target, float)
            zc = target - eye
            zc /= np.linalg.norm(zc)
            xc = np.cross(zc, [0, 0, -1.0])
            xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)
            # USD camera looks down -Z with +Y up; store a USD-convention row-vector matrix
            R = np.stack([xc, -yc, -zc], 1)
            M = np.eye(4)
            M[:3, :3] = R.T
            M[3, :3] = eye
            self.M = M

        def world_matrix(self):
            return self.M

    def inspection_views(p, theta):
        surf, n = p.surface_point(theta, 0.0)
        n2 = p.surface_point(theta + 0.35, 0.0)[1]
        n3 = p.surface_point(theta - 0.35, 0.0)[1]
        return [(surf + 0.85 * n + 0.15 * p.axis, surf), (surf + 0.7 * n2 + 0.12 * p.axis, surf + 0.03 * p.axis),
                (surf + 0.7 * n3 - 0.12 * p.axis, surf)]

    rs.find_main_pipe = lambda stage: None
    rs.read_pipe = lambda prim, n_z=40, n_theta=64: PipeGeometry(pipe.origin, pipe.axis, pipe.ref, pipe.radius,
                                                                 pipe.length, n_z, n_theta)
    rs.RustOverlay = lambda stage, p: types.SimpleNamespace(update=lambda world: None)
    rs.RobotPoser = lambda stage, model: types.SimpleNamespace(set=lambda base, q: None)
    rs.Camera = Camera
    rs.inspection_views = inspection_views
    rs.add_lens_mist = lambda stage, path: Anything()
    glare = types.SimpleNamespace(GetIntensityAttr=lambda: Attr())
    rs.add_glare_light = lambda stage, eye, target: glare
    rs.add_worker = lambda stage: (Attr(), Attr())
    rs.add_spray_cone = lambda stage: (None, None)
    rs.set_spray = lambda *a, **k: None
    rs.add_weld_bead = lambda *a, **k: None
    rs.UsdGeom = types.SimpleNamespace(Imageable=lambda prim: Anything())
    rs.Gf = types.SimpleNamespace(Vec3d=lambda *a: a)
    rs.INSPECTION_CAMERA = "/World/Cameras/repair_closeup"
    rs.PIPE_SCOPE = "/World/Pipes"
    sys.modules["integration.repair_scene"] = rs
    import integration
    integration.repair_scene = rs
    return rs


def make_bridge(theta: float | None = None):
    """-> (bridge, world, pipe). Inspection frames come from the CPU mock's ray-caster."""
    from policy.demo_mock import MockPerception, MockRustWorld, scene_geometry
    pipe, dmg = scene_geometry()
    theta = dmg if theta is None else theta
    install_stub_scene(pipe)
    sys.modules.pop("integration.isaac_bridge", None)
    from integration.isaac_bridge import Frames, IsaacRigBridge
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    world.set_rust(world.blob(theta, 0.0, size=0.05, level=0.85))
    world.damage_theta = theta
    holder = {}

    def render():
        b = holder["bridge"]
        cam = MockPerception(world, b._pipe, theta, W=640, H=360)
        cam.glare_views = set()
        cam.views = [cam.views[0]]
        T = b._camera_model().T_world_cam
        cam.views = [T]
        return {"inspection": cam.observe(), "hero": None}

    bridge = IsaacRigBridge(None, world, Frames(render), verbose=False)
    holder["bridge"] = bridge
    return bridge, world, pipe

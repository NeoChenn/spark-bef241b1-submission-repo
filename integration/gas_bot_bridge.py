from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pxr import Gf, UsdGeom

from integration import repair_scene as rs
from integration.isaac_bridge import IsaacRigBridge
from integration.kinematics import ARM_STOWED, TRUNK_HEIGHT, pose
from integration.leak_bridge import apply_leak_state, bolt_waypoint
from policy.gasbot import deck
from policy.gasbot.bot import GasBotMixin
from policy.interfaces import camera_from_usd

REPO = Path(__file__).resolve().parent.parent
OGI_CAMERA = "/World/Cameras/gasbot_ogi"
GASBOT = "/World/GasBot"


def add_deck_flanges(stage) -> dict:
    sys.path.insert(0, str(REPO / "sim"))
    from build_leak_flanges import BORE, CRACK, HEAD, RED, STEEL, build_flange
    from scene_builder import SceneBuilder
    UsdGeom.Xform.Define(stage, "/World/Leak")
    b = SceneBuilder(stage)
    mats = {"steel": b.plain("LeakSteel", STEEL, 0.55, 0.35), "head": b.plain("LeakBoltHead", HEAD, 0.5, 0.4),
            "bore": b.plain("LeakBore", BORE, 0.8), "red": b.plain("TorqueStripe", RED, 0.6),
            "crack": b.plain("LeakCrack", CRACK, 0.9)}
    return {g.id: build_flange(b, stage, g.id, g, mats) for g in deck.flanges()}


def _part(stage, name, kind, color, **attrs):
    prim = getattr(UsdGeom, kind).Define(stage, f"{GASBOT}/{name}")
    for k, v in attrs.items():
        getattr(prim, f"Create{k}Attr")(v)
    rs.bind(prim.GetPrim(), rs.preview_material(stage, f"gasbot_{name}", color, 0.4, 0.1))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    return prim, xf.AddTransformOp()


def _m4(R, t, s=(1.0, 1.0, 1.0)) -> Gf.Matrix4d:
    M = np.eye(4)
    M[:3, :3] = np.asarray(R) @ np.diag(s)
    M[:3, 3] = t
    return Gf.Matrix4d(*M.T.reshape(-1))


class IsaacGasBot(GasBotMixin, IsaacRigBridge):
    def __init__(self, stage, world, frames, leak_world, prims, wind, seed=0, flow_full=None, h2s_ratio=0.0,
                 start=(-5.5, 1.25), inspection_size=(640, 360), frame_dt=1 / 12, gas_stop=0.20, field=None):
        IsaacRigBridge.__init__(self, stage, world, frames, inspection_size=inspection_size)
        self.leak_prims = prims
        self.frame_dt = frame_dt
        self.gas_stop = gas_stop
        self.route = deck.Route()
        self.route.nearest(start)
        d = self.route.target() - np.asarray(start, float)
        self.state.base = pose(start[0], start[1], TRUNK_HEIGHT, math.atan2(d[1], d[0]))
        self.lap_done = False
        self.leak_mode = False
        self.leak_views: list = []
        self.phase = 0.0
        self.path = [np.asarray(start, float)]
        self.gas_init(leak_world, wind, seed, flow_full, h2s_ratio, field=field)
        self.ogi = rs.Camera(stage, OGI_CAMERA, focal=14.0, aperture=20.955)
        self.detector, self.det_op = _part(stage, "detector", "Cube", (0.95, 0.75, 0.05), Size=1.0)
        self.boom, self.boom_op = _part(stage, "boom", "Cylinder", (0.15, 0.15, 0.16), Radius=1.0, Height=1.0,
                                        Axis="Z")
        self.head, self.head_op = _part(stage, "sniffer", "Sphere", (0.95, 0.45, 0.05), Radius=1.0)
        apply_leak_state(stage, leak_world, prims)
        self._sync()

    def base_xy(self) -> np.ndarray:
        return self.state.base[:2, 3].copy()

    def base_yaw(self) -> float:
        return math.atan2(self.state.base[1, 0], self.state.base[0, 0])

    def _sync(self) -> None:
        super()._sync()
        if not hasattr(self, "head_op"):
            return
        R = self.state.base[:3, :3]
        trunk = self.trunk_pos()
        self.det_op.Set(_m4(R, trunk + [0, 0, 0.02], (0.12, 0.08, 0.06)))
        tip = self.tip_pos()
        self.head_op.Set(_m4(np.eye(3), tip, (0.035,) * 3))
        a = trunk + [0, 0, 0.05]
        d = tip - a
        if self.at_flange is None:
            z = d / np.linalg.norm(d)
            x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
            x /= np.linalg.norm(x)
            self.boom_op.Set(_m4(np.stack([x, np.cross(z, x), z], 1), (a + tip) / 2,
                                 (0.008, 0.008, float(np.linalg.norm(d)))))
        else:
            self.boom_op.Set(_m4(np.eye(3), [0, 0, -50.0], (0.001,) * 3))
        h = self.heading()
        eye = trunk + np.r_[0.12 * h, 0.32]
        look = (deck.leak_point(self.flange(self.at_flange)) if self.at_flange
                else eye + np.r_[3.0 * h, -0.55])
        self.ogi.look(eye, look)

    def tip_pos(self) -> np.ndarray:
        if self.at_flange is not None and not self.parked:
            return self.tool()[:3, 3]
        return super().tip_pos()

    def ogi_camera(self, w: int, h: int):
        return camera_from_usd(self.ogi.world_matrix(), self.ogi.focal, self.ogi.aperture, w, h)

    def _render(self, record: bool = True) -> dict:
        self.gas_sync(self.state.t)
        return super()._render(record)

    def advance(self, dt: float) -> None:
        n = max(1, int(round(dt / self.frame_dt)))
        for _ in range(n):
            self.state.t += dt / n
            self._fire_events()
            self._render()

    def _legs(self, moving: bool) -> None:
        stand = self.model.default_q()
        swing = 0.25 if moving else 0.0
        for leg, sign in (("FL", 1), ("RR", 1), ("FR", -1), ("RL", -1)):
            lift = max(0.0, sign * math.sin(self.phase))
            self.state.q[f"{leg}_thigh_joint"] = stand[f"{leg}_thigh_joint"] - swing * lift * 0.6
            self.state.q[f"{leg}_calf_joint"] = stand[f"{leg}_calf_joint"] - swing * lift * 0.9

    def _move(self, xy, yaw: Optional[float], dt: float) -> None:
        if not self.parked:
            self._park_arm()
        start, y0 = self.base_xy(), self.base_yaw()
        y1 = y0 if yaw is None else y0 + (yaw - y0 + math.pi) % (2 * math.pi) - math.pi
        n = max(1, int(round(dt / self.frame_dt)))
        dist = float(np.linalg.norm(np.asarray(xy) - start))
        for k in range(1, n + 1):
            s = k / n
            p = start + (np.asarray(xy, float) - start) * s
            self.phase += (dist + 0.3 * abs(y1 - y0)) / n / 0.18 * math.pi
            self._legs(dist > 1e-3 or abs(y1 - y0) > 1e-3)
            self.state.base = pose(p[0], p[1], TRUNK_HEIGHT, y0 + (y1 - y0) * s)
            self.state.t += dt / n
            self._fire_events()
            self._render()
        self.path.append(self.base_xy())

    def _stow(self) -> None:
        self.state.q.update(ARM_STOWED)
        self.parked, self.unpark_q = True, self.model.default_q()

    def patrol_step(self, dist: float, dt: float) -> None:
        p, yaw, lap = self.route.step(self.base_xy(), dist)
        self.lap_done |= lap
        self._move(p, yaw, dt)

    def seek_move(self, v, dt: float) -> bool:
        v = np.asarray(v, float)
        q, blocked = deck.clamp(self.base_xy() + v * dt)
        self._move(q, math.atan2(v[1], v[0]) if np.linalg.norm(v) > 1e-6 else None, dt)
        return blocked

    def walk_to(self, target, yaw=None, speed=0.6, guard=True) -> bool:
        target, _ = deck.clamp(target)
        for _ in range(200):
            d = target - self.base_xy()
            n = float(np.linalg.norm(d))
            if n < 1e-3:
                break
            step = min(n, speed * 0.5)
            self._move(self.base_xy() + d / n * step, math.atan2(d[1], d[0]), step / speed)
            g = self.safety_reading()
            if guard and (g is None or g >= self.gas_stop):
                return False
        if yaw is not None:
            turn = abs((yaw - self.base_yaw() + math.pi) % (2 * math.pi) - math.pi)
            self._move(self.base_xy(), yaw, max(0.4, turn / 1.2))
        self._legs(False)
        return True

    def flange(self, fid):
        return self.leak_world.flanges[fid].geom

    def nearest_flange(self):
        f = self.nearest_flange_to(self.base_xy())
        return f[0] if f else None

    def approach_flange(self, fid) -> bool:
        g = self.flange(fid)
        stand, yaw = deck.stand_pose(g)
        if not self.walk_to(stand, yaw):
            return False
        self.at_flange = fid
        self.leak_mode = True
        self.leak_views = g.view_poses(0.42, 0.3)
        self.inspection.camera.GetFocalLengthAttr().Set(15.0)
        self.inspection.look(*self.leak_views[0])
        self._sniff(g)
        return True

    def _sniff(self, g) -> None:
        self._move_tool(g.center + 0.22 * g.ref + 0.05 * g.normal, -g.normal)
        self.advance(1.0)

    def observe(self):
        if not self.leak_mode:
            return super().observe()
        images = self._render()
        rgb = images.get("inspection")
        if rgb is None or rgb.size == 0 or float(rgb.mean()) < 1.0:
            return None
        return rgb

    def request_view(self, index: int) -> bool:
        if not self.leak_mode:
            return super().request_view(index)
        if not 0 <= index < len(self.leak_views):
            return False
        self.inspection.look(*self.leak_views[index])
        return True

    def torque(self, fid: str, bolt: int) -> bool:
        g = self.flange(fid)
        wps = [bolt_waypoint(g, bolt, 0.06), bolt_waypoint(g, bolt, 0.0)]
        reached = self.move_to(wps)
        for _ in range(6):
            self.state.q["ur3_wrist_3_joint"] = self.state.q.get("ur3_wrist_3_joint", 0.0) + 0.35
            self.advance(self.frame_dt * 2)
        self.leak_world.tighten(fid, bolt)
        apply_leak_state(self.stage, self.leak_world, self.leak_prims)
        self.move_to([bolt_waypoint(g, bolt, 0.08)])
        self._sniff(g)
        return bool(reached[-1])

    def retreat(self, distance: float = 3.0) -> None:
        self.at_flange = None
        self.leak_mode = False
        self.inspection.camera.GetFocalLengthAttr().Set(35.0)
        self.walk_to(self.upwind_target(distance), guard=False)

    def return_to_work(self) -> None:
        self.at_flange = None
        self.leak_mode = False
        self.inspection.camera.GetFocalLengthAttr().Set(35.0)
        self._stow()
        self.route.nearest(self.base_xy())

    def go_to_patrol_pose(self, p) -> bool:
        return True

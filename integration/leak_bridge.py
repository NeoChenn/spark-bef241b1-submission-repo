from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from pxr import Usd, UsdGeom

from policy.interfaces import Waypoint
from policy.leak.geometry import FlangeGeometry
from policy.leak.world import LeakWorld


def load_flanges(meta_path: str) -> tuple[list[FlangeGeometry], dict]:
    meta = json.loads(Path(meta_path).read_text())
    geoms = [FlangeGeometry.from_meta(f) for f in meta["flanges"]]
    prims = {f["id"]: f for f in meta["flanges"]}
    return geoms, prims


def load_leak_world(meta_path: str) -> tuple[LeakWorld, dict]:
    """LeakWorld (all bolts tight, no cracks) and the prim index, from leak_flanges_meta.json."""
    geoms, prims = load_flanges(meta_path)
    return LeakWorld(geoms), prims


def apply_leak_state(stage: Usd.Stage, world: LeakWorld, prims: dict) -> None:
    for fid, s in world.flanges.items():
        info = prims[fid]
        for k, path in enumerate(info["bolt_prims"]):
            attr = stage.GetPrimAtPath(path).GetAttribute(info.get("loosen_attr", "xformOp:rotateX:loosen"))
            attr.Set(float(math.degrees(s.angles[k])))
        img = UsdGeom.Imageable(stage.GetPrimAtPath(info["crack_prim"]))
        if s.crack > 0:
            img.MakeVisible()
        else:
            img.MakeInvisible()
        root = stage.GetPrimAtPath(info["prim"])
        if root.HasAttribute("rig:crackSeverity"):
            root.GetAttribute("rig:crackSeverity").Set(float(s.crack))


def bolt_waypoint(geom: FlangeGeometry, k: int, standoff: float = 0.0) -> Waypoint:
    p = geom.bolt_center(k) + standoff * geom.normal
    return Waypoint(position=p, approach=-geom.normal, theta=geom.bolt_phi(k), z=0.0, cell=(k, 0), standoff=standoff)


class LeakBridgeMixin:
    """Leak methods for the Isaac bridge. The host class must provide:
        stage (Usd.Stage), leak_world (LeakWorld), leak_prims (dict from load_flanges),
        robot_position() -> (3,), move_base_to(xyz, facing) -> bool (one control step toward it; True when there),
        set_camera_pose(eye, target) (wrist camera), advance(dt), work_position() -> (3,),
        arm.move_to(waypoints) -> list[bool]   (the existing ArmController)
    The host's advance(dt) should call leak_advance(dt), so gas builds up and clears in sim time.
    """

    gas_stop: float = 0.20
    approach_distance: float = 1.1     # fits the whole flange in the 640x360 wrist camera (measured; 1.0 m clips)
    gas_override: Optional[float] = None   # video hook / test: a fixed reading; float("nan") = failed sensor
    gas_sensor_offset: tuple = (0.0, 0.0, 0.0)   # sensor position relative to robot_position(), world frame

    def gas_sensor_position(self) -> np.ndarray:
        return np.asarray(self.robot_position(), float) + np.asarray(self.gas_sensor_offset, float)

    def gas_reading(self) -> Optional[float]:
        """LeakWorld gas field at the sensor (fraction of LEL). None or NaN means a failed sensor: the job then
        retreats (fail safe), as in the mock."""
        if self.gas_override is not None:
            return self.gas_override
        return float(self.leak_world.gas_at(self.gas_sensor_position()))

    def leak_advance(self, dt: float) -> None:
        """Step the gas model by dt and push bolt angles / cracks to the stage. Call from the host's advance()."""
        self.leak_world.step(dt)
        apply_leak_state(self.stage, self.leak_world, self.leak_prims)

    def flanges(self) -> list[FlangeGeometry]:
        return [s.geom for s in self.leak_world.flanges.values()]

    def flange(self, fid: str) -> FlangeGeometry:
        return self.leak_world.flanges[fid].geom

    def nearest_flange(self) -> Optional[str]:
        """Id of the flange nearest the robot, or None if there are none."""
        p = np.asarray(self.robot_position(), float)
        return min(self.leak_world.flanges, key=lambda f: float(np.linalg.norm(self.flange(f).center - p)),
                   default=None)

    def approach_flange(self, fid: str, max_steps: int = 400, dt: float = 0.1) -> bool:
        g = self.flange(fid)
        target = g.center + self.approach_distance * g.normal
        for _ in range(max_steps):
            gas = self.gas_reading()
            if gas is None or gas >= self.gas_stop:
                return False
            if self.move_base_to(target, -g.normal):
                eye, look = g.view_poses(self.approach_distance)[0]
                self.set_camera_pose(eye, look)
                self._leak_views = g.view_poses(self.approach_distance)
                return True
            self.advance(dt)
        return False

    def request_flange_view(self, index: int) -> bool:
        views = getattr(self, "_leak_views", None)
        if not views or index >= len(views):
            return False
        self.set_camera_pose(*views[index])
        return True

    def retreat(self, distance: float = 3.0) -> None:
        """Walk ``distance`` m straight away from the nearest flange (horizontal), as the mock jumps there."""
        fid = self.nearest_flange()
        p = np.asarray(self.robot_position(), float)
        d = p - self.flange(fid).center if fid else np.array([-1.0, 0.0, 0.0])
        d[2] = 0.0
        target = p + d / (np.linalg.norm(d) or 1.0) * distance
        for _ in range(400):
            if self.move_base_to(target, d):
                break
            self.advance(0.1)

    def return_to_work(self) -> None:
        """Walk back to work_position() (the rust work pose or the current patrol stop)."""
        target = np.asarray(self.work_position(), float)
        for _ in range(400):
            if self.move_base_to(target, None):
                break
            self.advance(0.1)

    def torque_bolt(self, fid: str, bolt: int) -> bool:
        g = self.flange(fid)
        reached = self.arm.move_to([bolt_waypoint(g, bolt, 0.05), bolt_waypoint(g, bolt, 0.0)])
        if not all(reached):
            return False
        self.leak_world.tighten(fid, bolt)
        apply_leak_state(self.stage, self.leak_world, self.leak_prims)
        self.arm.move_to([bolt_waypoint(g, bolt, 0.05)])
        return True

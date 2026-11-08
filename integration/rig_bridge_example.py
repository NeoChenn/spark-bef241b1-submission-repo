"""How the team's Isaac bridge can take on our two jobs (gas leak + deck sweep): an example, not their bridge.

The team's bridge (SimBridge + Perception + ArmController inside the Isaac process, see INTERFACE.md) does not
exist yet. When it does, it mixes in our two halves:

    class RigIsaacBridge(LeakBridgeMixin, DeckBridgeMixin, TeamIsaacBridge): ...

and provides the few base methods below. This file shows that composition with the simplest possible base: the
robot is teleported along the deck by writing its USD transform, and ``step_fn`` advances the simulator. Every
method marked INTEGRATION SEAM is one the team replaces with the real thing (articulation pose, locomotion
policy, arm IK, their simulation loop). Everything else is ours and is tested on CPU (tests/test_deck_bridge.py).

No Isaac imports here: the only Isaac call (omni.replicator, for camera frames) is inside
DeckBridgeMixin.grab_camera.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from integration.deck_bridge import DeckBridgeMixin
from integration.leak_bridge import LeakBridgeMixin, apply_leak_state


class RigBridgeExample(LeakBridgeMixin, DeckBridgeMixin):
    """Leak + deck halves over a minimal base.

    What the host must provide (the integration seam):
      stage                       the open Usd.Stage (Isaac: omni.usd.get_context().get_stage())
      deck_camera_path,           camera prims: body camera looking down the deck ahead, body camera
      body_camera_path            looking ahead (may be the same prim), wrist camera for the flanges
      wrist_camera_path
      robot_position() / robot_yaw()        base pose in the world
      move_base_to(xyz, facing) -> bool     one control step toward a deck position; True when there
      set_camera_pose(eye, target)          point the wrist camera (Isaac: arm IK; here: move a free camera)
      advance(dt)                           step physics + render; must call leak_advance(dt)
      work_position(), arm (ArmController with move_to), pick_skill(target_xy_base, yaw) -> bool
    """

    walk_step: float = 0.1          # m per move_base_to call (0.1 s control step at 1 m/s)
    arrive_tol: float = 0.05

    def __init__(self, stage, robot_path: str, step_fn: Callable[[float], None], *, leak_world=None, leak_prims=None,
                 arm: Any = None, deck_z: float = 0.0, deck_camera_path: Optional[str] = None,
                 body_camera_path: Optional[str] = None, wrist_camera_path: Optional[str] = None,
                 pick_skill: Optional[Callable] = None):
        self.stage, self.robot_path, self.step_fn = stage, robot_path, step_fn
        self.leak_world, self.leak_prims, self.arm = leak_world, leak_prims or {}, arm
        self.deck_z = float(deck_z)
        if deck_camera_path:
            self.deck_camera_path = deck_camera_path
        if body_camera_path:
            self.body_camera_path = body_camera_path
        self.wrist_camera_path = wrist_camera_path
        if pick_skill is not None:
            self.pick_skill = pick_skill
        self.t = 0.0
        self.patrol_stop = 0
        self._work = self.robot_position().copy()
        if leak_world is not None:
            apply_leak_state(stage, leak_world, self.leak_prims)

    # ---------------------------------------------------------------- INTEGRATION SEAM: base pose
    def _base_op(self):
        from pxr import UsdGeom
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.robot_path))
        ops = xf.GetOrderedXformOps()
        if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
            return ops[0]
        xf.ClearXformOpOrder()
        return xf.AddTransformOp()

    def _base_matrix(self) -> np.ndarray:
        from pxr import Usd, UsdGeom
        return np.array(UsdGeom.Xformable(self.stage.GetPrimAtPath(self.robot_path))
                        .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    def set_base_pose(self, xyz, yaw: float) -> None:
        """Teleport the base (fine for the demo, INTERFACE.md section 2). Assumes the robot prim's parent has no
        transform of its own. Isaac Lab articulation: replace with robot.write_root_pose_to_sim(...); a USD
        write does not move a simulated articulation."""
        from pxr import Gf
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, z = (float(v) for v in xyz)
        self._base_op().Set(Gf.Matrix4d(c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, x, y, z, 1))

    def robot_position(self) -> np.ndarray:
        return self._base_matrix()[3, :3].copy()

    def robot_yaw(self) -> float:
        M = self._base_matrix()
        return float(math.atan2(M[0, 1], M[0, 0]))       # row-vector convention: row 0 = base x axis

    def move_base_to(self, xyz, facing) -> bool:
        """One walking step toward ``xyz`` on the deck (base height kept), turning to ``facing`` (or keeping the
        heading if None). Isaac: replace with the locomotion policy's velocity command."""
        p = self.robot_position()
        d = np.asarray(xyz, float)[:2] - p[:2]
        n = float(np.linalg.norm(d))
        yaw = self.robot_yaw() if facing is None else math.atan2(float(facing[1]), float(facing[0]))
        if n < self.arrive_tol:
            self.set_base_pose(p, yaw)
            return True
        p[:2] += d / n * min(self.walk_step, n)
        self.set_base_pose(p, yaw)
        return n <= self.walk_step

    # ---------------------------------------------------------------- INTEGRATION SEAM: arm camera + time
    def set_camera_pose(self, eye, target) -> None:
        """Point the wrist camera. Isaac: drive the arm with diff-IK so the wrist camera is at eye; here a free
        camera prim is moved, which is what the demo video can use too."""
        if not self.wrist_camera_path:
            return
        from pxr import Gf, UsdGeom
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.wrist_camera_path))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(view.GetInverse())

    def advance(self, dt: float) -> None:
        """Isaac: step_fn runs the simulation loop for dt (physics + render, as rig/sim/main.py does)."""
        self.step_fn(dt)
        self.t += dt
        if self.leak_world is not None:
            self.leak_advance(dt)

    def sim_time(self) -> float:
        return self.t

    def work_position(self) -> np.ndarray:
        return self._work

    def go_to_patrol_pose(self, pose) -> bool:
        """``pose``: (index, (x, y, z)) here; the team's patrol pose type will differ. Sets patrol_stop so the
        deck references are kept per stop."""
        index, xyz = pose
        self.patrol_stop = int(index)
        ok = self._walk_to(xyz, None, max_steps=2000)
        self._work = self.robot_position().copy()
        return ok

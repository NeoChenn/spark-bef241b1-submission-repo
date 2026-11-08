from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from integration import repair_scene as rs
from integration.kinematics import ARM_STOWED, LIFT_JOINT, RobotModel, pose
from integration.lab_bridge import LEGS, LabRepairBridge, smooth
from integration.loop_plan import PipeCyl, brush_fk, ik_error, plan_reposition, plan_staged, retract_target
from integration.loop_timeline import MissionTimeline, mission_phase
from integration.walk_obs import leg_targets, speed_profile, walk_obs

LEG_JOINTS = [f"{l}_{j}_joint" for l in LEGS for j in ("hip", "thigh", "calf")]
CLEAR_GOAL = 0.08


class LoopBridge(LabRepairBridge):
    def __init__(self, *a, walk_net=None, walk_speed: float = 0.5, paint_steps: int = 3, **kw):
        self.sub: Optional[str] = None
        self.timeline = MissionTimeline()
        self.motions: list[dict] = []
        self.reposition_log: list[dict] = []
        self.paint_misses: list[dict] = []
        super().__init__(*a, **kw)
        r = self.base._robot
        self.walk_net, self.walk_speed, self.paint_steps = walk_net, walk_speed, paint_steps
        self.walk_ids = r.find_joints(LEG_JOINTS, preserve_order=True)[0]
        self.walk_default = r.data.default_joint_pos[0, self.walk_ids].cpu().numpy()
        self.walk_last = np.zeros(12, np.float32)
        self.arm_names = [r.joint_names[i] for i in self.arm_ids]
        self.model = RobotModel.from_urdf()
        b = self.base
        self.pipe_cyl = PipeCyl(origin=tuple(b.pipe_o[0].cpu().numpy()), axis=tuple(b.pipe_a[0].cpu().numpy()),
                                radius=float(b.pipe_r[0]), half_length=2.0)
        self.fk_err = None

    def phase(self) -> str:
        return mission_phase(self.state_fn(), self.sub)

    def _base_env(self) -> np.ndarray:
        return (self.base._robot.data.root_pos_w[0].cpu().numpy() - self.origin).astype(float)

    def _fk(self):
        p = self._base_env()
        return brush_fk(self.model, pose(float(p[0]), float(p[1]), float(p[2])))

    def _q(self) -> dict:
        r = self.base._robot
        q = {n: float(v) for n, v in zip(self.arm_names, r.data.joint_pos[0, self.arm_ids].cpu().numpy())}
        if self.lift_ids:
            q[LIFT_JOINT] = float(r.data.joint_pos[0, self.lift_ids[0]])
        return q

    def check_fk(self) -> float:
        p = self._fk()(self._q())
        self.fk_err = float(np.linalg.norm(p - (self._brush() - self.origin)))
        self.log(f"   FK check: model brush vs sim brush {100 * self.fk_err:.1f} cm")
        return self.fk_err

    def _step(self, act=None):
        n0 = len(self.force_log)
        super()._step(act)
        ph = self.phase()
        if self.timeline.mark(self.t, ph):
            self.pending_events.append({"t": round(self.t, 3), "type": "phase", "phase": ph})
        for row in self.force_log[n0:]:
            row["phase"] = ph
            row["motion"] = self.sub

    def snapshot(self) -> dict:
        s = super().snapshot()
        s["phase"] = self.phase()
        s["base"] = self._base_env().tolist()
        return s

    def _with_sub(self, sub, fn, *a, **kw):
        prev, self.sub = self.sub, sub
        t0, f0 = self.t, len(self.force_log)
        try:
            return fn(*a, **kw)
        finally:
            seg = [r["force_n"] for r in self.force_log[f0:]] or [0.0]
            self.motions.append({"motion": sub, "t0": round(t0, 3), "t1": round(self.t, 3), "peak_n": round(max(seg), 3)})
            self.sub = prev

    def _walk_leg_step(self, vx: float):
        r = self.base._robot
        d = r.data
        jp = (d.joint_pos[0, self.walk_ids].cpu().numpy() - self.walk_default)
        jv = d.joint_vel[0, self.walk_ids].cpu().numpy()
        obs = walk_obs((vx, 0.0, 0.0), (vx, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), jp, jv, self.walk_last,
                       float(d.root_pos_w[0, 2]), 0.0)
        a = np.clip(self.walk_net(obs)[0], -4.0, 4.0)
        self.walk_last = a.astype(np.float32)
        tgt = torch.as_tensor(leg_targets(self.walk_default, a), device=self.dev, dtype=torch.float32)[None]
        r.set_joint_position_target(tgt, joint_ids=self.walk_ids)

    def _policy_walk(self, target):
        r = self.base._robot
        p = r.data.root_pos_w[:1].clone()
        d = (target - p)[0, :2]
        dist = float(d.norm())
        u = d / max(dist, 1e-6)
        s = 0.0
        while s < dist - 0.01:
            vx = speed_profile(s, dist, vmax=self.walk_speed)
            if vx <= 0:
                break
            if self.walk_net is not None:
                self._walk_leg_step(vx)
            s = min(dist, s + vx * self.dt)
            q = p.clone()
            q[0, :2] += u * s
            self._set_root(q)
            self._step()
        self._set_root(target)

    def _stance(self, steps: int = 24):
        r = self.base._robot
        q0 = r.data.joint_pos[:1, self.walk_ids].clone()
        q1 = torch.as_tensor(self.walk_default, device=self.dev, dtype=torch.float32)[None]
        for k in range(1, steps + 1):
            r.set_joint_position_target(q0 + (q1 - q0) * smooth(k / steps), joint_ids=self.walk_ids)
            self._step()
        self.walk_last[:] = 0.0

    def go_to_patrol_pose(self, p) -> bool:
        if not self.at_station:
            self._with_sub("WALK", self._policy_walk, self.station)
            self._with_sub("STANCE", self._stance)
            self.at_station = True
            self.check_fk()
        return True

    def _run_stages(self, stages, steps_arm: int, steps_lift: int, stop_force=None) -> bool:
        q_prev = self._q()
        for st in stages:
            arm = torch.tensor([[st[n] for n in self.arm_names]], device=self.dev)
            lift = st.get(LIFT_JOINT)
            moved_lift = lift is not None and abs(lift - q_prev.get(LIFT_JOINT, lift)) > 1e-3
            steps = steps_lift if moved_lift else steps_arm
            ok = self._interp_joints(arm, steps=steps, stop_force=stop_force,
                                     lift=None if lift is None else torch.tensor([[lift]], device=self.dev))
            for _ in range(3):
                self._step()
            if not ok:
                return False
            q_prev = st
        return True

    def _retract(self, goal: float = CLEAR_GOAL):
        b = self.base
        p = self._brush() - self.origin
        tgt = retract_target(p, self.pipe_cyl, self.brush_r, goal)
        if np.allclose(tgt, p):
            return
        self.joint_target = None
        b.p_tgt[0] = b.brush_w[0]
        self._goto(tgt + self.origin, 12)

    def _park(self):
        if self.parked:
            return
        self.press = None
        self._with_sub("PARK", self._safe_park)

    def _safe_park(self):
        if self.joint_target is None:
            self._retract()
        q_to = {n: float(ARM_STOWED[n]) for n in self.arm_names}
        q_to[LIFT_JOINT] = 0.0
        stages, c = plan_staged(self._fk(), self._q(), q_to, self.pipe_cyl, self.brush_r)
        self.log(f"   park: {len(stages)} stage(s), planned clearance {100 * c:.1f} cm")
        self._run_stages(stages, 30, 36)
        self.joint_target = self.q_stow.clone()
        self.parked = True
        self.force_now = 0.0

    def _unpark(self):
        if not self.parked:
            return
        self._with_sub("UNPARK", self._safe_unpark)

    def _safe_unpark(self):
        b = self.base
        q_to = {n: float(v) for n, v in zip(self.arm_names, self.q_work[0].cpu().numpy())}
        q_to[LIFT_JOINT] = float(self.lift_work)
        stages, c = plan_staged(self._fk(), self._q(), q_to, self.pipe_cyl, self.brush_r)
        self.log(f"   unpark: {len(stages)} stage(s), planned clearance {100 * c:.1f} cm")
        clear = self._run_stages(stages, 36, 36, stop_force=5.0)
        if not clear:
            self.joint_target = b._robot.data.joint_pos[:1, self.arm_ids].clone()
            self.log(f"   unpark: brush touched the pipe ({self.force_peak:.0f} N) -> lift off in task space")
        for _ in range(4):
            self._step()
        b.p_tgt[:] = b.brush_w
        self.joint_target = None
        if not clear:
            self._retract()
        self.parked = False
        self.chunk_peak = 0.0

    def _goto_wp(self, wp, steps: int) -> float:
        centre = np.asarray(wp.position, float) - self.brush_r * np.asarray(wp.approach, float)
        return self._goto(centre, steps)

    def move_to(self, waypoints) -> list[bool]:
        if self.state_fn() != "PAINT":
            return super().move_to(waypoints)
        self._unpark()
        errs = []
        for k, wp in enumerate(waypoints):
            rs.set_spray(self.spray_cone, self.spray_op, self._brush(), np.asarray(wp.approach, float), wp.standoff + 0.02,
                         True)
            errs.append(self._goto_wp(wp, 24 if k == 0 else self.paint_steps))
        rs.set_spray(self.spray_cone, self.spray_op, np.zeros(3), np.array([1.0, 0, 0]), 0.1, False)
        miss = [k for k, e in enumerate(errs) if e >= 0.03]
        for k in miss:
            self.paint_misses.append({"t": round(self.t, 2), "err": round(errs[k], 4),
                                      "target": [round(float(v), 4) for v in np.asarray(waypoints[k].position) - self.origin]})
        if miss:
            self._with_sub("REPOSITION", self._reposition, [waypoints[k] for k in miss])
            for k in miss:
                errs[k] = self._goto_wp(waypoints[k], 12)
        return [e < 0.03 for e in errs]

    def _reposition(self, wps):
        b = self.base
        targets = [np.asarray(w.position, float) - self.brush_r * np.asarray(w.approach, float) - self.origin for w in wps]
        base0, q0 = self._base_env(), self._q()
        lift0 = q0.get(LIFT_JOINT, 0.42)

        def err(dx, lift, tgt):
            return ik_error(self.model, pose(base0[0] + dx, base0[1], base0[2]), {**q0, LIFT_JOINT: lift}, tgt,
                            self.arm_names)

        plan = plan_reposition(err, targets, lift0=lift0, lifts=(lift0, min(0.45, lift0 + 0.03), lift0 - 0.04))
        plan["t"], plan["sim_misses"] = round(self.t, 2), len(wps)
        self.reposition_log.append(plan)
        self.log(f"   REPOSITION: {len(wps)} paint waypoints missed -> base dx {plan['dx']:+.2f} m, lift "
                 f"{plan['lift']:.2f} m (IK reaches {plan['reached']}/{plan['targets']}, before {plan['before']})")
        self.record_decision({"t": round(self.t, 3), "from": "PAINT", "to": "REPOSITION",
                              "reason": f"{len(wps)} paint waypoints out of reach -> base dx {plan['dx']:+.2f} m, "
                                        f"lift {plan['lift']:.2f} m", "evidence": plan})
        self._lift_off(0.06)
        if abs(plan["dx"]) > 1e-3 or abs(plan["lift"] - lift0) > 1e-3:
            r = b._robot
            p0 = r.data.root_pos_w[:1].clone()
            p1 = p0.clone()
            p1[0, 0] += plan["dx"]
            n = 30
            l0 = r.data.joint_pos[:1, self.lift_ids].clone() if self.lift_ids else None
            for k in range(1, n + 1):
                s = smooth(k / n)
                if self.walk_net is not None and abs(plan["dx"]) > 1e-3:
                    self._walk_leg_step(math.copysign(0.3, plan["dx"]))
                self._set_root(p0 + (p1 - p0) * s)
                if l0 is not None:
                    r.set_joint_position_target(l0 + (plan["lift"] - l0) * s, joint_ids=self.lift_ids)
                self._step()
            self._stance(12)
        for _ in range(6):
            self._step()


def summary_motions(motions: list[dict]) -> dict:
    out = {}
    for m in motions:
        k = m["motion"]
        o = out.setdefault(k, {"count": 0, "peak_n": 0.0, "time_s": 0.0})
        o["count"] += 1
        o["peak_n"] = max(o["peak_n"], m["peak_n"])
        o["time_s"] = round(o["time_s"] + m["t1"] - m["t0"], 3)
    return out


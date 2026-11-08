from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from integration import make_fixer as mf
from integration.kinematics import ARM_JOINTS, STANDING, TRUNK_HEIGHT, IKResult, RobotModel
from policy.leak.geometry import FlangeGeometry

FIXER_URDF = mf.DST
DECK_Z = 42.5
LIFT_JOINT = "lift_joint"
TOOL_LINK = mf.TOOL_LINK
SOCKET_JOINT = mf.SOCKET_JOINT
FINGER_JOINTS = mf.FINGER_JOINTS
RETORQUE_LIFT = mf.RETORQUE_LIFT
FIXER_HOME = {"ur3_shoulder_pan_joint": 0.0, "ur3_shoulder_lift_joint": -1.57, "ur3_elbow_joint": 1.57,
              "ur3_wrist_1_joint": 0.0, "ur3_wrist_2_joint": -1.57, "ur3_wrist_3_joint": 0.0}
PHASES = ("approach", "align", "engage", "torque", "disengage", "retract")


class Unreachable(RuntimeError):
    pass


@dataclass
class RetorqueConfig:
    torque_angle: float = math.radians(30.0)
    lift: float = RETORQUE_LIFT
    standoff: float = 0.12
    align_gap: float = 0.02
    engage_depth: float = 0.008
    max_joint_speed: float = 1.0
    max_lift_speed: float = 0.1
    max_socket_speed: float = 1.0
    min_segment: float = 0.4
    dt: float = 0.02
    cart_step: float = 0.004
    pos_tol: float = 0.003
    axis_tol: float = math.radians(3.0)
    sniff_gap: float = 0.05
    sniff_tilts: tuple = tuple(math.radians(a) for a in (0, 15, 30, 45, 60))


@dataclass
class RetorqueTrajectory:
    t: np.ndarray
    q: np.ndarray
    phase: list[str]
    joint_names: list[str]
    flange_id: str
    bolt: int
    bolt_center: np.ndarray
    bolt_axis: np.ndarray
    torque_angle: float

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, t: float) -> dict:
        return {n: float(np.interp(t, self.t, self.q[:, i])) for i, n in enumerate(self.joint_names)}

    def phase_span(self, name: str) -> tuple[float, float]:
        idx = [i for i, p in enumerate(self.phase) if p == name]
        return float(self.t[idx[0]]), float(self.t[idx[-1]])

    def end_of(self, name: str) -> dict:
        i = max(i for i, p in enumerate(self.phase) if p == name)
        return dict(zip(self.joint_names, map(float, self.q[i])))

    def waypoints(self) -> list[dict]:
        return [{"t": round(float(t), 4), "phase": p, "q": {n: float(v) for n, v in zip(self.joint_names, q)}}
                for t, p, q in zip(self.t, self.phase, self.q)]


@dataclass
class VerifyPose:
    q: dict
    tip: np.ndarray
    axis: np.ndarray
    target: np.ndarray
    tilt: float
    ik: IKResult


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def work_pose(flange: FlangeGeometry, deck_z: float = DECK_Z, standoff: float = mf.RETORQUE_STANDOFF,
              height: float = TRUNK_HEIGHT, side_on: bool = False) -> np.ndarray:
    return mf.stand_pose(flange.center, flange.normal, standoff, deck_z + height, side_on)


@dataclass
class RetorqueSkill:
    model: RobotModel | None = None
    urdf: str | Path = FIXER_URDF
    arm_joints: list[str] = field(default_factory=lambda: list(ARM_JOINTS))
    lift_joint: str = LIFT_JOINT
    socket_joint: str = SOCKET_JOINT
    finger_joints: tuple = FINGER_JOINTS
    finger_closed: float = mf.FINGER_CLOSED_ON_SOCKET
    tool_link: str = TOOL_LINK
    home: dict = field(default_factory=lambda: dict(FIXER_HOME))
    seeds: tuple = mf.IK_SEEDS
    config: RetorqueConfig = field(default_factory=RetorqueConfig)

    def __post_init__(self):
        if self.model is None:
            self.model = RobotModel.from_urdf(self.urdf)

    @property
    def joint_names(self) -> list[str]:
        return [self.lift_joint, *self.arm_joints, self.socket_joint, *self.finger_joints]

    def full_q(self, q: dict | None = None) -> dict:
        out = {n: 0.0 for n, j in self.model.joints.items() if j.kind != "fixed"}
        out.update(STANDING)
        out.update(self.home)
        out[self.lift_joint] = self.config.lift
        out.update({j: self.finger_closed for j in self.finger_joints})
        out.update(q or {})
        return out

    def tip_pose(self, base: np.ndarray, q: dict) -> np.ndarray:
        return self.model.tool_pose(base, self.full_q(q), self.tool_link)

    def tool_points(self, base: np.ndarray, q: dict) -> np.ndarray:
        poses = self.model.link_poses(base, self.full_q(q))
        chain = self.model.chain(self.tool_link)
        tip = poses[self.tool_link][:3, 3]
        grip = poses[chain[-4].child][:3, 3]
        links = [poses[j.child][:3, 3] for j in chain if j.name in self.arm_joints[3:]]
        return np.array(links + [grip + s * (tip - grip) for s in np.linspace(0, 1, 6)])

    def _ok(self, r: IKResult) -> bool:
        return r.position_error < self.config.pos_tol and r.axis_error < self.config.axis_tol

    def _solve(self, base, target, axis, seed: dict) -> IKResult:
        return self.model.solve_ik(base, self.full_q(seed), target, axis, self.arm_joints, iterations=150,
                                   damping=0.02, tip=self.tool_link)

    def solve(self, base: np.ndarray, target, axis, seed: dict | None = None) -> IKResult:
        seed = self.full_q(seed)
        tried = [self._solve(base, target, axis, seed)]
        if self._ok(tried[0]):
            return tried[0]
        for s in self.seeds:
            for pan in (0.0, math.pi / 2, -math.pi / 2):
                q = dict(seed, **dict(zip(self.arm_joints, s)))
                q[self.arm_joints[0]] += pan
                r = self._solve(base, target, axis, q)
                if self._ok(r):
                    return r
                tried.append(r)
        return min(tried, key=lambda r: r.position_error + 0.1 * r.axis_error)

    def _vec(self, q: dict) -> np.ndarray:
        full = self.full_q(q)
        return np.array([full[n] for n in self.joint_names])

    def _arm(self, v: np.ndarray) -> dict:
        return dict(zip(self.joint_names, map(float, v)))

    def _cartesian(self, base, start, end, axis, seed: np.ndarray, what: str) -> list[np.ndarray]:
        n = max(2, int(math.ceil(np.linalg.norm(end - start) / self.config.cart_step)) + 1)
        v, out = seed.copy(), []
        for s in np.linspace(0.0, 1.0, n):
            r = self._solve(base, start + s * (end - start), axis, self._arm(v))
            if not self._ok(r):
                raise Unreachable(f"{what}: IK off by {r.position_error * 1000:.1f} mm, "
                                  f"{math.degrees(r.axis_error):.1f} deg")
            v = v.copy()
            for i, name in enumerate(self.joint_names):
                if name in self.arm_joints:
                    v[i] = r.q[name]
            out.append(v)
        return out

    def _timed(self, path: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        P = np.asarray(path)
        limits = np.array([cfg.max_lift_speed if n == self.lift_joint else
                           cfg.max_socket_speed if n == self.socket_joint else cfg.max_joint_speed
                           for n in self.joint_names])
        step = (np.abs(np.diff(P, axis=0)) / limits).max(axis=1)
        length = float(step.sum())
        if length > 1e-9:
            s = np.concatenate([[0.0], np.cumsum(step) / length])
            keep = np.concatenate([[True], np.diff(s) > 1e-12])
            P, s = P[keep], s[keep]
        else:
            P, s = P[[0, -1]], np.array([0.0, 1.0])
        T = max(cfg.min_segment, 1.875 * length)
        n = int(math.ceil(T / cfg.dt))
        tt = np.arange(1, n + 1) * cfg.dt
        ss = min_jerk(tt / tt[-1])
        Q = np.stack([np.interp(ss, s, P[:, i]) for i in range(P.shape[1])], axis=1)
        return tt, Q

    def plan(self, flange: FlangeGeometry, bolt: int, base: np.ndarray, q_start: dict | None = None,
             torque_angle: float | None = None, seed: dict | None = None) -> RetorqueTrajectory:
        cfg = self.config
        angle = cfg.torque_angle if torque_angle is None else torque_angle
        if not 0 <= bolt < flange.n_bolts:
            raise ValueError(f"bolt {bolt} not on {flange.id} ({flange.n_bolts} bolts)")
        axis = -flange.normal
        head = flange.bolt_center(bolt)
        p_stand = head + cfg.standoff * flange.normal
        p_align = head + cfg.align_gap * flange.normal
        p_engage = head - min(cfg.engage_depth, flange.head_height) * flange.normal
        v0 = self._vec(q_start)
        k_lift, k_sock = self.joint_names.index(self.lift_joint), self.joint_names.index(self.socket_joint)
        work = v0.copy()
        work[k_lift] = cfg.lift

        r = self.solve(base, p_engage, axis, dict(self._arm(work), **(seed or {})))
        if not self._ok(r):
            raise Unreachable(f"{flange.id} bolt {bolt} out of reach at lift {cfg.lift:.2f} m: "
                              f"{r.position_error * 1000:.0f} mm, {math.degrees(r.axis_error):.1f} deg off")
        en = self._vec(dict(self._arm(work), **{n: r.q[n] for n in self.arm_joints}))
        inward = self._cartesian(base, p_engage, p_align, axis, en, "engage")[::-1]
        align = self._cartesian(base, p_align, p_stand, axis, inward[0], "align")[::-1]
        turned = inward[-1].copy()
        turned[k_sock] += angle
        disengage = self._cartesian(base, p_engage, p_align, axis, turned, "disengage")
        back = self._cartesian(base, p_align, p_stand, axis, disengage[-1], "retract")
        home = v0.copy()
        home[k_sock] = turned[k_sock]

        segments = [("approach", [v0, align[0]]), ("align", align), ("engage", inward),
                    ("torque", [inward[-1], turned]), ("disengage", disengage),
                    ("retract", back), ("retract", [back[-1], home])]
        ts, qs, phases, t0 = [0.0], [v0], ["approach"], 0.0
        for name, path in segments:
            tt, Q = self._timed(path)
            ts.extend(t0 + tt)
            qs.extend(Q)
            phases.extend([name] * len(tt))
            t0 += tt[-1]
        return RetorqueTrajectory(np.asarray(ts), np.asarray(qs), phases, self.joint_names, flange.id, bolt,
                                  head, axis, angle)

    def plan_bolts(self, flange: FlangeGeometry, bolts, base: np.ndarray, q_start: dict | None = None,
                   torque_angle: float | None = None) -> list[RetorqueTrajectory]:
        out, seed = [], None
        for k in bolts:
            tr = self.plan(flange, k, base, q_start, torque_angle, seed)
            seed = {n: v for n, v in tr.end_of("engage").items() if n in self.arm_joints}
            out.append(tr)
        return out

    def verify_pose(self, flange: FlangeGeometry, base: np.ndarray, bolt: int | None = None,
                    q_start: dict | None = None, gap: float | None = None) -> VerifyPose:
        cfg = self.config
        target = flange.center + (cfg.sniff_gap if gap is None else gap) * flange.normal
        if bolt is not None:
            target = target + flange.bolt_circle * flange.radial(bolt)
        down = np.array([0.0, 0.0, -1.0])
        side = down - (down @ flange.normal) * flange.normal
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else flange.ref
        best = None
        for tilt in cfg.sniff_tilts:
            axis = -math.cos(tilt) * flange.normal - math.sin(tilt) * side
            r = self.solve(base, target, axis, q_start)
            if best is None or r.position_error < best[0].position_error:
                best = (r, tilt)
            if self._ok(r):
                break
        r, tilt = best
        if not self._ok(r):
            raise Unreachable(f"sniff point at {flange.id} out of reach: {r.position_error * 1000:.0f} mm off")
        q = {n: float(r.q[n]) for n in self.joint_names}
        T = self.tip_pose(base, q)
        return VerifyPose(q, T[:3, 3], T[:3, 0], target, float(tilt), r)


def plan_retorque(center, axis, bolt_circle: float, bolt: int, base: np.ndarray, ref=(0.0, 0.0, 1.0),
                  n_bolts: int = 8, flange_id: str = "F", q_start: dict | None = None,
                  skill: RetorqueSkill | None = None, **config) -> RetorqueTrajectory:
    flange = FlangeGeometry(flange_id, center, axis, ref, bolt_circle=bolt_circle, n_bolts=n_bolts)
    skill = skill or RetorqueSkill(config=RetorqueConfig(**config))
    return skill.plan(flange, bolt, base, q_start)

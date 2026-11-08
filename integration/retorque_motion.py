from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from integration.kinematics import ARM_JOINTS
from policy.leak.geometry import FlangeGeometry
from policy.skills.retorque import LIFT_JOINT, SOCKET_JOINT, TOOL_LINK, RetorqueSkill, Unreachable, min_jerk, work_pose

REPO = Path(__file__).resolve().parent.parent
LEAK_META = REPO / "sim/output/leak_flanges_meta.json"
JOINTS = [LIFT_JOINT, *ARM_JOINTS, SOCKET_JOINT]
STOW = {LIFT_JOINT: 0.12, "ur3_shoulder_pan_joint": 0.0, "ur3_shoulder_lift_joint": -1.45, "ur3_elbow_joint": 2.55,
        "ur3_wrist_1_joint": -1.1, "ur3_wrist_2_joint": 1.52, "ur3_wrist_3_joint": 0.0, SOCKET_JOINT: 0.0}
PHASES = ("stow", "approach", "align", "engage", "torque", "disengage", "retract", "home")


@dataclass
class MotionConfig:
    lift: float = 0.40
    pre_m: float = 0.12
    align_gap_m: float = 0.02
    engage_m: float = 0.008
    turn_steps: int = 3
    turn_deg: float = 40.0
    approach_s: float = 2.6
    align_s: float = 1.2
    engage_s: float = 0.8
    turn_s: float = 0.7
    hold_s: float = 0.35
    disengage_s: float = 0.7
    retract_s: float = 1.0
    home_s: float = 2.2
    cart_step_m: float = 0.004
    dt: float = 0.02
    gas_tau_s: float = 2.0


def flange_from_meta(fid: str = "F5", path=LEAK_META) -> FlangeGeometry:
    meta = json.loads(Path(path).read_text())
    return FlangeGeometry.from_meta(next(f for f in meta["flanges"] if f["id"] == fid))


def _near(v: float, ref: float) -> float:
    return v + 2 * math.pi * round((ref - v) / (2 * math.pi))


class RetorqueMotion:
    def __init__(self, flange: Optional[FlangeGeometry] = None, bolt: int = 3, base: Optional[np.ndarray] = None,
                 t0: float = 0.0, config: Optional[MotionConfig] = None, skill: Optional[RetorqueSkill] = None,
                 stow: Optional[dict] = None, start_q: Optional[dict] = None):
        self.cfg = config or MotionConfig()
        self.flange = flange or flange_from_meta()
        self.bolt = int(bolt)
        self.base = work_pose(self.flange) if base is None else np.asarray(base, float)
        self.t0 = float(t0)
        self.skill = skill or RetorqueSkill()
        self.model = self.skill.model
        self.stow = dict(STOW, **(stow or {}))
        self.start_q = None if start_q is None else {k: float(start_q.get(k, v)) for k, v in self.stow.items()}
        self.axis = -self.flange.normal
        self.head = self.flange.bolt_center(self.bolt)
        self._build()

    def _full(self, v: np.ndarray) -> dict:
        return self.skill.full_q(dict(zip(JOINTS, map(float, v))))

    def _vec(self, q: dict) -> np.ndarray:
        return np.array([float(q[n]) for n in JOINTS])

    def _ik(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        r = self.model.solve_ik(self.base, self._full(seed), target, self.axis, ARM_JOINTS, iterations=200,
                                damping=0.02, tip=TOOL_LINK)
        if r.position_error > 0.002 or r.axis_error > math.radians(1.0):
            raise Unreachable(f"{self.flange.id} bolt {self.bolt}: IK off by {r.position_error * 1000:.1f} mm, "
                              f"{math.degrees(r.axis_error):.2f} deg")
        out = seed.copy()
        for i, n in enumerate(JOINTS):
            if n in ARM_JOINTS:
                out[i] = _near(float(r.q[n]), float(seed[i]))
        return out

    def _line(self, a: np.ndarray, b: np.ndarray, seed: np.ndarray) -> list[np.ndarray]:
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / self.cfg.cart_step_m)) + 1)
        out, v = [], seed
        for s in np.linspace(0.0, 1.0, n):
            v = self._ik(a + s * (b - a), v)
            out.append(v)
        return out

    def _pre(self, p: np.ndarray) -> np.ndarray:
        seed = self._vec(self.stow)
        seed[0] = self.cfg.lift
        best = None
        for start in (seed, self._vec(dict(self.stow, **{"ur3_shoulder_lift_joint": -0.6, "ur3_elbow_joint": 1.6,
                                                         "ur3_wrist_1_joint": -1.0}))):
            start = start.copy()
            start[0] = self.cfg.lift
            try:
                v = self._ik(p, start)
            except Unreachable:
                continue
            if self.elbow_up(v) and (best is None or np.abs(v - seed).sum() < np.abs(best - seed).sum()):
                best = v
        if best is None:
            raise Unreachable(f"{self.flange.id} bolt {self.bolt}: no elbow-up solution at the pre-align point")
        return best

    def _build(self) -> None:
        c = self.cfg
        n = self.flange.normal
        p_pre = self.head + c.pre_m * n
        p_align = self.head + c.align_gap_m * n
        p_engage = self.head - min(c.engage_m, self.flange.head_height) * n
        stow = self._vec(self.stow)
        first = stow if self.start_q is None else self._vec(self.start_q)
        pre = self._pre(p_pre)
        for i, nm in enumerate(JOINTS):
            if nm in ARM_JOINTS:
                pre[i] = _near(pre[i], first[i])
        align = self._line(p_pre, p_align, pre)
        engage = self._line(p_align, p_engage, align[-1])
        k = JOINTS.index(SOCKET_JOINT)
        segs = [("approach", [first, pre], c.approach_s), ("align", align, c.align_s),
                ("engage", engage, c.engage_s)]
        v = engage[-1]
        step = math.radians(c.turn_deg)
        for i in range(c.turn_steps):
            w = v.copy()
            w[k] += step
            segs += [("torque", [v, w], c.turn_s), ("torque", [w, w], c.hold_s)]
            v = w
        back = self._line(p_engage, p_align, v)
        out = self._line(p_align, p_pre, back[-1])
        home = stow.copy()
        home[k] = v[k]
        segs += [("disengage", back, c.disengage_s), ("retract", out, c.retract_s), ("home", [out[-1], home], c.home_s)]
        ts, qs, phases, t = [0.0], [first], ["approach"], 0.0
        self.spans: dict[str, list[float]] = {}
        for name, path, dur in segs:
            P = np.asarray(path)
            seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            s_of = seg / seg[-1] if seg[-1] > 1e-12 else np.linspace(0.0, 1.0, len(P))
            m = max(1, int(round(dur / c.dt)))
            tt = np.arange(1, m + 1) * (dur / m)
            ss = min_jerk(tt / dur)
            Q = np.stack([np.interp(ss, s_of, P[:, j]) for j in range(P.shape[1])], 1)
            span = self.spans.setdefault(name, [t, t])
            span[1] = t + dur
            ts.extend(t + tt)
            qs.extend(Q)
            phases.extend([name] * m)
            t += dur
        self.t = np.asarray(ts)
        self.q = np.asarray(qs)
        self.phases = phases
        self.duration = float(t)
        self.t_torque_done = self.spans["torque"][1] - c.hold_s
        self.engage_q = engage[-1]
        self.pre_q = pre

    @property
    def t_fix(self) -> float:
        return self.t0 + self.t_torque_done

    @property
    def t_end(self) -> float:
        return self.t0 + self.duration

    def _local(self, t: float) -> float:
        return float(np.clip(t - self.t0, 0.0, self.duration))

    def vector(self, t: float) -> np.ndarray:
        s = self._local(t)
        return np.array([np.interp(s, self.t, self.q[:, j]) for j in range(len(JOINTS))])

    def at(self, t: float) -> dict:
        return dict(zip(JOINTS, map(float, self.vector(t))))

    def phase(self, t: float) -> str:
        if t < self.t0:
            return "stow"
        if t >= self.t_end:
            return "done"
        return self.phases[min(int(np.searchsorted(self.t, t - self.t0)), len(self.phases) - 1)]

    def socket_angle(self, t: float) -> float:
        return float(self.vector(t)[JOINTS.index(SOCKET_JOINT)])

    def tool_pose(self, t_or_q) -> np.ndarray:
        v = t_or_q if isinstance(t_or_q, np.ndarray) else self.vector(float(t_or_q))
        return self.model.tool_pose(self.base, self._full(v), TOOL_LINK)

    def elbow_up(self, v: np.ndarray) -> bool:
        P = self.model.link_poses(self.base, self._full(v))
        sh, el, wr = (P[k][:3, 3] for k in ("ur3_upper_arm_link", "ur3_forearm_link", "ur3_wrist_1_link"))
        d = wr - sh
        s = float(np.clip((el - sh) @ d / max(float(d @ d), 1e-9), 0.0, 1.0))
        return float(el[2] - (sh + s * d)[2]) > 0.0

    def coaxial_error_deg(self) -> float:
        T = self.tool_pose(self.engage_q)
        return math.degrees(math.acos(float(np.clip(T[:3, 0] @ self.axis, -1.0, 1.0))))

    def gas_scale(self, t: float) -> float:
        if t <= self.t_fix:
            return 1.0
        return float(math.exp(-(t - self.t_fix) / self.cfg.gas_tau_s))

    def rows(self, fps: float = 30.0, start: Optional[float] = None, end: Optional[float] = None) -> list[dict]:
        a = self.t0 if start is None else start
        b = self.t_end if end is None else end
        out = []
        for i in range(int(math.floor((b - a) * fps + 1e-9)) + 1):
            t = a + i / fps
            T = self.tool_pose(t)
            out.append({"t": round(t, 4), "phase": self.phase(t), "q": {k: round(v, 5) for k, v in self.at(t).items()},
                        "socket_tip": [round(float(x), 4) for x in T[:3, 3]], "gas_scale": round(self.gas_scale(t), 4)})
        return out

    def summary(self) -> dict:
        b = self.base
        return {"flange": self.flange.id, "bolt": self.bolt, "head": [round(float(x), 4) for x in self.head],
                "base": [round(float(b[0, 3]), 4), round(float(b[1, 3]), 4), round(math.atan2(b[1, 0], b[0, 0]), 4)],
                "t0": self.t0, "duration": round(self.duration, 3), "t_fix": round(self.t_fix, 3),
                "spans": {k: [round(self.t0 + v[0], 3), round(self.t0 + v[1], 3)] for k, v in self.spans.items()},
                "coaxial_deg": round(self.coaxial_error_deg(), 3), "gas_tau_s": self.cfg.gas_tau_s, "joints": JOINTS}


def fit(t_engaged: float, t_torque_done: float, t_leave: Optional[float] = None, flange=None, bolt: int = 3,
        config: Optional[MotionConfig] = None, **kw) -> RetorqueMotion:
    c = config or MotionConfig()
    span = t_torque_done - t_engaged
    n = c.turn_steps
    hold = min(c.hold_s, 0.4 * span / max(n - 1, 1)) if n > 1 else 0.0
    turn = (span - (n - 1) * hold) / n
    if turn < 0.3:
        raise ValueError(f"torque span {span:.2f} s too short for {n} visible turns")
    c = replace(c, turn_s=turn, hold_s=hold)
    if t_leave is not None:
        room = t_leave - t_torque_done - hold - c.disengage_s - c.retract_s - 0.05
        c = replace(c, home_s=max(1.0, min(c.home_s, room)))
    t0 = t_engaged - (c.approach_s + c.align_s + c.engage_s)
    return RetorqueMotion(flange, bolt, t0=t0, config=c, **kw)


def timeline_marks(path, robot: str = "unit2") -> dict:
    marks: dict = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "torque_done" and r.get("robot", robot) == robot:
            marks.setdefault("torque_done", float(r["t"]))
            marks.setdefault("bolt", int(r.get("bolt", 3)))
            marks.setdefault("flange", r.get("flange", "F5"))
        if r.get("type") == "tick":
            st = r.get("robots", {}).get(robot, {}).get("state")
            if st == "RETORQUE":
                marks.setdefault("engaged", float(r["t"]))
            if st == "RETURN" and "torque_done" in marks:
                marks.setdefault("leave", float(r["t"]))
    return marks


def from_timeline(path, robot: str = "unit2", **kw) -> RetorqueMotion:
    m = timeline_marks(path, robot)
    return fit(m["engaged"], m["torque_done"], m.get("leave"), flange_from_meta(m.get("flange", "F5")),
               m.get("bolt", 3), **kw)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flange", default="F5")
    ap.add_argument("--bolt", type=int, default=3)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--timeline", default="", help="fit to a recorded run: engage at RETORQUE, turn ends at torque_done")
    ap.add_argument("--out", default="runs/retorque_motion.json")
    a = ap.parse_args(argv)
    m = from_timeline(a.timeline) if a.timeline else RetorqueMotion(flange_from_meta(a.flange), a.bolt, t0=a.t0)
    data = {**m.summary(), "rows": m.rows(a.fps)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, separators=(",", ":")))
    print(json.dumps(m.summary()))
    return data


if __name__ == "__main__":
    main()

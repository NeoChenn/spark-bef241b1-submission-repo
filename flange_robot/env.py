"""Skeleton MuJoCo environment: a platform deck, one bolted flange, a point-mass robot.

STATUS: skeleton for the team to extend. It is NOT the final world and has no reward
or training loop yet. It exists so perception, safety and the console have something
real to run against from the first hour.

Bolt "torque stripe" misalignment is the angle of each bolt's hinge joint, in radians
(0 means the stripe is aligned). The robot only observes it through a noisy sensor
whose confidence drops with distance and with `dust`.

Live scene-change hooks for the unscripted change in the video:
- trigger_gas_leak()    loosens a bolt and starts a leak; the gas reading rises
- loosen_bolt(i, rad)   a bolt turns, so its stripe misaligns (and a leak grows if active)
- trigger_crack(sev)    a crack or burst in the pipe: it leaks, the camera sees it, and
                        tightening a bolt does NOT fix it
- set_dust(level)       dust degrades sensor confidence (0 to 1)

All noise and gas values are ILLUSTRATIVE.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

N_BOLTS = 8
FLANGE_XY = (2.0, 0.0)
BOLT_RING_RADIUS = 0.4
SENSOR_RANGE_M = 2.5
REACH_M = 0.6

# Gas model (ILLUSTRATIVE). The leak strength follows how loose the bolts are, so
# tightening a bolt really does reduce the reading and verification means something.
GAS_BASE = 0.02          # background reading, fraction of LEL
LEAK_COEFF = 0.4         # peak extra reading at the flange for a full-strength leak
LEAK_FULL_RAD = 0.6      # total bolt misalignment (rad) that gives a full-strength leak
LEAK_RELAX = 0.2         # per-step relaxation of the leak level (gas takes time to clear)
GAS_NOISE = 0.004        # sensor noise on the reading

# Simulated camera feature for a crack or burst (ILLUSTRATIVE). This is a stand-in for a real
# image model: a crack-likeness score plus a confidence that drops with distance and dust.
CRACK_SCORE_BASE = 0.55  # score for the faintest visible crack
CRACK_SCORE_GAIN = 0.45  # added in proportion to severity


def _build_xml() -> str:
    bolts = []
    for i in range(N_BOLTS):
        a = 2 * np.pi * i / N_BOLTS
        x, y = BOLT_RING_RADIUS * np.cos(a), BOLT_RING_RADIUS * np.sin(a)
        bolts.append(
            f'<body name="bolt{i}" pos="{x:.4f} {y:.4f} 0.08">'
            f'<joint name="bolt{i}" type="hinge" axis="0 0 1" range="-1 1" damping="1"/>'
            f'<geom type="cylinder" size="0.06 0.04" rgba="0.8 0.7 0.2 1"/>'
            f'<geom type="box" pos="0.045 0 0.04" size="0.02 0.008 0.002" rgba="1 0 0 1"/>'
            f"</body>"
        )
    return f"""
<mujoco model="flange_platform">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="deck" type="plane" size="6 6 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="robot" pos="-3 0 0.15">
      <joint name="rx" type="slide" axis="1 0 0" range="-5.5 5.5" damping="20"/>
      <joint name="ry" type="slide" axis="0 1 0" range="-5.5 5.5" damping="20"/>
      <geom type="box" size="0.2 0.2 0.1" mass="10" rgba="0.2 0.5 0.9 1"/>
    </body>
    <body name="flange" pos="{FLANGE_XY[0]} {FLANGE_XY[1]} 0.4">
      <geom type="cylinder" size="0.5 0.05" rgba="0.6 0.6 0.6 1"/>
      {''.join(bolts)}
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="rx" kv="40"/>
    <velocity name="vy" joint="ry" kv="40"/>
  </actuator>
</mujoco>
"""


@dataclass
class Action:
    kind: str  # "move_to", "move_xy", "retorque", "retreat" or "wait"
    bolt: int | None = None
    xy: tuple[float, float] | None = None  # target for "move_xy"


class FlangeEnv:
    def __init__(self, substeps: int = 10, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_string(_build_xml())
        self.data = mujoco.MjData(self.model)
        self.substeps = substeps
        self.rng = np.random.default_rng(seed)
        self._bolt_body = [self.model.body(f"bolt{i}").id for i in range(N_BOLTS)]
        self._bolt_qpos = [self.model.joint(f"bolt{i}").qposadr[0] for i in range(N_BOLTS)]
        self._bolt_dof = [self.model.joint(f"bolt{i}").dofadr[0] for i in range(N_BOLTS)]
        self.reset(seed)

    # -- scene control -------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._cam_rng = np.random.default_rng(seed + 10_000)  # separate stream: camera noise
        mujoco.mj_resetData(self.model, self.data)
        self.leak_active = False
        self.crack_severity = 0.0
        self._leak_level = 0.0
        self.dust = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def trigger_gas_leak(self, bolt: int = 0, radians: float = 0.5) -> None:
        """Start a leak by loosening one bolt. A larger angle gives a stronger leak."""
        self.loosen_bolt(bolt, radians)
        self.leak_active = True

    def trigger_crack(self, severity: float = 0.3) -> None:
        """A crack or burst at the flange (0 to 1). It adds to the leak and cannot be fixed by tightening."""
        self.crack_severity = float(np.clip(severity, 0.0, 1.0))
        self.leak_active = True

    def loosen_bolt(self, i: int, radians: float) -> None:
        self.data.qpos[self._bolt_qpos[i]] = radians
        mujoco.mj_forward(self.model, self.data)

    def set_dust(self, level: float) -> None:
        self.dust = float(np.clip(level, 0.0, 1.0))

    # -- state helpers -------------------------------------------------------
    @property
    def robot_xy(self) -> np.ndarray:
        return np.array([self.data.qpos[self.model.joint("rx").qposadr[0]] - 3.0,
                         self.data.qpos[self.model.joint("ry").qposadr[0]]])

    def bolt_world_xy(self, i: int) -> np.ndarray:
        return self.data.xpos[self._bolt_body[i]][:2].copy()

    def true_bolt_angle(self, i: int) -> float:
        return float(self.data.qpos[self._bolt_qpos[i]])

    def _leak_target(self) -> float:
        """Leak strength (0 to 1) implied by the current bolt looseness."""
        if not self.leak_active:
            return 0.0
        total = sum(abs(self.true_bolt_angle(i)) for i in range(N_BOLTS))
        return float(np.clip(total / LEAK_FULL_RAD + self.crack_severity, 0.0, 1.0))

    def gas_fraction_lel(self) -> float:
        """True gas at the robot (no sensor noise): background plus a decaying leak plume."""
        d = float(np.linalg.norm(self.robot_xy - np.array(FLANGE_XY)))
        return GAS_BASE + LEAK_COEFF * self._leak_level * float(np.exp(-d / 1.5))

    # -- perception ----------------------------------------------------------
    def observe(self) -> dict:
        """What the robot's sensors report. Ground truth is never exposed here."""
        pos = self.robot_xy
        seen = []
        for i in range(N_BOLTS):
            d = float(np.linalg.norm(self.bolt_world_xy(i) - pos))
            if d > SENSOR_RANGE_M:
                continue
            confidence = float(np.clip(1.0 - d / SENSOR_RANGE_M - 0.6 * self.dust, 0.0, 1.0))
            noise = float(self.rng.normal(0.0, 0.02 + 0.15 * (1.0 - confidence)))
            seen.append({"bolt": i, "stripe_misalignment_rad": self.true_bolt_angle(i) + noise,
                         "confidence": confidence, "distance_m": d})
        gas = self.gas_fraction_lel() + float(self.rng.normal(0.0, GAS_NOISE))

        # Simulated camera: crack-likeness of the flange region. Never exposes the true severity.
        d_flange = float(np.linalg.norm(pos - np.array(FLANGE_XY)))
        if d_flange <= SENSOR_RANGE_M:
            cam_conf = float(np.clip(1.0 - d_flange / SENSOR_RANGE_M - 0.6 * self.dust, 0.0, 1.0))
            true_score = CRACK_SCORE_BASE + CRACK_SCORE_GAIN * self.crack_severity if self.crack_severity > 0 else 0.0
            noise = float(self._cam_rng.normal(0.0, 0.05 + 0.2 * (1.0 - cam_conf)))
            camera = {"crack_score": float(np.clip(true_score + noise, 0.0, 1.0)), "crack_confidence": cam_conf}
        else:
            camera = {"crack_score": 0.0, "crack_confidence": 0.0}
        return {"robot_xy": pos.tolist(), "gas_fraction_lel": gas, "bolts": seen, "camera": camera}

    # -- dynamics ------------------------------------------------------------
    def step(self, action: Action) -> dict:
        target = None
        if action.kind == "move_to" and action.bolt is not None:
            target = self.bolt_world_xy(action.bolt) - np.array([0.5, 0.0])  # stand near, not on top
        elif action.kind == "move_xy" and action.xy is not None:
            target = np.array(action.xy, dtype=float)
        elif action.kind == "retreat":
            target = np.array([-5.0, 0.0])

        retorqued = False
        if action.kind == "retorque" and action.bolt is not None:
            if np.linalg.norm(self.bolt_world_xy(action.bolt) - self.robot_xy) <= REACH_M:
                self.data.qpos[self._bolt_qpos[action.bolt]] = 0.0
                self.data.qvel[self._bolt_dof[action.bolt]] = 0.0
                retorqued = True

        for _ in range(self.substeps):
            if target is None:
                self.data.ctrl[:] = 0.0
            else:
                delta = target - self.robot_xy
                dist = float(np.linalg.norm(delta))
                self.data.ctrl[:2] = 0.0 if dist < 0.05 else 1.0 * delta / dist
            mujoco.mj_step(self.model, self.data)
        self._leak_level += LEAK_RELAX * (self._leak_target() - self._leak_level)
        obs = self.observe()
        obs["retorqued"] = retorqued
        return obs

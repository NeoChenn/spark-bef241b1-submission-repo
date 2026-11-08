from __future__ import annotations

import numpy as np

from policy.weld.env import V_NOM


class SeamTracker:
    def __init__(self, kp: float = 0.6, kd: float = 0.15, hi_target: float = 1.0, i_nom: float = 165.0,
                 t_soft: float = 180.0, eta: float = 0.8):
        self.kp, self.kd, self.hi_target, self.i_nom, self.t_soft, self.eta = kp, kd, hi_target, i_nom, t_soft, eta
        self.bx: list[float] = []
        self.bc: list[float] = []
        self.prev = None
        self.target = None

    def lateral(self, obs: dict) -> float:
        c = obs["center_est"]
        if c is not None and (not self.bx or obs["x_look"] > self.bx[-1]):
            self.bx.append(obs["x_look"])
            self.bc.append(c)
        if not self.bx:
            return 0.0
        x = obs["x"]
        k = max(0, np.searchsorted(self.bx, x) - 3)
        if x >= self.bx[0]:
            self.target = float(np.median(np.interp(x + np.array([-1.0, 0.0, 1.0]), self.bx[k:], self.bc[k:])))
        else:
            self.target = float(np.median(self.bc[:5]))
        err = self.target - obs["y"]
        d = 0.0 if self.prev is None else err - self.prev
        self.prev = err
        return self.kp * err + self.kd * d

    def act(self, obs: dict):
        dy = self.lateral(obs)
        i = self.i_nom * (1 - float(np.clip((obs["temp_c"] - self.t_soft) / 60.0, 0, 0.4)))
        tele = obs["tele"]
        v_arc = tele["voltage_v"] if tele["voltage_v"] > 5 else V_NOM
        i_meas = tele["current_a"] if tele["current_a"] > 20 else i
        v = self.eta * v_arc * i_meas / (self.hi_target * 1000.0)
        return dy, v, i


class OpenLoop:
    def __init__(self, i_nom: float = 165.0, speed: float = 2.9):
        self.i_nom, self.speed = i_nom, speed

    def act(self, obs: dict):
        return 0.0, self.speed, self.i_nom

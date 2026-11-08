from __future__ import annotations

import math
from dataclasses import dataclass

R = 8.314462618
P_ATM = 101325.0
T_STD = 288.15
T_NORM = 273.15
M_AIR = 0.0289647
G = 9.80665
GASES = {"CH4": (0.016043, 1.304, 0.05), "H2S": (0.034081, 1.32, 0.04)}


@dataclass(frozen=True)
class Gas:
    molar_mass: float
    gamma: float
    lel: float

    @classmethod
    def mix(cls, h2s: float = 0.0) -> "Gas":
        parts = [("CH4", 1.0 - h2s), ("H2S", h2s)]
        m = sum(x * GASES[g][0] for g, x in parts)
        cp = sum(x * GASES[g][1] * R / (GASES[g][1] - 1.0) for g, x in parts)
        lel = 1.0 / sum(x / GASES[g][2] for g, x in parts if x > 0)
        return cls(m, cp / (cp - R), lel)

    def density(self, p: float = P_ATM, t: float = T_STD) -> float:
        return p * self.molar_mass / (R * t)

    @property
    def buoyancy(self) -> float:
        return 1.0 - self.molar_mass / M_AIR


@dataclass(frozen=True)
class Release:
    mass_flow: float
    choked: bool
    exit_velocity: float
    exit_pressure: float
    momentum: float
    gas: Gas

    @property
    def volume_flow(self) -> float:
        return self.mass_flow / self.gas.density(P_ATM, T_STD)

    @property
    def sm3_h(self) -> float:
        return self.volume_flow * 3600.0

    @property
    def nm3_h(self) -> float:
        return self.mass_flow / self.gas.density(P_ATM, T_NORM) * 3600.0

    def to_json(self) -> dict:
        return {"kg_s": round(self.mass_flow, 6), "sm3_h": round(self.sm3_h, 3), "nm3_h": round(self.nm3_h, 3),
                "choked": self.choked, "exit_velocity_m_s": round(self.exit_velocity, 1),
                "momentum_n": round(self.momentum, 3)}


def critical_ratio(gamma: float) -> float:
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def orifice(hole_mm: float, bar_g: float, t_k: float = T_STD, cd: float = 0.62, h2s: float = 0.0,
            p_amb: float = P_ATM) -> Release:
    gas = Gas.mix(h2s)
    g, m = gas.gamma, gas.molar_mass
    area = cd * math.pi * (hole_mm * 1e-3) ** 2 / 4.0
    p0 = bar_g * 1e5 + p_amb
    if p0 <= p_amb or area <= 0:
        return Release(0.0, False, 0.0, p_amb, 0.0, gas)
    rc = critical_ratio(g)
    if p_amb / p0 <= rc:
        mdot = area * p0 * math.sqrt(g * m / (R * t_k) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
        te = t_k * 2.0 / (g + 1.0)
        pe = p0 * rc
        ue = math.sqrt(g * R * te / m)
        return Release(mdot, True, ue, pe, mdot * ue + (pe - p_amb) * area, gas)
    r = p_amb / p0
    mdot = area * p0 * math.sqrt(2.0 * m / (R * t_k) * g / (g - 1.0) * (r ** (2.0 / g) - r ** ((g + 1.0) / g)))
    te = t_k * r ** ((g - 1.0) / g)
    ue = mdot / (gas.density(p_amb, te) * area)
    return Release(mdot, False, ue, p_amb, mdot * ue, gas)

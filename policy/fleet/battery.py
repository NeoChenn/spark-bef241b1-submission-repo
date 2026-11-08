from __future__ import annotations

from dataclasses import dataclass

POWER_W = {"idle": 40.0, "walk": 180.0, "stairs": 300.0, "work": 240.0}


@dataclass
class Battery:
    capacity_wh: float = 432.0
    wh: float = 432.0
    reserve_frac: float = 0.15
    return_frac: float = 0.3
    resume_frac: float = 0.8
    charger_w: float = 1500.0
    used_wh: float = 0.0
    charged_wh: float = 0.0

    @property
    def soc(self) -> float:
        return self.wh / self.capacity_wh

    @property
    def reserve_wh(self) -> float:
        return self.reserve_frac * self.capacity_wh

    def drain(self, mode: str, dt: float) -> None:
        e = min(self.wh, POWER_W[mode] * dt / 3600.0)
        self.wh -= e
        self.used_wh += e

    def charge(self, dt: float) -> None:
        e = min(self.capacity_wh - self.wh, self.charger_w * dt / 3600.0)
        self.wh += e
        self.charged_wh += e

    def can_afford(self, wh_needed: float) -> bool:
        return self.wh - wh_needed >= self.reserve_wh

    def must_return(self, wh_to_dock: float) -> bool:
        return self.wh - wh_to_dock < self.reserve_wh

    @staticmethod
    def path_energy(t_total: float, t_stairs: float) -> float:
        return (POWER_W["walk"] * (t_total - t_stairs) + POWER_W["stairs"] * t_stairs) / 3600.0

    @staticmethod
    def work_energy(t_work: float) -> float:
        return POWER_W["work"] * t_work / 3600.0

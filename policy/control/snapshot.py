from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

PPM_PER_LEL_PCT = 500.0


def _pick(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names}


@dataclass
class Robot:
    id: str
    state: str = ""
    caption: str = ""
    pose: Optional[tuple] = None
    battery: Optional[float] = None
    task: Optional[str] = None
    zone: str = ""
    rust_conf: Optional[float] = None
    rust_pct: Optional[float] = None
    force_n: Optional[float] = None
    person_m: Optional[float] = None
    sensors: list = field(default_factory=list)
    deck: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Robot":
        r = cls(**_pick(cls, d))
        if r.pose is not None:
            r.pose = tuple(float(v) for v in r.pose)
        r.sensors = list(r.sensors or [])
        return r


@dataclass
class GasSensor:
    id: str
    zone: str = ""
    lel: float = 0.0
    acoustic_db: Optional[float] = None
    lel_trend: float = 0.0
    db_trend: float = 0.0
    group: str = ""
    pos: Optional[tuple] = None

    @property
    def ppm(self) -> float:
        return self.lel * PPM_PER_LEL_PCT

    @classmethod
    def from_dict(cls, d: dict) -> "GasSensor":
        return cls(**_pick(cls, d))


@dataclass
class FixedSensor:
    id: str
    kind: str
    value: Optional[float] = None
    zone: str = ""
    group: str = ""
    unit: str = ""
    pos: Optional[tuple] = None

    @classmethod
    def from_dict(cls, d: dict) -> "FixedSensor":
        return cls(**_pick(cls, d))


@dataclass
class SensorHealth:
    id: str
    kind: str
    status: str
    zone: str = ""
    since: float = 0.0
    evidence: dict = field(default_factory=dict)
    hazard: bool = False
    candidates: list = field(default_factory=list)
    job: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SensorHealth":
        return cls(**_pick(cls, d))


@dataclass
class Verdict:
    robot: str
    verdict: str
    confidence: float
    reason: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(**_pick(cls, d))


@dataclass
class SensorSnapshot:
    t: float = 0.0
    robots: list[Robot] = field(default_factory=list)
    gas: list[GasSensor] = field(default_factory=list)
    plume_kg_h: Optional[float] = None
    plume_source: Optional[str] = None
    verdicts: list[Verdict] = field(default_factory=list)
    storm_level: float = 0.0
    wind_deg: Optional[float] = None
    jobs: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    fixed: list[FixedSensor] = field(default_factory=list)
    health: list[SensorHealth] = field(default_factory=list)

    def faults(self) -> list[SensorHealth]:
        return [h for h in self.health if h.status != "OK"]

    def sensor_ids(self) -> set[str]:
        return {g.id for g in self.gas} | {f.id for f in self.fixed} | {h.id for h in self.health}

    def robot(self, rid: str) -> Optional[Robot]:
        return next((r for r in self.robots if r.id == rid), None)

    def max_lel(self) -> float:
        return max((g.lel for g in self.gas), default=0.0)

    def zones(self) -> set[str]:
        return {r.zone for r in self.robots if r.zone} | {g.zone for g in self.gas if g.zone} | \
               {j.get("zone") for j in self.jobs if j.get("zone")} | {f.zone for f in self.fixed if f.zone}

    def to_dict(self, images: bool = True) -> dict:
        d = asdict(self)
        for r in d["robots"]:
            if r["pose"] is not None:
                r["pose"] = list(r["pose"])
        for g, s in zip(d["gas"], self.gas):
            g["ppm"] = round(s.ppm, 1)
            if g["pos"] is not None:
                g["pos"] = list(g["pos"])
        for f in d["fixed"]:
            if f["pos"] is not None:
                f["pos"] = list(f["pos"])
        if not images:
            d["images"] = len(self.images)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensorSnapshot":
        s = cls(**_pick(cls, d))
        s.robots = [Robot.from_dict(r) for r in d.get("robots") or []]
        s.gas = [GasSensor.from_dict(g) for g in d.get("gas") or []]
        s.verdicts = [Verdict.from_dict(v) for v in d.get("verdicts") or []]
        s.jobs = [dict(j) for j in d.get("jobs") or []]
        s.events = [dict(e) for e in d.get("events") or []]
        s.images = list(d.get("images") or [])
        s.sources = list(d.get("sources") or [])
        s.fixed = [FixedSensor.from_dict(f) for f in d.get("fixed") or []]
        s.health = [SensorHealth.from_dict(h) for h in d.get("health") or []]
        return s

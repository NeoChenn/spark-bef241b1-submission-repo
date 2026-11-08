from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LEL_PPM = 44000.0
NATURAL_GAS = {"ethane": 0.06, "propane": 0.02}
IR_CROSS = {"ethane": 0.35, "propane": 0.9}
CAT_CROSS = {"ethane": 0.8, "propane": 0.65}


def gain(cross: dict, composition: dict, ch4_frac: float = 0.9) -> float:
    return 1.0 + sum(cross.get(k, 0.0) * f / ch4_frac for k, f in composition.items())


def lag(y: float, x: float, dt: float, t90: float) -> float:
    return y + (1.0 - math.exp(-dt * math.log(10.0) / t90)) * (x - y)


@dataclass
class PointGasSensor:
    name: str
    kind: str = "ir"
    t90: float = 6.0
    noise_ppm: float = 60.0
    drift_ppm_rt_s: float = 1.5
    full_scale_ppm: float = LEL_PPM
    cross: dict = field(default_factory=lambda: dict(IR_CROSS))
    composition: dict = field(default_factory=lambda: dict(NATURAL_GAS))
    poison_dose: float = 5000.0
    seed: int = 0
    y: float = 0.0
    drift: float = 0.0
    dose: float = 0.0
    reading: float = 0.0
    saturated: bool = False

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    @property
    def sensitivity(self) -> float:
        g = gain(self.cross, self.composition)
        return g * math.exp(-self.dose / self.poison_dose) if self.kind == "catalytic" else g

    def update(self, ppm: float, dt: float, h2s_ppm: float = 0.0) -> float:
        self.dose += max(h2s_ppm, 0.0) * dt / 60.0
        self.y = lag(self.y, max(ppm, 0.0) * self.sensitivity, dt, self.t90)
        self.drift += self.drift_ppm_rt_s * math.sqrt(dt) * self.rng.normal()
        r = self.y + self.drift + self.noise_ppm * self.rng.normal()
        self.saturated = r >= self.full_scale_ppm
        self.reading = float(np.clip(r, 0.0, self.full_scale_ppm))
        return self.reading

    @property
    def lel(self) -> float:
        return self.reading / LEL_PPM


@dataclass
class H2SCell:
    name: str = "h2s_ec"
    t90: float = 25.0
    noise_ppm: float = 0.25
    full_scale_ppm: float = 100.0
    seed: int = 0
    y: float = 0.0
    reading: float = 0.0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def update(self, h2s_ppm: float, dt: float) -> float:
        self.y = lag(self.y, max(h2s_ppm, 0.0), dt, self.t90)
        self.reading = float(np.clip(self.y + self.noise_ppm * self.rng.normal(), 0.0, self.full_scale_ppm))
        return self.reading


@dataclass
class AcousticReading:
    db: float
    background_db: float
    snr_db: float
    bearing: float | None
    bearing_sigma: float


def spl(l1m_db: float, r: float, absorb_db_m: float) -> float:
    r = max(r, 0.3)
    return l1m_db - 20 * math.log10(r) - absorb_db_m * r


@dataclass
class AcousticSensor:
    l1m_db: float = 78.0
    background_db: float = 58.0
    noise_db: float = 1.2
    absorb_db_m: float = 1.0
    min_snr_db: float = 3.0
    multipath: float = 0.08
    machinery: tuple = ()
    machinery_err_db: float = 0.0
    seed: int = 0
    last: AcousticReading | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def level(self, r: float, q: float) -> float:
        return spl(self.l1m_db + 10 * math.log10(max(q, 1e-9)), r, self.absorb_db_m)

    def expected_db(self, pos) -> float:
        p = np.asarray(pos, float)[:2]
        e = 10 ** (self.background_db / 10)
        for c, l1 in self.machinery:
            e += 10 ** (spl(l1, float(np.linalg.norm(np.asarray(c)[:2] - p)), self.absorb_db_m) / 10)
        return 10 * math.log10(e)

    def measure(self, pos, leaks) -> AcousticReading:
        p = np.asarray(pos, float)[:2]
        srcs = [(np.asarray(c, float)[:2] - p, l1 + self.machinery_err_db) for c, l1 in self.machinery]
        srcs += [(np.asarray(c, float)[:2] - p, self.l1m_db + 10 * math.log10(q)) for c, q in leaks if q > 0]
        energy = 10 ** (self.background_db / 10)
        best, best_e = None, 0.0
        for d, l1 in srcs:
            e = 10 ** (spl(l1, float(np.linalg.norm(d)), self.absorb_db_m) / 10)
            energy += e
            if e > best_e:
                best, best_e = d, e
        db = 10 * math.log10(energy) + self.noise_db * self.rng.normal()
        exp_db = self.expected_db(p)
        excess = 10 ** (db / 10) - 10 ** (exp_db / 10)
        snr = 10 * math.log10(excess) - exp_db if excess > 0 else -30.0
        bearing, sigma = None, math.pi
        if snr >= self.min_snr_db and best is not None:
            sigma = max(math.radians(6.0), math.radians(70.0) / (1.0 + 10 ** (snr / 10)) ** 0.5)
            if self.rng.random() < self.multipath:
                bearing = float(self.rng.uniform(-math.pi, math.pi))
            else:
                bearing = float(math.atan2(best[1], best[0]) + sigma * self.rng.normal())
        self.last = AcousticReading(round(db, 2), round(exp_db, 2), round(snr, 2), bearing, sigma)
        return self.last


def column_density(plume, origins: np.ndarray, dirs: np.ndarray, max_range: float = 12.0, n: int = 28) -> np.ndarray:
    s = (np.arange(n) + 0.5) / n * max_range
    o = np.asarray(origins, float).reshape(-1, 3)
    d = np.asarray(dirs, float).reshape(-1, 3)
    pts = o[:, None, :] + d[:, None, :] * s[None, :, None]
    ppm = plume.sample(pts.reshape(-1, 3)).reshape(len(d), n) - plume.background
    return np.maximum(ppm, 0.0).sum(1) * (max_range / n)


OGI_RAMP = np.array([[40, 0, 80], [70, 10, 120], [200, 40, 60], [255, 150, 20], [255, 245, 200]], np.float32)


def ogi_colour(column: np.ndarray, threshold: float, top: float) -> tuple[np.ndarray, np.ndarray]:
    u = np.clip((np.log10(np.maximum(column, 1e-9)) - math.log10(threshold)) /
                (math.log10(top) - math.log10(threshold)), 0.0, 1.0)
    x = u * (len(OGI_RAMP) - 1)
    k = np.minimum(x.astype(int), len(OGI_RAMP) - 2)
    f = (x - k)[..., None]
    rgb = OGI_RAMP[k] * (1 - f) + OGI_RAMP[k + 1] * f
    return rgb, column >= threshold


def ogi_view(rgb_bg: np.ndarray, camera, plume, threshold: float = 2000.0, top: float = 60000.0,
             grid: tuple[int, int] = (96, 54), max_range: float = 12.0) -> tuple[np.ndarray, dict]:
    from PIL import Image
    from detection.localize import pixel_rays
    bg = np.asarray(rgb_bg, np.float32)[..., :3]
    H, W = bg.shape[:2]
    gw, gh = grid
    o, d = pixel_rays(camera)
    ys = np.linspace(0, d.shape[0] - 1, gh).astype(int)
    xs = np.linspace(0, d.shape[1] - 1, gw).astype(int)
    oo, dd = np.broadcast_to(o, d.shape)[ys][:, xs], d[ys][:, xs]
    col = column_density(plume, oo.reshape(-1, 3), dd.reshape(-1, 3), max_range).reshape(gh, gw)
    up = np.asarray(Image.fromarray(col.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    grey = bg.mean(-1, keepdims=True)
    grey = (grey - grey.min()) / max(float(np.ptp(grey)), 1.0) * 170 + 40
    colour, mask = ogi_colour(up, threshold, top)
    a = np.where(mask, 0.85, 0.0)[..., None]
    out = grey * (1 - a) + colour * a
    return out.clip(0, 255).astype(np.uint8), {"frac": round(float(mask.mean()), 4),
                                                "peak_ppm_m": round(float(col.max()), 1)}


@dataclass
class SensorSuite:
    seed: int = 0
    h2s_ratio: float = 0.0
    trunk: PointGasSensor = None
    tip: PointGasSensor = None
    h2s: H2SCell = None
    acoustic: AcousticSensor = None
    machinery: tuple = ()
    machinery_err_db: float = 0.0

    def __post_init__(self):
        s = self.seed
        self.trunk = self.trunk or PointGasSensor("trunk_ir", "ir", t90=6.0, noise_ppm=60.0, seed=s + 1)
        self.tip = self.tip or PointGasSensor("tip_cat", "catalytic", t90=9.0, noise_ppm=90.0,
                                              drift_ppm_rt_s=3.0, cross=dict(CAT_CROSS), seed=s + 2)
        self.h2s = self.h2s or H2SCell(seed=s + 3)
        self.acoustic = self.acoustic or AcousticSensor(machinery=self.machinery,
                                                         machinery_err_db=self.machinery_err_db, seed=s + 4)

    def update(self, plume, trunk_pos, tip_pos, dt: float) -> None:
        ppm = plume.sample(np.stack([trunk_pos, tip_pos]))
        h2s = ppm * self.h2s_ratio
        self.trunk.update(float(ppm[0]), dt, float(h2s[0]))
        self.tip.update(float(ppm[1]), dt, float(h2s[1]))
        self.h2s.update(float(h2s[0]), dt)

    def listen(self, pos, leaks) -> AcousticReading:
        return self.acoustic.measure(pos, leaks)

    def lel_max(self) -> float:
        return max(self.trunk.lel, self.tip.lel)

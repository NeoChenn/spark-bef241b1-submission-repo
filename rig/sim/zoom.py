import math
from dataclasses import dataclass


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def interp(x: float, xs, ys) -> float:
    if x <= xs[0]:
        return ys[0]
    for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def spherical(target, d: float, az_deg: float, el_deg: float) -> tuple:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (target[0] + d * math.cos(el) * math.cos(az), target[1] + d * math.cos(el) * math.sin(az),
            target[2] + d * math.sin(el))


def _lerp(a, b, s):
    return tuple(x + (y - x) * s for x, y in zip(a, b))


@dataclass(frozen=True)
class ZoomPath:
    wide_target: tuple
    close_target: tuple
    wide_dist: float = 330.0
    close_dist: float = 1.9
    orbit_s: float = 6.0
    orbit_sweep: float = 50.0
    descend_s: float = 4.0
    hold_s: float = 8.0
    hold_sweep: float = 8.0
    wide_focal: float = 28.0
    close_focal: float = 30.0
    az_keys: tuple = ((1.9, 62.0), (9.0, 62.0), (60.0, 15.0), (330.0, -40.0))
    el_keys: tuple = ((1.9, 26.0), (5.0, 12.0), (12.0, 11.0), (40.0, 16.0), (330.0, 19.0))

    @property
    def period(self) -> float:
        return self.orbit_s + 2 * self.descend_s + self.hold_s

    def _angles(self, d: float) -> tuple:
        ld = math.log(d)
        az = interp(ld, [math.log(k) for k, _ in self.az_keys], [v for _, v in self.az_keys])
        el = interp(ld, [math.log(k) for k, _ in self.el_keys], [v for _, v in self.el_keys])
        return az, el

    def _descent(self, s: float, az_offset: float = 0.0) -> tuple:
        d = math.exp(math.log(self.wide_dist) + (math.log(self.close_dist) - math.log(self.wide_dist)) * s)
        w = (self.wide_dist - d) / (self.wide_dist - self.close_dist)
        target = _lerp(self.wide_target, self.close_target, w)
        az, el = self._angles(d)
        focal = self.wide_focal + (self.close_focal - self.wide_focal) * w
        return spherical(target, d, az + az_offset, el), target, focal

    def at(self, t: float) -> tuple:
        if math.isfinite(self.period):
            t %= self.period
        if t < self.orbit_s:
            u = t / self.orbit_s
            return self._descent(0.0, -self.orbit_sweep * (1 - math.cos(2 * math.pi * u)) / 2)
        t -= self.orbit_s
        if t < self.descend_s:
            return self._descent(ease(t / self.descend_s))
        t -= self.descend_s
        if t < self.hold_s:
            u = t / self.hold_s
            return self._descent(1.0, self.hold_sweep * (1 - math.cos(2 * math.pi * u)) / 2)
        t -= self.hold_s
        return self._descent(ease(1.0 - t / self.descend_s))

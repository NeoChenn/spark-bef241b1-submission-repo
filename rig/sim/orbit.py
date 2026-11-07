import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Rows = tuple[tuple[float, float, float, float], ...]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a: Vec3) -> Vec3:
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / n, a[1] / n, a[2] / n)


def look_at_rows(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0)) -> Rows:
    z = _unit(_sub(eye, target))
    x = _unit(_cross(up, z))
    y = _cross(z, x)
    return ((*x, 0.0), (*y, 0.0), (*z, 0.0), (*eye, 1.0))


def view_target(rows, z: float) -> Vec3:
    eye = tuple(rows[3][:3])
    fwd = tuple(-c for c in rows[2][:3])
    s = (z - eye[2]) / fwd[2]
    return (eye[0] + s * fwd[0], eye[1] + s * fwd[1], z)


@dataclass(frozen=True)
class Orbit:
    target: Vec3
    radius: float
    z: float
    phase: float
    period: float

    @classmethod
    def from_eye(cls, eye: Vec3, target: Vec3, period: float, radius: float = 0.0, z: float | None = None) -> "Orbit":
        dx, dy = eye[0] - target[0], eye[1] - target[1]
        r = radius or math.hypot(dx, dy)
        return cls(target, r, eye[2] if z is None else z, math.atan2(dy, dx), period)


def orbit_eye(orbit: Orbit, t: float) -> Vec3:
    a = orbit.phase + 2 * math.pi * t / orbit.period
    return (orbit.target[0] + orbit.radius * math.cos(a), orbit.target[1] + orbit.radius * math.sin(a), orbit.z)

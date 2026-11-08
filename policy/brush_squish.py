from __future__ import annotations

import numpy as np

F_REF = 40.0
C_MAX = 0.6
EPS = 0.005


def compression(force, f_ref: float = F_REF, c_max: float = C_MAX):
    return c_max * np.clip(np.asarray(force, float) / f_ref, 0.0, 1.0)


def pad_thickness(spring_k: float, f_ref: float = F_REF, c_max: float = C_MAX) -> float:
    return f_ref / (spring_k * c_max)


def revolve(profile, n_seg: int = 40):
    prof = np.asarray(profile, float)
    phi = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    x = np.repeat(prof[:, 0], n_seg)
    rho = np.repeat(np.maximum(prof[:, 1], 1e-4), n_seg)
    ph = np.tile(phi, len(prof))
    pts = np.stack([x, rho * np.cos(ph), rho * np.sin(ph)], -1)
    i = np.arange(len(prof) - 1)[:, None] * n_seg
    j = np.arange(n_seg)[None, :]
    a, b = i + j, i + (j + 1) % n_seg
    quads = np.stack([a, b, b + n_seg, a + n_seg], -1).reshape(-1, 4)
    return pts, quads


def sponge_profile(radius: float, thickness: float, layers: int = 8, rings: int = 6, bevel: float = 0.35):
    rb = radius * (1 - bevel * 0.3)
    back = [(-thickness, radius * k / rings) for k in range(rings)]
    side = [(-thickness + thickness * k / layers, radius) for k in range(layers)]
    edge = [(-thickness * 0.12 * (1 - np.sin(a)), rb + (radius - rb) * np.cos(a))
            for a in np.linspace(0.0, np.pi / 2, 4)]
    face = [(0.0, rb * (1 - k / rings)) for k in range(1, rings + 1)]
    return back + side + edge + face


def hub_profile(hub_radius: float, thickness: float, shaft_radius: float = 0.015, shaft_back: float = 0.12,
                hub_back: float = 0.03):
    return [(-shaft_back, 0.0), (-shaft_back, shaft_radius), (-hub_back, shaft_radius), (-hub_back, hub_radius),
            (-thickness - 0.001, hub_radius), (-thickness - 0.001, 0.0)]


def squish_local(pts: np.ndarray, thickness: float, c: float, bulge: float = 0.5) -> np.ndarray:
    u = np.clip((pts[:, 0] + thickness) / thickness, 0.0, 1.0)
    out = pts.copy()
    out[:, 0] = -thickness + u * thickness * (1 - c)
    swell = 1.0 + bulge * c * (0.6 * np.sin(np.pi * u) + 0.4 * u)
    out[:, 1:] *= swell[:, None]
    return out


def to_world(pts: np.ndarray, pos, rot) -> np.ndarray:
    return pts @ np.asarray(rot, float).T + np.asarray(pos, float)


def conform(pts: np.ndarray, u: np.ndarray, pipe_o, pipe_a, pipe_r: float, contact: bool, eps: float = EPS,
            reach: float = 0.03) -> np.ndarray:
    a = np.asarray(pipe_a, float)
    a = a / np.linalg.norm(a)
    d = pts - np.asarray(pipe_o, float)
    foot = np.asarray(pipe_o, float) + (d @ a)[:, None] * a
    radial = pts - foot
    rho = np.linalg.norm(radial, axis=1)
    n = radial / np.maximum(rho, 1e-9)[:, None]
    gap = rho - (pipe_r + eps)
    if contact:
        w = np.clip((u - 0.5) / 0.5, 0.0, 1.0) ** 1.5 * np.clip(1.0 - gap / reach, 0.0, 1.0)
        gap = gap * (1.0 - w)
    gap = np.maximum(gap, 0.0)
    return foot + (pipe_r + eps + gap)[:, None] * n


def vertex_normals(pts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    p = pts[quads]
    fn = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    nrm = np.zeros_like(pts)
    for k in range(4):
        np.add.at(nrm, quads[:, k], fn)
    return nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)


class BrushPad:

    def __init__(self, spring_k: float = 4000.0, radius: float = 0.036, hub_radius: float = 0.031, n_seg: int = 40,
                 bulge: float = 0.5, eps: float = EPS):
        self.thickness = pad_thickness(spring_k)
        self.bulge, self.eps = bulge, eps
        self.rest, self.quads = revolve(sponge_profile(radius, self.thickness), n_seg)
        self.u = np.clip((self.rest[:, 0] + self.thickness) / self.thickness, 0.0, 1.0)
        self.hub_rest, self.hub_quads = revolve(hub_profile(hub_radius, self.thickness), n_seg)

    def sponge(self, pos, rot, pipe_o, pipe_a, pipe_r, force):
        c = float(compression(force))
        local = squish_local(self.rest, self.thickness, c, self.bulge)
        return conform(to_world(local, pos, rot), self.u, pipe_o, pipe_a, pipe_r, force > 0.0, self.eps)

    def hub(self, pos, rot, pipe_o, pipe_a, pipe_r):
        return conform(to_world(self.hub_rest, pos, rot), np.zeros(len(self.hub_rest)), pipe_o, pipe_a, pipe_r,
                       False, self.eps)


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class UsdBrush:

    def __init__(self, stage, root: str, pad: BrushPad, sponge_mat, hub_mat):
        from pxr import UsdGeom, UsdShade, Vt

        self.pad, self.Vt = pad, Vt
        self.meshes = {}
        for name, quads, mat in (("sponge", pad.quads, sponge_mat), ("hub", pad.hub_quads, hub_mat)):
            mesh = UsdGeom.Mesh.Define(stage, f"{root}/{name}")
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.reshape(-1).astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            self.meshes[name] = (mesh, quads)

    def _set(self, name, pts):
        mesh, quads = self.meshes[name]
        f32 = lambda a: self.Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, np.float32))
        mesh.GetPointsAttr().Set(f32(pts))
        mesh.GetNormalsAttr().Set(f32(vertex_normals(pts, quads)))
        lo, hi = pts.min(0), pts.max(0)
        mesh.GetExtentAttr().Set(f32(np.stack([lo, hi])))

    def update(self, pos, quat, pipe_o, pipe_a, pipe_r, force):
        rot = quat_to_rot(quat)
        self._set("sponge", self.pad.sponge(pos, rot, pipe_o, pipe_a, pipe_r, force))
        self._set("hub", self.pad.hub(pos, rot, pipe_o, pipe_a, pipe_r))

import math

import numpy as np
from PIL import Image, ImageDraw

BG = np.array((14, 17, 22), np.float32)
FREE = np.array((44, 54, 66), np.float32)
OCC = np.array((120, 228, 255), np.float32)
PATH = (217, 119, 87)
ROBOT = (250, 250, 245)
RING = (34, 40, 48)


class Bev:
    def __init__(self, cell: float = 0.2, bounds=(-60.0, -60.0, 60.0, 60.0), decay: float = 0.985):
        self.cell, self.bounds, self.decay = cell, tuple(map(float, bounds)), decay
        n = (int(round((self.bounds[2] - self.bounds[0]) / cell)), int(round((self.bounds[3] - self.bounds[1]) / cell)))
        self.free = np.zeros(n, np.float32)
        self.occ = np.zeros(n, np.float32)
        self.z = None

    def ij(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.bounds[0]) // self.cell), int((y - self.bounds[1]) // self.cell)

    def update(self, points, floor_z: float) -> None:
        if self.z is None or abs(floor_z - self.z) > 1.0:
            self.free[:] = 0
            self.occ[:] = 0
        self.z = floor_z
        self.free *= self.decay
        self.occ *= self.decay
        p = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(p):
            return
        ix = np.floor((p[:, 0] - self.bounds[0]) / self.cell).astype(np.int64)
        iy = np.floor((p[:, 1] - self.bounds[1]) / self.cell).astype(np.int64)
        h = p[:, 2] - floor_z
        ok = (ix >= 0) & (iy >= 0) & (ix < self.free.shape[0]) & (iy < self.free.shape[1])
        floor = ok & (np.abs(h) < 0.2)
        wall = ok & (h >= 0.2) & (h < 2.2)
        np.add.at(self.free, (ix[floor], iy[floor]), 1.0)
        np.add.at(self.occ, (ix[wall], iy[wall]), 1.0)

    def render(self, pose, path=None, size: int = 192, span: float = 12.0, fov: float = 0.0) -> Image.Image:
        x0, y0, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        s = (np.arange(size) + 0.5) * (2 * span / size) - span
        xs, ys = x0 + s, y0 - s
        ix = np.floor((xs - self.bounds[0]) / self.cell).astype(np.int64)
        iy = np.floor((ys - self.bounds[1]) / self.cell).astype(np.int64)
        okx = (ix >= 0) & (ix < self.free.shape[0])
        oky = (iy >= 0) & (iy < self.free.shape[1])
        f = np.zeros((size, size), np.float32)
        o = np.zeros((size, size), np.float32)
        sub = np.ix_(iy[oky].clip(0), ix[okx].clip(0))
        rows, cols = np.nonzero(oky)[0], np.nonzero(okx)[0]
        f[np.ix_(rows, cols)] = np.minimum(self.free.T[sub] / 3.0, 1.0)
        o[np.ix_(rows, cols)] = np.minimum(self.occ.T[sub] / 2.0, 1.0)
        img = BG + f[..., None] * (FREE - BG)
        img = img + o[..., None] * (OCC - img)
        im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        px = size / (2 * span)
        c = size / 2
        for r in range(4, int(span) + 1, 4):
            d.ellipse([c - r * px, c - r * px, c + r * px, c + r * px], outline=RING)
        if fov:
            for a in (yaw - fov / 2, yaw + fov / 2):
                d.line([c, c, c + math.cos(a) * span * px, c - math.sin(a) * span * px], fill=(70, 80, 92))
        if path and len(path) > 1:
            pts = [(c + (p[0] - x0) * px, c - (p[1] - y0) * px) for p in path]
            d.line(pts, fill=PATH, width=max(2, size // 64))
            for q in pts[1:]:
                d.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=PATH)
        k = max(5, size // 24)
        tri = [(c + math.cos(yaw) * k, c - math.sin(yaw) * k),
               (c + math.cos(yaw + 2.5) * k * 0.7, c - math.sin(yaw + 2.5) * k * 0.7),
               (c + math.cos(yaw - 2.5) * k * 0.7, c - math.sin(yaw - 2.5) * k * 0.7)]
        d.polygon(tri, fill=ROBOT)
        return im

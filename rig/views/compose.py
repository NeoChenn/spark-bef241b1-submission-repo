import math

import numpy as np
from PIL import Image

from rig.views.bev import Bev
from rig.camera import project
from rig.views.draw import agc, depth_rgb, draw_boxes, draw_targets, encode, label, sparkline
from rig.views.geom import backproject
from rig.views.ogi import gas_column, ogi_image
from rig.views.rust import boxes, rust_heat

ROLES = ("rust", "gas", "inspection")
UNITS = {"rust": "Unit 1 · rust", "gas": "Unit 2 · gas", "inspection": "Unit 3 · inspection"}
RUST_CELL = 16


def role_of(index: int) -> str:
    return ROLES[index % len(ROLES)]


def _rust(rgb, info):
    heat = rust_heat(rgb, RUST_CELL)
    h, w = rgb.shape[:2]
    up = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32) / 255
    a = (np.clip((up - 0.35) / 0.5, 0, 1) * 0.6)[..., None]
    base = np.asarray(rgb, np.float32)[..., :3] * 0.75
    im = Image.fromarray((base * (1 - a) + np.array([255, 40, 30], np.float32) * a).astype(np.uint8))
    bs = boxes(heat, 0.6, RUST_CELL)
    info["rust_boxes"], info["rust_max"] = len(bs), round(float(heat.max()), 3)
    return label(draw_boxes(im, bs[:6]), "RUST LOCALIZER (colour proxy)", f"{len(bs)} patch"), "rust localizer (colour proxy) · heatmap + boxes"


def _gas(rgb, depth, cam, field, info):
    col = gas_column(field, cam, depth=depth, stride=4) if field is not None else np.zeros((1, 1))
    info["gas_column"] = round(float(col.max()), 2)
    sub = f"peak {col.max():.0f} %LEL·m" if field is not None else "no plume"
    return label(ogi_image(rgb, col), "OGI FALSE COLOUR", sub), "OGI false colour · plume column density along camera rays"


def _depth(depth, info):
    z = np.asarray(depth, np.float32)
    ok = np.isfinite(z)
    info["depth_min"] = round(float(z[ok].min()), 2) if ok.any() else None
    return label(depth_rgb(z), "DEPTH", f"min {info['depth_min'] or 0:.1f} m"), "depth (m) · distance_to_image_plane"


def robot_views(rid: str, role: str, rgb, depth, cam: dict, pose, path=None, field=None, bev: Bev | None = None,
                force=None, seq: int = 0, span: float = 12.0, targets=None) -> dict:
    rgb = agc(rgb)
    x, y, z, yaw = (float(v) for v in pose)
    h, w = rgb.shape[:2]
    pts = backproject(depth, cam, stride=2)
    bev = bev if bev is not None else Bev()
    bev.update(pts, floor_z=z)
    info = {"role": role, "unit": UNITS.get(role, role), "pose": [round(x, 2), round(y, 2), round(z, 2), round(yaw, 3)],
            "points": int(len(pts))}
    if role == "rust":
        pol, cap = _rust(rgb, info)
    elif role == "gas":
        pol, cap = _gas(rgb, depth, cam, field, info)
    else:
        pol, cap = _depth(depth, info)
    seen = []
    for xyz, name in targets or []:
        u, v, zc = project(cam, xyz)
        if zc > 0.3 and 0 <= u < w and 0 <= v < h:
            seen.append((u, v, max(6.0, 0.25 * cam["K"][0][0] / zc), name))
    if seen:
        draw_targets(pol, seen)
        info["targets"] = [s[3] for s in seen]
    if force is not None and len(force):
        f = list(force)[-60:]
        sparkline(pol, f, (4, h - 26, w // 2, h - 4), 0.0, 60.0, ref=40.0)
        info["force_n"] = round(float(f[-1]), 1)
    fov = 2 * math.atan(w / 2 / cam["K"][0][0])
    top = label(bev.render((x, y, yaw), path, size=h, span=span, fov=fov), "BEV · depth -> occupancy", f"±{span:.0f} m")
    fpv = label(Image.fromarray(rgb), f"{rid} · {UNITS.get(role, role)}", "head cam")
    return {"fpv": encode(fpv, f"{rid} head camera", seq), "policy": encode(pol, cap, seq),
            "bev": encode(top, "BEV occupancy from depth points + planned path", seq), "info": info}

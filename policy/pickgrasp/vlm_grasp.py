from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np

from policy import gemini_verifier as gv

MODELS = ("gemini-robotics-er-2-preview", "gemini-robotics-er-2-streaming-preview", "gemini-3.8-flash")
PROMPT = ("You are the vision system of a robot arm with a suction gripper clearing loose objects off an offshore "
          "rig deck. Point to the single best grasp point on the {query}: the centre of its top surface as seen "
          "from above, where a top-down suction cup would seal. Ignore every other object. If no {query} is "
          "visible, return an empty points list.")
FORMAT = (" The answer follows the format: {\"points\": [{\"point\": [y, x], \"label\": <label>}], "
          "\"approach\": \"top\" or \"side\"}. The point is in [y, x] format normalized to 0-1000. One point only.")
DOWN = [0.0, 0.0, -1.0]


@dataclass
class GraspTarget:
    xyz: list
    approach: list
    pixel: list
    source: str
    label: str = ""
    model: str = ""
    latency_ms: float = 0.0

    def summary(self) -> dict:
        d = asdict(self)
        d["xyz"] = [round(float(x), 4) for x in self.xyz]
        d["approach"] = [round(float(x), 4) for x in self.approach]
        d["pixel"] = [round(float(x), 1) for x in self.pixel]
        d["latency_ms"] = round(float(self.latency_ms))
        return d


def parse_points(text: str, w: int, h: int) -> tuple[list, str]:
    m = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    if not m:
        return [], "top"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        yx = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text)
        side = re.search(r'"approach"\s*:\s*"side"', text)
        data = {"points": [[float(yx.group(1)), float(yx.group(2))]] if yx else [], "approach": "side" if side else "top"}
    items = data.get("points") or [] if isinstance(data, dict) else data
    ap = str(data.get("approach", "top")).lower() if isinstance(data, dict) else "top"
    out = []
    for p in items if isinstance(items, list) else []:
        yx = p.get("point") if isinstance(p, dict) else p
        if not (isinstance(yx, (list, tuple)) and len(yx) == 2):
            continue
        try:
            y, x = (float(np.clip(float(c), 0.0, 1000.0)) for c in yx)
        except (TypeError, ValueError):
            continue
        label = str(p.get("label", "")) if isinstance(p, dict) else ""
        out.append((label, x / 1000.0 * w, y / 1000.0 * h))
    return out, ap if ap in ("top", "side") else "top"


def _rays(us, vs, K, T):
    K, T = np.asarray(K, float), np.asarray(T, float)
    d = np.stack([(np.asarray(us, float) - K[0, 2]) / K[0, 0], (np.asarray(vs, float) - K[1, 2]) / K[1, 1],
                  np.ones(np.shape(us))], -1)
    return T[:3, 3], d @ T[:3, :3].T


def project(xyz, K, T) -> tuple[float, float]:
    K, T = np.asarray(K, float), np.asarray(T, float)
    p = T[:3, :3].T @ (np.asarray(xyz, float) - T[:3, 3])
    return float(K[0, 0] * p[0] / p[2] + K[0, 2]), float(K[1, 1] * p[1] / p[2] + K[1, 2])


def on_plane(u: float, v: float, K, T, z0: float = 0.0):
    o, d = _rays(u, v, K, T)
    if abs(d[2]) < 1e-9 or (z0 - o[2]) / d[2] <= 0:
        return None
    return list(o + (z0 - o[2]) / d[2] * d)


def from_depth(u: float, v: float, depth, K, T, win: int = 5, band: float = 0.01):
    depth = np.asarray(depth, float)
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    vs, us = np.mgrid[max(vi - win, 0):min(vi + win + 1, h), max(ui - win, 0):min(ui + win + 1, w)]
    vs, us = vs.ravel(), us.ravel()
    z = depth[vs, us] if len(vs) else np.array([])
    ok = np.isfinite(z) & (z > 0)
    if not ok.any():
        return None
    o, d = _rays(us[ok], vs[ok], K, T)
    P = o + z[ok, None] * d
    sel = P[P[:, 2] >= P[:, 2].max() - band]
    return list(np.median(sel, 0))


def approach_vec(kind: str, xyz, T) -> list:
    if kind != "side":
        return list(DOWN)
    d = np.asarray(xyz, float) - np.asarray(T, float)[:3, 3]
    d[2] = 0.0
    n = np.linalg.norm(d)
    return list(d / n) if n > 1e-9 else list(DOWN)


def detection_target(det: dict, K, T, source: str = "detector") -> GraspTarget:
    x, y, z = (float(c) for c in det["xyz"])
    top = z + (float(det["diameter"]) / 2 if det.get("lying") else float(det["length"]) / 2)
    return GraspTarget([x, y, top], list(DOWN), list(project([x, y, top], K, T)), source, det.get("colour", ""))


def load_key(env_file: str | None = None) -> str:
    k = gv.load_api_key()
    path = env_file or os.environ.get("GEMINI_ENV_FILE")
    if not k and path and os.path.exists(path):
        m = re.search(r"(?m)^\s*(?:GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*['\"]?([^'\"\s]+)", open(path).read())
        k = m.group(1) if m else None
    return k or ""


class VlmGrasp:

    def __init__(self, api_key: str | None = None, models=None, transport=None, timeout: float = 60.0,
                 max_calls: int = 60, max_height: float = 0.6, max_range: float = 3.0):
        self.api_key = load_key() if api_key is None else api_key
        self.models = list(models or ([os.environ["GRASP_MODEL"]] if os.environ.get("GRASP_MODEL") else MODELS))
        self.model = self.models[0]
        self.transport = transport or self._post
        self.timeout, self.max_calls = timeout, max_calls
        self.max_height, self.max_range = max_height, max_range
        self.calls: list[dict] = []
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.api_key) and len(self.calls) < self.max_calls

    def _post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers={
            "Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def point(self, rgb, query: str):
        jpeg, _ = gv.encode_jpeg(rgb)
        h, w = np.asarray(rgb).shape[:2]
        parts = [{"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}},
                 {"text": PROMPT.format(query=query) + FORMAT}]
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}}
        while True:
            t0 = time.perf_counter()
            try:
                reply = self.transport(f"{gv.API_ROOT}/models/{self.model}:generateContent", body)
            except urllib.error.HTTPError as e:
                self.calls.append({"model": self.model, "error": f"HTTP {e.code}"})
                i = self.models.index(self.model)
                if e.code == 404 and i + 1 < len(self.models):
                    self.model = self.models[i + 1]
                    continue
                raise
            u = reply.get("usageMetadata") or {}
            text = gv._extract_text(reply)
            pts, ap = parse_points(text, w, h)
            self.calls.append({"model": self.model, "tokens_in": int(u.get("promptTokenCount", 0)),
                               "tokens_out": int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0)),
                               "latency_ms": round((time.perf_counter() - t0) * 1000.0), "text": text[:300]})
            return pts, ap, self.calls[-1]["latency_ms"]

    def _plausible(self, xyz, T, plane_z) -> bool:
        if xyz is None:
            return False
        dz = xyz[2] - plane_z
        return -0.03 <= dz <= self.max_height and np.linalg.norm(np.subtract(xyz, np.asarray(T)[:3, 3])) <= self.max_range

    def grasp(self, rgb, K, T, query: str, depth=None, plane_z: float = 0.0, fallback=None) -> GraspTarget | None:
        def fb():
            t = fallback(rgb, depth, K, T) if fallback else None
            if t is not None:
                t.source = f"fallback:{t.source}"
            return t

        if not self.available:
            self.last_error = "no GEMINI_API_KEY" if not self.api_key else "call budget spent"
            return fb()
        try:
            pts, ap, lat = self.point(rgb, query)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return fb()
        if not pts:
            return fb()
        label, u, v = pts[0]
        xyz = from_depth(u, v, depth, K, T) if depth is not None else on_plane(u, v, K, T, plane_z)
        if not self._plausible(xyz, T, plane_z):
            return fb()
        return GraspTarget([float(c) for c in xyz], [float(c) for c in approach_vec(ap, xyz, T)], [u, v], "vlm",
                           label or query, self.model, lat)

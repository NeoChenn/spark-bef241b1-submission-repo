"""Analog pressure-gauge perception: synthetic camera crops plus two independent readers.

Dial geometry (sim/tools/make_generated_assets.py, make_gauge_dial): in image coordinates (x right, y down)
the scale runs clockwise from 135 deg (lower left, value 0) through the top to 405 = 45 deg (lower right,
value 1), 270 deg in all, 51 ticks with a major tick every 10. The red band covers values 0.8 to 1.0.
The rig gauges read 0 to 16 bar (sim/rig_details.py GAUGE_RANGE_BAR).

Readers
  ClassicalGaugeReader  OpenCV: fit the light dial face as an ellipse, undo the perspective tilt with an
                        affine map, then take the circular mean direction of the red needle pixels.
  GeminiGaugeReader     Gemini vision: structured JSON reading + confidence. Returns None when there is no
                        key, no network, the call budget is spent or the reply is unusable (offline fallback).
"""
from __future__ import annotations

import base64
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAL_TEXTURE = REPO / "sim/assets/generated/gauge_dial.png"
START_DEG = 135.0          # value 0, image coordinates (y down)
SWEEP_DEG = 270.0
RED_ZONE = (0.8, 1.0)
RANGE_BAR = 16.0
NEEDLE_FRACTION = 0.75     # needle tip radius / dial radius (inside the ticks and the red band)
NEEDLE_RED = (191, 16, 12)


def value_to_angle(value: float) -> float:
    return START_DEG + SWEEP_DEG * float(value)


def angle_to_value(angle_deg: float) -> float:
    """Image-frame needle angle (deg, y down) -> dial fraction. Angles in the dead zone snap to the nearer end."""
    rel = (angle_deg - START_DEG) % 360.0
    if rel <= SWEEP_DEG:
        return rel / SWEEP_DEG
    return 1.0 if rel - SWEEP_DEG < (360.0 - rel) else 0.0


# ---------------------------------------------------------------------------------------------- synthetic crops

@dataclass
class CropParams:
    yaw_deg: float = 0.0       # camera off the dial axis, left/right
    pitch_deg: float = 0.0     # camera above/below the dial axis
    roll_deg: float = 0.0      # small camera roll (body sway)
    blur_sigma: float = 0.0
    glare: float = 0.0         # 0..1 peak of a white specular blob
    noise_std: float = 0.0     # gaussian sensor noise, 8-bit units
    scale: float = 1.0         # dial size in the crop
    shift: tuple[float, float] = (0.0, 0.0)   # dial centre offset, fraction of crop size
    glare_at: tuple[float, float] = (0.35, 0.3)

    @classmethod
    def random(cls, rng: np.random.Generator, hard: bool = False) -> "CropParams":
        k = 1.5 if hard else 1.0
        return cls(
            yaw_deg=float(rng.uniform(-35, 35) * k),
            pitch_deg=float(rng.uniform(-25, 25) * k),
            roll_deg=float(rng.uniform(-2.5, 2.5)),
            blur_sigma=float(rng.uniform(0.0, 2.0 * k)),
            glare=float(rng.uniform(0.0, 0.6 * k)),
            noise_std=float(rng.uniform(0.0, 10.0 * k)),
            scale=float(rng.uniform(0.8, 1.1)),
            shift=(float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-0.06, 0.06))),
            glare_at=(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))),
        )


_TEXTURE: Optional[np.ndarray] = None


def dial_texture() -> np.ndarray:
    global _TEXTURE
    if _TEXTURE is None:
        bgr = cv2.imread(str(DIAL_TEXTURE), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(DIAL_TEXTURE)
        _TEXTURE = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _TEXTURE.copy()


def dial_face(value: float) -> np.ndarray:
    """512 px dial texture with the needle drawn at ``value`` (RGB)."""
    face = dial_texture()
    h, w = face.shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    r = w / 2.0
    a = math.radians(value_to_angle(value))
    d = np.array([math.cos(a), math.sin(a)])
    tip = c + d * r * NEEDLE_FRACTION
    tail = c - d * r * 0.12
    cv2.line(face, tuple(np.round(tail).astype(int)), tuple(np.round(tip).astype(int)), NEEDLE_RED, 9, cv2.LINE_AA)
    cv2.circle(face, tuple(np.round(c).astype(int)), int(r * 0.06), (25, 25, 25), -1, cv2.LINE_AA)
    return face


def _tilt_homography(size: int, yaw: float, pitch: float, roll: float) -> np.ndarray:
    f = 2.2 * size
    cx = cy = size / 2.0
    ry, rp, rr = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rp), -math.sin(rp)], [0, math.sin(rp), math.cos(rp)]])
    Rz = np.array([[math.cos(rr), -math.sin(rr), 0], [math.sin(rr), math.cos(rr), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    dst = []
    for x, y in src:
        p = R @ np.array([x - cx, y - cy, 0.0]) + np.array([0, 0, f])
        dst.append([f * p[0] / p[2] + cx, f * p[1] / p[2] + cy])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def render_gauge_crop(value: float, params: Optional[CropParams] = None, size: int = 256,
                      seed: int = 0) -> np.ndarray:
    """A camera crop of one gauge: grey pipework background, black bezel, dial with needle at ``value``,
    then perspective tilt, blur, glare and sensor noise. Returns RGB uint8 (size x size)."""
    p = params or CropParams()
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bg = 70 + 40 * (yy / size) + rng.normal(0, 4, (size, size))
    img = np.dstack([bg * 0.95, bg, bg * 1.05]).clip(0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    radius = size * 0.31 * p.scale
    centre = (size / 2.0 + p.shift[0] * size, size / 2.0 + p.shift[1] * size)
    ci = (int(round(centre[0])), int(round(centre[1])))
    cv2.circle(img, ci, int(radius * 1.13), (18, 18, 20), -1, cv2.LINE_AA)
    face = cv2.resize(dial_face(value), (int(2 * radius), int(2 * radius)), interpolation=cv2.INTER_AREA)
    mask = np.zeros(face.shape[:2], np.uint8)
    cv2.circle(mask, (face.shape[1] // 2, face.shape[0] // 2), int(radius) - 1, 255, -1, cv2.LINE_AA)
    x0, y0 = ci[0] - face.shape[1] // 2, ci[1] - face.shape[0] // 2
    roi = img[y0:y0 + face.shape[0], x0:x0 + face.shape[1]]
    alpha = (mask[..., None] / 255.0)
    roi[:] = (face * alpha + roi * (1 - alpha)).astype(np.uint8)

    H = _tilt_homography(size, p.yaw_deg, p.pitch_deg, p.roll_deg)
    img = cv2.warpPerspective(img, H, (size, size), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    if p.glare > 0:
        gx, gy = p.glare_at[0] * size, p.glare_at[1] * size
        sigma = size * 0.12
        blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2)) * 255.0 * p.glare
        img = img + blob[..., None]
    if p.blur_sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), p.blur_sigma)
    if p.noise_std > 0:
        img = img + rng.normal(0, p.noise_std, img.shape)
    return img.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- classical reader

@dataclass
class GaugeReading:
    value: Optional[float]              # 0..1 across the dial, None if unreadable
    confidence: float
    method: str
    reading_bar: Optional[float] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.value is not None:
            d["value"] = round(float(self.value), 4)
        d["confidence"] = round(float(self.confidence), 3)
        return d


class ClassicalGaugeReader:
    """Needle-angle reader. Works on any RGB crop with one gauge roughly centred."""

    method = "classical_opencv"

    def __init__(self, range_bar: float = RANGE_BAR, canon: int = 128):
        self.range_bar = range_bar
        self.canon = canon          # dial radius in the rectified image, px

    def _dial_ellipse(self, rgb: np.ndarray):
        return self._bezel_ellipse(rgb) or self._light_ellipse(rgb)

    def _bezel_ellipse(self, rgb: np.ndarray):
        """Dial edge = inner boundary of the dark bezel ring. A black-hat filter keeps thin dark structures and
        is unaffected by smooth additive glare, which shifts the bezel and the dial face by the same amount."""
        gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        k = max(9, rgb.shape[0] // 9) | 1
        hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask = (hat > 35).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is None:
            return None
        hier = hier[0]
        best = None
        min_area = 0.03 * rgb.shape[0] * rgb.shape[1]
        for i, c in enumerate(contours):
            if hier[i][3] != -1:
                continue
            child = hier[i][2]
            while child != -1:
                hole = contours[child]
                area = cv2.contourArea(hole)
                if area > min_area and len(hole) >= 20 and (best is None or area > best[0]):
                    best = (area, hole)
                child = hier[child][0]
        if best is None:
            return None
        hole = best[1]
        ellipse = cv2.fitEllipse(hole)
        fill = best[0] / (math.pi * ellipse[1][0] * ellipse[1][1] / 4.0)
        return ellipse if 0.85 < fill < 1.1 else None

    def _light_ellipse(self, rgb: np.ndarray):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        light = ((hsv[..., 2] > 150) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
        k = max(3, rgb.shape[0] // 40) | 1
        light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        light = cv2.morphologyEx(light, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 0.03 * rgb.shape[0] * rgb.shape[1] or len(cnt) < 20:
            return None
        hull = cv2.convexHull(cnt)
        if len(hull) < 5:
            return None
        return cv2.fitEllipse(hull)

    def _rectify(self, rgb: np.ndarray, ellipse):
        (cx, cy), (w, h), ang = ellipse
        a, b = w / 2.0, h / 2.0
        t = math.radians(ang)
        Rm = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        S = np.diag([self.canon / a, self.canon / b])
        A = Rm @ S @ Rm.T
        out = 2 * self.canon + 8
        M = np.hstack([A, (np.array([out / 2.0, out / 2.0]) - A @ np.array([cx, cy]))[:, None]])
        return cv2.warpAffine(rgb, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _scale_gap_offset(self, rect: np.ndarray, rad: np.ndarray, dx: np.ndarray, dy: np.ndarray):
        """The ellipse fit leaves an in-plane twist (combined yaw + pitch, camera roll). The tick-free gap at
        the bottom of the scale must sit at 90 deg: measure where it is and return the offset (or None)."""
        gray = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
        hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        ticks = (hat > 40) & (rad > 0.70) & (rad < 0.82)
        if ticks.sum() < 50:
            return None
        bins = np.zeros(180, bool)
        ang = (np.degrees(np.arctan2(dy[ticks], dx[ticks])) % 360.0) // 2
        bins[ang.astype(int)] = True
        if bins.all():
            return None
        start = int(np.argmax(bins))             # rotate so the run search starts on an occupied bin
        rolled = np.roll(bins, -start)
        best_len, best_mid, run = 0, 0.0, 0
        for i, occ in enumerate(np.append(rolled, True)):
            if not occ:
                run += 1
                continue
            if run > best_len:
                best_len, best_mid = run, (i - run / 2.0)
            run = 0
        gap_deg = best_len * 2.0
        if not 60.0 <= gap_deg <= 115.0:
            return None
        centre = ((best_mid + start) * 2.0) % 360.0
        offset = (centre - 90.0 + 180.0) % 360.0 - 180.0
        return offset if abs(offset) < 25.0 else None

    def read(self, rgb: np.ndarray) -> GaugeReading:
        rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8)
        ellipse = self._dial_ellipse(rgb)
        if ellipse is None:
            return GaugeReading(None, 0.0, self.method, detail={"reason": "dial face not found"})
        rect = self._rectify(rgb, ellipse).astype(np.int16)
        r, g, b = rect[..., 0], rect[..., 1], rect[..., 2]
        red = (r - np.maximum(g, b) > 45) & (r > 90)
        n = rect.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        dx, dy = xx - n / 2.0, yy - n / 2.0
        rad = np.hypot(dx, dy) / self.canon
        sel = red & (rad > 0.18) & (rad < 0.70)
        count = int(sel.sum())
        expected = 0.52 * self.canon * 9 / 256 * 2 * self.canon   # needle length x width in canon px
        if count < 0.12 * expected:
            return GaugeReading(None, 0.0, self.method,
                                detail={"reason": "needle not found", "needle_px": count})
        ang = np.arctan2(dy[sel], dx[sel])
        weights = rad[sel]                      # outer pixels pin the direction better
        cs, sn = float((np.cos(ang) * weights).sum()), float((np.sin(ang) * weights).sum())
        rbar = math.hypot(cs, sn) / float(weights.sum())
        raw_angle = math.degrees(math.atan2(sn, cs)) % 360.0
        twist = self._scale_gap_offset(rect.astype(np.uint8), rad, dx, dy)
        angle = (raw_angle - (twist or 0.0)) % 360.0
        value = angle_to_value(angle)
        in_dead_zone = ((angle - START_DEG) % 360.0) > SWEEP_DEG
        tilt = min(ellipse[1]) / max(ellipse[1])
        conf = min(1.0, count / expected) * rbar ** 8 * (0.5 if in_dead_zone else 1.0) * (0.6 + 0.4 * tilt)
        return GaugeReading(value, float(conf), self.method, round(value * self.range_bar, 2), detail={
            "needle_angle_deg": round(angle, 2), "twist_correction_deg": None if twist is None else round(twist, 2),
            "needle_px": count, "concentration": round(rbar, 4),
            "dial_aspect": round(tilt, 3), "dead_zone": bool(in_dead_zone)})


# ---------------------------------------------------------------------------------------------- Gemini reader

GAUGE_PROMPT = (
    "You are the instrument technician for an inspection robot on an offshore oil rig. The image is a camera "
    "crop of one analog pressure gauge, possibly seen at an angle, blurred, with glare or noise. The dial has "
    "no numbers: its scale is 0 to {range:g} bar over a 270 degree arc running CLOCKWISE from the lower-left "
    "end (0 bar) over the top to the lower-right end ({range:g} bar). There are 51 ticks; the 6 long major "
    "ticks mark 0, {m1:g}, {m2:g}, {m3:g}, {m4:g} and {range:g} bar; the major tick at the top is not a scale "
    "mark (the top of the dial is {mid:g} bar). The red band on the right covers {redlo:g} to {range:g} bar "
    "(high alarm). Read where the thin red needle points (ignore its short tail behind the hub). "
    "Answer only with JSON: {{\"reading_bar\": number, \"fraction\": 0..1 position along the scale, "
    "\"confidence\": 0..1, \"readable\": true|false, \"reason\": \"one short sentence\"}}. "
    "If glare or blur hides the needle, set readable false and confidence low."
)
GAUGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reading_bar": {"type": "NUMBER"},
        "fraction": {"type": "NUMBER"},
        "confidence": {"type": "NUMBER"},
        "readable": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["reading_bar", "confidence", "readable"],
}


class GeminiGaugeReader:
    """Gemini vision reader with a hard call budget. ``read`` returns None whenever Gemini can't be used,
    so callers always keep the classical reading (offline fallback)."""

    method = "gemini_vision"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 30.0,
                 budget: Optional[int] = None, transport: Optional[Callable[[str, dict], dict]] = None,
                 range_bar: float = RANGE_BAR):
        from policy.gemini_verifier import DEFAULT_MODEL, load_api_key
        offline = os.environ.get("FLEET_OFFLINE", "").strip() not in ("", "0")
        self.api_key = None if offline else (api_key or load_api_key())
        self.model = model or os.environ.get("GEMINI_GAUGE_MODEL") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.budget = int(os.environ.get("FLEET_GEMINI_BUDGET", "24")) if budget is None else budget
        self.transport = transport
        self.range_bar = range_bar
        self.calls: list[dict] = []
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key or self.transport) and len(self.calls) < self.budget

    def prompt(self) -> str:
        R = self.range_bar
        return GAUGE_PROMPT.format(range=R, m1=R * 0.2, m2=R * 0.4, m3=R * 0.6, m4=R * 0.8, mid=R * 0.5,
                                   redlo=R * RED_ZONE[0])

    def read(self, rgb: np.ndarray) -> Optional[GaugeReading]:
        if not self.available:
            self.last_error = "Gemini unavailable (no key, FLEET_OFFLINE or budget spent)"
            return None
        from policy.gemini_verifier import API_ROOT, _extract_text, encode_jpeg, post_json
        import json
        import re
        jpeg, _ = encode_jpeg(np.asarray(rgb)[..., :3], max_side=512)
        body = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}},
                {"text": self.prompt()}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": GAUGE_SCHEMA,
                                 "temperature": 0.0},
        }
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        started = time.perf_counter()
        try:
            send = self.transport or (lambda u, bd: post_json(u, bd, self.api_key, self.timeout))
            text = _extract_text(send(url, body))
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0) if m else text)
            bar = float(data["reading_bar"])
            frac = float(np.clip(bar / self.range_bar, 0.0, 1.0))
            conf = float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0))
            readable = bool(data.get("readable", True))
        except Exception as e:   # network, HTTP, schema: fall back to the classical reader
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            self.calls.append({"error": self.last_error})
            return None
        latency = (time.perf_counter() - started) * 1000.0
        self.calls.append({"reading_bar": bar, "confidence": conf, "latency_ms": round(latency)})
        self.last_error = None
        return GaugeReading(frac if readable else None, conf if readable else min(conf, 0.2), self.method,
                            round(bar, 2) if readable else None,
                            detail={"reason": str(data.get("reason", ""))[:200], "model": self.model,
                                    "latency_ms": round(latency), "readable": readable})

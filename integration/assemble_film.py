from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integration.assemble_fleet_video import FFMPEG, IMAGES
from policy.hud import font

FPS = 30
SR = 48000
XFADE_S = 0.5
BLUR = ("board", "conclusion", "control", "outro", "end")
READOUT = ("unit1", "unit2", "job", "fixer", "scrubber")
FF = ("walk", "return", "dock")
TEXT = (238, 242, 246)
TEAL = (46, 196, 182)
CHIME_DBFS = -18.0
CHIME = ((659.25, 0.0), (523.25, 0.42))


@dataclass
class Shot:
    name: str
    start: float
    end: float
    kind: str = ""
    rate: float = 1.0
    xfade: float = 0.0
    frames: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def blur(self) -> bool:
        return any(k in self.kind for k in BLUR)

    @property
    def readout(self) -> bool:
        return any(k in self.kind for k in READOUT)

    @property
    def ff(self) -> bool:
        return any(k in self.kind for k in FF)


def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _span(d: dict, keys) -> Optional[tuple]:
    v = _get(d, *keys)
    return tuple(float(x) for x in v[:2]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None


def shot_from(d, prev_end: float = 0.0) -> Shot:
    if dataclasses.is_dataclass(d):
        d = dataclasses.asdict(d)
    elif not isinstance(d, dict):
        d = {k: getattr(d, k) for k in dir(d) if not k.startswith("_") and not callable(getattr(d, k))}
    name = str(_get(d, "name", "id", "shot", default="shot"))
    span = _span(d, ("film", "film_span", "span"))
    start = float(span[0] if span else _get(d, "film_start", "film_t0", "start", "t0", default=prev_end))
    end = span[1] if span else _get(d, "film_end", "film_t1", "end", "t1")
    if end is None:
        end = start + float(_get(d, "duration", "dur", default=0.0))
    tr = _get(d, "transition", "transition_in", "in", default="cut")
    xf = 0.0
    if isinstance(tr, dict):
        xf = float(_get(tr, "duration", "s", default=XFADE_S)) if "cut" not in str(_get(tr, "kind", "type", default="")) else 0.0
    elif isinstance(tr, (int, float)):
        xf = float(tr)
    elif any(k in str(tr).lower() for k in ("fade", "dissolve", "xf", "cross")):
        xf = XFADE_S
    kind = str(_get(d, "kind", "type", "role", default=name)).lower()
    known = {"name", "id", "shot", "film", "film_span", "span", "film_start", "film_t0", "start", "t0", "film_end",
             "film_t1", "end", "t1", "duration", "dur", "transition", "transition_in", "in", "kind", "type", "role",
             "rate", "playback", "speed", "frames"}
    return Shot(name, start, float(end), kind, float(_get(d, "rate", "playback", "speed", default=1.0)), xf,
                _get(d, "frames"), {k: v for k, v in d.items() if k not in known})


def load_edl(spec: Optional[str] = None) -> tuple[list[Shot], dict]:
    if spec and Path(spec).exists():
        raw = json.loads(Path(spec).read_text())
    else:
        mod = importlib.import_module((spec or "integration.film_edl").split(":")[0])
        attr = spec.split(":")[1] if spec and ":" in spec else None
        names = [attr] if attr else ["edl", "EDL", "SHOTS", "shots", "build_edl"]
        raw = next(getattr(mod, n) for n in names if hasattr(mod, n))
        raw = raw() if callable(raw) else raw
    return load_edl_rows(raw)


def load_edl_rows(raw) -> tuple[list[Shot], dict]:
    meta = {}
    if dataclasses.is_dataclass(raw):
        raw = dataclasses.asdict(raw)
    if isinstance(raw, dict):
        meta = {k: v for k, v in raw.items() if k != "shots"}
        raw = raw.get("shots", [])
    shots, end = [], 0.0
    for d in raw:
        s = shot_from(d, end)
        shots.append(s)
        end = s.end
    return sorted(shots, key=lambda s: s.start), meta


def inject_time(shots: list[Shot], meta: dict) -> Optional[float]:
    v = _get(meta, "inject", "inject_at", "alarm", "alarm_at", "scenario_at")
    if v is not None:
        return float(v)
    for s in shots:
        v = _get(s.extra, "inject", "inject_at", "alarm", "alarm_at", "scenario_at")
        if v is not None:
            v = float(v)
            return v if v >= s.start else s.start + v
    board = next((s for s in shots if "board" in s.kind), None)
    return None if board is None else board.start + 2.5


def _images(d: Path) -> list[Path]:
    return sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGES) if d.is_dir() else []


def shot_frames(root: Optional[Path], shot: Shot) -> list[Path]:
    if shot.frames:
        p = Path(shot.frames)
        if not p.is_absolute() and root is not None:
            p = root / p
        for c in (p / "frames", p):
            if _images(c):
                return _images(c)
    if root is None:
        return []
    for c in (root / shot.name / "frames", root / shot.name, root / "shots" / shot.name / "frames",
              root / "shots" / shot.name, root / "frames" / shot.name):
        if _images(c):
            return _images(c)
    for c in (root / "frames", root):
        if _images(c):
            return _images(c)
    return []


@dataclass
class Overlay:
    name: str
    dir: Path
    offset: int

    def at(self, k: int) -> Optional[Path]:
        p = self.dir / f"{k - self.offset:05d}.png"
        return p if p.exists() else None


def load_overlays(root: Optional[Path]) -> list[Overlay]:
    if root is None or not root.is_dir():
        return []
    out = []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        m = d / "manifest.json"
        off = 0
        if m.exists():
            j = json.loads(m.read_text())
            off = int(_get(j, "start_frame", default=round(float(_get(j, "start", "film_start", default=0.0)) * FPS)))
        out.append(Overlay(d.name, d, off))
    return out


@lru_cache(maxsize=64)
def _load(path: str, size: tuple[int, int], mode: str) -> Image.Image:
    im = Image.open(path).convert(mode)
    return im if im.size == size else im.resize(size, Image.LANCZOS)


def background(shot: Shot, frames: list[Path], k: int, size: tuple[int, int], fallback: Optional[Image.Image]):
    n_shot = max(1, round((shot.end - shot.start) * FPS))
    rel = min(max(k - round(shot.start * FPS), 0), n_shot - 1)
    if frames:
        path = str(frames[min(len(frames) - 1, rel * len(frames) // n_shot)])
        return _blurred(path, size) if shot.blur else _load(path, size, "RGB")
    im = fallback if fallback is not None else Image.new("RGB", size, (8, 11, 16))
    return blur_dim(im) if shot.blur else im


def blur_dim(im: Image.Image) -> Image.Image:
    return Image.blend(im.filter(ImageFilter.GaussianBlur(im.height / 24)), Image.new("RGB", im.size, (5, 8, 12)), 0.55)


@lru_cache(maxsize=16)
def _blurred(path: str, size: tuple[int, int]) -> Image.Image:
    return blur_dim(_load(path, size, "RGB"))


def ff_badge(im: Image.Image, rate: float) -> Image.Image:
    w, h = im.size
    s = h / 720
    base = im.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font("sans", int(24 * s), "Semibold")
    txt = f"{rate:g}×"
    tw = d.textlength(txt, font=f)
    bw, bh = tw + 64 * s, 44 * s
    x1, y0 = w - 32 * s, 32 * s
    x0 = x1 - bw
    d.rounded_rectangle([x0, y0, x1, y0 + bh], radius=bh / 2, fill=(10, 14, 20, 120), outline=(255, 255, 255, 90),
                        width=max(1, int(1.5 * s)))
    cy = y0 + bh / 2
    for j in range(2):
        ox = x0 + 18 * s + j * 11 * s
        d.polygon([(ox, cy - 8 * s), (ox + 11 * s, cy), (ox, cy + 8 * s)], fill=TEXT + (170,))
    d.text((x0 + 50 * s, cy), txt, font=f, fill=TEXT + (190,), anchor="lm")
    return Image.alpha_composite(base, layer).convert("RGB")


def compose(shot: Shot, frames: list[Path], overlays: list[Overlay], k: int, size: tuple[int, int],
            fallback: Optional[Image.Image] = None) -> Image.Image:
    im = background(shot, frames, k, size, fallback)
    if shot.ff and not any(o.at(k) for o in overlays if "ff" in o.name or "fast" in o.name):
        im = ff_badge(im, shot.rate)
    return im


def overlay_all(im: Image.Image, overlays: list[Overlay], k: int) -> Image.Image:
    layers = [o.at(k) for o in overlays]
    layers = [p for p in layers if p is not None]
    if not layers:
        return im
    base = im.convert("RGBA")
    for p in layers:
        base = Image.alpha_composite(base, _load(str(p), im.size, "RGBA"))
    return base.convert("RGB")


def active(shots: list[Shot], t: float) -> list[tuple[Shot, float]]:
    for i, s in enumerate(shots):
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        if nxt is not None and nxt.xfade > 0 and abs(t - nxt.start) < nxt.xfade / 2:
            a = (t - (nxt.start - nxt.xfade / 2)) / nxt.xfade
            return [(s, 1.0 - a), (nxt, a)]
        if s.start <= t < s.end or (nxt is None and t >= s.start):
            return [(s, 1.0)]
    return [(shots[-1], 1.0)]


def tone(f: float, dur: float, attack: float = 0.03, release: float = 0.9) -> np.ndarray:
    t = np.arange(int((dur + release) * SR)) / SR
    env = np.minimum(t / attack, 1.0) * np.exp(-np.maximum(t - attack, 0.0) / (release / 3.0))
    env *= np.clip((dur + release - t) / 0.05, 0.0, 1.0)
    return np.sin(2 * np.pi * f * t) * env + 0.25 * np.sin(4 * np.pi * f * t) * env


def chime(repeat: int = 2, gap: float = 1.6, dbfs: float = CHIME_DBFS) -> np.ndarray:
    n = int((gap * repeat + 2.0) * SR)
    out = np.zeros(n)
    for r in range(repeat):
        for f, at in CHIME:
            x = tone(f, 0.35)
            i = int((r * gap + at) * SR)
            out[i:i + len(x)] += x[: max(0, n - i)]
    return out / np.abs(out).max() * 10 ** (dbfs / 20)


def _smooth(x: np.ndarray, m: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo, hi = np.clip(i - m // 2, 0, len(x)), np.clip(i + m // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def ambient(seconds: float, dbfs: float = -38.0, seed: int = 7) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    wind = np.cumsum(rng.standard_normal(n))
    wind = _smooth(wind - _smooth(wind, SR // 2), 400)
    t = np.arange(n) / SR
    wind *= 0.6 + 0.4 * np.sin(2 * np.pi * t / 9.0) ** 2
    hum = 0.35 * np.sin(2 * np.pi * 55 * t) + 0.15 * np.sin(2 * np.pi * 110 * t)
    x = wind / (np.abs(wind).max() + 1e-9) + hum
    fade = np.clip(np.minimum(t, seconds - t) / 1.5, 0.0, 1.0)
    return x / np.abs(x).max() * 10 ** (dbfs / 20) * fade


def soundtrack(seconds: float, chime_at: Optional[float], bed: bool = True) -> np.ndarray:
    out = ambient(seconds) if bed else np.zeros(int(seconds * SR))
    if chime_at is not None and 0 <= chime_at < seconds:
        c = chime()
        i = int(chime_at * SR)
        out[i:i + len(c)] += c[: max(0, len(out) - i)]
    return np.clip(out, -1.0, 1.0)


def write_wav(path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.abs(x).max() + 1e-12))


def _encode(cmd_out, w: int, h: int, wav, crf: int = 18) -> subprocess.Popen:
    return subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(FPS), "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                             "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a",
                             "160k", "-shortest", "-movflags", "+faststart", str(cmd_out)], stdin=subprocess.PIPE)


def teaser_start(shots: list[Shot], total: float, dur: float = 10.0) -> float:
    s = next((s for s in shots if "unit2" in s.kind or "fixer" in s.kind), None)
    s = s or next((s for s in shots if "board" in s.kind), shots[0])
    mid = (s.start + s.end) / 2
    return max(0.0, min(mid - dur / 2, total - dur))


def assemble(edl, out, frames_root=None, overlays_root=None, size=(1920, 1080), chime_at: Optional[float] = None,
             bed: bool = True, teaser: bool = True) -> dict:
    shots, meta = load_edl(edl) if isinstance(edl, (str, type(None))) else (edl, {})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(frames_root) if frames_root else None
    srcs = {s.name: shot_frames(root, s) for s in shots}
    overlays = load_overlays(Path(overlays_root) if overlays_root else None)
    total = max(s.end for s in shots)
    n = int(round(total * FPS))
    chime_at = inject_time(shots, meta) if chime_at is None else chime_at
    wav = out.with_suffix(".wav")
    write_wav(wav, soundtrack(n / FPS, chime_at, bed))
    w, h = size
    hd = out.with_name(out.stem + "_1080p.mp4") if h >= 1080 else None
    enc = _encode(hd or out, w, h, wav)
    last = None
    for k in range(n):
        t = k / FPS
        parts = active(shots, t)
        ims = [(compose(s, srcs[s.name], overlays, k, size, last), a) for s, a in parts]
        im = ims[0][0] if len(ims) == 1 else Image.blend(ims[0][0], ims[1][0], ims[1][1])
        im = overlay_all(im, overlays, k)
        last = im
        enc.stdin.write(np.asarray(im, np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg exited {enc.returncode}")
    if hd is not None:
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(hd), "-vf", "scale=1280:720:flags=lanczos", "-c:v",
                        "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
                        "-movflags", "+faststart", str(out)], check=True)
    report = {"out": str(out), "hd": str(hd) if hd else None, "seconds": round(n / FPS, 2), "shots": len(shots),
              "frames": {k: len(v) for k, v in srcs.items()}, "overlays": [o.name for o in overlays],
              "chime_at": chime_at, "crossfades": [s.name for s in shots if s.xfade > 0]}
    if teaser:
        st = teaser_start(shots, n / FPS)
        mp4, gif = out.with_name(out.stem + "_teaser.mp4"), out.with_name(out.stem + "_teaser.gif")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{st:.3f}", "-i", str(out), "-t", "10", "-c:v",
                        "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-movflags", "+faststart", str(mp4)], check=True)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(mp4), "-vf", "fps=12,scale=640:-2:flags=lanczos,split"
                        "[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(gif)], check=True)
        report.update(teaser=str(mp4), gif=str(gif), teaser_start_s=round(st, 2))
    wav.unlink(missing_ok=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the fleet film from the EDL, Isaac shot frames and overlays.")
    ap.add_argument("--edl", default=None, help="EDL JSON path or module[:attr]; default integration.film_edl")
    ap.add_argument("--frames", help="Isaac --film output root: <root>/<shot>/frames/*.jpg (or shots/<shot>/...)")
    ap.add_argument("--overlays", help="overlay root: <root>/<name>/<film_frame:05d>.png, optional manifest.json")
    ap.add_argument("--out", default="runs/film/film.mp4")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--chime-at", type=float, default=None)
    ap.add_argument("--no-bed", action="store_true")
    ap.add_argument("--no-teaser", action="store_true")
    a = ap.parse_args(argv)
    w, h = (int(x) for x in a.size.split("x"))
    r = assemble(a.edl, a.out, a.frames, a.overlays, (w, h), a.chime_at, not a.no_bed, not a.no_teaser)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

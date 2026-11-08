from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from policy.hud import font

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FPS = 30
TITLE = "Argon · rig fleet"
TITLE_S = 1.5
END_S = 3.5
BUDGET_S = 60.0
SPEEDS = (4.0, 6.0, 8.0, 12.0, 16.0)
PAD_S = 1.0
RAMP_S = 0.5
TEASER_S = 10.0
BEAT_ROWS = ("finding", "command", "ticket", "vetoed", "argon_call", "link")
TEXT = (238, 242, 246)
DIM = (150, 160, 172)
ACCENT = (255, 184, 64)


def read_jsonl(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            if isinstance(r, dict):
                rows.append(r)
    return rows


def load_headers(path) -> list[dict]:
    rows = [r for r in read_jsonl(path) if r.get("type", "frame") in ("frame", "tick")]
    key = next((k for k in ("seq", "frame") if all(k in r for r in rows)), None)
    return sorted(rows, key=lambda r: r[key]) if key else sorted(rows, key=lambda r: float(r.get("t", 0.0)))


def align(headers: list[dict], n: int, fps_in: float) -> list[dict]:
    if n == len(headers) or not headers:
        return headers
    ts = np.array([float(h.get("t", 0.0)) for h in headers])
    at = ts[0] + np.arange(n) / fps_in
    k = np.clip(np.searchsorted(ts, at + 1e-6, side="right") - 1, 0, len(headers) - 1)
    return [dict(headers[j], t=round(float(t), 4)) for j, t in zip(k, at)]


def times(headers: list[dict], fps_in: float) -> np.ndarray:
    return np.array([float(h.get("t", i / fps_in)) for i, h in enumerate(headers)], float)


def argon(h: dict) -> dict:
    a = h.get("argon") or (h.get("control") if isinstance(h.get("control"), dict) else None) or {}
    return a if isinstance(a, dict) else {}


def waits_from_timeline(rows: list[dict]) -> list[tuple[float, float]]:
    open_, spans = {}, []
    for r in rows:
        if r.get("type") == "argon_request":
            open_.setdefault(r.get("key"), float(r["t"]))
        elif r.get("type") == "argon_call" and r.get("key") in open_:
            t0 = open_.pop(r["key"])
            t0 = float(r.get("t_request", t0))
            spans.append((t0, float(r["t"])))
    return sorted(spans)


def beats_from_timeline(rows: list[dict]) -> list[float]:
    return sorted(float(r["t"]) for r in rows if r.get("type") in BEAT_ROWS and float(r.get("t", -1)) >= 0)


def _robots(h: dict) -> list:
    rs = h.get("robots") or []
    if isinstance(rs, dict):
        rs = [dict(v, id=k) for k, v in rs.items()]
    return [(r.get("id"), r.get("state")) for r in rs if isinstance(r, dict)]


def _detectors(h: dict) -> list:
    ds = h.get("detectors") or []
    if isinstance(ds, dict):
        return sorted((k, v.get("led") if isinstance(v, dict) else v) for k, v in ds.items())
    return [(d.get("id"), d.get("led")) for d in ds if isinstance(d, dict)]


def signature(h: dict) -> str:
    a = argon(h)
    sig = {k: a.get(k) for k in ("held", "link", "last_command", "verdicts", "tickets")}
    sig.update(robots=_robots(h), detectors=_detectors(h), beat=h.get("beat"), job=h.get("job"))
    return json.dumps(sig, sort_keys=True, default=str)


def beats_from_headers(headers: list[dict]) -> list[int]:
    out, prev = [], None
    for i, h in enumerate(headers):
        s = signature(h)
        if prev is not None and s != prev:
            out.append(i)
        prev = s
    return out


def wait_mask(headers: list[dict], ts: np.ndarray, spans: Optional[list] = None) -> np.ndarray:
    if spans:
        m = np.zeros(len(ts), bool)
        for t0, t1 in spans:
            m |= (ts >= t0) & (ts < t1)
        return m
    return np.array([bool(argon(h).get("pending") or h.get("argon_pending") or h.get("wait")) for h in headers])


def beat_mask(ts: np.ndarray, beat_idx: Iterable[int], beat_t: Iterable[float], pad_s: float) -> np.ndarray:
    m = np.zeros(len(ts), bool)
    for t in [*(ts[i] for i in beat_idx if 0 <= i < len(ts)), *beat_t]:
        m |= np.abs(ts - t) <= pad_s
    return m


def ramp(fast: np.ndarray, top: float, ramp_n: int) -> np.ndarray:
    n = len(fast)
    d = np.full(n, 0.0)
    run = 0
    for i in range(n):
        run = run + 1 if fast[i] else 0
        d[i] = run
    run = 0
    for i in range(n - 1, -1, -1):
        run = run + 1 if fast[i] else 0
        d[i] = min(d[i], run)
    return 1.0 + (top - 1.0) * np.clip(d / max(ramp_n, 1), 0.0, 1.0)


def schedule(speed: np.ndarray, fps_in: float, fps_out: float = FPS) -> list[int]:
    out, p, n = [], 0.0, len(speed)
    step = fps_in / fps_out
    while p < n:
        i = int(p)
        out.append(i)
        p += step * speed[i]
    return out


@dataclass
class Plan:
    idx: list[int]
    speed: np.ndarray
    top: float
    mode: str
    fast: np.ndarray
    beats: np.ndarray
    seconds: float = 0.0
    notes: list = field(default_factory=list)


def plan(headers: list[dict], fps_in: float = FPS, timeline: Optional[list[dict]] = None, budget_s: float = BUDGET_S,
         title_s: float = TITLE_S, end_s: float = END_S, pad_s: float = PAD_S, ramp_s: float = RAMP_S,
         speeds: Iterable[float] = SPEEDS) -> Plan:
    ts = times(headers, fps_in)
    rows = timeline or []
    waits = wait_mask(headers, ts, waits_from_timeline(rows))
    beats = beat_mask(ts, beats_from_headers(headers), beats_from_timeline(rows), pad_s)
    room = budget_s - title_s - end_s
    ramp_n = int(round(ramp_s * fps_in))
    best = None
    for mode, fast in (("waits", waits & ~beats), ("idle", ~beats)):
        for top in speeds:
            sp = np.where(fast, ramp(fast, top, ramp_n), 1.0)
            idx = schedule(sp, fps_in)
            best = Plan(idx, sp, top, mode, fast, beats, len(idx) / FPS)
            if best.seconds <= room:
                return best
    best.notes.append(f"over budget: {best.seconds:.1f}s of footage > {room:.1f}s")
    return best


def frames_from_video(path, w: int, h: int) -> Iterator[np.ndarray]:
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def probe_size(path) -> tuple[int, int]:
    out = subprocess.run([shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams",
                          "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


IMAGES = (".png", ".jpg", ".jpeg")


def count_frames(path) -> int:
    out = subprocess.run([shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams",
                          "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0])


def source(path, headers: list[dict], fps_in: float) -> tuple[Iterator[np.ndarray], tuple[int, int], int]:
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGES)
        w, h = Image.open(files[0]).size
        return (np.asarray(Image.open(f).convert("RGB")) for f in files), (w, h), len(files)
    if p.suffix.lower() in IMAGES:
        still = np.asarray(Image.open(p).convert("RGB"))
        ts = [float(x.get("t", 0.0)) for x in headers]
        n = len(headers) if "seq" in headers[0] else int(round((ts[-1] - ts[0]) * fps_in)) + 1
        return (still for _ in range(n)), (still.shape[1], still.shape[0]), n
    w, h = probe_size(p)
    return frames_from_video(p, w, h), (w, h), count_frames(p)


def pick(frames: Iterator[np.ndarray], idx: list[int]) -> Iterator[tuple[int, int, np.ndarray]]:
    want = iter(enumerate(idx))
    k, i = next(want, (None, None))
    for j, f in enumerate(frames):
        while i is not None and i == j:
            yield k, i, f
            k, i = next(want, (None, None))
        if i is None:
            return


def resolve(spec: Optional[str]) -> Optional[Callable]:
    if not spec:
        return None
    mod, _, name = spec.partition(":")
    try:
        return getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as e:
        print(f"assemble: {spec} unavailable ({e}); skipping", file=sys.stderr)
        return None


def labels_for(fn: Optional[Callable], header: dict, size: tuple[int, int]) -> dict:
    if fn is None:
        return {}
    try:
        return fn(header, size) or {}
    except Exception as e:
        print(f"assemble: labels failed at t={header.get('t')}: {e}", file=sys.stderr)
        return {}


def world_positions(header: dict) -> dict:
    rs = header.get("robots") or []
    items = rs.items() if isinstance(rs, dict) else ((r.get("id"), r) for r in rs if isinstance(r, dict))
    out = {}
    for rid, r in items:
        p = r.get("world_pos") or r.get("pos") or (r.get("pose")[:3] if isinstance(r.get("pose"), list) else None)
        if rid and p is not None and len(p) >= 3:
            out[rid] = tuple(float(c) for c in p[:3])
    return out


def camera_labels(beats=None) -> Optional[Callable]:
    try:
        fc = importlib.import_module("integration.fleet_camera")
    except ImportError as e:
        print(f"assemble: integration.fleet_camera unavailable ({e}); no labels", file=sys.stderr)
        return None
    b = fc.Beats.from_events(beats) if beats else fc.DEFAULT_BEATS

    def fn(header: dict, size: tuple[int, int]) -> dict:
        raw = fc.labels(float(header.get("t", 0.0)), b, positions=world_positions(header) or None, w=size[0],
                        h=size[1])
        return {k: {"xy": (a["x"], a["y"]), "visible": a.get("mode") in ("on", "edge"), "mode": a.get("mode")}
                for k, a in raw.items() if a.get("x") is not None}
    return fn


def for_overlay(header: dict) -> dict:
    rs = header.get("robots")
    if isinstance(rs, list):
        header = dict(header, robots={r["id"]: r for r in rs if isinstance(r, dict) and r.get("id")})
    return header


def badge(img: np.ndarray, top: float, alpha: float) -> np.ndarray:
    if alpha <= 0.02:
        return img
    h, w = img.shape[:2]
    s = h / 720
    im = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font("sans", int(22 * s), "Semibold")
    txt = f"×{top:g}"
    tw = d.textlength(txt, font=f)
    bw, bh = tw + 28 * s, 36 * s
    x1, y0 = w - 24 * s, h - 24 * s - bh
    x0 = x1 - bw - 22 * s
    a = int(255 * alpha)
    d.rounded_rectangle([x0, y0, x1, y0 + bh], radius=bh / 2, fill=(12, 16, 22, int(190 * alpha)),
                        outline=ACCENT + (a,), width=max(1, int(2 * s)))
    cx, cy = x0 + 18 * s, y0 + bh / 2
    for k in range(2):
        ox = cx + k * 8 * s
        d.polygon([(ox - 4 * s, cy - 6 * s), (ox + 4 * s, cy), (ox - 4 * s, cy + 6 * s)], fill=ACCENT + (a,))
    d.text((x0 + 34 * s, cy), txt, font=f, fill=TEXT + (a,), anchor="lm")
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def _backdrop(frame: np.ndarray, dim: float) -> Image.Image:
    im = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(frame.shape[0] / 90))
    return Image.blend(im, Image.new("RGB", im.size, (6, 9, 14)), dim)


def title_card(frame: np.ndarray, title: str = TITLE, sub: str = "two robots · one command centre · one take",
               k: float = 1.0) -> np.ndarray:
    h, w = frame.shape[:2]
    s = h / 720
    im = _backdrop(frame, 0.55 + 0.3 * k)
    d = ImageDraw.Draw(im)
    big, small = font("sans", int(64 * s), "Bold"), font("sans", int(24 * s))
    d.text((w / 2, h / 2 - 18 * s), title, font=big, fill=TEXT, anchor="ms")
    d.line([(w / 2 - 60 * s, h / 2 + 4 * s), (w / 2 + 60 * s, h / 2 + 4 * s)], fill=ACCENT, width=max(2, int(3 * s)))
    d.text((w / 2, h / 2 + 46 * s), sub, font=small, fill=DIM, anchor="ms")
    return np.asarray(im)


def end_card(frame: np.ndarray, stats: list[tuple[str, str]], title: str = TITLE) -> np.ndarray:
    h, w = frame.shape[:2]
    s = h / 720
    im = _backdrop(frame, 0.8)
    d = ImageDraw.Draw(im)
    d.text((w / 2, h * 0.27), title, font=font("sans", int(34 * s), "Semibold"), fill=DIM, anchor="ms")
    big, small = font("sans", int(72 * s), "Bold"), font("sans", int(22 * s))
    cols = len(stats) or 1
    for i, (value, label) in enumerate(stats):
        cx = w * (i + 0.5) / cols
        d.text((cx, h * 0.53), value, font=big, fill=TEXT, anchor="ms")
        d.line([(cx - 28 * s, h * 0.53 + 16 * s), (cx + 28 * s, h * 0.53 + 16 * s)], fill=ACCENT,
               width=max(2, int(3 * s)))
        d.text((cx, h * 0.53 + 52 * s), label, font=small, fill=DIM, anchor="ms")
    return np.asarray(im)


def _secs(x: float) -> str:
    return f"{x:.0f} s" if x >= 10 else f"{x:.1f} s"


def key_numbers(timeline: list[dict], headers: list[dict]) -> list[tuple[str, str]]:
    rows = timeline or []
    out = []
    alarm = next((r for r in rows if r.get("type") == "finding" and r.get("kind") not in (None, "OK")), None)
    if alarm is not None:
        cleared = next((r for r in rows if r.get("type") == "finding" and r.get("kind") == "OK"
                        and r.get("sensor") == alarm.get("sensor") and r["t"] > alarm["t"]), None)
        if cleared is not None:
            out.append((_secs(cleared["t"] - alarm["t"]), f"alarm to {alarm.get('sensor', 'flange')} green"))
    calls = [r for r in rows if r.get("type") == "argon_call" and r.get("status") == "ok"]
    last = argon(headers[-1]) if headers else {}
    if calls:
        lat = [float(r["latency_ms"]) for r in calls if r.get("latency_ms") is not None]
        out.append((str(len(calls)), f"Argon calls · {np.mean(lat) / 1000:.1f} s avg" if lat else "Argon calls"))
        cost = sum(float(r.get("cost_usd", 0.0)) for r in calls)
        out.append((f"${cost:.3f}", "cloud cost, whole job"))
    elif last:
        if last.get("latency_ms") is not None:
            out.append((_secs(last["latency_ms"] / 1000), "last Argon decision"))
        if last.get("cost_usd") is not None:
            out.append((f"${last['cost_usd']:.3f}", "cloud cost, whole job"))
    tickets = [r for r in rows if r.get("type") == "ticket"]
    if len(out) < 3 and tickets:
        out.append((str(len(tickets)), "maintenance ticket" + ("s" if len(tickets) > 1 else "")))
    commands = [r for r in rows if r.get("type") == "command"]
    if len(out) < 3 and commands:
        out.append((str(len(commands)), "fleet commands issued"))
    if len(out) < 3 and headers:
        out.append((_secs(times(headers, FPS)[-1] - times(headers, FPS)[0]), "real time, one take"))
    return out[:3]


def teaser_index(headers: list[dict], timeline: Optional[list[dict]] = None, fps_in: float = FPS) -> int:
    for i, h in enumerate(headers):
        if any("TORQUE" in str(st).upper() for _, st in _robots(h)) or "TORQUE" in str(h.get("beat", "")).upper():
            return i
    ts = times(headers, fps_in)
    fix = next((r for r in timeline or [] if r.get("type") == "command" and r.get("kind") == "fix_leak"), None)
    if fix is not None:
        return int(np.searchsorted(ts, float(fix["t"])))
    return len(headers) // 2


def writer(path, w: int, h: int, fps: int = FPS, crf: int = 18) -> subprocess.Popen:
    return subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def ffmpeg(*args) -> None:
    subprocess.run([FFMPEG, "-v", "error", "-y", *map(str, args)], check=True)


def export_1080(master, out) -> None:
    ffmpeg("-i", master, "-vf", "scale=-2:1080:flags=lanczos", "-c:v", "libx264", "-preset", "medium", "-crf", 18,
           "-r", FPS, "-pix_fmt", "yuv420p", "-movflags", "+faststart", out)


def export_teaser(master, start_s: float, mp4, gif, dur: float = TEASER_S) -> None:
    ffmpeg("-ss", f"{start_s:.3f}", "-i", master, "-t", dur, "-c:v", "libx264", "-preset", "medium", "-crf", 18,
           "-r", FPS, "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4)
    ffmpeg("-i", mp4, "-vf", "fps=12,scale=640:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
           "[b][p]paletteuse=dither=bayer:bayer_scale=4", gif)


def _even(f: np.ndarray) -> np.ndarray:
    h, w = f.shape[:2]
    return f[: h - h % 2, : w - w % 2]


def assemble(raw, headers_path, out, timeline_path=None, fps_in: float = FPS,
             overlay: Optional[str] = "policy.fleet_overlay:draw", labels: Optional[str] = None, beats=None,
             hd: bool = True, teaser: bool = True, budget_s: float = BUDGET_S) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = load_headers(headers_path)
    timeline = read_jsonl(timeline_path) if timeline_path else []
    frames, size, n = source(raw, headers, fps_in)
    headers = align(headers, n, fps_in)
    p = plan(headers, fps_in, timeline, budget_s)
    draw = resolve(overlay)
    label_fn = resolve(labels) if labels else camera_labels(beats)
    w, h = size[0] - size[0] % 2, size[1] - size[1] % 2
    enc = writer(out, w, h)
    stats = key_numbers(timeline, headers)
    n_title, n_end = int(round(TITLE_S * FPS)), int(round(END_S * FPS))
    last = None
    written = 0
    t_out = {}
    for k, i, f in pick(frames, p.idx):
        f = _even(f)
        hd_ = dict(headers[i], timelapse=round(float(p.speed[i]), 2))
        g = draw(f.copy(), for_overlay(hd_), labels_for(label_fn, hd_, (w, h))) if draw else f
        g = np.ascontiguousarray(np.asarray(g, np.uint8)[:h, :w, :3])
        g = badge(g, p.top, (p.speed[i] - 1.0) / max(p.top - 1.0, 1e-6))
        if last is None:
            for j in range(n_title):
                k_ = min(1.0, (n_title - j) / (0.4 * FPS))
                card = title_card(g, k=k_)
                if j >= n_title - int(0.4 * FPS):
                    a = (n_title - j) / (0.4 * FPS)
                    card = (card * a + g * (1 - a)).astype(np.uint8)
                enc.stdin.write(card.tobytes())
        enc.stdin.write(g.tobytes())
        t_out.setdefault(i, n_title + written)
        written += 1
        last = g
    if last is not None:
        card = end_card(last, stats)
        fade = int(0.4 * FPS)
        for j in range(n_end):
            a = min(1.0, (j + 1) / fade)
            enc.stdin.write((card * a + last * (1 - a)).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg exited {enc.returncode}")
    total = (n_title + written + n_end) / FPS if last is not None else 0.0
    report = {"out": str(out), "seconds": round(total, 2), "frames_in": len(headers), "frames_used": written,
              "speed": p.top, "mode": p.mode, "fast_frames": int(p.fast.sum()), "beat_frames": int(p.beats.sum()),
              "end_card": stats, "size": [w, h], "overlay": bool(draw), "labels": bool(label_fn), "notes": p.notes}
    if hd and last is not None:
        hd_path = out.with_name(out.stem + "_1080p.mp4")
        export_1080(out, hd_path)
        report["hd"] = str(hd_path)
    if teaser and last is not None:
        ti = teaser_index(headers, timeline, fps_in)
        near = min(t_out, key=lambda i: abs(i - ti))
        start = max(0.0, min(t_out[near] / FPS - TEASER_S * 0.35, total - TEASER_S))
        mp4, gif = out.with_name(out.stem + "_teaser.mp4"), out.with_name(out.stem + "_teaser.gif")
        export_teaser(out, start, mp4, gif)
        report.update(teaser=str(mp4), gif=str(gif), teaser_start_s=round(start, 2))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the fleet video: HUD burn-in, Argon-wait time-lapse, cards.")
    ap.add_argument("raw", help="Isaac render: MP4, a frames directory, or one still to loop")
    ap.add_argument("headers", help="per-frame header JSONL")
    ap.add_argument("--timeline", help="dual_argon Timeline JSONL (argon_request/argon_call spans)")
    ap.add_argument("--out", default="runs/fleet/fleet_video.mp4")
    ap.add_argument("--fps-in", type=float, default=FPS)
    ap.add_argument("--overlay", default="policy.fleet_overlay:draw")
    ap.add_argument("--labels", default="", help="module:fn(header, size); default projects via fleet_camera")
    ap.add_argument("--beats", help="JSON of fleet_camera beat times (dict or event list) used for the render")
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    ap.add_argument("--no-hd", action="store_true")
    ap.add_argument("--no-teaser", action="store_true")
    a = ap.parse_args(argv)
    beats = json.loads(Path(a.beats).read_text()) if a.beats else None
    r = assemble(a.raw, a.headers, a.out, a.timeline, a.fps_in, a.overlay or None, a.labels or None, beats,
                 not a.no_hd, not a.no_teaser, a.budget)
    print(json.dumps(r, indent=2))
    return 0 if r["seconds"] <= a.budget + 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())

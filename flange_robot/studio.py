"""Studio feed: show the gas-leak robot on the team's web page.

Renders a 2D dashboard of what OUR module perceives and decides, JPEG-encodes it, and sends it with
the studio's own frame protocol (`rig.frames`, `rig.sim.sender`) to the studio's web relay. It is a
2D schematic, not the 3D rig: our module is not inside Isaac Sim.

IMPORTANT: the relay accepts ONE source at a time and a new connection kicks the old one off. Both
senders retry every second, so pointing this at the same port as the Isaac sim makes the two fight
and the page flicker. Run it against its OWN relay (default port 9001). See docs/studio.md.

Live scene changes for the unscripted change in the video: type a command on stdin while it runs.

    leak [bolt] [radians]   loosen a bolt and start a leak     e.g.  leak 2 0.15
    crack [severity]        a crack at the flange              e.g.  crack 0.3
    dust [level]            degrade the camera (0 to 1)        e.g.  dust 0.5
    clear                   remove the dust
    next                    skip to the next scenario

Run:  python -m flange_robot.studio --sim 127.0.0.1:9001
"""

import argparse
import io
import math
import queue
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SafetyLimits
from .env import FLANGE_XY, N_BOLTS, Action, FlangeEnv
from .policy import Phase, RepairPolicy
from .scenarios import TOLERANCE_RAD, ScenarioSpec, apply_scenario

W, H = 1280, 720
BG, PANEL, TEXT, MUTED = (20, 22, 27), (30, 33, 40), (230, 232, 236), (140, 146, 158)
GREEN, RED, ORANGE, BLUE, GREY, YELLOW = (70, 200, 120), (235, 80, 80), (245, 160, 60), (90, 160, 255), (95, 100, 112), (240, 210, 90)
PHASE_COLOUR = {"approach": BLUE, "inspect": YELLOW, "request_permit": YELLOW, "repair": ORANGE, "verify": ORANGE,
                "done": GREEN, "escalated": RED}
SEVERITY_COLOUR = {"info": GREEN, "warning": ORANGE, "critical": RED}
GAS_MAX = 0.30

_fonts: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int):
    if size not in _fonts:
        _fonts[size] = ImageFont.load_default(size)
    return _fonts[size]


@dataclass
class View:
    """Everything the dashboard needs for one frame. Built from policy state, never from ground truth
    (except the optional debug markers)."""

    scenario: str = ""
    step: int = 0
    robot_xy: tuple[float, float] = (-3.0, 0.0)
    gas: float = 0.0
    gas_hist: list[float] = field(default_factory=list)
    phase: str = "approach"
    action: str = "wait"
    verdict: str = "allow"
    reason: str = ""
    diagnosis: str | None = None
    diagnosis_reason: str = ""
    crack_score: float = 0.0
    camera_conf: float = 0.0
    bolts: dict[int, tuple[float, float]] = field(default_factory=dict)  # bolt -> (est angle, confidence)
    alert: dict | None = None
    event: str = ""
    rereading: int | None = None  # a bolt just tightened, whose readings are being refreshed
    truth_loose: set[int] | None = None  # debug markers, only if requested
    limits: SafetyLimits = field(default_factory=SafetyLimits)


# -- drawing -----------------------------------------------------------------
def _text(d: ImageDraw.ImageDraw, xy, s: str, size: int = 18, fill=TEXT, anchor="la") -> None:
    d.text(xy, s, font=font(size), fill=fill, anchor=anchor)


def _wrap(d: ImageDraw.ImageDraw, s: str, size: int, width_px: int) -> list[str]:
    chars = max(10, int(width_px / (size * 0.52)))
    return textwrap.wrap(s, chars) or [""]


def _corridor(img: Image.Image, d: ImageDraw.ImageDraw, v: View) -> None:
    x0, y0, x1, y1 = 24, 64, 664, 196
    d.rounded_rectangle((x0, y0, x1, y1), 10, fill=PANEL)
    _text(d, (x0 + 12, y0 + 8), "Approach (top-down)", 15, MUTED)
    lo, hi = -5.5, 3.0
    sx = lambda x: x0 + 20 + (x - lo) / (hi - lo) * (x1 - x0 - 40)
    cy = (y0 + y1) // 2 + 10
    fx = sx(FLANGE_XY[0])
    if v.gas > 0.03:  # gas plume: rings that grow with the reading
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        reach = 25 + 380 * (v.gas - 0.03)
        for k in range(8):
            r = reach * (1 - k / 8)
            ld.ellipse((fx - r, cy - r, fx + r, cy + r), fill=(255, 110, 40, 16))
        img.alpha_composite(layer)
    d.line((x0 + 20, cy + 22, x1 - 20, cy + 22), fill=GREY, width=2)
    d.ellipse((fx - 20, cy - 20, fx + 20, cy + 20), outline=TEXT, width=3)
    _text(d, (fx, cy + 28), "flange", 14, MUTED, "ma")
    rx = sx(v.robot_xy[0])
    d.rectangle((rx - 14, cy - 8, rx + 14, cy + 8), fill=BLUE)
    near = abs(rx - fx) < 45  # keep the label off the flange ring when it arrives
    _text(d, (rx, cy - (34 if near else 14)), "robot", 14, BLUE, "ms")
    _text(d, (sx(-5.0), cy + 28), "standoff", 13, MUTED, "ma")


def _flange(img: Image.Image, d: ImageDraw.ImageDraw, v: View) -> None:
    x0, y0, x1, y1 = 24, 212, 664, 664
    d.rounded_rectangle((x0, y0, x1, y1), 10, fill=PANEL)
    _text(d, (x0 + 12, y0 + 8), "Flange close-up: what the robot believes", 15, MUTED)
    cx, cy, scale = (x0 + x1) // 2, (y0 + y1) // 2 + 2, 330.0
    d.ellipse((cx - 165, cy - 165, cx + 165, cy + 165), fill=(58, 62, 72), outline=(110, 116, 130), width=3)
    d.ellipse((cx - 90, cy - 90, cx + 90, cy + 90), fill=(26, 28, 34))
    for i in range(N_BOLTS):
        a = 2 * math.pi * i / N_BOLTS
        bx, by = cx + 0.4 * scale * math.cos(a), cy - 0.4 * scale * math.sin(a)
        est = v.bolts.get(i)
        if est is None and i == v.rereading:
            col, note = YELLOW, "re-reading"
        elif est is None:
            col, note = GREY, "unseen"
        elif est[1] < v.limits.min_detection_confidence:
            col, note = GREY, "unsure"
        elif abs(est[0]) <= TOLERANCE_RAD:
            col, note = GREEN, "ok"
        else:
            col, note = RED, "loose"
        d.ellipse((bx - 24, by - 24, bx + 24, by + 24), fill=(40, 43, 51), outline=col, width=4)
        ang = (est[0] * 6.0 if est else 0.0) + a  # offset exaggerated so a small turn is visible
        d.line((bx, by, bx + 22 * math.cos(ang), by - 22 * math.sin(ang)), fill=col, width=4)
        _text(d, (bx, by + 30), f"{i}", 15, TEXT, "ma")
        _text(d, (bx, by + 48), f"{note} {int(est[1] * 100)}%" if est else note, 13, col, "ma")
        if v.truth_loose is not None and i in v.truth_loose:
            d.ellipse((bx - 30, by - 30, bx + 30, by + 30), outline=(255, 255, 255), width=1)
    crack_known = v.diagnosis == "crack_or_burst" or (
        v.crack_score >= 0.5 and v.camera_conf >= v.limits.min_detection_confidence)
    if crack_known:
        pts = [(cx - 140, cy - 65), (cx - 55, cy - 10), (cx - 18, cy + 28), (cx + 45, cy - 5), (cx + 130, cy + 75)]
        d.line(pts, fill=RED, width=6)
        _text(d, (cx, cy + 196), f"CRACK diagnosed (score {v.crack_score:.2f})", 18, RED, "ma")
    elif v.crack_score >= 0.5:
        _text(d, (cx, cy + 196), f"possible crack, camera unsure (score {v.crack_score:.2f})", 15, YELLOW, "ma")
    else:
        _text(d, (cx, cy + 196), f"camera: no crack seen (score {v.crack_score:.2f})", 15, MUTED, "ma")
    if v.robot_xy[0] > FLANGE_XY[0] - 0.4:
        d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=BLUE)
        _text(d, (cx, cy), "R", 16, (10, 10, 14), "mm")


def _gauge(d: ImageDraw.ImageDraw, v: View, x0: int, y0: int, width: int) -> None:
    _text(d, (x0, y0), "Gas at the robot (fraction of LEL)", 15, MUTED)
    by = y0 + 28
    d.rounded_rectangle((x0, by, x0 + width, by + 22), 6, fill=(18, 19, 24))
    frac = min(v.gas / GAS_MAX, 1.0)
    col = GREEN if v.gas < 0.05 else (YELLOW if v.gas < v.limits.gas_stop_fraction_lel * 0.75 else RED)
    d.rounded_rectangle((x0, by, x0 + max(8, width * frac), by + 22), 6, fill=col)
    for val, label, c in ((0.05, "clear", GREEN), (v.limits.gas_stop_fraction_lel, "STOP", RED)):
        mx = x0 + width * val / GAS_MAX
        d.line((mx, by - 4, mx, by + 26), fill=c, width=2)
        _text(d, (mx, by + 28), label, 13, c, "ma")
    _text(d, (x0 + width, y0), f"{v.gas:.3f}", 18, TEXT, "ra")


def _sparkline(d: ImageDraw.ImageDraw, v: View, x0: int, y0: int, width: int, height: int) -> None:
    d.rounded_rectangle((x0, y0, x0 + width, y0 + height), 6, fill=(18, 19, 24))
    ly = y0 + height - height * v.limits.gas_stop_fraction_lel / GAS_MAX
    d.line((x0, ly, x0 + width, ly), fill=(120, 50, 50), width=1)
    hist = v.gas_hist[-160:]
    if len(hist) > 1:
        pts = [(x0 + 4 + (width - 8) * k / 159, y0 + height - 4 - (height - 8) * min(g, GAS_MAX) / GAS_MAX)
               for k, g in enumerate(hist)]
        d.line(pts, fill=ORANGE, width=2)


def render_frame(v: View) -> Image.Image:
    """Draw one dashboard frame."""
    img = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(img)
    _text(d, (24, 14), "Flange gas-leak response", 26, TEXT)
    _text(d, (W - 24, 18), f"{v.scenario}   step {v.step}", 18, MUTED, "ra")
    _corridor(img, d, v)
    d = ImageDraw.Draw(img)
    _flange(img, d, v)

    px, pw = 688, 568
    d.rounded_rectangle((px - 8, 64, W - 24, 664), 10, fill=PANEL)
    colour = PHASE_COLOUR.get(v.phase, GREY)
    d.rounded_rectangle((px + 8, 78, px + 8 + 190, 112), 17, fill=colour)
    _text(d, (px + 8 + 95, 95), v.phase.upper().replace("_", " "), 17, (10, 10, 14), "mm")
    diag = v.diagnosis or "gathering evidence"
    _text(d, (px + 216, 86), f"Diagnosis: {diag}", 19, TEXT)
    if v.diagnosis_reason:
        _text(d, (px + 216, 110), v.diagnosis_reason[: int(pw / 8.6)], 14, MUTED)

    _gauge(d, v, px + 8, 140, pw - 16)
    _sparkline(d, v, px + 8, 218, pw - 16, 70)

    _text(d, (px + 8, 302), "Perception", 15, MUTED)
    _text(d, (px + 8, 324), f"camera confidence {v.camera_conf * 100:.0f}%   crack score {v.crack_score:.2f}", 17)
    seen = len([b for b in v.bolts.values() if b[1] >= v.limits.min_detection_confidence])
    _text(d, (px + 8, 350), f"bolts read confidently {seen}/{N_BOLTS}   loose "
          f"{sum(1 for a, c in v.bolts.values() if c >= v.limits.min_detection_confidence and abs(a) > TOLERANCE_RAD)}", 17)

    _text(d, (px + 8, 392), "Decision", 15, MUTED)
    _text(d, (px + 8, 414), f"action: {v.action}    safety: {v.verdict}", 18)
    for k, line in enumerate(_wrap(d, v.reason, 15, pw - 16)[:3]):
        _text(d, (px + 8, 442 + 20 * k), line, 15, MUTED)

    if v.alert:
        col = SEVERITY_COLOUR.get(v.alert["severity"], GREY)
        d.rounded_rectangle((px + 8, 512, px + pw - 8, 652), 10, fill=(24, 26, 32), outline=col, width=3)
        _text(d, (px + 22, 522), f"ALERT TO SUPERVISOR  [{v.alert['severity'].upper()}]"
              + ("  ack required" if v.alert["requires_ack"] else ""), 16, col)
        for k, line in enumerate(_wrap(d, v.alert["message"], 15, pw - 44)[:3]):
            _text(d, (px + 22, 548 + 19 * k), line, 15, TEXT)
        for k, line in enumerate(_wrap(d, "Recommended: " + v.alert["recommended_action"], 14, pw - 44)[:2]):
            _text(d, (px + 22, 612 + 18 * k), line, 14, MUTED)
    if v.event:
        d.rounded_rectangle((24, 676, W - 24, 710), 8, fill=(120, 40, 40))
        _text(d, (W // 2, 693), f"LIVE CHANGE: {v.event}", 18, TEXT, "mm")
    return img.convert("RGB")


def to_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# -- the feed ------------------------------------------------------------------
class Sink(Protocol):
    def send(self, header: dict, payload: bytes) -> None: ...


SCENARIOS: list[tuple[str, ScenarioSpec, Callable[[int, FlangeEnv], str | None] | None]] = [
    ("loose bolts", ScenarioSpec("loose", bolts=((3, 0.12), (5, 0.09)), env_seed=1), None),
    ("crack", ScenarioSpec("crack", crack=0.3, env_seed=2), None),
    ("dusty camera", ScenarioSpec("loose", bolts=((3, 0.12),), dust=0.4, env_seed=3), None),
    ("big leak", ScenarioSpec("big", crack=0.8, env_seed=4), None),
    ("leak starts mid-run", ScenarioSpec("none", env_seed=5),
     lambda t, e: (e.trigger_gas_leak(2, 0.15), "a bolt works loose and a leak starts")[1] if t == 30 else None),
]


def apply_command(env: FlangeEnv, text: str) -> str | None:
    """Apply a live command. Returns a banner text, 'NEXT' to skip, or None if not understood."""
    parts = text.split()
    if not parts:
        return None
    cmd, args = parts[0].lower(), parts[1:]
    try:
        if cmd == "leak":
            bolt, ang = (int(args[0]) if args else 0), (float(args[1]) if len(args) > 1 else 0.15)
            env.trigger_gas_leak(bolt, ang)
            return f"bolt {bolt} loosened by {ang:.2f} rad, leak starting"
        if cmd == "crack":
            sev = float(args[0]) if args else 0.3
            env.trigger_crack(sev)
            return f"crack at the flange (severity {sev:.2f})"
        if cmd == "dust":
            lvl = float(args[0]) if args else 0.5
            env.set_dust(lvl)
            return f"dust on the camera ({lvl:.2f})"
        if cmd == "clear":
            env.set_dust(0.0)
            return "camera cleared"
        if cmd == "next":
            return "NEXT"
    except (ValueError, IndexError):
        return None
    return None


class StudioFeed:
    def __init__(self, sink: Sink, fps: float = 10.0, show_truth: bool = False, realtime: bool = True,
                 settle_steps: int = 25, hold_seconds: float = 4.0, event_frames: int = 40):
        self.sink, self.fps, self.show_truth, self.realtime = sink, fps, show_truth, realtime
        self.settle_steps, self.hold_seconds, self.event_frames = settle_steps, hold_seconds, event_frames
        self.commands: "queue.Queue[str]" = queue.Queue()
        self.results: list[dict] = []
        self.frames = 0
        self._t0 = time.monotonic()
        self._next_tick = self._t0

    def _emit(self, v: View) -> None:
        jpeg = to_jpeg(render_frame(v))
        mean = float(np.asarray(Image.open(io.BytesIO(jpeg)).convert("L")).mean())
        self.sink.send({"type": "frame", "w": W, "h": H, "seq": self.frames, "t": time.monotonic() - self._t0,
                        "mean": round(mean, 1), "scenario": v.scenario}, jpeg)
        self.frames += 1
        if self.realtime:
            self._next_tick += 1.0 / self.fps
            delay = self._next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_tick = time.monotonic()

    def run_scenario(self, name: str, spec: ScenarioSpec, hook, max_frames: int | None = None) -> None:
        env = FlangeEnv(seed=spec.env_seed)
        apply_scenario(env, spec)
        policy = RepairPolicy()
        obs = env.observe()
        gas_hist: list[float] = []
        event, event_left = "", 0
        settled, skipped = 0, False
        t, action, info = 0, Action("wait"), {"verdict": "allow", "reason": ""}  # defined even if skipped at once
        for t in range(400):
            if max_frames is not None and self.frames >= max_frames:
                break
            try:
                while True:
                    banner = apply_command(env, self.commands.get_nowait())
                    if banner == "NEXT":
                        skipped = True
                    elif banner:
                        event, event_left = banner, self.event_frames
            except queue.Empty:
                pass
            if hook:
                banner = hook(t, env)
                if banner:
                    event, event_left = banner, self.event_frames
            if skipped:
                break
            action, info = policy.act(obs)
            gas_hist.append(env.gas_fraction_lel())
            obs = env.step(action)
            event_left = max(0, event_left - 1)
            self._emit(self._view(name, t + 1, obs, action, info, policy, env, gas_hist, event if event_left else ""))
            if policy.phase in (Phase.DONE, Phase.ESCALATED):
                settled += 1
                if policy.phase == Phase.DONE or settled >= self.settle_steps:
                    break
        last = self._view(name, t + 1, obs, action, info, policy, env, gas_hist, "")
        for _ in range(int(self.hold_seconds * self.fps) if self.realtime else 2):
            if max_frames is not None and self.frames >= max_frames:
                break
            self._emit(last)
        self.results.append({"scenario": name, "alert": policy.alerts[0].kind if policy.alerts else None,
                             "phase": policy.phase.value})

    def _view(self, name, step, obs, action, info, policy, env, gas_hist, event) -> View:
        diag = policy.diagnosis
        truth = {i for i in range(N_BOLTS) if abs(env.true_bolt_angle(i)) > TOLERANCE_RAD} if self.show_truth else None
        return View(
            scenario=name, step=step, robot_xy=tuple(obs["robot_xy"]), gas=obs["gas_fraction_lel"],
            gas_hist=list(gas_hist), phase=policy.phase.value,
            action=action.kind + (f":{action.bolt}" if action.bolt is not None else ""),
            verdict=info["verdict"], reason=info["reason"],
            diagnosis=diag.kind.value if diag else None, diagnosis_reason=diag.reason if diag else "",
            crack_score=policy.diagnoser.crack_score, camera_conf=obs["camera"]["crack_confidence"],
            bolts=policy.bolt_estimates(), alert=policy.alerts[0].to_dict() if policy.alerts else None,
            rereading=policy.target_bolt if policy.phase == Phase.VERIFY else None,
            event=event, truth_loose=truth,
        )

    def run(self, scenarios=None, loop: bool = True, max_frames: int | None = None) -> None:
        scenarios = scenarios or SCENARIOS
        while True:
            for name, spec, hook in scenarios:
                self.run_scenario(name, spec, hook, max_frames)
                if max_frames is not None and self.frames >= max_frames:
                    return
            if not loop:
                return


def _read_stdin(feed: StudioFeed) -> None:
    for line in sys.stdin:
        feed.commands.put(line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="127.0.0.1:9001", help="host:port of the web relay's frame ingest (NOT the Isaac relay)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--once", action="store_true", help="run each scenario once and stop")
    ap.add_argument("--show-truth", action="store_true", help="debug: ring the bolts that are truly loose")
    ap.add_argument("--snapshot", default="", help="save the first frame of each scenario as PNG files with this prefix")
    ap.add_argument("--no-stdin", action="store_true")
    a = ap.parse_args()

    from rig.sim.sender import FrameSender

    host, port = a.sim.rsplit(":", 1)
    sender = FrameSender(host, int(port)).start()

    class Sink:
        def send(self, header, payload):
            sender.send(header, payload)

    feed = StudioFeed(Sink(), fps=a.fps, show_truth=a.show_truth)
    if not a.no_stdin and sys.stdin and not sys.stdin.isatty():
        pass  # piped input is allowed too
    if not a.no_stdin:
        threading.Thread(target=_read_stdin, args=(feed,), daemon=True).start()
    print(f"studio feed -> {a.sim} at {a.fps} fps. Commands: leak, crack, dust, clear, next", flush=True)
    try:
        feed.run(loop=not a.once)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"sent={sender.sent} dropped={sender.dropped} frames={feed.frames}", flush=True)
        sender.stop()


if __name__ == "__main__":
    main()

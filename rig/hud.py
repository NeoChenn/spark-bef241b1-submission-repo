import base64
import io
import json
import random
from pathlib import Path

import numpy as np

HOOKS = ("spawn_patch", "set_glare", "occlude_camera", "force_spike", "person_at")
EXTRA = ("camera",)
FLEET = ("drop_obstacle", "close_walkway", "close_stairs", "clear", "set_weather")
RATES = {"SOAK": 0.15, "SCRUB": 0.35}
DWELL = 1.3
THUMB_W = 320
PATCH_OFFSET = (0.45, 0.1)


def _bits(mask) -> list[str]:
    m = np.asarray(mask, bool)
    return ["".join("1" if v else "0" for v in row) for row in m]


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def thumb(img: np.ndarray, mask=None) -> dict:
    from PIL import Image
    rgb = np.ascontiguousarray(np.asarray(img)[..., :3]).astype(np.uint8).copy()
    if mask is not None and np.shape(mask) == rgb.shape[:2]:
        m = np.asarray(mask, bool)
        inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
        edge = m & ~inner
        edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        rgb[edge] = (255, 59, 48)
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb).resize((THUMB_W, round(THUMB_W * h / w)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return {"jpg": base64.b64encode(buf.getvalue()).decode(), "w": w, "h": h}


class HudFeed:
    def __init__(self, orch):
        self.orch = orch
        self.seen = 0
        self.last_perceived = None
        self.verdict_seen = None
        self.source = "EDGE"

    def header(self) -> dict:
        o = self.orch
        world, pipe = o.world, o.pipe
        records = o.records[self.seen:]
        self.seen = len(o.records)
        verdict = getattr(o, "advisor_last", None)
        if records:
            self.source = "CLOUD" if verdict is not None and verdict is not self.verdict_seen else "EDGE"
            self.verdict_seen = verdict
        conf = None
        if o.last is not None:
            cs = [p.mean_conf for p in o.last.result.patches]
            conf = round(float(max(cs) if cs else o.last.result.max_conf), 3)
        theta, z = o.arm.ee_theta_z()
        i, j = pipe.theta_z_to_cell(theta, z)
        out = {
            "t": round(float(o.t), 2),
            "state": o.state.value,
            "hud": o.overlay_text(),
            "events": [json.loads(json.dumps(r.to_json(), default=_plain)) for r in records],
            "rust_grid": np.round(np.clip(_np(world.rust[0]), 0, 1), 2).tolist(),
            "sprayed": _bits(_np(world.sprayed[0])),
            "painted": _bits(_np(world.painted[0])),
            "target": _bits(o.target),
            "brush": [int(i), int(j)],
            "force_n": round(float(o.arm.current_force()), 2),
            "force_limit": float(o.cfg.force_limit),
            "detector_conf": conf,
            "rust_fraction": None if o.rust_frac is None else round(float(o.rust_frac), 4),
            "gemini": self._gemini(verdict),
            "source": self.source,
            "soak": {"left": round(max(0.0, o.soak_until - o.t), 2), "total": float(o.cfg.soak_s)}
            if o.state.value == "SOAK" else None,
        }
        if o.last is not None and o.last is not self.last_perceived and o.last_image is not None:
            self.last_perceived = o.last
            out["inspection"] = thumb(o.last_image, o.last.result.mask)
        return out

    def _gemini(self, verdict):
        if verdict is None:
            return None
        calls = getattr(self.orch.advisor, "calls", None) or [{}]
        return {**verdict.summary(), "question": calls[-1].get("question")}


def _plain(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def is_fleet_event(name: str, args: dict) -> bool:
    return name in FLEET or (name == "person_at" and any(k in args for k in ("xy", "xyz", "at")))


def apply_event(bridge, event: dict, fleet=None) -> None:
    name = event.get("name")
    if name not in HOOKS + EXTRA + FLEET:
        raise ValueError(f"unknown event {name!r}")
    args = dict(event.get("args") or {})
    if is_fleet_event(name, args):
        if fleet is not None:
            fleet.event(name, args)
        hook = getattr(bridge, name, None) if name == "set_weather" else None
        if hook is not None:
            hook(**{k: v for k, v in args.items() if k in ("level", "wind_dir", "gust")})
        return
    if name == "spawn_patch" and ("theta" not in args or "z" not in args):
        theta, z = bridge.ee_theta_z()
        args.setdefault("theta", float(theta) + PATCH_OFFSET[0])
        args.setdefault("z", float(z) + PATCH_OFFSET[1])
    getattr(bridge, name)(**args)


def playback_times(snaps: list[dict]) -> list[float]:
    pts, pt = [], 0.0
    for k, s in enumerate(snaps):
        if k:
            prev = snaps[k - 1]
            pt += (s["t"] - prev["t"]) * RATES.get(prev["state"], 1.0)
            if s["state"] != prev["state"]:
                pt += DWELL
            if s.get("gemini") and s["gemini"] != prev.get("gemini"):
                pt += s["gemini"].get("latency_ms", 0) / 1000
        pts.append(round(pt, 3))
    return pts


class ScriptedAdvisor:
    model = "gemini-robotics-er-2-preview (scripted)"

    def __init__(self, veto: int = 1):
        self.veto, self.calls, self.last_error = veto, [], None
        self.rng = random.Random(7)

    def verify(self, image, question="rust"):
        from policy.gemini_verifier import GeminiVerdict
        img = np.asarray(image, np.int32)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        score = r - b
        ys, xs = np.nonzero((score > 55) & (r > g + 10))
        if not len(xs):
            ys, xs = np.nonzero(score >= np.sort(score, axis=None)[-40])
        rust = question == "rust" or self.veto > 0
        if question == "clean" and self.veto > 0:
            self.veto -= 1
        points = []
        if rust and len(xs):
            order = np.argsort(xs)
            for q in (0.15, 0.5, 0.85):
                k = order[int(q * (len(order) - 1))]
                points.append((float(xs[k]), float(ys[k])))
        verdict = GeminiVerdict("rust" if rust else "clean", 0.82 if rust else 0.91,
                                "orange-brown residue still visible along the scrubbed band" if rust
                                else "uniform bare steel, no oxide texture left", points,
                                latency_ms=self.rng.uniform(900, 1500), model=self.model)
        self.calls.append({"question": question, **verdict.summary()})
        return verdict

    def confident(self, verdict):
        return verdict is not None and verdict.verdict != "cannot_tell" and verdict.confidence >= 0.6


def record_mock(path, log_path: str = "runs/repair/decisions.jsonl") -> dict:
    from policy.demo_mock import build
    from policy.orchestrator import S
    orch, bridge, world, perception, arm, pipe = build(log_path, verbose=False)
    orch.advisor = ScriptedAdvisor()
    feed = HudFeed(orch)
    snaps: list[dict] = []
    last = {"t": -1.0}

    def snap(force=False):
        if force or orch.t - last["t"] >= 0.5:
            last["t"] = orch.t
            snaps.append(feed.header())

    advance = bridge.advance

    def tracked(dt):
        advance(dt)
        snap()
    bridge.advance = tracked
    arm.spike_at_step = 40
    flags = set()
    snap(True)
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN) and len(orch.records) < 400:
        prev = orch.state
        orch.step()
        if prev == S.SPRAY and "patch" not in flags:
            flags.add("patch")
            th = bridge.dmg_theta + 0.75
            bridge.events.append((bridge.t + 12.0, "new rust", lambda: world.inject_patch(th, 0.13, size=0.04, level=0.8)))
        if orch.state == S.VERIFY and prev == S.SCRUB and "occl" not in flags:
            flags.add("occl")
            perception.occluded_frames = 2
            bridge.person_window = (bridge.t + 0.5, bridge.t + 2.5)
        snap(True)
    for k, pt in enumerate(playback_times(snaps)):
        snaps[k]["pt"] = pt
    for k in range(len(snaps) - 1, 0, -1):
        for key in ("rust_grid", "sprayed", "painted", "target"):
            if snaps[k][key] == snaps[k - 1][key]:
                del snaps[k][key]
    session = {"snapshots": snaps, "duration": snaps[-1]["pt"]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(session, separators=(",", ":")))
    return session


if __name__ == "__main__":
    import sys
    s = record_mock(sys.argv[1] if len(sys.argv) > 1 else "runs/hud/mock_session.json")
    print(f"{len(s['snapshots'])} snapshots, {s['duration']:.1f} s playback")

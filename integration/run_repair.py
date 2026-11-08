"""Run the rust repair loop end to end inside Isaac Sim 6.2 on the rig deck.

    C:\\isaacsim\\python.bat integration\\run_repair.py [--detector runs/detector/best] [--max-steps 400]

Needs torch in Isaac's Python: install it once into a separate folder and point ISAAC_DEPS at it
(default %USERPROFILE%\\.isaac-deps):
    C:\\isaacsim\\python.bat -m pip install --target %USERPROFILE%\\.isaac-deps --index-url https://download.pytorch.org/whl/cpu torch

Writes runs/repair/: decisions.jsonl, repair.mp4 (hero view, HUD, detector inset) and inspection stills.
Without --detector (a folder with model.onnx + meta.json) it uses the colour stand-in classifier from the
mock demo, and the scrub policy is the mock's greedy policy; both are printed at start-up.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
deps = Path(os.environ.get("ISAAC_DEPS", Path.home() / ".isaac-deps"))
if deps.exists():
    sys.path.insert(0, str(deps))

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--detector", default="", help="folder with model.onnx + meta.json (default: colour stand-in)")
parser.add_argument("--out", default=str(REPO / "runs/repair"))
parser.add_argument("--max-steps", type=int, default=400, help="orchestrator steps before giving up")
parser.add_argument("--subframes", type=int, default=4, help="RTX subframes per rendered frame")
parser.add_argument("--fps", type=int, default=12, help="video frame rate")
parser.add_argument("--hero-size", default="1280x720")
parser.add_argument("--no-events", action="store_true", help="skip the unscripted scene changes")
parser.add_argument("--glare", action="store_true", help="add glare on the nominal inspection view")
parser.add_argument("--stream", default="", help="host:port of the web viewer's sim socket, e.g. 127.0.0.1:9000")
parser.add_argument("--no-gemini", action="store_true",
                    help="skip the Gemini second opinion even if GEMINI_API_KEY is set")
args = parser.parse_args()

from isaacsim import SimulationApp

hero_w, hero_h = (int(v) for v in args.hero_size.split("x"))
app = SimulationApp({"headless": True, "width": hero_w, "height": hero_h})

import numpy as np
import omni.kit.app
import omni.usd
import torch
from PIL import Image, ImageDraw

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.replicator.core", True)
manager.set_extension_enabled_immediate("omni.videoencoding", True)
import omni.replicator.core as rep
from pxr import UsdGeom
from video_encoding import get_video_encoding_interface

from detection.localize import RustLocalizer, fake_colour_classifier
from integration import repair_scene as rs
from integration.isaac_bridge import Frames, IsaacRigBridge
from policy.demo_mock import GreedyScrub, MockRustWorld
from policy.gemini_verifier import GeminiVerifier
from policy.orchestrator import Orchestrator, OrchestratorConfig, S, draw_hud

OUT = Path(args.out)
FRAMES = OUT / "frames"
INSPECTION_SIZE = (640, 360)


def open_deck():
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(rs.DECK_SCENE)):
        raise SystemExit(f"could not open {rs.DECK_SCENE}")
    for _ in range(600):
        app.update()
        if ctx.get_stage_loading_status()[2] == 0:
            break
    return ctx.get_stage()


def setup_scene(stage):
    meta = json.loads((REPO / "sim/output/rig_deck_meta.json").read_text())
    pipe_meta, damage = meta["main_pipe"], meta["damage"]
    origin = (float(damage["position_m"][0]), float(pipe_meta["y_m"]), float(pipe_meta["z_m"]))
    rs.write_pipe_attrs(rs.find_main_pipe(stage), origin, radius=float(pipe_meta["radius_m"]))
    patch = stage.GetPrimAtPath(f"{rs.PIPE_SCOPE}/damage_patch")
    if patch.IsValid():
        UsdGeom.Imageable(patch).MakeInvisible()
    for robot in ("fitter", "welder"):
        prim = stage.GetPrimAtPath(f"/World/Robots/{robot}")
        if prim.IsValid():
            prim.SetActive(False)
    n = damage["surface_normal"]
    theta = math.atan2(float(n[2]), float(n[1])) % (2 * math.pi)
    hero = rs.Camera(stage, rs.HERO_CAMERA, focal=24.0, aperture=20.955, clip=(0.05, 4000.0))
    shots = {1: ((origin[0] + 9.0, 9.5, 6.5), (origin[0] - 0.5, 0.0, 0.8)),
             2: ((origin[0] + 2.6, 2.9, 2.15), (origin[0] - 0.1, 0.25, 0.95))}
    hero.look(*shots[2])
    return theta, hero, shots


def make_world(pipe, theta):
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    world.set_rust(world.blob(theta, 0.0, size=0.05, level=0.85))
    world.damage_theta = theta
    return world


def make_localizer():
    if args.detector and (Path(args.detector) / "model.onnx").exists():
        print(f"DETECTOR onnx {args.detector}", flush=True)
        return RustLocalizer.from_checkpoint(args.detector, patch=96, stride=48)
    print("DETECTOR colour stand-in (no --detector model.onnx given)", flush=True)
    return RustLocalizer(fake_colour_classifier, patch=32, stride=16, threshold=0.5, temperature=1.0)


def inset(frame: np.ndarray, inspection: np.ndarray | None, mask: np.ndarray | None, label: str) -> np.ndarray:
    """Detector view in the bottom-right corner: the inspection frame with the rust mask outlined in red."""
    if inspection is None:
        return frame
    view = inspection[..., :3].astype(np.float32).copy()
    if mask is not None and mask.shape == view.shape[:2]:
        view[mask] = view[mask] * 0.45 + np.array([255.0, 30.0, 30.0]) * 0.55
    w = frame.shape[1] // 3
    h = int(w * view.shape[0] / view.shape[1])
    small = np.asarray(Image.fromarray(view.clip(0, 255).astype(np.uint8)).resize((w, h)))
    out = frame.copy()
    y0, x0 = out.shape[0] - h - 12, out.shape[1] - w - 12
    out[y0 - 2:y0 + h + 2, x0 - 2:x0 + w + 2] = 235
    out[y0:y0 + h, x0:x0 + w] = small
    im = Image.fromarray(out)
    ImageDraw.Draw(im).text((x0 + 6, y0 + 4), label, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(exist_ok=True)
    for old in FRAMES.glob("*.jpg"):
        old.unlink()
    stage = open_deck()
    theta, hero, shots = setup_scene(stage)
    pipe = rs.read_pipe(rs.find_main_pipe(stage))
    world = make_world(pipe, theta)

    state = {"orch": None, "frame": 0, "shot": 2, "feed": None}
    sender = None
    if args.stream:
        from rig.hud import HudFeed, apply_event
        from rig.sim.sender import FrameSender
        host, port = args.stream.rsplit(":", 1)
        sender = FrameSender(host, int(port)).start()
        print(f"STREAM frames + HUD header to {args.stream}", flush=True)
    annotators = {}

    def render() -> dict:
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        out = {}
        for key, annotator in annotators.items():
            data = np.asarray(annotator.get_data())
            out[key] = np.ascontiguousarray(data[..., :3]).astype(np.uint8) if data.size else None
        return out

    def record(rgb: np.ndarray, hud: list) -> None:
        orch = state["orch"]
        mask = orch.last.result.mask if orch is not None and orch.last is not None else None
        frame = inset(rgb, bridge.last_inspection, mask, "inspection camera + detector mask")
        frame = draw_hud(frame, hud) if hud else frame
        Image.fromarray(frame).save(FRAMES / f"{state['frame']:05d}.jpg", quality=92)
        state["frame"] += 1
        if sender is not None and state["feed"] is not None:
            stream(rgb)

    def stream(rgb: np.ndarray) -> None:
        while not sender.inbox.empty():
            event = sender.inbox.get_nowait()
            try:
                apply_event(bridge, event)
                print(f"   >>> [web event t={bridge.sim_time():5.1f}s] {event}", flush=True)
            except Exception as e:
                print(f"   >>> [web event rejected] {event}: {e}", flush=True)
        shot = bridge.state.shot
        if shot != state["shot"]:
            state["shot"] = shot
            if shot in shots:
                hero.look(*shots[shot])
        img = bridge.last_inspection if shot == 3 and bridge.last_inspection is not None else rgb
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(buf, "JPEG", quality=85)
        header = {"type": "frame", "w": img.shape[1], "h": img.shape[0], "seq": state["frame"],
                  "t": round(bridge.sim_time(), 2), "mean": round(float(img.mean()), 1), "shot": shot}
        header.update(state["feed"].header())
        sender.send(header, buf.getvalue())

    bridge = IsaacRigBridge(stage, world, Frames(render, record), inspection_size=INSPECTION_SIZE)
    for _ in range(20):
        app.update()
    inspection_rp = rep.create.render_product(rs.INSPECTION_CAMERA, INSPECTION_SIZE)
    hero_rp = rep.create.render_product(rs.HERO_CAMERA, (hero_w, hero_h))
    annotators.update(inspection=rep.AnnotatorRegistry.get_annotator("rgb"),
                      hero=rep.AnnotatorRegistry.get_annotator("rgb"))
    annotators["inspection"].attach(inspection_rp)
    annotators["hero"].attach(hero_rp)
    for _ in range(6):
        render()

    localizer = make_localizer()
    print("SCRUB POLICY greedy stand-in (policy.demo_mock.GreedyScrub)", flush=True)
    advisor = None
    if not args.no_gemini:
        advisor = GeminiVerifier()
        if advisor.available:
            print(f"GEMINI second opinion on, model {advisor.model}", flush=True)
        else:
            print("GEMINI off (no GEMINI_API_KEY in the environment or .env)", flush=True)
            advisor = None
    orch = Orchestrator(bridge, bridge, localizer, world, bridge, GreedyScrub(bridge.pipe()),
                        OrchestratorConfig(log_path=str(OUT / "decisions.jsonl"), scrub_steps=200,
                                           scrub_redetect_every=40), advisor=advisor)
    state["orch"] = orch
    if sender is not None:
        state["feed"] = HudFeed(orch)
    if args.glare:
        bridge.set_glare(True)
        original = bridge.request_view

        def request_view(index):
            bridge.set_glare(index == 0)
            return original(index)
        bridge.request_view = request_view

    fired = set()
    started = time.time()
    for step in range(args.max_steps):
        if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN):
            break
        prev = orch.state
        orch.step()
        now = bridge.sim_time()
        if args.no_events:
            continue
        if prev == S.SPRAY and "patch" not in fired:
            fired.add("patch")
            th_new, z_new = theta + 0.28, 0.12
            bridge.schedule(now + 12.0, f"NEW rust patch at theta={th_new:.2f}, z={z_new:+.2f}",
                            lambda: bridge.spawn_patch(th_new, z_new, size=0.035, level=0.85))
        if orch.state == S.SCRUB and "bead" not in fired:
            fired.add("bead")
            bridge.schedule(now + 4.0, "weld bead under the brush (force spike)", bridge.force_spike)
        if prev == S.SCRUB and orch.state == S.VERIFY and "mist" not in fired:
            fired.add("mist")
            bridge.occlude_camera(True)
            bridge.schedule(now + 1.5, "lens mist cleared", lambda: bridge.occlude_camera(False))
            bridge.schedule(now + 0.5, "worker walks up to 2.4 m", lambda: bridge.person_at(2.4))
            bridge.schedule(now + 2.5, "worker leaves", lambda: bridge.person_at(None))
        if step % 10 == 0:
            print(f"step {step} t={now:.1f}s state={orch.state.value} frames={state['frame']} "
                  f"wall={time.time() - started:.0f}s", flush=True)

    for r in orch.records:
        ev = r.evidence
        print(f"t={r.t:6.1f}s {r.frm:>13} -> {r.to:<17} conf={ev.get('detector_conf')} "
              f"rust={ev.get('rust_fraction')} F={ev.get('force')} | {r.reason}", flush=True)
    true_left = float(world.fraction_remaining()[0])
    print(f"FINAL {orch.state.value} decisions={len(orch.records)} true_rust_left={true_left:.3f} "
          f"painted={int(world.painted.sum())} frames={state['frame']} wall={time.time() - started:.0f}s", flush=True)
    if bridge.last_inspection is not None:
        Image.fromarray(bridge.last_inspection).save(OUT / "inspection_last.png")
    if advisor is not None:
        (OUT / "gemini_calls.json").write_text(json.dumps(advisor.calls, indent=2))
        print(f"GEMINI calls={len(advisor.calls)} -> {OUT / 'gemini_calls.json'}", flush=True)
    encode_video(state["frame"])


def encode_video(count: int) -> None:
    if not count:
        print("no frames recorded", flush=True)
        return
    path = OUT / "repair.mp4"
    encoder = get_video_encoding_interface()
    if not encoder.start_encoding(str(path), float(args.fps), count, True):
        raise RuntimeError(f"could not start video encoding for {path}")
    written = 0
    for k in range(count):
        rgb = np.asarray(Image.open(FRAMES / f"{k:05d}.jpg").convert("RGBA"))
        if encoder.encode_next_frame_from_buffer(np.ascontiguousarray(rgb), rgb.shape[1], rgb.shape[0]):
            written += 1
    encoder.finalize_encoding()
    if not path.exists() or path.stat().st_size == 0 or written == 0:
        raise RuntimeError(f"no video written to {path}")
    print(f"VIDEO_WRITTEN {path} frames={written}/{count} seconds={count / args.fps:.1f}", flush=True)


main()
os._exit(0)

from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from detection.leak import LeakLocalizer
from detection.localize import RustLocalizer, fake_colour_classifier
from integration import repair_scene as rs
from integration.gas_bot_bridge import OGI_CAMERA, IsaacGasBot, add_deck_flanges
from integration.isaac_bridge import Frames
from policy.demo_mock import GreedyScrub, MockRustWorld
from policy.gasbot import deck
from policy.gasbot.bot import GasBotOrchestrator
from policy.gasbot.mock import make_leak
from policy.gasbot.plume import Wind, deck_solver_field
from policy.gasbot.sensors import OGI_RAMP, column_density, ogi_view
from policy.interfaces import camera_from_usd
from policy.leak.geometry import project
from policy.leak.job import LeakConfig, LeakJob
from policy.leak.world import LeakWorld
from policy.orchestrator import TERMINAL, OrchestratorConfig, S, draw_hud

OGI_SIZE = (480, 270)
INSPECTION_SIZE = (640, 360)
ACCENT = (217, 119, 87)


def gas_panel(img: Image.Image, g: dict) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    W = img.width
    x0, y0, w = W - 300, 12, 288
    rows = [("CH4 trunk IR", f"{g['sensors'][0]['lel']:.1f} %LEL", g["sensors"][0]["lel"]),
            ("CH4 arm tip", f"{g['sensors'][1]['lel']:.1f} %LEL", g["sensors"][1]["lel"]),
            ("H2S", f"{g['h2s_ppm']:.1f} ppm", None),
            ("Ultrasound", "--" if not g["acoustic"] else f"{g['acoustic']['db']:.0f} dB (+{max(g['acoustic']['snr_db'], 0):.0f})", None),
            ("Source", "--" if not g["source"] else
             f"{g['source']['x']:.1f}, {g['source']['y']:.1f} +/-{g['source']['sigma_m']:.1f} m", None),
            ("Flow", "--" if g["flow_gs"] is None else f"{g['flow_gs']:.1f} g/s", None),
            ("Wind", f"{g['wind']['dir_deg']:.0f} deg {g['wind']['speed_ms']:.1f} m/s", None)]
    h = 34 + 20 * len(rows) + 8
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], 12, fill=(38, 38, 36, 215))
    col = {"OK": (63, 185, 80), "DETECT": (227, 179, 65), "ALARM": (248, 81, 73), "RETREAT": (248, 81, 73)}
    d.text((x0 + 12, y0 + 9), f"GAS SENSORS  {g['unit']}", fill=(245, 244, 238))
    d.text((x0 + w - 70, y0 + 9), g["alarm"], fill=col[g["alarm"]])
    for k, (name, val, bar) in enumerate(rows):
        y = y0 + 32 + 20 * k
        d.text((x0 + 12, y), name, fill=(190, 189, 182))
        d.text((x0 + 130, y), val, fill=(245, 244, 238))
        if bar is not None:
            bw = int(min(1.0, bar / 100.0) * 50)
            c = (248, 81, 73) if bar >= 20 else (227, 179, 65) if bar >= 10 else (63, 185, 80)
            d.rectangle([x0 + w - 62, y + 5, x0 + w - 12, y + 8], fill=(80, 80, 76))
            d.rectangle([x0 + w - 62, y + 5, x0 + w - 62 + bw, y + 8], fill=c)


def paste(img: Image.Image, small: np.ndarray, corner: str, label: str, width: int) -> None:
    s = Image.fromarray(small).resize((width, int(width * small.shape[0] / small.shape[1])))
    W, H = img.size
    x = 12 if corner == "left" else W - s.width - 12
    y = H - s.height - 12
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([x - 2, y - 2, x + s.width + 2, y + s.height + 2], fill=(235, 235, 235))
    img.paste(s, (x, y))
    d.rectangle([x, y, x + 8 + 7 * len(label), y + 16], fill=(0, 0, 0, 150))
    d.text((x + 4, y + 2), label, fill=(255, 255, 255))


def bolts_overlay(rgb: np.ndarray, reading, tol: float) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    if reading is None:
        return np.asarray(im)
    d = ImageDraw.Draw(im)
    for b in reading.bolts:
        cx, cy = b.center_px
        if b.angle is None or not np.isfinite(cx):
            continue
        c = (63, 185, 80) if abs(b.angle) <= tol else (248, 81, 73)
        d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], outline=c, width=2)
        d.text((cx + 15, cy - 6), f"{math.degrees(b.angle):+.0f}", fill=c)
    return np.asarray(im)


def plume_tint(rgb: np.ndarray, col: np.ndarray, threshold: float, top: float) -> np.ndarray:
    u = np.clip((np.log10(np.maximum(col, 1e-9)) - math.log10(threshold)) / math.log10(top / threshold), 0, 1)
    ramp = OGI_RAMP[1:]
    x = u * (len(ramp) - 1)
    k = np.minimum(x.astype(int), len(ramp) - 2)
    f = (x - k)[..., None]
    colour = ramp[k] * (1 - f) + ramp[k + 1] * f
    a = np.where(col >= threshold, 0.18 + 0.42 * u, 0.0)[..., None]
    return (rgb.astype(np.float32) * (1 - a) + colour * a).clip(0, 255).astype(np.uint8)


def run(app, rep, stage, args, out: Path, hero, hero_size, encode) -> None:
    frames_dir = out / "frames"
    prims = add_deck_flanges(stage)
    for _ in range(10):
        app.update()
    geoms = deck.flanges()
    lw = LeakWorld(geoms)
    rng = np.random.default_rng(args.seed)
    flow_full = {g.id: 0.0 for g in geoms}
    flow_full[args.leak_flange] = make_leak(rng, lw, args.leak, args.leak_flange)
    lw.settle()
    if args.leak_flow > 0:
        flow_full[args.leak_flange] = args.leak_flow / max(lw.flanges[args.leak_flange].level, 1e-6)
    wind = Wind(args.wind_speed, math.radians(args.wind_dir), meander_deg=10.0, seed=args.seed)
    pipe = rs.read_pipe(rs.find_main_pipe(stage))
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    import torch
    world.set_rust(torch.zeros(1, pipe.n_z, pipe.n_theta))
    world.damage_theta = 1.0
    annotators = {}
    cam = {"eye": None, "look": None}
    st = {"frame": 0, "orch": None, "t0": time.time()}

    def follow():
        b = st["bridge"]
        p = b.base_xy()
        if b.at_flange:
            g = b.flange(b.at_flange)
            look = np.array([g.center[0] - 0.2, 0.9, 0.75])
            eye = np.array([g.center[0] + 2.3, 2.85, 1.9])
        else:
            look = np.array([p[0] - 1.6, 1.0, 0.55])
            eye = np.array([min(p[0] + 4.4, 8.2), 2.9, 2.8])
        k = 1.0 if cam["eye"] is None else 0.12
        cam["eye"] = eye if cam["eye"] is None else cam["eye"] + k * (eye - cam["eye"])
        cam["look"] = look if cam["look"] is None else cam["look"] + k * (look - cam["look"])
        hero.look(cam["eye"], cam["look"])

    def render() -> dict:
        if "bridge" in st:
            follow()
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        res = {}
        for key, a in annotators.items():
            data = np.asarray(a.get_data())
            res[key] = np.ascontiguousarray(data[..., :3]).astype(np.uint8) if data.size else None
        st["images"] = res
        return res

    def record(rgb: np.ndarray, hud: list) -> None:
        b, orch = st["bridge"], st["orch"]
        if orch is None:
            return
        H, W = rgb.shape[:2]
        cm = camera_from_usd(hero.world_matrix(), hero.focal, hero.aperture, W, H)
        from detection.localize import pixel_rays
        o, dirs = pixel_rays(cm)
        gy, gx = np.linspace(0, H - 1, 54).astype(int), np.linspace(0, W - 1, 96).astype(int)
        col = column_density(b.field, np.broadcast_to(o, dirs.shape)[gy][:, gx].reshape(-1, 3),
                             dirs[gy][:, gx].reshape(-1, 3), 14.0, 20).reshape(54, 96)
        up = np.asarray(Image.fromarray(col.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
        im = Image.fromarray(plume_tint(rgb, up, 8000.0, 200000.0))
        d = ImageDraw.Draw(im, "RGBA")
        if orch.estimate is not None and orch.state in (S.GAS_SEEK, S.LEAK_APPROACH):
            m, sig, _ = orch.estimate
            ring = np.array([[m[0] + max(sig, 0.15) * math.cos(t), m[1] + max(sig, 0.15) * math.sin(t), 0.03]
                             for t in np.linspace(0, 2 * math.pi, 40)])
            uv, z = project(cm, ring)
            if (z > 0.1).all():
                d.line([tuple(p) for p in uv] + [tuple(uv[0])], fill=ACCENT + (255,), width=3)
                c, zc = project(cm, np.array([[m[0], m[1], 0.03]]))
                d.text((c[0][0] + 8, c[0][1] - 18), f"source est +/-{sig:.1f} m", fill=ACCENT)
        im = Image.fromarray(draw_hud(np.asarray(im), orch.overlay_text()))
        gas_panel(im, orch.gas_telemetry())
        imgs = st.get("images", {})
        if imgs.get("ogi") is not None:
            ogi, stats = ogi_view(imgs["ogi"], b.ogi_camera(*OGI_SIZE), b.field, threshold=8000.0, top=200000.0,
                                  grid=(80, 45))
            b.ogi_stats = stats
            paste(im, ogi, "left", "unit 2 OGI camera (gas false colour)", W // 4)
        if b.leak_mode and b.last_inspection is not None:
            ins = bolts_overlay(b.last_inspection, orch.last_leak_reading, orch.lcfg.tolerance_rad)
            paste(im, ins, "right", "flange camera: torque stripes", W // 3)
        im.save(frames_dir / f"{st['frame']:05d}.jpg", quality=90)
        st["frame"] += 1

    field = (deck_solver_field(lw.flanges[args.leak_flange].geom, args.leak_flange, wind, args.wind_speed,
                               args.wind_dir, args.hole_mm, args.bar, stage) if args.plume == "solver" else None)
    bridge = IsaacGasBot(stage, world, Frames(render, record), lw, prims, wind, seed=args.seed, flow_full=flow_full,
                         h2s_ratio=args.h2s_ratio, start=tuple(args.start), inspection_size=INSPECTION_SIZE,
                         frame_dt=1.0 / args.fps, field=field)
    st["bridge"] = bridge
    for _ in range(20):
        app.update()
    rps = {"inspection": rep.create.render_product(rs.INSPECTION_CAMERA, INSPECTION_SIZE),
           "hero": rep.create.render_product(rs.HERO_CAMERA, hero_size),
           "ogi": rep.create.render_product(OGI_CAMERA, OGI_SIZE)}
    for k, rp in rps.items():
        annotators[k] = rep.AnnotatorRegistry.get_annotator("rgb")
        annotators[k].attach(rp)
    for _ in range(6):
        render()
    orch = GasBotOrchestrator(bridge, bridge, RustLocalizer(fake_colour_classifier, patch=32, stride=16),
                              world, bridge, GreedyScrub(bridge.pipe()),
                              OrchestratorConfig(log_path=str(out / "decisions.jsonl")),
                              leak=LeakJob(LeakLocalizer(), LeakConfig(max_settle_reads=15)))
    st["orch"] = orch
    sender, feed = None, None
    if args.stream:
        from rig.hud import HudFeed
        from rig.sim.sender import FrameSender
        host, port = args.stream.rsplit(":", 1)
        sender, feed = FrameSender(host, int(port)).start(), HudFeed(orch)
    truth = {"fid": args.leak_flange, "flow_gs": flow_full[args.leak_flange] * lw.flanges[args.leak_flange].level,
             "source": deck.leak_point(lw.flanges[args.leak_flange].geom).tolist(), "wind_dir": args.wind_dir,
             "wind_speed": args.wind_speed}
    print("GASBOT truth", json.dumps(truth), flush=True)
    tele = []
    tail = None
    for step in range(args.max_steps):
        if orch.state in TERMINAL or bridge.sim_time() > args.max_sim_s:
            break
        if tail is None and orch.state == S.PATROL and orch.fixed_t is not None:
            tail = bridge.sim_time()
        if tail is not None and bridge.sim_time() - tail > 2.5:
            break
        orch.step()
        tele.append({"t": round(bridge.sim_time(), 2), "state": orch.state.value, **orch.gas_telemetry()})
        if sender is not None:
            rgb = st["images"].get("hero")
            if rgb is not None:
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, "JPEG", quality=85)
                header = {"type": "frame", "w": rgb.shape[1], "h": rgb.shape[0], "seq": st["frame"],
                          "t": round(bridge.sim_time(), 2)}
                header.update(feed.header())
                sender.send(header, buf.getvalue())
        if step % 10 == 0:
            print(f"step {step} t={bridge.sim_time():.1f}s state={orch.state.value} frames={st['frame']} "
                  f"wall={time.time() - st['t0']:.0f}s", flush=True)
    for r in orch.records:
        print(f"t={r.t:6.1f}s {r.frm:>14} -> {r.to:<14} | {r.reason}", flush=True)
    src = np.asarray(truth["source"][:2])
    err = None if orch.estimate is None else float(np.linalg.norm(orch.estimate[0] - src))
    summary = {"truth": truth, "final": orch.state.value, "t_detect": orch.detect_t, "t_found": orch.found_t,
               "t_fixed": orch.fixed_t, "loc_err_m": err, "flange": orch.flange_id, "alerts": orch.alerts,
               "torques": [h for s in lw.flanges.values() for h in s.history], "frames": st["frame"]}
    (out / "gasbot_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "gasbot_telemetry.json").write_text(json.dumps(tele, default=float))
    print("GASBOT summary", json.dumps(summary, default=float), flush=True)
    encode(st["frame"])

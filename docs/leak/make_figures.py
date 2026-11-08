from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detection.leak import LeakLocalizer, _project
from policy.interfaces import CameraModel
from policy.leak.geometry import FlangeGeometry, look_at
from policy.leak.mock import build_rig
from policy.leak.render import render_flange
from policy.leak.world import LeakWorld
from policy.orchestrator import S, draw_hud

G = FlangeGeometry("F1", [0, 0, 1], [0, -1, 0], [0, 0, 1])
LOC = LeakLocalizer.default()          # the trained crack model, as on the robot


def font(size):
    return ImageFont.load_default(size)


def isaac_cam(view=0, distance=1.1):
    W, H = 640, 360
    f = 35.0 / 20.955 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return CameraModel(K, look_at(*G.view_poses(distance)[view]), W, H)


def flange_state(angles=None, crack=0.0):
    s = LeakWorld([G]).flanges["F1"]
    if angles is not None:
        s.angles[:] = angles
    s.crack = crack
    return s


def caption(img: Image.Image, lines: list[str], size=18) -> Image.Image:
    h = 12 + (size + 6) * len(lines)
    out = Image.new("RGB", (img.width, img.height + h), (250, 250, 250))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((10, img.height + 6 + i * (size + 6)), line, fill=(20, 20, 20), font=font(size))
    return out


def fig_stripes():
    angles = np.array([0.0, 0.15, 0.0, 0.04, 0.0, 0.12, 0.0, 0.0])
    cam = isaac_cam()
    img = render_flange(cam, flange_state(angles))
    r = LOC.read(img, cam, G)
    scale = 2
    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for b in r.bolts:
        c = G.bolt_center(b.bolt)
        (p0, p1), _ = _project(cam, np.stack([c, c + G.head_radius * G.radial(b.bolt)]))
        d_exp = (p1 - p0) / np.linalg.norm(p1 - p0)
        ang = b.angle or 0.0
        d_meas = np.array([d_exp[0] * math.cos(ang) - d_exp[1] * math.sin(ang),
                           d_exp[0] * math.sin(ang) + d_exp[1] * math.cos(ang)])
        cx, cy = p0 * scale
        L = 46
        d.line([cx - L * d_exp[0], cy - L * d_exp[1], cx + L * d_exp[0], cy + L * d_exp[1]], fill=(0, 200, 255), width=2)
        d.line([cx - L * d_meas[0], cy - L * d_meas[1], cx + L * d_meas[0], cy + L * d_meas[1]], fill=(255, 220, 0), width=2)
        loose = b.conf >= 0.7 and abs(ang) > 0.06
        col = (230, 40, 40) if loose else (40, 170, 70)
        d.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=col, width=3)
        tx = cx + (G.radial(b.bolt)[2] * 0 + (60 if d_exp[0] >= 0 else -150))
        d.text((tx, cy - 22), f"#{b.bolt} {abs(ang):.2f} rad", fill=col, font=font(18))
        d.text((tx, cy), f"conf {b.conf:.2f}", fill=(235, 235, 235), font=font(15))
    out = caption(im, [
        "Torque-stripe reading, 640x360 wrist camera at 1.1 m (rendered by the mock, read by detection/leak.py).",
        "Blue: where the stripe should point. Yellow: the stripe the camera measured. "
        f"Truth: #1 0.15, #3 0.04, #5 0.12 rad.",
        "Red = loose (turned > 0.06 rad, confidence >= 0.7). Green = within tolerance.",
    ])
    out.save(HERE / "stripe_reading.png")


def fig_crack():
    from scipy import ndimage

    panels = []
    cases = [("Clean flange", 0.0, 0.0), ("Cracked flange", 0.3, 0.0), ("Cracked, dirty lens", 0.3, 0.35)]
    for title, crack, dust in cases:
        cam = isaac_cam(view=1)
        img = render_flange(cam, flange_state(np.zeros(8), crack), dust=dust)
        r = LOC.read(img, cam, G)
        face = LOC.face_mask(cam, G, img.shape[:2], bolt_clear=1.4)
        R, Gc, B = (img[..., i].astype(np.float32) for i in range(3))
        face &= ~ndimage.binary_dilation((R > 110) & (R > 1.5 * Gc) & (R > 1.5 * B), iterations=3)
        tint = img.astype(np.float32)
        tint[face] = tint[face] * 0.75 + np.array([60, 120, 255]) * 0.25
        im = Image.fromarray(tint.astype(np.uint8))
        d = ImageDraw.Draw(im)
        for x, y, p in r.heat:                    # only windows the model scores as crack-like
            if p < 0.35:
                continue
            col = (230, 40, 40) if p >= LOC.threshold else (255, 160, 0)
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=col, width=2)
        if r.crack_prob >= LOC.threshold:
            verdict = f"CRACK (p={r.crack_prob:.2f})"
        elif r.crack_unsure:
            verdict = f"UNSURE: view quality {r.view_quality:.2f} < 0.85"
        else:
            verdict = f"no crack (p={r.crack_prob:.2f})"
        panels.append(caption(im, [title, verdict, f"view quality {r.view_quality:.2f}"], size=17))
    W = sum(p.width for p in panels) + 20 * (len(panels) - 1)
    H = max(p.height for p in panels)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 20
    sheet = caption(sheet, ["Crack model (MobileNetV4 + LoRA, trained like the rust model). Blue tint: the flange face it "
                            "looks at. Circles: windows it scores as crack; red = above threshold, orange = unsure band.",
                            "A dirty lens cannot say 'no crack': if the stripes do not read cleanly, 'no crack' becomes "
                            "UNSURE and the robot takes another view, then asks a human."], size=17)
    sheet.save(HERE / "crack_check.png")


def fig_timeline():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orch, bridge, world, leak, perc, arm, pipe = build_rig(None, leak="none")
    trace = []
    real_advance = bridge.advance

    def advance(dt):
        real_advance(dt)
        trace.append((bridge.t, bridge.gas_reading(), orch.state.value))

    bridge.advance = advance
    scheduled = False
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN) and len(orch.records) < 500:
        prev = orch.state
        orch.step()
        trace.append((bridge.t, bridge.gas_reading(), orch.state.value))
        if not scheduled and orch.state == S.SCRUB and prev != S.SCRUB:
            scheduled = True
            t_event = bridge.t + 5.0
            bridge.events.append((t_event, "bolt #2 works loose", lambda: leak.loosen("F1", 2, 0.18)))
    t = np.array([x[0] for x in trace])
    g = np.array([x[1] for x in trace])
    st = [x[2] for x in trace]
    leak_states = {"GAS_CHECK", "LEAK_APPROACH", "LEAK_INSPECT", "LEAK_SECOND_VIEW", "RETORQUE", "LEAK_VERIFY"}
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=130)
    in_leak = np.array([s in leak_states for s in st])
    ax.fill_between(t, 0, 0.26, where=in_leak, color="#ffd9a8", step="post", label="leak job")
    ax.fill_between(t, 0, 0.26, where=np.array([s in ("SCRUB", "SOAK", "SPRAY") for s in st]), color="#d6e8ff",
                    step="post", label="rust job (spray / soak / scrub)")
    ax.step(t, g, where="post", color="#c0392b", lw=2, label="gas at the robot")
    ax.axhline(0.05, color="#27ae60", ls="--", lw=1)
    ax.text(t[0] + 1, 0.055, "clear 0.05", color="#27ae60", fontsize=8)
    ax.axhline(0.20, color="#7f0000", ls="--", lw=1)
    ax.text(t[0] + 1, 0.205, "stop 0.20", color="#7f0000", fontsize=8)
    marks = {("SCRUB", "GAS_CHECK"): ("gas above clear: pause scrub", 0.245, "right"),
             ("RETORQUE", "LEAK_VERIFY"): ("re-torque bolt #2", 0.225, "left"),
             ("LEAK_VERIFY", "SCRUB"): ("leak fixed: resume scrub", 0.150, "left"),
             ("VERIFY_PAINT", "DONE"): ("rust repair done", 0.245, "right")}
    for r in orch.records:
        if (r.frm, r.to) in marks:
            label, y, ha = marks[(r.frm, r.to)]
            ax.annotate(f"{label} (t={r.t:.1f}s)", (r.t, y), fontsize=8, ha=ha,
                        xytext=(-4 if ha == "right" else 4, 0), textcoords="offset points")
            ax.axvline(r.t, color="#555", lw=0.6, ls=":")
    ax.axvline(t_event, color="#c0392b", lw=1)
    ax.annotate(f"unscripted: bolt #2 works loose (t={t_event:.1f}s)", (t_event, 0.185), fontsize=8, ha="right",
                xytext=(-4, 0), textcoords="offset points", color="#c0392b")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 0.26)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("gas (fraction of LEL)")
    ax.set_title("A leak starts while the robot scrubs rust (CPU mock, policy/leak/mock.py)")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "timeline_leak_mid_scrub.png")
    plt.close(fig)


def fig_hud():
    def run_until(leak, dust, stop_state, after=None):
        orch, bridge, world, lw, perc, arm, pipe = build_rig(None, leak=leak, rust=False, dust=dust)
        while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN):
            prev = orch.state
            orch.step()
            if orch.state == stop_state and (after is None or prev == after):
                break
        frame = perc.leak.observe()
        return draw_hud(frame, orch.overlay_text())

    shots = [("Loose bolts: reading stripes", run_until("loose", 0.0, S.LEAK_INSPECT)),
             ("Bolt #1 re-torqued, verifying", run_until("loose", 0.0, S.LEAK_VERIFY, after=S.RETORQUE)),
             ("Crack: no repair, escalate", run_until("crack", 0.0, S.ESCALATE_TO_HUMAN)),
             ("Dirty lens: unsure, ask a human", run_until("loose", 0.4, S.ESCALATE_TO_HUMAN))]
    tiles = [caption(Image.fromarray(img), [title], size=18) for title, img in shots]
    w, h = tiles[0].width, tiles[0].height
    sheet = Image.new("RGB", (2 * w + 16, 2 * h + 16), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * (w + 16), (i // 2) * (h + 16)))
    sheet.save(HERE / "hud_frames.png")


if __name__ == "__main__":
    fig_stripes()
    fig_crack()
    fig_timeline()
    fig_hud()
    print("wrote", sorted(p.name for p in HERE.glob("*.png")))

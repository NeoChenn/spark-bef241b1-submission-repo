import argparse
import json
import math
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from policy.fleet.render import render_png

SHOW = ("drop_obstacle", "set_weather", "close_walkway", "close_stairs", "clear", "reroute", "needs_human")
HOLD_S = 32.0
CORAL = (217, 119, 87)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def rain(img, wet: float, wind_dir: float, seed: int) -> None:
    if wet <= 0:
        return
    rng = random.Random(seed)
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([0, 0, W, H], fill=(40, 48, 56, int(55 * wet)))
    slant = 18 * math.cos(math.radians(wind_dir))
    for _ in range(int(700 * wet)):
        x, y, L = rng.uniform(-40, W + 40), rng.uniform(-40, H), rng.uniform(14, 34)
        d.line([x, y, x + slant * L / 30, y + L], fill=(200, 210, 222, rng.randint(40, 110)), width=1)
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))


def captions(ticks):
    out, seen = [], []
    for tk in ticks:
        for e in tk.get("events", []):
            if e["kind"] in SHOW:
                seen.append(e)
        out.append([e for e in seen if tk["t"] - e["t"] <= HOLD_S][-3:])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, nargs="+")
    p.add_argument("--inset-w", type=float, default=0.46)
    p.add_argument("--fleet-log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=24.0)
    a = p.parse_args(argv)
    lines = [json.loads(x) for x in open(a.fleet_log)]
    mp = next(x for x in lines if x["type"] == "map")
    ticks = [x for x in lines if x["type"] == "tick"]
    by_t = {tk["t"]: k for k, tk in enumerate(ticks)}
    caps = captions(ticks)
    rows = {}
    for fd in a.frames:
        for r in json.loads((Path(fd) / "index.json").read_text()):
            rows[r["frame"]] = {**r, "dir": fd}
    idx = [rows[k] for k in sorted(rows)]
    f14, f12 = font(15), font(13)
    insets = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n, row in enumerate(idx):
            k = by_t[row["sim_t"]]
            img = Image.open(Path(row["dir"]) / f"{row['frame']:05d}.png").convert("RGB")
            W, H = img.size
            if k not in insets:
                png = Path(tmp) / f"m{k}.png"
                render_png(mp, ticks[k], str(png))
                m = Image.open(png).convert("RGB")
                iw = int(W * a.inset_w)
                insets[k] = m.resize((iw, int(m.height * iw / m.width)), Image.LANCZOS)
            ins = insets[k]
            w = (ticks[k].get("hazards") or {}).get("weather") or {}
            rain(img, float(w.get("wet", 0.0)), float(w.get("wind_dir", 90.0)), n)
            d = ImageDraw.Draw(img, "RGBA")
            if w.get("level", "calm") != "calm":
                badge = f"{w['level'].upper()}  wind {w['wind']:.1f}  walk x{w['speed']['walk']:.2f}  sight {w['perception_range_m']:.0f} m"
                bw = d.textlength(badge, font=f14)
                d.rounded_rectangle([W - bw - 44, 16, W - 16, 52], 10, fill=(150, 40, 30, 200))
                d.text((W - bw - 30, 25), badge, fill=(255, 240, 230), font=f14)
            x0, y0 = W - ins.width - 16, H - ins.height - 16
            d.rectangle([x0 - 3, y0 - 3, x0 + ins.width + 3, y0 + ins.height + 3], fill=(0, 0, 0, 160))
            img.paste(ins, (x0, y0))
            d.rounded_rectangle([16, 16, 470, 52], 10, fill=(20, 20, 19, 190))
            d.text((30, 25), "FLEET PLANNER  ·  live re-routing around obstacles & weather", fill=(245, 244, 238), font=f14)
            y = 62
            for e in caps[k]:
                msg = e["message"]
                msg = msg if len(msg) < 96 else msg[:93] + "..."
                col = CORAL if e["kind"] == "reroute" else (245, 244, 238)
                w = d.textlength(f"t={e['t']:.0f}s  {msg}", font=f12)
                d.rounded_rectangle([16, y, 34 + w, y + 26], 8, fill=(20, 20, 19, 170))
                d.text((26, y + 6), f"t={e['t']:.0f}s  {msg}", fill=col, font=f12)
                y += 32
            img.save(Path(tmp) / f"c{n:05d}.png")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(Path(tmp) / "c%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", a.out],
                       check=True)
    print(f"-> {a.out} ({len(idx)} frames)")


if __name__ == "__main__":
    main()

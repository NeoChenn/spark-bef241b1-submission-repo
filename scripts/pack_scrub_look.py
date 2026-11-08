import argparse
import json
import shutil
import subprocess
from pathlib import Path

KEYS = ("t", "frac", "force", "state")


def pack(src: Path, dst: Path, shot: str) -> dict:
    raw = json.loads((src / f"raw_{shot}.json").read_text())
    head = {k: raw[k] for k in ("wall_s", "frames", "fps", "ckpt", "renderer", "spp", "subframes") if k in raw}
    log = [{k: e[k] for k in KEYS} for e in raw["log"]]
    (dst / f"demo_{shot}.json").write_text(json.dumps({**head, "log": log}))
    shutil.copyfile(src / f"raw_{shot}.mp4", dst / f"demo_{shot}.mp4")
    return {"frames": len(log), "fps": raw["fps"], "peak_force": max(e["force"] for e in log),
            "frac_end": log[-1]["frac"]}


def key_frame(video: Path, out: Path, t: float) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                    str(out)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--shots", default="push,medium,close")
    ap.add_argument("--keys", default="", help="shot@seconds,... key frames to extract as PNG")
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    for shot in filter(None, a.shots.split(",")):
        if (a.src / f"raw_{shot}.json").exists():
            print(shot, pack(a.src, a.dst, shot))
    for item in filter(None, a.keys.split(",")):
        shot, t = item.split("@")
        key_frame(a.dst / f"demo_{shot}.mp4", a.dst / f"key_{shot}_{float(t):04.1f}s.png", float(t))

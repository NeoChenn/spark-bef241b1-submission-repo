from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from policy.weld.env import run_pass
from policy.weld.job import make_job
from policy.weld.overlay import render
from policy.weld.policy import OpenLoop, SeamTracker
from policy.weld.seam import Seam

KEYS = ("track_rmse_mm", "track_max_mm", "coverage", "defect_rate", "lack_of_fusion", "heat_out_of_band",
        "temp_max_c")


def evaluate(n: int = 50) -> dict:
    out = {}
    for name, mk, blind in (("seam_tracker", SeamTracker, False), ("open_loop", OpenLoop, False),
                            ("tracker_laser_off", SeamTracker, True)):
        ms = [run_pass(Seam(seed=s), mk(), *Seam(seed=s).crack_extent(), seed=s, blind=blind)["metrics"]
              for s in range(n)]
        out[name] = {k: round(float(np.mean([m[k] for m in ms])), 4) for k in KEYS}
        out[name]["pass_rate"] = round(float(np.mean([m["defect_rate"] < 0.05 for m in ms])), 3)
    jobs = {}
    for sc in ("crack", "open_loop", "gas", "gas_clears", "clean"):
        res = []
        for s in range(min(n, 20)):
            j = make_job(sc, seed=s)
            j.run()
            res.append((j.state, j.t))
        jobs[sc] = {"done": round(float(np.mean([r[0] == "DONE" for r in res])), 3),
                    "escalated": round(float(np.mean([r[0] == "ESCALATE_TO_HUMAN" for r in res])), 3),
                    "mean_s": round(float(np.mean([r[1] for r in res])), 1)}
    out["jobs"] = jobs
    out["n_seeds"] = n
    return out


def film(scenario: str, every_s: float = 1.0, seed: int = 1):
    j = make_job(scenario, seed=seed)
    frames, last = [], [-1e9]

    def grab(job):
        if job.t - last[0] >= every_s:
            last[0] = job.t
            frames.append((job.state, job.t, render(job)))
    j.on_frame = grab
    j.run()
    frames.append((j.state, j.t, render(j)))
    return j, frames


def sheet(frames, cols: int = 4) -> Image.Image:
    rows = (len(frames) + cols - 1) // cols
    im = Image.new("RGB", (cols * 324, rows * 244), (0, 0, 0))
    for k, f in enumerate(frames):
        im.paste(Image.fromarray(f), ((k % cols) * 324 + 2, (k // cols) * 244 + 2))
    return im


def pick(frames):
    out, seen = [], {}
    for st, t, img in frames:
        seen.setdefault(st, []).append(img)
    for st, imgs in seen.items():
        out += [imgs[len(imgs) // 3], imgs[(2 * len(imgs)) // 3]] if len(imgs) > 2 else imgs[-1:]
    return out


def hero(frames) -> Image.Image:
    by = {}
    for st, t, img in frames:
        by.setdefault(st, []).append(img)
    want = [("INSPECT", 0.4), ("HOT_WORK_HOLD", 0.5), ("WELD", 0.6), ("INSPECT_WELD", 0.5)]
    imgs = [by[st][int(len(by[st]) * q)] for st, q in want if st in by]
    return sheet(imgs, cols=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/weld")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = evaluate(a.n)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
    for sc in ("crack", "gas_clears", "open_loop"):
        j, frames = film(sc)
        sheet(pick(frames)).save(out / f"sheet_{sc}.png")
        imgs = [Image.fromarray(f[2]) for f in frames]
        imgs[0].save(out / f"weld_{sc}.gif", save_all=True, append_images=imgs[1:], duration=120, loop=0)
        weld = [f for f in frames if f[0] == "WELD"]
        Image.fromarray((weld[len(weld) // 2] if weld else frames[-1])[2]).save(out / f"still_{sc}.png")
        if sc == "gas_clears":
            hero(frames).save(out / "mission_weld.png")
        (out / f"trace_{sc}.json").write_text(json.dumps(
            {"records": [{"t": r.t, "to": r.to, "reason": r.reason, "evidence": r.evidence} for r in j.records],
             "result": j.result, "alerts": j.alerts}, indent=1))


if __name__ == "__main__":
    main()

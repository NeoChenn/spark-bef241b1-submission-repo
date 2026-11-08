from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from fleet.contract import World
from fleet.perception.gauge import (RANGE_BAR, RED_ZONE, ClassicalGaugeReader, CropParams, GeminiGaugeReader,
                                    render_gauge_crop)

OUT = Path(__file__).resolve().parents[1] / "testdata/rounds"


def stats(errors: list[float], failures: int, total: int) -> dict:
    e = np.abs(np.array(errors)) * 100.0
    if not len(e):
        return {"n": total, "unreadable": failures}
    return {"n": total, "unreadable": failures, "mae_pct_fs": round(float(e.mean()), 2),
            "median_pct_fs": round(float(np.median(e)), 2), "p95_pct_fs": round(float(np.percentile(e, 95)), 2),
            "max_pct_fs": round(float(e.max()), 2), "within_2pct": round(float((e <= 2).mean()), 3),
            "within_5pct": round(float((e <= 5).mean()), 3),
            "mae_bar": round(float(e.mean()) / 100.0 * RANGE_BAR, 3)}


def classical_eval(n: int, seed: int = 11) -> dict:
    reader = ClassicalGaugeReader()
    out = {}
    for hard in (False, True):
        rng = np.random.default_rng(seed + hard)
        errs, fails, alarm_ok, conf_err = [], 0, 0, []
        for i in range(n):
            v = float(rng.uniform(0.0, 1.0))
            r = reader.read(render_gauge_crop(v, CropParams.random(rng, hard), seed=i))
            if r.value is None:
                fails += 1
                continue
            errs.append(r.value - v)
            alarm_ok += (r.value >= RED_ZONE[0]) == (v >= RED_ZONE[0])
            conf_err.append((r.confidence, abs(r.value - v)))
        s = stats(errs, fails, n)
        s["red_band_classification_acc"] = round(alarm_ok / max(1, len(errs)), 3)
        c = np.array(conf_err)
        hi = c[c[:, 0] >= 0.5]
        s["mae_pct_fs_when_conf_ge_0.5"] = round(float(hi[:, 1].mean() * 100), 2) if len(hi) else None
        s["share_conf_ge_0.5"] = round(len(hi) / max(1, len(c)), 3)
        out["hard" if hard else "normal"] = s
    return out


def gemini_sample(k: int, seed: int = 23) -> list[tuple[str, float, CropParams, str]]:
    rng = np.random.default_rng(seed)
    cases = []
    for g in sorted(World.from_rig_meta().of_type("gauge"), key=lambda a: a["id"]):
        cases.append((g["id"], g["reading_bar"] / RANGE_BAR, CropParams.random(rng), "rig"))
    i = 0
    while len(cases) < k:
        hard = i % 3 == 2
        cases.append((f"synth_{i:02d}", float(rng.uniform(0, 1)), CropParams.random(rng, hard),
                      "hard" if hard else "normal"))
        i += 1
    return cases[:k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gemini", type=int, default=0, help="number of live Gemini calls (0 = not measured)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"classical": classical_eval(args.n), "gemini": None,
              "note": "synthetic crops rendered from sim/assets/generated/gauge_dial.png; errors in % of full scale"}
    print(json.dumps(result["classical"], indent=1))

    if args.gemini:
        gem = GeminiGaugeReader(budget=args.gemini)
        if not gem.available:
            print("Gemini unavailable: not measured")
        else:
            reader = ClassicalGaugeReader()
            manifest, g_err, c_err, flags, flagged_big, big, fails = [], [], [], 0, 0, 0, 0
            for idx, (name, v, p, group) in enumerate(gemini_sample(args.gemini)):
                img = render_gauge_crop(v, p, seed=100 + idx)
                path = args.out / f"gauge_{name}.jpg"
                cv2.imwrite(str(path), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                img = cv2.imread(str(path))[..., ::-1]
                c = reader.read(img)
                gr = gem.read(img)
                rec = {"file": path.name, "group": group, "truth": round(v, 4), "truth_bar": round(v * RANGE_BAR, 2),
                       "params": asdict(p), "classical": c.to_dict(),
                       "gemini": gr.to_dict() if gr else {"error": gem.last_error}}
                manifest.append(rec)
                if gr is None or gr.value is None:
                    fails += 1
                else:
                    g_err.append(gr.value - v)
                if c.value is not None:
                    c_err.append(c.value - v)
                if gr and gr.value is not None and c.value is not None:
                    flag = abs(gr.value - c.value) > 0.05
                    worst = max(abs(gr.value - v), abs(c.value - v)) > 0.05
                    flags += flag
                    big += worst
                    flagged_big += flag and worst
                print(name, group, rec["truth_bar"], c.reading_bar, gr.reading_bar if gr else gem.last_error)
            result["gemini"] = {
                "model": gem.model, "calls": len(gem.calls), **stats(g_err, fails, len(manifest)),
                "classical_on_same_crops": stats(c_err, len(manifest) - len(c_err), len(manifest)),
                "disagreement_flags": int(flags), "cases_with_a_reader_off_by_gt_5pct": int(big),
                "of_which_flagged": int(flagged_big),
                "mean_latency_ms": round(float(np.mean([c["latency_ms"] for c in gem.calls if "latency_ms" in c]))),
            }
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (args.out / "gauge_accuracy.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["gemini"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

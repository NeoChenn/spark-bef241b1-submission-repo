from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from fleet.aerial import thermal
from fleet.aerial.gemini import GeminiVision
from fleet.aerial.patches import TESTDATA, accuracy, grade_classical, load_image, test_set, write_test_set

THERMAL_CASES = {
    "healthy": [560, 548, 571, 556, 563, 552, 566, 559],
    "hot": [560, 548, 571, 556, 840, 552, 566, 559],
    "cold": [560, 548, 140, 556, 563, 552, 566, 559],
}
THERMAL_DISTANCE_M = 26.0


def thermal_frames() -> dict[str, np.ndarray]:
    out = {}
    for name, temps in THERMAL_CASES.items():
        frame = thermal.render(thermal.FlareTipState(temps), THERMAL_DISTANCE_M, 0.5, seed=7)
        out[name] = thermal.false_colour(frame, len(temps), 0.45, THERMAL_DISTANCE_M)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    rows = write_test_set()
    for row in rows:
        g, _ = grade_classical(load_image(TESTDATA / row["file"]))
        row["classical"] = g.summary()
    repaired = []
    for spec in test_set():
        if spec.ri >= 3:
            spec.repaired = True
            img, _ = spec.render()
            name = f"{spec.id}_repaired.jpg"
            Image.fromarray(img).save(TESTDATA / name, quality=95, subsampling=0)
            repaired.append({"id": f"{spec.id}_repaired", "file": name, "true_ri": 0,
                             "classical": grade_classical(img)[0].summary()})
    (TESTDATA / "patches.json").write_text(json.dumps({"patches": rows, "repaired": repaired}, indent=1),
                                           encoding="utf-8")
    frames = thermal_frames()
    for name, rgb in frames.items():
        Image.fromarray(rgb).save(TESTDATA / f"thermal_{name}.png")
    print("classical", accuracy([(r["true_ri"], r["classical"]["ri"]) for r in rows]))
    if not args.gemini:
        return 0
    only = {s for s in args.only.split(",") if s}
    vision = GeminiVision(model=args.model, record=TESTDATA / "gemini_replies.json")
    if not vision.available:
        print("no GEMINI_API_KEY")
        return 2
    for row in rows + repaired:
        if only and row["id"] not in only:
            continue
        g = vision.grade_ri(load_image(TESTDATA / row["file"]))
        print(row["id"], row["true_ri"], g.summary() if g else vision.last_error, flush=True)
    for name in frames:
        if only and f"thermal_{name}" not in only:
            continue
        rgb = np.asarray(Image.open(TESTDATA / f"thermal_{name}.png").convert("RGB"))
        print("thermal", name, vision.thermal_opinion(rgb) or vision.last_error, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

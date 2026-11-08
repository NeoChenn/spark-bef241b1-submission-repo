"""Train a binary rig detector with the team's detection/train.py, unchanged (same model, LoRA, calibration,
recall-first threshold, unsure band, ONNX export and meta.json). Only the positive class name changes.

    python -m detection.train_task --task debris --data data/debris_v1 --out runs/debris_detector/mnv4s_lora
    python -m detection.train_task --task person --data data/person_v1 --out runs/person_detector/mnv4s_lora

(The crack model has its own wrapper, detection/train_leak.py, which does the same for task "crack".)
The result loads with detection.rig_vision.WindowDetector.from_checkpoint(<out>).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import detection.train as T

TASKS = {"debris": "loose object on the deck (can, bottle, packaging, tool)", "person": "a person in view",
         "crack": "crack on a flange face"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=T.PRIMARY)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    T.CLASSES = {"clean": 0, a.task: 1}          # list_split reads this module global
    res = T.train(model_name=a.model, data_dir=a.data, out_dir=a.out, epochs=a.epochs, batch_size=a.batch_size,
                  num_workers=a.workers, threads=a.threads, pretrained=True, export=True)
    meta_path = Path(a.out) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update({"task": a.task, "classes": T.CLASSES, "positive": TASKS[a.task],
                 "output": f"single logit; p_{a.task} = sigmoid(logit / temperature)",
                 "decision": f"{a.task} if p >= threshold; if unsure_band[0] <= p <= unsure_band[1] take another view"})
    meta_path.write_text(json.dumps(meta, indent=2))
    return res


if __name__ == "__main__":
    main()

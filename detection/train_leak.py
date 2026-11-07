from __future__ import annotations

import argparse
import json
from pathlib import Path

import detection.train as T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/leak_v1")
    ap.add_argument("--out", default="runs/leak_detector/mnv4s_lora")
    ap.add_argument("--model", default=T.PRIMARY)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    T.CLASSES = {"clean": 0, "crack": 1}
    res = T.train(model_name=a.model, data_dir=a.data, out_dir=a.out, epochs=a.epochs, batch_size=a.batch_size,
                  num_workers=a.workers, threads=a.threads, pretrained=True, export=True)
    meta_path = Path(a.out) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update({"task": "crack", "classes": T.CLASSES,
                 "output": "single logit; p_crack = sigmoid(logit / temperature)",
                 "decision": "crack if p_crack >= threshold; if unsure_band[0] <= p_crack <= unsure_band[1] take another view"})
    meta_path.write_text(json.dumps(meta, indent=2))
    return res


if __name__ == "__main__":
    main()

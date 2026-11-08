from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/repair")
    ap.add_argument("--dst", default="runs_sample/repair")
    ap.add_argument("--hz", type=float, default=4.0)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(src / "telemetry.jsonl")]
    out, carry, next_t = [], [], -1.0
    for r in rows:
        carry += r["events"]
        if r["t"] + 1e-9 >= next_t or r["state"] in ("BACK_OFF",) or r["events"]:
            r = dict(r, events=carry)
            carry = []
            out.append(r)
            next_t = r["t"] + 1.0 / a.hz
    with open(dst / "telemetry.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    for name in ("decisions.jsonl", "summary.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    print(f"{len(rows)} -> {len(out)} frames, {(dst / 'telemetry.jsonl').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

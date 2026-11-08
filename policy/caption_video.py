import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rig.captions import robot_entry
from rig.overlay import burn

NAME = "Unit 1 · rust"


def demo_robot(entry: dict) -> dict:
    word = entry["state"].split()[0]
    state = {"RE-PLAN": "REPLAN"}.get(word, word)
    ev = {"force": entry.get("force"), "rust_fraction": entry.get("frac"), "job": "scrub"}
    return robot_entry("unit-1", NAME, state, ev, None)


def aligned(log: list[dict], n: int) -> list[dict]:
    return log[len(log) - n:] if n <= len(log) else log + [log[-1]] * (n - len(log))


def caption_frames(frames, log: list[dict]):
    return [burn(f, [demo_robot(e)]) for f, e in zip(frames, aligned(log, len(frames)))]


def main(argv=None) -> None:
    import imageio.v2 as imageio
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("log")
    p.add_argument("out")
    a = p.parse_args(argv)
    meta = json.load(open(a.log))
    reader = imageio.get_reader(a.video)
    log = aligned(meta["log"], reader.count_frames())
    writer = imageio.get_writer(a.out, fps=meta.get("fps", 30), quality=8, macro_block_size=8)
    for f, e in zip(reader, log):
        writer.append_data(burn(f, [demo_robot(e)]))
    writer.close()
    print(f"{a.out}: {len(log)} frames captioned")


if __name__ == "__main__":
    main()

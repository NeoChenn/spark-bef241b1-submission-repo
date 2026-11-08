import argparse
import io
import json
import queue
import subprocess
import threading
import time

from PIL import Image, ImageDraw

from policy.control.backend import ArgonBackend
from policy.control.centre import from_env
from policy.control.mock import FleetScenario
from policy.demo_mock import build
from policy.orchestrator import S
from rig.hud import HudFeed, ScriptedAdvisor, apply_event
from rig.sim.sender import FrameSender


def frame(text: str) -> bytes:
    img = Image.new("RGB", (960, 540), (18, 22, 26))
    ImageDraw.Draw(img).text((40, 40), text, fill=(120, 200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


CLIP = "runs/scrub_look/demo_close"


def video_frames(clip):
    while True:
        log = json.load(open(clip + ".json"))["log"]
        cmd = ["ffmpeg", "-loglevel", "error", "-re", "-i", clip + ".mp4", "-vf", "scale=1280:720",
               "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        buf, k = b"", 0
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b"\xff\xd8")
                j = buf.find(b"\xff\xd9", i + 2)
                if i < 0 or j < 0:
                    break
                yield buf[i:j + 2], log[min(k, len(log) - 1)]
                buf, k = buf[j + 2:], k + 1
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--sim-port", type=int, default=9077)
    ap.add_argument("--web-port", type=int, default=8077)
    args = ap.parse_args()
    orch, bridge, world, perception, arm, pipe = build("runs/demo/decisions.jsonl", verbose=False)
    orch.advisor = ScriptedAdvisor()
    fleet = FleetScenario()
    t0 = time.time()
    control = from_env("live", "runs/demo/control_calls.jsonl", sync=False, port=fleet.port, timeout_s=120)
    control.backend = ArgonBackend(timeout=120)
    feed = HudFeed(orch, control=control, extra=lambda: fleet.at(time.time() - t0))
    sender = FrameSender("127.0.0.1", args.sim_port).start()
    print(f"streaming to the web HUD; open http://localhost:{args.web_port}", flush=True)
    seq, last_step = 0, 0.0
    latest = {"jpeg": frame("loading video"), "log": None}

    def pump():
        for j, entry in video_frames(args.clip):
            latest["jpeg"], latest["log"] = j, entry
    threading.Thread(target=pump, daemon=True).start()
    while True:
        tick = time.time()
        while True:
            try:
                ev = sender.inbox.get_nowait()
            except queue.Empty:
                break
            try:
                if ev.get("name") == "intent":
                    threading.Thread(target=apply_event, args=(bridge, ev, control), daemon=True).start()
                else:
                    apply_event(bridge, ev, control)
                print("event", ev, flush=True)
            except Exception as e:
                print("event failed", ev, e, flush=True)
        if tick - last_step >= 0.2 and orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN):
            orch.step()
            last_step = tick
        h = feed.header()
        e = latest["log"]
        if e is not None:
            h = {**h, "force_n": round(float(e["force"]), 2), "force_limit": 40.0,
                 "rust_fraction": round(float(e["frac"]), 4)}
        seq += 1
        sender.send({"type": "frame", "w": 1280, "h": 720, "seq": seq, "mean": 90.0, **h}, latest["jpeg"])
        time.sleep(max(0.0, 1 / 30 - (time.time() - tick)))

if __name__ == "__main__":
    main()

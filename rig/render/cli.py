import argparse
import os


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render-only Isaac app: mirrors the sim's scene state and renders the main view")
    p.add_argument("--relay", default=os.environ.get("RENDER_RELAY") or "192.168.1.190:9000",
                   help="host:port of the web relay ingest (state in, JPEG frames out)")
    p.add_argument("--token-env", default="INGEST_TOKEN", help="env var holding the shared ingest token")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--renderer", default="RealTimePathTracing")
    p.add_argument("--spp", type=int, default=1)
    p.add_argument("--subframes", type=int, default=1)
    p.add_argument("--bounces", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--wait-render", action="store_true")
    p.add_argument("--fps", type=float, default=30.0, help="frame cap (live) or output rate for --offline (0 = one frame per record)")
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--jpeg", choices=("pil", "cv2"), default="pil")
    p.add_argument("--keepalive", type=float, default=1.0, help="re-render this often with no new state")
    p.add_argument("--record", default="", help="append every received state message to this JSONL file")
    p.add_argument("--offline", default="", help="render a recorded JSONL state log instead of the live stream")
    p.add_argument("--out", default="/out/frames", help="--offline: PNG output directory")
    p.add_argument("--start", type=float, default=0.0, help="--offline: skip records before this sim time")
    p.add_argument("--end", type=float, default=0.0, help="--offline: stop after this sim time; 0 = end of log")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--snapshot", default="")
    p.add_argument("--snapshot-every", type=float, default=0.0)
    return p.parse_args(argv)

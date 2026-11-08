import argparse
import os

DEFAULT_SCENE = "/rig/sim/output/full_rig.usda"
DEFAULT_CAMERA = "/World/Cameras/wide"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless Isaac Sim camera stream for the rig")
    p.add_argument("--sim", default="127.0.0.1:9000", help="host:port of the web relay's frame ingest")
    p.add_argument("--scene", default=DEFAULT_SCENE, help="scene USD path, or 'test' for the primitive scene")
    p.add_argument("--camera", default=DEFAULT_CAMERA, help="camera prim path in the scene")
    p.add_argument("--orbit", type=float, default=0.0, help="orbit period in seconds around the camera's view target; 0 = fixed")
    p.add_argument("--orbit-z", type=float, default=1.0, help="height of the plane the orbit target is picked on")
    p.add_argument("--env", default="", help="sim.environment preset to add (ocean/sky); empty = none")
    p.add_argument("--cinematic", default="", help="sim/cinematic.py preset (night|golden|storm): cinematic scene, effects and camera sequence")
    p.add_argument("--merged", action=argparse.BooleanOptionalAction, default=os.environ.get("RIG_MERGED", "") == "1", help="with --cinematic: open sim/output/rig_merged.usda (repair zone on the top deck) and fly wide orbit -> pipe close-up")
    p.add_argument("--merged-hold", type=float, default=float(os.environ.get("RIG_MERGED_HOLD", "8")), help="seconds at the pipe close-up before flying back out; 0 = stay at the close-up")
    p.add_argument("--merged-start", type=float, default=0.0, help="time offset into the merged camera path")
    p.add_argument("--orbit-target", type=vec3, default=None, metavar="X,Y,Z", help="orbit target; default = view ray hit on --orbit-z")
    p.add_argument("--orbit-radius", type=float, default=0.0, help="orbit radius; 0 = keep the camera's distance")
    p.add_argument("--orbit-eye-z", type=float, default=None, help="orbit eye height; default = the camera's height")
    p.add_argument("--sea-level", type=float, default=0.0, help="sea level z passed to add_environment")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--renderer", default="RealTimePathTracing")
    p.add_argument("--spp", type=int, default=1, help="path tracing samples per pixel per subframe")
    p.add_argument("--subframes", type=int, default=1, help="rt_subframes accumulated per streamed frame")
    p.add_argument("--bounces", type=int, default=0, help="path tracing max bounces; 0 = renderer default")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra carb setting, repeatable")
    p.add_argument("--wait-render", action="store_true", help="block each step until the frame is rendered")
    p.add_argument("--fps", type=float, default=30.0, help="frame rate cap")
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 = forever")
    p.add_argument("--snapshot", default="", help="save frames as PNG here (first good frame, then every --snapshot-every s)")
    p.add_argument("--snapshot-every", type=float, default=0.0)
    p.add_argument("--render-mode", choices=("local", "remote"), default=os.environ.get("RENDER_MODE") or "local",
                   help="remote: skip the main view render and publish scene state for rig.render on another box")
    return p.parse_args(argv)


def vec3(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return (x, y, z)


def parse_setting(item: str) -> tuple[str, object]:
    key, _, raw = item.partition("=")
    low = raw.lower()
    if low in ("true", "false"):
        return key, low == "true"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    return key, raw

"""Live cinematic demo in the Isaac Sim window.

    C:\\isaacsim\\python.bat sim\\run_demo.py

Keys: 1 Night Operations, 2 Golden Hour, 3 Storm, C plays the camera sequence.
Right mouse and WASD fly the camera when the sequence isn't playing.
Add --self-test to run through every preset without a window, then exit.
"""

from __future__ import annotations

import argparse
import time as clock

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--preset", default="night")
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.self_test, "width": 1600, "height": 900})

import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import cinematic  # noqa: E402
import ocean  # noqa: E402

CAMERA_PATH = "/World/Cinematic/Cameras/sequence"
PRESET_KEYS = {
    carb.input.KeyboardInput.KEY_1: "night",
    carb.input.KeyboardInput.KEY_2: "golden",
    carb.input.KeyboardInput.KEY_3: "storm",
}

context = omni.usd.get_context()
context.open_stage(str(cinematic.SCENE_PATH))
for _ in range(600):
    simulation_app.update()
    if context.get_stage_loading_status()[2] == 0:
        break
stage = context.get_stage()

camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.5, 60000))
camera.CreateFocalLengthAttr(28.0)
UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(1))
cinematic.set_camera(stage, CAMERA_PATH, 1.0)

state = {"preset": args.preset, "sequence_start": None, "pending": None}
cinematic.apply_preset(stage, state["preset"])
rest, distance, _, _ = ocean.make_grid()


def on_key(event, *_):
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return True
    if event.input in PRESET_KEYS:
        state["pending"] = PRESET_KEYS[event.input]
    elif event.input == carb.input.KeyboardInput.C:
        state["sequence_start"] = clock.time()
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().camera_path = CAMERA_PATH
    return True


if args.self_test:
    for name in cinematic.PRESETS:
        cinematic.apply_preset(stage, name)
        for step in range(5):
            cinematic.animate_ocean(stage, rest, distance, step * 0.1, cinematic.PRESETS[name]["wave_height"])
            cinematic.animate_effects(stage, step * 0.1)
            cinematic.set_camera(stage, CAMERA_PATH, step * 4.0)
            simulation_app.update()
        print(f"SELF_TEST_OK {name}", flush=True)
    simulation_app.close()
    raise SystemExit(0)

keyboard = omni.appwindow.get_default_app_window().get_keyboard()
input_interface = carb.input.acquire_input_interface()
subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)

started = clock.time()
while simulation_app.is_running():
    now = clock.time() - started
    if state["pending"]:
        state["preset"] = state.pop("pending")
        state["pending"] = None
        cinematic.apply_preset(stage, state["preset"])
    cinematic.animate_ocean(stage, rest, distance, now, cinematic.PRESETS[state["preset"]]["wave_height"])
    cinematic.animate_effects(stage, now)
    if state["sequence_start"] is not None:
        elapsed = clock.time() - state["sequence_start"]
        if elapsed < cinematic.SEQUENCE_SECONDS:
            cinematic.set_camera(stage, CAMERA_PATH, elapsed)
        else:
            state["sequence_start"] = None
    simulation_app.update()

input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
simulation_app.close()

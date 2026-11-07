"""Build the cinematic scene and render stills for each preset.

    C:\\isaacsim\\python.bat sim\\build_cinematic.py --preset night

Run sim/build_full_rig.py and sim/tools/make_preset_skies.py first. Writes
sim/output/cinematic.usda and sim/output/cine_<preset>_<camera>.png.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--preset", default="night", help="night, storm, golden or all")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import cinematic  # noqa: E402
import ocean  # noqa: E402
from scene_builder import OUTPUT  # noqa: E402

CAMERAS = {
    "hero": ((110.0, -235.0, 9.0), (4.0, -6.0, 36.0), 28.0),
    "aerial": ((230.0, -260.0, 105.0), (4.0, -4.0, 22.0), 36.0),
    "waterline": ((75.0, -105.0, 4.0), (2.0, -8.0, 24.0), 18.0),
}


def add_camera(stage, name, eye, target, focal):
    camera = UsdGeom.Camera.Define(stage, f"/World/Cinematic/Cameras/{name}")
    camera.CreateFocalLengthAttr(focal)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.5, 60000))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    return str(camera.GetPath())


stage = cinematic.build_stage()
UsdGeom.Scope.Define(stage, "/World/Cinematic/Cameras")
paths = {name: add_camera(stage, name, *values) for name, values in CAMERAS.items()}
cinematic.apply_preset(stage, "night")
layer = stage.GetRootLayer()
layer_data = dict(layer.customLayerData)
layer_data["renderSettings"] = cinematic.render_settings("night")
layer.customLayerData = layer_data
layer.Save()
print(f"SCENE_WRITTEN {cinematic.SCENE_PATH}")

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.replicator.core", True)
import omni.replicator.core as rep  # noqa: E402

context = omni.usd.get_context()
context.open_stage(str(cinematic.SCENE_PATH))
for _ in range(600):
    simulation_app.update()
    if context.get_stage_loading_status()[2] == 0:
        break
live = context.get_stage()
for _ in range(60):
    simulation_app.update()

names = list(cinematic.PRESETS) if args.preset == "all" else [args.preset]
for preset in names:
    cinematic.apply_preset(live, preset)
    for _ in range(40):
        simulation_app.update()
    for camera, path in paths.items():
        product = rep.create.render_product(path, (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=24, delta_time=0.0)
        data = np.asarray(annotator.get_data())
        out = OUTPUT / f"cine_{preset}_{camera}.png"
        Image.fromarray(data[:, :, :3].astype("uint8")).save(out)
        print(f"PREVIEW_WRITTEN {out.name} mean={data[:, :, :3].mean():.1f}")
        annotator.detach(product)
        product.destroy()
rep.orchestrator.wait_until_complete()
simulation_app.close()

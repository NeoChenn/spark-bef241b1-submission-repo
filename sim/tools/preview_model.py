"""Path-traced wide and deck-level previews of an imported model.

    /isaac-sim/python.sh sim/tools/preview_model.py sim/assets/rig_model/rig/rig.usd --out sim/output/import --sea-z -20

Writes <out>/<name>_wide.png and <out>/<name>_deck.png at 1280x720.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
parser.add_argument("--out", type=Path, default=SIM_DIR / "output" / "import")
parser.add_argument("--name", help="image prefix (default: USD stem)")
parser.add_argument("--sea-z", type=float, help="sea level in metres (default: just below the model)")
parser.add_argument("--hdri", default="aristea_wreck")
parser.add_argument("--subframes", type=int, default=64)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "renderer": "RealTimePathTracing", "width": args.width, "height": args.height}
)

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_fix import preview_cameras  # noqa: E402


def material(stage, path, color, roughness, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def build(stage) -> dict:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    rig = stage.DefinePrim("/World/Rig", "Xform")
    rig.GetReferences().AddReference(str(args.usd.resolve()))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = box.ComputeWorldBound(rig).ComputeAlignedRange()
    lo, hi = list(rng.GetMin()), list(rng.GetMax())

    sea_z = args.sea_z if args.sea_z is not None else lo[2] - 0.05
    sea = UsdGeom.Mesh.Define(stage, "/World/Env/sea")
    s = 5000.0
    sea.CreatePointsAttr([(-s, -s, sea_z), (s, -s, sea_z), (s, s, sea_z), (-s, s, sea_z)])
    sea.CreateFaceVertexCountsAttr([4])
    sea.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdShade.MaterialBindingAPI.Apply(sea.GetPrim()).Bind(
        material(stage, "/World/Env/Looks/sea", (0.02, 0.08, 0.10), 0.06)
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/Env/sky")
    hdri = SIM_DIR / "assets" / "third_party" / "hdris" / f"{args.hdri}.hdr"
    if hdri.exists():
        dome.CreateTextureFileAttr().Set(str(hdri))
        dome.CreateTextureFormatAttr().Set("latlong")
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Env/sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.8)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(48, 12, 35))

    cams = {}
    for name, view in preview_cameras(lo, hi).items():
        cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
        cam.CreateFocalLengthAttr(view["focal"])
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20000))
        look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view["eye"]), Gf.Vec3d(*view["target"]), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
        cams[name] = str(cam.GetPath())
    print("PREVIEW_BOUNDS " + json.dumps({"min": lo, "max": hi, "sea_z": sea_z}), flush=True)
    return cams


def render(cams: dict) -> None:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep

    for _ in range(120):
        simulation_app.update()
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = args.name or args.usd.stem
    for name, path in cams.items():
        product = rep.create.render_product(path, (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
        start = time.time()
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False)
        seconds = time.time() - start
        data = np.asarray(annotator.get_data())[:, :, :3].astype("uint8")
        out = args.out / f"{prefix}_{name}.png"
        Image.fromarray(data).save(out)
        print(f"PREVIEW_WRITTEN {out} mean={data.mean():.1f} seconds={seconds:.1f}", flush=True)
        annotator.detach(product)
        product.destroy()


code = 0
try:
    context = omni.usd.get_context()
    context.new_stage()
    simulation_app.update()
    render(build(context.get_stage()))
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
sys.stdout.flush()
os._exit(code)

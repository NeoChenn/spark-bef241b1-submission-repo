"""Render the ocean environment with an optional rig scene, and time frames.

    /isaac-sim/python.sh sim/tools/render_environment.py --preset clear --scene sim/output/rig_deck.usda \
        --camera 60,-80,10 --target 0,0,-10 --spp 32 --out sim/output/ocean/clear_wide.png
"""

from __future__ import annotations

import argparse
import json
import os
import time


def vec3(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return x, y, z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default="clear")
    p.add_argument("--scene", default="", help="usda to reference at /World/Rig (deck at z=0)")
    p.add_argument("--fixture", action="store_true", help="add placeholder jacket legs and cellar deck")
    p.add_argument("--camera", type=vec3, default=(120.0, -110.0, 25.0))
    p.add_argument("--target", type=vec3, default=(0.0, -5.0, 20.0))
    p.add_argument("--focal", type=float, default=24.0)
    p.add_argument("--spp", type=int, default=16, help="render subframes accumulated per output frame")
    p.add_argument("--renderer", default="RealTimePathTracing")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--time", type=float, default=10.0, help="sea time of the still")
    p.add_argument("--sea-level", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--view", action="append", default=[],
                   help="extra shot name:x,y,z:x,y,z[:focal], saved as <out stem>_<name>.png; repeatable")
    p.add_argument("--variant", action="append", default=[],
                   help="carb settings 'key=json~key=json' applied cumulatively, re-rendering all shots as _v<i>")
    p.add_argument("--timing-frames", type=int, default=0, help="also time N animated frames")
    p.add_argument("--timing-spp", default="", help="comma list of subframe counts to time (default --spp)")
    p.add_argument("--set", action="append", default=[], help="extra carb setting key=json, repeatable")
    p.add_argument("--out", default="sim/output/ocean/render.png")
    return p.parse_args()


def add_fixture(stage, sea_level: float) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mat = UsdShade.Material.Define(stage, "/World/Fixture/Looks/steel")
    sh = UsdShade.Shader.Define(stage, "/World/Fixture/Looks/steel/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.42, 0.36, 0.12))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    dark = UsdShade.Material.Define(stage, "/World/Fixture/Looks/dark")
    sd = UsdShade.Shader.Define(stage, "/World/Fixture/Looks/dark/Shader")
    sd.CreateIdAttr("UsdPreviewSurface")
    sd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.12, 0.13))
    sd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    dark.CreateSurfaceOutput().ConnectToSource(sd.ConnectableAPI(), "surface")

    top, bottom = -6.0, sea_level - 40.0
    for i, (x, y) in enumerate([(-11, -7), (11, -7), (-11, 7), (11, 7)]):
        c = UsdGeom.Cylinder.Define(stage, f"/World/Fixture/leg_{i}")
        c.CreateRadiusAttr(1.3)
        c.CreateHeightAttr(top - bottom)
        c.CreateAxisAttr("Z")
        c.AddTranslateOp().Set(Gf.Vec3d(x, y, (top + bottom) / 2))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    for i, z in enumerate((sea_level + 6.0, -12.0)):
        for j, (a, b) in enumerate([((-11, -7), (11, -7)), ((-11, 7), (11, 7)), ((-11, -7), (-11, 7)), ((11, -7), (11, 7))]):
            c = UsdGeom.Cylinder.Define(stage, f"/World/Fixture/brace_{i}_{j}")
            c.CreateRadiusAttr(0.45)
            length = abs(b[0] - a[0]) + abs(b[1] - a[1])
            c.CreateHeightAttr(length)
            c.CreateAxisAttr("X" if a[1] == b[1] else "Y")
            c.AddTranslateOp().Set(Gf.Vec3d((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z))
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    cellar = UsdGeom.Cube.Define(stage, "/World/Fixture/cellar_deck")
    cellar.CreateSizeAttr(1.0)
    cellar.AddTranslateOp().Set(Gf.Vec3d(0, 0, -4.5))
    cellar.AddScaleOp().Set(Gf.Vec3f(26, 18, 3))
    UsdShade.MaterialBindingAPI.Apply(cellar.GetPrim()).Bind(dark)


def aim_camera(stage, eye, target, focal):
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, "/World/RenderCam")
    cam.CreateFocalLengthAttr().Set(focal)
    cam.CreateHorizontalApertureAttr().Set(36.0)
    cam.CreateVerticalApertureAttr().Set(20.25)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e6))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    ops = xf.GetOrderedXformOps()
    (ops[0] if ops else xf.AddTransformOp()).Set(view.GetInverse())
    return str(cam.GetPath())


def parse_view(text: str):
    parts = text.split(":")
    return parts[0], vec3(parts[1]), vec3(parts[2]), float(parts[3]) if len(parts) > 3 else 24.0


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})

    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Sdf, UsdGeom

    from sim.environment import add_environment

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    if args.scene:
        rig = stage.DefinePrim("/World/Rig", "Xform")
        rig.GetReferences().AddReference(os.path.abspath(args.scene))
        old_env = stage.GetPrimAtPath("/World/Rig/Environment")
        if old_env:
            old_env.SetActive(False)
    if args.fixture:
        add_fixture(stage, args.sea_level)
    cam_path = aim_camera(stage, args.camera, args.target, args.focal)

    t0 = time.monotonic()
    env = add_environment(stage, preset=args.preset, sea_level_z=args.sea_level)
    print(f"legs={[tuple(round(v, 1) for v in l) for l in env.legs]}", flush=True)
    print(f"environment built in {time.monotonic() - t0:.2f}s", flush=True)
    settings = carb.settings.get_settings()
    for item in args.set:
        key, value = item.split("=", 1)
        settings.set(key, json.loads(value))

    rp = rep.create.render_product(cam_path, (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    env.update(args.time)
    for _ in range(args.warmup):
        app.update()
    stem, ext = os.path.splitext(args.out)
    shots = [("", args.camera, args.target, args.focal)] + [parse_view(v) for v in args.view]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    variants = [("", "")] + [(f"v{i}", v) for i, v in enumerate(args.variant)]
    for (vname, vset), (name, eye, target, focal) in [(v, s) for v in variants for s in shots]:
        for item in filter(None, vset.split("~")):
            key, value = item.split("=", 1)
            settings.set(key, json.loads(value))
        name = "_".join(filter(None, [name, vname]))
        aim_camera(stage, eye, target, focal)
        env.update(args.time, camera=eye)
        for _ in range(8 if name else 0):
            app.update()
        t0 = time.monotonic()
        rep.orchestrator.step(rt_subframes=args.spp, pause_timeline=False)
        rgb = np.asarray(ann.get_data())[..., :3]
        still_s = time.monotonic() - t0
        path = f"{stem}_{name}{ext}" if name else args.out
        Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path)
        print(f"saved {path} mean={rgb.mean():.1f} still={still_s:.2f}s spp={args.spp}", flush=True)

    aim_camera(stage, args.camera, args.target, args.focal)
    for spp in [int(v) for v in args.timing_spp.split(",") if v] or ([args.spp] if args.timing_frames else []):
        upd = 0.0
        rep.orchestrator.step(rt_subframes=spp, pause_timeline=False)
        t0 = time.monotonic()
        for i in range(args.timing_frames):
            u0 = time.monotonic()
            env.update(args.time + i / 10.0, camera=args.camera)
            upd += time.monotonic() - u0
            rep.orchestrator.step(rt_subframes=spp, pause_timeline=False, wait_for_render=False)
            frame = np.asarray(ann.get_data())
        dt = (time.monotonic() - t0) / args.timing_frames
        if frame.ndim == 3:
            Image.fromarray(np.ascontiguousarray(frame[..., :3], dtype=np.uint8)).save(f"{stem}_anim_spp{spp}{ext}")
        w0 = time.monotonic()
        for i in range(10):
            env.displacer(args.time + i)
        wave_ms = (time.monotonic() - w0) * 100
        print(f"TIMING spp={spp} {args.width}x{args.height} renderer={args.renderer} "
              f"frame={dt:.3f}s fps={1 / dt:.2f} env_update={upd / args.timing_frames * 1000:.1f}ms "
              f"(waves {wave_ms:.1f}ms)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

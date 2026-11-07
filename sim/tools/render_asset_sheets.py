from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["swatches", "hdris", "props"])
parser.add_argument("--only", action="append", help="asset id (repeatable)")
parser.add_argument("--source", help="only assets from this source (polyhaven, ambientcg, nvidia, ...)")
parser.add_argument("--subframes", type=int, default=64)
parser.add_argument("--sky", default="kloofendal_48d_partly_cloudy_puresky")
parser.add_argument("--exposure", type=float, default=None, help="fixed exposure (EV) instead of auto-exposure")
parser.add_argument("--intensity", type=float, default=1000.0, help="dome light intensity")
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": 1280, "height": 720})

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from asset_manifest import load_assets
from pbr_material import texture_material

SIM = Path(__file__).resolve().parents[1]
ASSETS = SIM / "assets" / "third_party"
OUT = SIM / "output" / "assets"
W, H = 1280, 720
FONT = ImageFont.load_default(size=13)

SETTINGS = carb.settings.get_settings()
for key in ("/rtx/post/motionblur/enabled", "/rtx/post/dof/enabled", "/rtx/post/lensFlares/enabled"):
    SETTINGS.set(key, False)
if args.exposure is not None:
    SETTINGS.set("/rtx/post/histogram/enabled", False)
    SETTINGS.set("/rtx/post/tonemap/exposureBias", args.exposure)


def manifest():
    return load_assets(json.loads((SIM / "assets" / "manifest.json").read_text()))


def new_stage() -> Usd.Stage:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def plain(stage, path, color, roughness, metallic=0.0):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def mesh(stage, path, points, uvs, counts, indices, material, normals=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(indices)
    m.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
    if normals:
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def plate(stage, path, cx, cy, z, size, material, uv=1.0):
    h = size / 2
    pts = [(cx - h, cy - h, z), (cx + h, cy - h, z), (cx + h, cy + h, z), (cx - h, cy + h, z)]
    uvs = [(0, 0), (uv, 0), (uv, uv), (0, uv)]
    return mesh(stage, path, pts, uvs, [4], [0, 1, 2, 3], material, [(0, 0, 1)] * 4)


def sphere(stage, path, centre, radius, material, seg=48, rings=24, uv=2.0):
    pts, uvs, nrm, idx = [], [], [], []
    for r in range(rings + 1):
        v = r / rings
        phi = math.pi * v
        for s in range(seg + 1):
            u = s / seg
            th = 2 * math.pi * u
            n = (math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), -math.cos(phi))
            pts.append(tuple(centre[i] + radius * n[i] for i in range(3)))
            nrm.append(n)
            uvs.append((u * uv, v * uv / 2))
    for r in range(rings):
        for s in range(seg):
            a = r * (seg + 1) + s
            idx += [a, a + 1, a + seg + 2, a + seg + 1]
    return mesh(stage, path, pts, uvs, [4] * (rings * seg), idx, material, nrm)


def dome(stage, sky_id):
    sky = next(a for a in manifest() if a.id == sky_id)
    folder = ASSETS / sky.path
    hdr = folder if sky.path.endswith(".hdr") else sorted(folder.glob(f"*_{sky.resolution}.hdr"))[0]
    light = UsdLux.DomeLight.Define(stage, "/World/sky")
    light.CreateTextureFileAttr().Set(hdr.as_posix())
    light.CreateTextureFormatAttr().Set("latlong")
    light.CreateIntensityAttr(args.intensity)
    return light


def camera(stage, eye, target, focal):
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * H / W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    return cam


def project(cam, point):
    frustum = cam.GetCamera(Usd.TimeCode.Default()).frustum
    ndc = (frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()).Transform(Gf.Vec3d(*point))
    return (ndc[0] + 1) / 2 * W, (1 - ndc[1]) / 2 * H


class Capture:
    def __init__(self, size=(W, H)):
        self.product = rep.create.render_product("/World/cam", size)
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach(self.product)

    def warm(self):
        for _ in range(20):
            if np.asarray(self.grab(4)).mean() > 1:
                return

    def grab(self, subframes) -> Image.Image:
        start = time.monotonic()
        rep.orchestrator.step(rt_subframes=subframes, pause_timeline=False)
        data = np.asarray(self.annotator.get_data())
        if data.ndim != 3:
            data = np.zeros((2, 2, 3), dtype=np.uint8)
        print(f"FRAME {data.shape} mean={data[..., :3].mean():.1f} {time.monotonic() - start:.1f}s", flush=True)
        return Image.fromarray(np.ascontiguousarray(data[..., :3], dtype=np.uint8))


def settle(frames=60):
    for _ in range(frames):
        app.update()
    for _ in range(600):
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            break
        app.update()


def label(draw, x, y, text):
    box = draw.textbbox((x, y), text, font=FONT, anchor="mt")
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, font=FONT, fill=(255, 255, 255), anchor="mt")


def render_swatches():
    textures = [a for a in manifest() if a.kind == "texture" and (not args.only or a.id in args.only)]
    cols, rows, sx, sy = 6, 3, 1.5, 2.1
    pages = [textures[i : i + cols * rows] for i in range(0, len(textures), cols * rows)]
    for n, page in enumerate(pages, 1):
        stage = new_stage()
        dome(stage, args.sky)
        floor = plain(stage, "/World/Looks/floor", (0.35, 0.35, 0.35), 0.7)
        plate(stage, "/World/floor", 0, 0, 0, 40, floor)
        centres = []
        for i, asset in enumerate(page):
            c, r = i % cols, i // cols
            x, y = (c - (cols - 1) / 2) * sx, ((rows - 1) / 2 - r) * sy
            mat = texture_material(stage, f"/World/Looks/m{i}", ASSETS / asset.path)
            plate(stage, f"/World/plate{i}", x, y - 0.15, 0.12, 1.2, mat)
            sphere(stage, f"/World/ball{i}", (x, y + 0.35, 0.12 + 0.36), 0.36, mat)
            centres.append((asset.id, (x, y - 0.8, 0.12)))
        cam = camera(stage, (0, -17.0, 14.5), (0, -0.45, 0.2), 40)
        settle()
        capture = Capture()
        capture.warm()
        shot = capture.grab(args.subframes)
        draw = ImageDraw.Draw(shot)
        for asset_id, point in centres:
            x, y = project(cam, point)
            label(draw, x, y, asset_id)
        shot.save(OUT / f"swatches_{n}.png")
        print(f"WROTE swatches_{n}.png", flush=True)


def render_hdris():
    skies = [a for a in manifest() if a.kind == "hdri" and a.resolution != "8k" and (not args.only or a.id in args.only)]
    cw, ch, cols = 320, 180, 4
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    stage = new_stage()
    light = dome(stage, skies[0].id)
    ground = plain(stage, "/World/Looks/ground", (0.18, 0.18, 0.18), 0.6)
    chrome = plain(stage, "/World/Looks/chrome", (0.95, 0.95, 0.95), 0.03, 1.0)
    grey = plain(stage, "/World/Looks/grey", (0.18, 0.18, 0.18), 0.5)
    plate(stage, "/World/ground", 0, 0, 0, 3, ground)
    sphere(stage, "/World/chrome", (-0.55, 0, 0.5), 0.5, chrome)
    sphere(stage, "/World/grey", (0.55, 0, 0.5), 0.5, grey)
    camera(stage, (0, -4.2, 1.1), (0, 0, 0.75), 28)
    settle()
    capture = Capture((cw * 2, ch * 2))
    capture.warm()
    for i, sky in enumerate(skies):
        folder = ASSETS / sky.path
        hdr = folder if sky.path.endswith(".hdr") else sorted(folder.glob(f"*_{sky.resolution}.hdr"))[0]
        light.GetTextureFileAttr().Set(hdr.as_posix())
        settle(30)
        capture.grab(8)
        shot = capture.grab(args.subframes).resize((cw, ch), Image.LANCZOS)
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(shot, (x, y))
        label(ImageDraw.Draw(sheet), x + cw // 2, y + ch - 20, sky.id)
    sheet.save(OUT / "hdris.png")
    print("WROTE hdris.png", flush=True)


def localise_textures(prim, folder):
    local = {f.stem: f for f in folder.rglob("*.*")}
    for p in Usd.PrimRange(prim):
        for attr in p.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if value and value.path.startswith("/") and Path(value.path).stem in local:
                attr.Set(Sdf.AssetPath(local[Path(value.path).stem].as_posix()))


def model_source(asset):
    if asset.remote:
        return asset.path
    return sorted((ASSETS / asset.path).glob("*.usd*"))[0].as_posix()


def render_props():
    models = [a for a in manifest() if a.kind == "model" and "legacy" not in a.category]
    models = [a for a in models if (not args.only or a.id in args.only) and (not args.source or a.source == args.source)]
    cw, ch, cols, per = 320, 240, 4, 12
    pages = [models[i : i + per] for i in range(0, len(models), per)]
    for n, page in enumerate(pages, 1):
        sheet = Image.new("RGB", (W, H), (20, 20, 20))
        stage = new_stage()
        dome(stage, args.sky)
        ground = plain(stage, "/World/Looks/ground", (0.3, 0.3, 0.3), 0.7)
        plate(stage, "/World/ground", 0, 0, 0, 400, ground)
        holders = []
        for i, asset in enumerate(page):
            src = model_source(asset)
            layer = Usd.Stage.Open(src)
            scale = UsdGeom.GetStageMetersPerUnit(layer) if layer else 1.0
            y_up = layer is not None and UsdGeom.GetStageUpAxis(layer) == UsdGeom.Tokens.y
            holder = UsdGeom.Xform.Define(stage, f"/World/p{i}")
            fit = UsdGeom.Xform.Define(stage, f"/World/p{i}/fit")
            inner = UsdGeom.Xform.Define(stage, f"/World/p{i}/fit/model")
            inner.GetPrim().GetReferences().AddReference(src)
            ops = UsdGeom.Xformable(fit)
            ops.AddScaleOp().Set(Gf.Vec3f(scale))
            if y_up:
                ops.AddRotateXOp().Set(90.0)
            if not asset.remote:
                localise_textures(inner.GetPrim(), ASSETS / asset.path)
            UsdGeom.Imageable(holder).MakeInvisible()
            holders.append((asset, holder))
        camera(stage, (3, -4, 2), (0, 0, 0.5), 30)
        settle(120)
        capture = Capture((cw * 2, ch * 2))
        capture.warm()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for i, (asset, holder) in enumerate(holders):
            UsdGeom.Imageable(holder).MakeVisible()
            box = cache.ComputeWorldBound(holder.GetPrim()).ComputeAlignedRange()
            if box.IsEmpty():
                print(f"EMPTY {asset.id}", flush=True)
                UsdGeom.Imageable(holder).MakeInvisible()
                continue
            lift = -box.GetMin()[2]
            UsdGeom.Xformable(holder).ClearXformOpOrder()
            UsdGeom.Xformable(holder).AddTranslateOp().Set(Gf.Vec3d(-box.GetMidpoint()[0], -box.GetMidpoint()[1], lift))
            size = box.GetSize()
            radius = max(size[0], size[1], size[2]) * 0.75 + 0.05
            mid_z = size[2] / 2
            camera(stage, (radius * 1.6, -radius * 2.2, mid_z + radius * 1.0), (0, 0, mid_z), 30)
            settle(20)
            capture.grab(8)
            shot = capture.grab(args.subframes).resize((cw, ch), Image.LANCZOS)
            x, y = (i % cols) * cw, (i // cols) * ch
            sheet.paste(shot, (x, y))
            dims = f"{asset.id} {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f}m"
            label(ImageDraw.Draw(sheet), x + cw // 2, y + ch - 20, dims)
            print(f"PROP {dims}", flush=True)
            UsdGeom.Imageable(holder).MakeInvisible()
        name = f"props_{args.source or 'all'}_{n}.png"
        sheet.save(OUT / name)
        print(f"WROTE {name}", flush=True)


OUT.mkdir(parents=True, exist_ok=True)
{"swatches": render_swatches, "hdris": render_hdris, "props": render_props}[args.mode]()
import os

os._exit(0)

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
LAYER = SIM_DIR / "output" / "flare_fx.usda"
TEX_DIR = SIM_DIR / "assets" / "generated"
FLAME_TEX = TEX_DIR / "flare_flame_atlas.png"
SMOKE_TEX = TEX_DIR / "flare_smoke_puff.png"
MDL = SIM_DIR / "assets" / "mdl" / "flare_fx.mdl"
HALO_TEX = TEX_DIR / "flare_halo.png"
ROOT = "/World/FlareFx"
LIGHT = "/World/Cinematic/Lamps/lamp_400"
LIGHT_RADIUS, LIGHT_POWER = 6.0, 0.15
LIGHT_SCALE = LIGHT_POWER * (3.0 / LIGHT_RADIUS) ** 2
FLOW = "/World/Atmos/Flow"
CAMERAS = ("/World/Judge/Cam", "/World/Cinematic/Cameras/stream")
TIP = (-58.4, 8.0, 70.6)
JET = (-0.75, 0.0, 0.66)
COLS, ROWS, TILE_W, TILE_H = 8, 4, 256, 512
FRAMES = COLS * ROWS
FPS = 18.0
HEIGHT, WIDTH = 24.0, 14.0
WIND = (2.6, 1.5)
EMISSION = float(os.environ.get("RIG_FLARE_GAIN", "6000"))
CARDS = (
    (0.0, 1.0, 1.0, 0.0, 1.0),
    (52.0, 0.75, 0.8, 0.37, 1.27),
    (-48.0, 0.68, 0.72, 0.71, 0.83),
)
SEGMENTS = 12
SMOKE_N, SMOKE_LIFE, SMOKE_LEVELS = 40, 10.0, 5
HALO_SIZE, HALO_GAIN, HALO_ALPHA = 80.0, EMISSION * 0.2, 0.5
SIDE_GAIN = 0.8
STOPS = (
    (0.0, (0.7, 0.1, 0.01)),
    (0.3, (1.0, 0.38, 0.05)),
    (0.55, (1.0, 0.6, 0.16)),
    (0.75, (1.0, 0.8, 0.42)),
    (0.9, (1.0, 0.9, 0.7)),
    (1.0, (1.0, 0.97, 0.92)),
)


def tile_noise(h: int, w: int, cy: int, cx: int, rng) -> np.ndarray:
    g = rng.random((cy, cx))
    y = np.arange(h) * cy / h
    x = np.arange(w) * cx / w
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
    a, b = g[y0][:, x0], g[y0][:, x1]
    c, d = g[y1][:, x0], g[y1][:, x1]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(h: int, w: int, cy: int, cx: int, octaves: int, rng) -> np.ndarray:
    out, amp, total = np.zeros((h, w)), 1.0, 0.0
    for k in range(octaves):
        out += amp * tile_noise(h, w, cy << k, cx << k, rng)
        total += amp
        amp *= 0.5
    return out / total


def smoothstep(a: float, b: float, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def ramp(t: np.ndarray) -> np.ndarray:
    xs = np.array([s[0] for s in STOPS])
    cs = np.array([s[1] for s in STOPS])
    return np.stack([np.interp(t, xs, cs[:, i]) for i in range(3)], -1)


def flame_frames(n: int = FRAMES, w: int = TILE_W, h: int = TILE_H, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = fbm(h, w, 3, 4, 4, rng)
    fine = fbm(h, w, 5, 9, 4, rng)
    warp = fbm(h, w, 2, 3, 3, rng)
    side = fbm(h, w, 3, 4, 3, rng)
    u = np.linspace(-1.0, 1.0, w)[None, :]
    v = np.linspace(1.0, 0.0, h)[:, None]
    prof = (v + 0.03) ** 0.75 * (1.02 - v) ** 0.8
    width = 0.85 * prof / prof.max()
    edge = smoothstep(0.0, 0.08, 1.0 - np.abs(u)) * smoothstep(0.0, 0.06, 1.0 - v)
    out = np.zeros((n, h, w, 4))
    for k in range(n):
        roll = lambda f, s: np.roll(f, -int(round(k * s * h / n)), axis=0)
        wv = (roll(warp, 1) - 0.5) * 2.0
        sv = (roll(side, 2) - 0.5) * 2.0
        uu = u + 0.5 * wv * v**1.2 + 0.18 * sv * v**0.8
        body = np.clip(1.0 - (np.abs(uu) / np.maximum(width, 1e-3)) ** 2, 0.0, 1.0)
        turb = 0.5 * roll(coarse, 1) + 0.5 * (1.0 - np.abs(roll(fine, 3) - 0.5) * 2.0)
        env = np.clip(1.0 - (np.abs(uu) / np.maximum(1.35 * width, 1e-3)) ** 2, 0.0, 1.0)
        i = body**0.8 * (1.1 - 0.85 * v) + (turb - 0.55) * (0.6 + 1.6 * v) * env - 0.25 * v
        i = np.clip(i * env**0.5, 0.0, 1.0) * smoothstep(0.0, 0.03, v) * edge
        heat = np.clip(i * (1.0 - 0.8 * v) * body**0.4, 0.0, 1.0)
        streak = roll(fine, 3)
        glow = (0.07 + 0.93 * heat**2.4) * (0.35 + 1.3 * streak**1.5)
        out[k, ..., 3] = smoothstep(0.06, 0.3, i)
        out[k, ..., :3] = ramp(heat) * glow[..., None]
    return out


def srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def atlas(frames: np.ndarray, cols: int = COLS) -> np.ndarray:
    n, h, w, _ = frames.shape
    rows = -(-n // cols)
    img = np.zeros((rows * h, cols * w, 4))
    for k in range(n):
        r, c = divmod(k, cols)
        img[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[k]
    img[..., :3] = srgb(img[..., :3])
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def smoke_puff(size: int = 256, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = fbm(size, size, 4, 4, 5, rng)
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.hypot(x, y) + (n - 0.5) * 0.7
    a = np.clip(1.0 - r, 0.0, 1.0) ** 1.4 * smoothstep(0.25, 0.75, n + 0.25 * (1 - r))
    g = 0.35 + 0.65 * n
    img = np.dstack([g, g * 0.95, g * 0.9, a / max(a.max(), 1e-6)])
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def halo(size: int = 256) -> np.ndarray:
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.hypot(x, y)
    a = (np.exp(-(r / 0.22) ** 2) * 0.5 + np.exp(-(r / 0.45) ** 2) * 0.5) * smoothstep(1.0, 0.55, r)
    img = np.dstack([a, a * 0.45, a * 0.12, a])
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def flicker(t: float, seed: float = 0.0) -> float:
    s = (math.sin(t * 11.3 + seed) * 0.5 + math.sin(t * 17.9 + 2.0 * seed + 1.3) * 0.3
         + math.sin(t * 5.1 + 3.0 * seed + 0.4) * 0.6 + math.sin(t * 29.3 + seed) * 0.2)
    return 0.92 + 0.11 * s / 1.6


def lean_vector(wind) -> np.ndarray:
    speed = math.hypot(*wind)
    if speed < 1e-6:
        return np.zeros(3)
    k = 0.2 + 0.5 * min(1.0, speed / 15.0)
    return np.array([wind[0] / speed * k, wind[1] / speed * k, 0.0])


def centerline(v: np.ndarray, t: float, wind=WIND, tip=TIP, height: float = HEIGHT) -> np.ndarray:
    v = np.asarray(v, float)[:, None]
    jet = np.asarray(JET, float)
    jet = jet / np.linalg.norm(jet)
    lean = lean_vector(wind)
    a = 0.25 * (1 - np.exp(-4.0 * v)) / (1 - math.exp(-4.0))
    sway = 0.09 * math.sin(t * 1.3) + 0.05 * math.sin(t * 2.9 + 1.0)
    side = np.array([-lean[1], lean[0], 0.0])
    return (np.asarray(tip, float) + height * (jet * a + np.array([0, 0, 0.82]) * v + lean * v**1.7)
            + height * side * sway * v**1.5)


def card_points(line: np.ndarray, right: np.ndarray, width: float) -> np.ndarray:
    half = right * width * 0.5
    return np.stack([line - half, line, line + half], 1).reshape(-1, 3)


def facing_right(centre, eye, yaw_deg: float = 0.0) -> np.ndarray:
    d = np.asarray(eye, float) - np.asarray(centre, float)
    d[2] = 0.0
    if np.linalg.norm(d) < 1e-6:
        d = np.array([0.4, -1.0, 0.0])
    d /= np.linalg.norm(d)
    a = math.radians(yaw_deg)
    d = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a), 0.0])
    return np.array([-d[1], d[0], 0.0])


def smoke_state(t: float, wind=WIND, n: int = SMOKE_N, life: float = SMOKE_LIFE, origin=None):
    origin = centerline(np.array([1.0]), t, wind)[0] if origin is None else np.asarray(origin, float)
    w = np.array([wind[0], wind[1], 0.0])
    speed = max(np.linalg.norm(w), 1.5)
    w = w / max(np.linalg.norm(w), 1e-6) * speed
    i = np.arange(n)
    shift = t - i * life / n
    gen = np.floor(shift / life)
    age = shift - gen * life
    h = np.sin((i[:, None] * 12.9898 + gen[:, None] * 78.233 + np.array([0.0, 37.7, 91.3])) * 43758.5453)
    jitter = (h * 1e4 - np.floor(h * 1e4)) * 2 - 1
    pos = origin + w[None, :] * age[:, None] + np.array([0, 0, 2.2]) * age[:, None] ** 0.8
    pos = pos + jitter * (0.8 + 0.55 * age[:, None])
    scale = 3.0 + 2.1 * age
    u = age / life
    alpha = smoothstep(0.0, 0.08, u) * (1 - smoothstep(0.45, 1.0, u))
    return pos, scale, alpha


def fade_level(alpha: np.ndarray, levels: int = SMOKE_LEVELS) -> np.ndarray:
    return np.clip(np.round(np.asarray(alpha) * levels - 0.5), 0, levels - 1).astype(int)


def write_textures() -> None:
    from PIL import Image

    TEX_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas(flame_frames())).save(FLAME_TEX, optimize=True)
    Image.fromarray(smoke_puff()).save(SMOKE_TEX, optimize=True)
    Image.fromarray(halo()).save(HALO_TEX, optimize=True)


def _mdl_material(stage, path: str, sub: str, inputs: dict, rel):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.SetSourceAsset(Sdf.AssetPath(rel(MDL)), "mdl")
    sh.SetSourceAssetSubIdentifier(sub, "mdl")
    for k, v in inputs.items():
        if isinstance(v, Path):
            sh.CreateInput(k, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel(v)))
            sh.GetInput(k).GetAttr().SetColorSpace("raw")
        elif isinstance(v, tuple):
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        else:
            sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(v))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateSurfaceOutput().ConnectToSource(out)
    return mat


def _quad(stage, path: str):
    from pxr import Sdf, UsdGeom

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)])
    return m


def build_layer(path: Path = LAYER) -> Path:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    if not all(p.exists() for p in (FLAME_TEX, SMOKE_TEX, HALO_TEX)):
        write_textures()
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    rel = lambda p: os.path.relpath(p, path.parent)
    stage.OverridePrim("/World")
    for i in range(5):
        stage.OverridePrim(f"/World/Cinematic/Flare/flame_{i}").CreateAttribute(
            "visibility", Sdf.ValueTypeNames.Token, custom=False).Set("invisible")
    light = stage.OverridePrim(f"{LIGHT}/light")
    light.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float, custom=False).Set(LIGHT_RADIUS)
    light.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Scope.Define(stage, f"{ROOT}/Looks")
    rows = SEGMENTS + 1
    st = [(s, v / SEGMENTS) for v in range(rows) for s in (0.0, 0.5, 1.0)]
    faces = []
    for r in range(SEGMENTS):
        for c in range(2):
            a = r * 3 + c
            faces += [a, a + 1, a + 4, a + 3]
    line = centerline(np.linspace(0, 1, rows), 0.0)
    for k, (yaw, ws, hs, phase, rate) in enumerate(CARDS):
        mat = _mdl_material(stage, f"{ROOT}/Looks/flame_{k}", "flare_card",
                            {"atlas": FLAME_TEX, "frame": 0.0, "cols": COLS, "rows": ROWS, "gain": EMISSION}, rel)
        m = UsdGeom.Mesh.Define(stage, f"{ROOT}/flame_{k}")
        m.CreatePointsAttr(card_points(line, facing_right(line[0], (0, -200, 50), yaw), WIDTH * ws).tolist())
        m.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
        m.CreateFaceVertexIndicesAttr(faces)
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(st)
        m.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    h = _quad(stage, f"{ROOT}/halo")
    h.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI.Apply(h.GetPrim()).Bind(
        _mdl_material(stage, f"{ROOT}/Looks/halo", "flare_card",
                      {"atlas": HALO_TEX, "cols": 1.0, "rows": 1.0, "gain": HALO_GAIN, "alpha_gain": HALO_ALPHA}, rel))
    inst = UsdGeom.PointInstancer.Define(stage, f"{ROOT}/smoke")
    protos = []
    for k in range(SMOKE_LEVELS):
        q = _quad(stage, f"{ROOT}/smoke/proto_{k}")
        a = 0.85 * (k + 1) / SMOKE_LEVELS
        mat = _mdl_material(stage, f"{ROOT}/Looks/smoke_{k}", "smoke_card",
                            {"puff": SMOKE_TEX, "tint": (0.06, 0.055, 0.05), "alpha_gain": a}, rel)
        UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).Bind(mat)
        protos.append(q.GetPath())
    inst.CreatePrototypesRel().SetTargets(protos)
    inst.CreatePositionsAttr([])
    inst.CreateProtoIndicesAttr([])
    inst.CreateScalesAttr([])
    inst.CreateOrientationsAttr([])
    stage.GetRootLayer().GetPrimAtPath("/World").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()
    return path


RTX = {"/rtx/raytracing/fractionalCutoutOpacity": True, "/rtx/pathtracing/fractionalCutoutOpacity": True,
       "/rtx/shadows/fractionalCutoutOpacity": True}


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_FLARE_FX", "1") == "0":
        return False
    try:
        import carb.settings

        for k, v in RTX.items():
            carb.settings.get_settings().set(k, v)
    except ImportError:
        pass
    root = stage.GetRootLayer()
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    if any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        return True
    root.subLayerPaths.insert(0, target)
    return True


def _eye(stage):
    from pxr import Usd, UsdGeom

    for path in CAMERAS:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return np.array(m.ExtractTranslation())
    return np.array([0.0, -200.0, 50.0])


def billboard_quats(pos: np.ndarray, eye) -> np.ndarray:
    d = np.asarray(eye, float)[None, :] - pos
    yaw = np.arctan2(d[:, 0], -d[:, 1])
    return np.column_stack([np.cos(yaw / 2), np.zeros(len(pos)), np.zeros(len(pos)), np.sin(yaw / 2)])


def update(stage, t: float, wind=WIND) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    root = stage.GetPrimAtPath(ROOT)
    if not root or not root.IsValid():
        return
    flow = stage.GetPrimAtPath(FLOW)
    flow_flame = bool(flow and flow.IsValid()) and os.environ.get("RIG_FLARE_FLOW", "0") == "1"
    shown = "invisible" if flow_flame else "inherited"
    emitter = stage.GetPrimAtPath(f"{FLOW}/flowEmitterSphere")
    if emitter and emitter.IsValid() and not flow_flame:
        emitter.GetAttribute("fuel").Set(0.0)
        emitter.GetAttribute("temperature").Set(0.0)
    for name in [f"flame_{k}" for k in range(len(CARDS))]:
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{ROOT}/{name}")).GetVisibilityAttr().Set(shown)
    eye = _eye(stage)
    rows = SEGMENTS + 1
    line = centerline(np.linspace(0, 1, rows), t, wind)
    level = flicker(t)
    for k, (yaw, ws, hs, phase, rate) in enumerate(CARDS):
        v = np.linspace(0, 1, rows)
        sub = centerline(v * hs, t + phase, wind)
        away = np.asarray(eye, float) - line[rows // 3]
        away = -away / max(np.linalg.norm(away), 1e-6) * 1.6 * abs(np.sign(yaw))
        sub = sub + np.outer(1.8 * v**1.5, facing_right(line[rows // 3], eye)) * np.sign(yaw) + away
        pts = card_points(sub, facing_right(line[rows // 3], eye, yaw), WIDTH * ws * (0.96 + 0.06 * level))
        UsdGeom.Mesh(stage.GetPrimAtPath(f"{ROOT}/flame_{k}")).GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        frame = int(math.floor((t + phase * 7.0) * FPS * rate))
        sh = UsdShade.Shader.Get(stage, f"{ROOT}/Looks/flame_{k}/shader")
        sh.GetInput("frame").Set(float(frame % FRAMES))
        sh.GetInput("gain").Set(EMISSION * flicker(t, k + 1.0) * (1.0 if yaw == 0 else SIDE_GAIN))
    pos, scale, alpha = smoke_state(t, wind)
    inst = UsdGeom.PointInstancer(stage.GetPrimAtPath(f"{ROOT}/smoke"))
    inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
    inst.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(scale[:, None], 3, 1).astype(np.float32)))
    inst.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(fade_level(alpha).astype(np.int32)))
    inst.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(*map(float, q)) for q in billboard_quats(pos, eye)]))
    centre = line[int(rows * 0.4)]
    right = facing_right(centre, eye) * HALO_SIZE * 0.5
    toward = np.asarray(eye, float) - centre
    toward /= max(np.linalg.norm(toward), 1e-6)
    up = np.cross(toward, right / max(np.linalg.norm(right), 1e-6)) * HALO_SIZE * 0.5
    up = up if up[2] >= 0 else -up
    base = centre - toward * 8.0
    quad = np.array([base - right - up, base + right - up, base + right + up, base - right + up])
    UsdGeom.Mesh(stage.GetPrimAtPath(f"{ROOT}/halo")).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(quad.astype(np.float32)))
    UsdShade.Shader.Get(stage, f"{ROOT}/Looks/halo/shader").GetInput("gain").Set(
        HALO_GAIN * level * (0.4 if shown == "invisible" else 1.0))
    lamp = stage.GetPrimAtPath(LIGHT)
    if lamp:
        c = line[rows // 4]
        op = lamp.GetAttribute("xformOp:translate")
        if op:
            op.Set(Gf.Vec3d(*map(float, c)))
        light = stage.GetPrimAtPath(f"{LIGHT}/light")
        base = light.GetAttribute("spark:baseIntensity") if light else None
        if base and base.IsValid() and base.Get():
            if not light.GetAttribute("spark:flareScaled").IsValid():
                base.Set(float(base.Get()) * LIGHT_SCALE)
                light.CreateAttribute("spark:flareScaled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            light.GetAttribute("inputs:intensity").Set(float(base.Get()) * level)


if __name__ == "__main__":
    write_textures()
    print(build_layer())

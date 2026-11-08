from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

REPO = Path(__file__).resolve().parent.parent
RUST_DIR = REPO / "sim/assets/third_party/textures/rust_coarse_01"
RUST_FULL = 0.35
EPS = 1e-4
TILE_M = 0.3
PRIMER_RGB = (0.80, 0.79, 0.75)
WET_RGB = (0.025, 0.03, 0.035)
WET_MAX = 0.6
RADII = (0.1522, 0.1528, 0.1534)
CROSS = np.ones((3, 3), bool)


def region_of(*grids, margin: int = 3) -> tuple[int, int, int, int]:
    on = np.zeros(np.asarray(grids[0]).shape, bool)
    for g in grids:
        on |= np.asarray(g, np.float32) > 0
    n_z, n_t = on.shape
    if not on.any():
        return 0, n_z, 0, n_t
    rows, cols = np.flatnonzero(on.any(1)), np.flatnonzero(on.any(0))
    gaps = np.diff(np.r_[cols, cols[0] + n_t])
    k = int(gaps.argmax())
    j0, j1 = int(cols[(k + 1) % len(cols)]), int(cols[k])
    if j1 < j0:
        j1 += n_t
    span = min(j1 - j0 + 1 + 2 * margin, n_t)
    j0 = j0 - margin if span < n_t else 0
    return max(0, int(rows[0]) - margin), min(n_z, int(rows[-1]) + 1 + margin), j0, j0 + span


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(h: int, w: int, cell_px: float, seed: int, octaves: int = 5, gain: float = 0.55) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total, size = 1.0, 0.0, float(cell_px)
    for _ in range(octaves):
        gh, gw = int(math.ceil(h / size)) + 2, int(math.ceil(w / size)) + 2
        lat = rng.random((gh, gw), np.float32)
        y, x = (np.arange(h) + 0.5) / size, (np.arange(w) + 0.5) / size
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = smoothstep(0, 1, y - y0)[:, None], smoothstep(0, 1, x - x0)[None, :]
        top = lat[y0][:, x0] * (1 - fx) + lat[y0][:, x0 + 1] * fx
        bot = lat[y0 + 1][:, x0] * (1 - fx) + lat[y0 + 1][:, x0 + 1] * fx
        out += amp * (top * (1 - fy) + bot * fy)
        total += amp
        amp *= gain
        size = max(size / 2.0, 1.0)
    out /= total
    return ((out - out.min()) / max(float(out.max() - out.min()), 1e-6)).astype(np.float32)


def tiled(path: Path, h: int, w: int, tile_px: tuple[float, float], mode: str = "RGB") -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert(mode)
    th, tw = max(1, round(tile_px[0])), max(1, round(tile_px[1]))
    a = np.asarray(img.resize((tw, th), Image.LANCZOS))
    return a[np.arange(h) % th][:, np.arange(w) % tw]


def normal_png(height: np.ndarray, strength: float) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float32))
    n = np.stack([-gx * strength, gy * strength, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)


class SurfaceMasks:

    def __init__(self, pipe, region=None, mm_per_px: float = 0.6, seed: int = 7, textures: Path = RUST_DIR):
        self.shape = (pipe.n_z, pipe.n_theta)
        self.region = tuple(region) if region is not None else (0, pipe.n_z, 0, pipe.n_theta)
        i0, i1, j0, j1 = self.region
        self.dz = pipe.length / pipe.n_z
        self.dt = 2 * math.pi * pipe.radius / pipe.n_theta
        self.ppz = max(4, round(1000 * self.dz / mm_per_px))
        self.ppt = max(4, round(1000 * self.dt / mm_per_px))
        self.H, self.W = (i1 - i0) * self.ppz, (j1 - j0) * self.ppt
        zc = i1 - (np.arange(self.H) + 0.5) / self.ppz
        tc = j0 + (np.arange(self.W) + 0.5) / self.ppt
        self.rows_cell = np.clip(np.floor(zc).astype(int), 0, pipe.n_z - 1)
        self.cols_cell = np.floor(tc).astype(int) % pipe.n_theta
        y = np.clip(zc - 0.5, 0, pipe.n_z - 1)
        x = tc - 0.5
        self._y0 = np.floor(y).astype(int)
        self._y1 = np.minimum(self._y0 + 1, pipe.n_z - 1)
        self._fy = (y - self._y0).astype(np.float32)[:, None]
        x0 = np.floor(x).astype(int)
        self._fx = (x - x0).astype(np.float32)[None, :]
        self._x0, self._x1 = x0 % pipe.n_theta, (x0 + 1) % pipe.n_theta
        self.mm = mm = mm_per_px
        self.noise_coarse = value_noise(self.H, self.W, 14.0 / mm, seed)
        self.noise_fine = value_noise(self.H, self.W, 5.0 / mm, seed + 1, octaves=4)
        self.noise_paint = value_noise(self.H, self.W, 40.0 / mm, seed + 2, octaves=3)
        tile = (1000 * TILE_M / mm, 1000 * TILE_M / mm)
        self.textures = Path(textures)
        self.rust_albedo = tiled(self.textures / "diff.jpg", self.H, self.W, tile).astype(np.float32)
        self.tile = tile
        self.cell_px = 0.5 * (self.ppz + self.ppt)

    def up(self, grid) -> np.ndarray:
        g = np.asarray(grid, np.float32)
        top = g[self._y0][:, self._x0] * (1 - self._fx) + g[self._y0][:, self._x1] * self._fx
        bot = g[self._y1][:, self._x0] * (1 - self._fx) + g[self._y1][:, self._x1] * self._fx
        return top * (1 - self._fy) + bot * self._fy

    def near(self, cells) -> np.ndarray:
        return np.asarray(cells, bool)[self.rows_cell][:, self.cols_cell]

    def _gate(self, cells, grow: bool = True) -> np.ndarray:
        c = np.asarray(cells, bool)
        if grow:
            c = binary_dilation(np.pad(c, ((0, 0), (1, 1)), mode="wrap"), CROSS)[:, 1:-1] | c
        return self.near(c)

    def _blur(self, a: np.ndarray, cells: float) -> np.ndarray:
        return gaussian_filter(a, (cells * self.ppz, cells * self.ppt), mode="nearest")

    def rust_alpha(self, rust) -> np.ndarray:
        r = np.asarray(rust, np.float32)
        level = self._blur(self.up(np.clip(r / RUST_FULL, 0, 1)), 0.25)
        s = level * 1.35 - 0.7 * self.noise_coarse - 0.3 * self.noise_fine
        return smoothstep(0.1, 0.16, s) * self._gate(r > 0), level

    def masks(self, rust, sprayed, painted):
        r = np.asarray(rust, np.float32)
        paint_cells = np.asarray(painted, bool) & ~(r > EPS)
        wet_cells = np.asarray(sprayed, bool) & ~np.asarray(painted, bool)
        ra, level = self.rust_alpha(r)
        shade = 1.15 - 0.18 * level - 0.3 * (4 * ra * (1 - ra))
        rgb = np.clip(self.rust_albedo * shade[..., None], 0, 255)
        rust_rgba = np.concatenate([rgb, 255 * ra[..., None]], -1).round().astype(np.uint8)
        wf = self._blur(self.up(wet_cells.astype(np.float32)), 0.45)
        wet = smoothstep(0.15, 0.55, wf + 0.3 * (self.noise_coarse - 0.5)) * self._gate(wet_cells)
        pf = self._blur(self.up(paint_cells.astype(np.float32)), 0.5)
        primer = smoothstep(0.3, 0.7, pf + 0.2 * (self.noise_coarse - 0.5) + 0.1 * (self.noise_fine - 0.5))
        primer *= self._gate(paint_cells) & ~self.near(r > EPS)
        return rust_rgba, wet.astype(np.float32), primer.astype(np.float32)

    def __call__(self, rust, sprayed, painted):
        return self.masks(rust, sprayed, painted)

    def wet_rgba(self, wet: np.ndarray) -> np.ndarray:
        rgb = np.broadcast_to(np.asarray(WET_RGB, np.float32) ** (1 / 2.2) * 255, wet.shape + (3,))
        return np.concatenate([rgb, 255 * WET_MAX * wet[..., None]], -1).round().astype(np.uint8)

    def primer_rgba(self, primer: np.ndarray) -> np.ndarray:
        tone = 1.0 + 0.06 * (self.noise_paint - 0.5) + 0.03 * (self.noise_fine - 0.5)
        rgb = np.clip(np.asarray(PRIMER_RGB, np.float32) ** (1 / 2.2) * 255 * tone[..., None], 0, 255)
        return np.concatenate([rgb, 255 * primer[..., None]], -1).round().astype(np.uint8)

    def statics(self, out_dir: Path) -> dict:
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        nor = tiled(self.textures / "nor_gl.jpg", self.H, self.W, self.tile)
        rough = tiled(self.textures / "rough.jpg", self.H, self.W, self.tile, "L")
        peel = value_noise(self.H, self.W, 1.6 / self.mm, 11, octaves=3)
        for name, arr in (("rust_normal", nor), ("rust_rough", rough), ("primer_normal", normal_png(peel, 0.9))):
            files[name] = out_dir / f"{name}.png"
            Image.fromarray(arr).save(files[name])
        return files


def sleeve(pipe, region, radius: float, theta_sub: int = 6, z_sub: int = 1):
    i0, i1, j0, j1 = region
    th = np.linspace(j0, j1, (j1 - j0) * theta_sub + 1) * 2 * math.pi / pipe.n_theta
    z = np.linspace(i0, i1, (i1 - i0) * z_sub + 1) * pipe.length / pipe.n_z - pipe.length / 2
    Z, T = np.meshgrid(z, th, indexing="ij")
    n = np.cos(T)[..., None] * pipe.ref + np.sin(T)[..., None] * pipe.binormal
    pts = pipe.origin + Z[..., None] * pipe.axis + radius * n
    uv = np.stack([(T - th[0]) / (th[-1] - th[0]), (Z - z[0]) / (z[-1] - z[0])], -1)
    w = len(th)
    iz, it = np.meshgrid(np.arange(len(z) - 1), np.arange(w - 1), indexing="ij")
    k = (iz * w + it).ravel()
    idx = np.stack([k, k + w, k + w + 1, k + 1], -1).ravel()
    return pts.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2), idx


class _Sink:

    def __init__(self, name: str, out_dir: Path, dynamic: bool):
        self.name, self.out_dir, self.provider, self.k = name, out_dir, None, 0
        if dynamic:
            try:
                import omni.ui as ui

                self.provider = ui.DynamicTextureProvider(name)
            except Exception:
                self.provider = None

    @property
    def asset(self) -> str:
        if self.provider is not None:
            return f"dynamic://{self.name}"
        return str(self.out_dir / f"{self.name}_{self.k:05d}.png")

    def push(self, rgba: np.ndarray) -> str | None:
        if self.provider is not None:
            lin = rgba.copy()
            lin[..., :3] = ((rgba[..., :3] / 255.0) ** 2.2 * 255).round().astype(np.uint8)
            self.provider.set_data_array(np.ascontiguousarray(lin), [rgba.shape[1], rgba.shape[0]])
            return None
        from PIL import Image

        old = self.out_dir / f"{self.name}_{self.k:05d}.png"
        self.k += 1
        Image.fromarray(rgba, "RGBA").save(self.asset, compress_level=1)
        old.unlink(missing_ok=True)
        return self.asset


def omnipbr(stage, path: str, values: dict):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    kinds = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for k, v in values.items():
        if isinstance(v, (str, Path)):
            inp = sh.CreateInput(k, Sdf.ValueTypeNames.Asset)
            inp.Set(Sdf.AssetPath(str(v).replace("\\", "/")))
            if "normal" in k or "rough" in k:
                inp.GetAttr().SetColorSpace("raw")
        elif isinstance(v, tuple) and len(v) == 3:
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        elif isinstance(v, tuple):
            sh.CreateInput(k, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*v))
        else:
            sh.CreateInput(k, kinds[type(v)]).Set(v)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    return mat, sh


class SurfaceLook:

    def __init__(self, stage, pipe, root: str = "/World/SurfaceLook", region=None, textures: Path = RUST_DIR,
                 out_dir: Path | None = None, mm_per_px: float = 0.6, dynamic: bool = True):
        from pxr import Sdf, UsdGeom, UsdShade, Vt

        self.stage, self.pipe, self.root = stage, pipe, root
        self.m = SurfaceMasks(pipe, region, mm_per_px, textures=textures)
        self.out_dir = Path(out_dir or tempfile.mkdtemp(prefix="surface_look_"))
        statics = self.m.statics(self.out_dir)
        tag = root.strip("/").replace("/", "_")
        self.sinks = {k: _Sink(f"{tag}_{k}", self.out_dir, dynamic) for k in ("rust", "wet", "primer")}
        looks = {
            "rust": {"reflection_roughness_constant": 0.85, "reflectionroughness_texture": statics["rust_rough"],
                     "reflection_roughness_texture_influence": 0.8, "metallic_constant": 0.0,
                     "normalmap_texture": statics["rust_normal"], "bump_factor": 0.8,
                     "opacity_threshold": 0.5},
            "wet": {"reflection_roughness_constant": 0.05, "metallic_constant": 0.0, "specular_level": 0.9,
                    "opacity_threshold": 0.0},
            "primer": {"reflection_roughness_constant": 0.42, "metallic_constant": 0.0, "specular_level": 0.45,
                       "normalmap_texture": statics["primer_normal"], "bump_factor": 0.25,
                       "opacity_threshold": 0.0},
        }
        UsdGeom.Scope.Define(stage, root)
        self.meshes, self.shaders, self._key = {}, {}, None
        for (name, look), radius in zip(looks.items(), (RADII[0], RADII[1], RADII[2])):
            tex = self.sinks[name].asset
            mat, sh = omnipbr(stage, f"{root}/Looks/{name}", {
                "diffuse_texture": tex, "opacity_texture": tex, "enable_opacity": True,
                "enable_opacity_texture": True, "opacity_mode": 0, "texture_scale": (1.0, 1.0), **look})
            pts, nrm, uv, idx = sleeve(pipe, self.m.region, radius)
            mesh = UsdGeom.Mesh.Define(stage, f"{root}/{name}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (len(idx) // 4)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx.astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(False)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.vertex).Set(
                Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
            self.meshes[name], self.shaders[name] = mesh, sh
        z = np.zeros(self.m.shape)
        self.update(z, z, z)

    def masks(self, rust, sprayed, painted):
        return self.m(rust, sprayed, painted)

    def update(self, rust_grid, sprayed, painted, t: float | None = None) -> bool:
        from pxr import Sdf, UsdGeom

        r = np.asarray(rust_grid, np.float32)
        s, p = np.asarray(sprayed) > 0.5, np.asarray(painted) > 0.5
        key = r.tobytes() + s.tobytes() + p.tobytes()
        if key == self._key:
            return False
        self._key = key
        rust, wet, primer = self.m(r, s, p)
        layers = {"rust": (rust, rust[..., 3].max() > 127), "wet": (self.m.wet_rgba(wet), wet.max() > 0),
                  "primer": (self.m.primer_rgba(primer), primer.max() > 0)}
        for name, (rgba, show) in layers.items():
            img = UsdGeom.Imageable(self.meshes[name])
            if not show:
                img.MakeInvisible()
                continue
            path = self.sinks[name].push(rgba)
            if path is not None:
                for k in ("diffuse_texture", "opacity_texture"):
                    self.shaders[name].GetInput(k).Set(Sdf.AssetPath(path))
            img.MakeVisible()
        return True


def composite(m: SurfaceMasks, rust_rgba, wet, primer, pipe_rgb=(0.16, 0.22, 0.16)) -> np.ndarray:
    th = (m.region[2] + (np.arange(m.W) + 0.5) / m.ppt) / m.shape[1] * 2 * math.pi
    light = 0.55 + 0.45 * np.cos(th - th.mean())[None, :, None]
    col = np.ones((m.H, m.W, 3), np.float32) * np.asarray(pipe_rgb, np.float32) ** (1 / 2.2)
    ra = (rust_rgba[..., 3:] > 127).astype(np.float32)
    col = col * (1 - ra) + rust_rgba[..., :3] / 255.0 * ra
    w = WET_MAX * wet[..., None]
    col = col * (1 - 0.85 * w) + 0.05 * w
    pr = m.primer_rgba(primer)
    pa = pr[..., 3:] / 255.0
    col = col * (1 - pa) + pr[..., :3] / 255.0 * pa
    return (np.clip(col * light, 0, 1) * 255).round().astype(np.uint8)


def blocky(m: SurfaceMasks, rust, sprayed, painted) -> np.ndarray:
    p = m.near(np.asarray(painted) > 0.5)
    wet = m.near((np.asarray(sprayed) > 0.5) & ~(np.asarray(painted) > 0.5)).astype(np.float32)
    rr = m.near((np.clip(np.asarray(rust) / 0.1, 0, 1) > 0.5) & ~(np.asarray(painted) > 0.5))
    rgba = np.zeros((m.H, m.W, 4), np.uint8)
    rgba[..., :3] = (205, 105, 40)
    rgba[..., 3] = 255 * rr
    return composite(m, rgba, wet / WET_MAX * 0.9, p.astype(np.float32))


def main(argv=None) -> None:
    from PIL import Image

    from integration.scrub_motion import film_job

    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=str(REPO / "runs_sample/film/timeline.jsonl"))
    ap.add_argument("--out", default=str(REPO / "runs/surface_look"))
    a = ap.parse_args(argv)
    job = film_job(a.timeline)
    p, w = job.plan_, job.plan_.windows
    region = region_of(p.rust[0], p.sprayed[-1], p.painted[-1])
    m = SurfaceMasks(job.pipe, region)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = [("0_before", w["spray"] - 0.5), ("1_soak", w["soak"] + 1), ("2_scrub3", w["scrub"] + 3),
             ("3_scrub8", w["scrub"] + 8), ("4_scrub14", w["scrub"] + 14), ("5_paint1", w["paint"] + 1.2),
             ("6_painted", w["stow"])]
    for name, t in picks:
        s = job.at(t)
        new = composite(m, *m(s["rust"], s["sprayed"], s["painted"]))
        old = blocky(m, s["rust"], s["sprayed"], s["painted"])
        gap = np.full((m.H, 12, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([old, gap, new], 1)).save(out / f"surface_{name}.png")
        print(out / f"surface_{name}.png", f"t={t:.1f}")


if __name__ == "__main__":
    main()

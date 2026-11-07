"""Sky, sun, animated ocean and renderer look for the offshore rig. Call after SimulationApp has started.

    env = add_environment(stage, preset="clear", sea_level_z=-25.0)
    env.update(t)   # once per frame, t = sim time in seconds
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from sim.ocean_mdl import MODULE, build_mdl
from sim.ocean_presets import Preset, get_preset, render_settings
from sim.ocean_waves import EARTH_RADIUS, Waves, make_waves, polar_grid

ROOT = "/Environment"
HDRI_DIR = Path(__file__).resolve().parent / "assets" / "third_party" / "hdris"
WAVE_FULL_RADIUS = 600.0
WAVE_END_RADIUS = 2000.0


def _smooth_fade(r: np.ndarray) -> np.ndarray:
    s = np.clip((WAVE_END_RADIUS - r) / (WAVE_END_RADIUS - WAVE_FULL_RADIUS), 0.0, 1.0)
    return (s * s * (3 - 2 * s)).astype(np.float32)


class _Displacer:
    """Gerstner displacement of the active ring of the mesh; Warp on GPU when available, else numpy."""

    def __init__(self, waves: Waves, rest: np.ndarray, base_z: np.ndarray, fade: np.ndarray, spacing: np.ndarray):
        self.waves, self.rest, self.base_z, self.fade, self.spacing = waves, rest, base_z, fade, spacing
        self.wp = None
        try:
            import warp as wp

            wp.init()
            self.wp = wp
            self._setup_warp()
        except Exception as exc:  # noqa: BLE001
            print(f"[environment] warp unavailable ({exc}); using numpy waves", flush=True)

    def _setup_warp(self) -> None:
        wp = self.wp
        w = self.waves

        @wp.kernel
        def gerstner(
            rest: wp.array(dtype=wp.vec2),
            base_z: wp.array(dtype=float),
            fade: wp.array(dtype=float),
            spacing: wp.array(dtype=float),
            wa: wp.array(dtype=wp.vec4),
            wb: wp.array(dtype=wp.vec4),
            count: int,
            t: float,
            out: wp.array(dtype=wp.vec3),
        ):
            i = wp.tid()
            p = rest[i]
            sp = spacing[i]
            ax = float(0.0)
            ay = float(0.0)
            az = float(0.0)
            for j in range(count):
                a = wa[j]
                b = wb[j]
                s = wp.clamp((6.2831853 / a[2] / sp - 3.0) / 3.0, 0.0, 1.0)
                amp = a[3] * s * s * (3.0 - 2.0 * s)
                th = a[2] * (a[0] * p[0] + a[1] * p[1]) - (b[0] * t - b[1])
                c = wp.cos(th)
                qa = b[2] * amp * c
                ax += qa * a[0]
                ay += qa * a[1]
                az += amp * wp.sin(th)
            f = fade[i]
            out[i] = wp.vec3(p[0] + ax * f, p[1] + ay * f, base_z[i] + az * f)

        self.kernel = gerstner
        dev = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.dev = dev
        self.d_rest = wp.array(self.rest.astype(np.float32), dtype=wp.vec2, device=dev)
        self.d_base = wp.array(self.base_z, dtype=float, device=dev)
        self.d_fade = wp.array(self.fade, dtype=float, device=dev)
        self.d_spacing = wp.array(self.spacing.astype(np.float32), dtype=float, device=dev)
        wa = np.stack([w.dir[:, 0], w.dir[:, 1], w.k, w.amp], 1).astype(np.float32)
        wb = np.stack([w.omega, w.phase, w.q, np.zeros_like(w.q)], 1).astype(np.float32)
        self.d_wa = wp.array(wa, dtype=wp.vec4, device=dev)
        self.d_wb = wp.array(wb, dtype=wp.vec4, device=dev)
        self.d_out = wp.zeros(len(self.rest), dtype=wp.vec3, device=dev)

    def __call__(self, t: float) -> np.ndarray:
        if self.wp is not None:
            self.wp.launch(
                self.kernel,
                dim=len(self.rest),
                inputs=[self.d_rest, self.d_base, self.d_fade, self.d_spacing, self.d_wa, self.d_wb,
                        len(self.waves.k), float(t)],
                outputs=[self.d_out],
                device=self.dev,
            )
            return self.d_out.numpy()
        from sim.ocean_waves import displace

        d = displace(self.waves, self.rest[:, 0], self.rest[:, 1], t, self.fade, self.spacing)
        out = np.empty((len(self.rest), 3), np.float32)
        out[:, :2] = self.rest + d[:, :2]
        out[:, 2] = self.base_z + d[:, 2]
        return out


class Environment:
    def __init__(self, preset: Preset, mesh: UsdGeom.Mesh, points: np.ndarray, active: int,
                 displacer: _Displacer, time_input: UsdShade.Input):
        self.preset = preset
        self.mesh = mesh
        self.points = points
        self.active = active
        self.displacer = displacer
        self.time_input = time_input
        self._points_attr = mesh.GetPointsAttr()

    def update(self, t: float) -> None:
        self.points[: self.active] = self.displacer(t)
        self._points_attr.Set(Vt.Vec3fArray.FromNumpy(self.points))
        self.time_input.Set(float(t))


def find_legs(stage: Usd.Stage, sea_level_z: float, max_width: float = 8.0) -> list[tuple[float, float, float]]:
    """Vertical members that pierce the sea surface, as (x, y, radius), for foam rings."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    legs = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim) or str(prim.GetPath()).startswith(ROOT):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        lo, hi = box.GetMin(), box.GetMax()
        wx, wy = hi[0] - lo[0], hi[1] - lo[1]
        if lo[2] < sea_level_z - 0.5 and hi[2] > sea_level_z + 0.5 and max(wx, wy) < max_width:
            legs.append(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max(wx, wy) / 2))
    return legs


def _disable_other_lights(stage: Usd.Stage) -> None:
    for prim in stage.Traverse():
        if (prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight)) and not str(prim.GetPath()).startswith(ROOT):
            prim.SetActive(False)


def _add_sky(stage: Usd.Stage, preset: Preset) -> None:
    dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/Sky")
    dome.CreateTextureFileAttr(str(HDRI_DIR / f"{preset.hdri}.hdr"))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(preset.dome_intensity)
    UsdGeom.Xformable(dome).AddRotateZOp().Set(preset.dome_rotate_z)


def _add_material(stage: Usd.Stage, preset: Preset, mdl_path: Path) -> tuple[UsdShade.Material, UsdShade.Input]:
    mat = UsdShade.Material.Define(stage, f"{ROOT}/Looks/OceanWater")
    shader = UsdShade.Shader.Define(stage, f"{ROOT}/Looks/OceanWater/Shader")
    shader.SetSourceAsset(Sdf.AssetPath(str(mdl_path)), "mdl")
    shader.SetSourceAssetSubIdentifier(MODULE, "mdl")
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    color = Sdf.ValueTypeNames.Color3f
    for name, value in (("deep_color", preset.deep_color), ("crest_color", preset.crest_color),
                        ("haze_radiance", preset.haze_radiance)):
        shader.CreateInput(name, color).Set(Gf.Vec3f(*value))
    for name in ("foam_threshold", "foam_amount", "roughness", "haze_distance"):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(getattr(preset, name)))
    time_input = shader.CreateInput("time", Sdf.ValueTypeNames.Float)
    time_input.Set(0.0)
    return mat, time_input


def _apply_render_settings(preset: Preset) -> None:
    try:
        import carb
    except ImportError:
        return
    s = carb.settings.get_settings()
    for key, value in render_settings(preset).items():
        s.set(key, value)


def add_environment(
    stage: Usd.Stage,
    preset: str = "clear",
    sea_level_z: float = -25.0,
    legs: list[tuple[float, float, float]] | None = None,
    seed: int = 7,
) -> Environment:
    p = get_preset(preset)
    UsdGeom.Scope.Define(stage, ROOT)
    _disable_other_lights(stage)
    _add_sky(stage, p)
    _apply_render_settings(p)

    waves = make_waves(p.wind_speed, p.wind_dir_deg, seed=seed, swell=p.swell, choppiness=p.choppiness)
    if legs is None:
        legs = find_legs(stage, sea_level_z)
    mdl_dir = Path(tempfile.mkdtemp(prefix="rig_ocean_"))
    mdl_path = mdl_dir / f"{MODULE}.mdl"
    mdl_path.write_text(build_mdl(waves, legs))
    mat, time_input = _add_material(stage, p, mdl_path)

    grid = polar_grid()
    r = np.hypot(grid.xy[:, 0], grid.xy[:, 1])
    base_z = (sea_level_z - r * r / (2 * EARTH_RADIUS)).astype(np.float32)
    points = np.concatenate([grid.xy, base_z[:, None]], 1).astype(np.float32)
    active = int(np.searchsorted(r, WAVE_END_RADIUS))
    fade = _smooth_fade(r[:active])

    mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/Ocean")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(grid.counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(grid.indices))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(grid.xy.astype(np.float32)))
    mesh.CreateExtentAttr([Gf.Vec3f(-40000, -40000, sea_level_z - 200), Gf.Vec3f(40000, 40000, sea_level_z + 20)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    displacer = _Displacer(waves, grid.xy[:active].astype(np.float32), base_z[:active], fade,
                           grid.spacing[:active].astype(np.float32))
    env = Environment(p, mesh, points, active, displacer, time_input)
    env.update(0.0)
    return env

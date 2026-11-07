"""Pure-Python decisions for import_model.py: units, up axis, deck level, walkable meshes, texture paths."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path, PureWindowsPath

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
SCALE_CANDIDATES = (1.0, 0.01, 0.001, 0.0254, 0.3048, 0.1, 10.0, 100.0)
PLAUSIBLE_EXTENT_M = (8.0, 400.0)
SANE_EXTENT_M = (1.0, 2000.0)
TYPICAL_EXTENT_M = 60.0

CONVERTERS = {
    "assimp": {".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl", ".lxo", ".md5", ".bvh"},
    "usd": {".usd", ".usda", ".usdc"},
    "usdz": {".usdz"},
    "hoops": {".step", ".stp", ".iges", ".igs"},
    "blender": {".blend", ".dae", ".3ds", ".x3d", ".wrl", ".abc"},
}

WALKABLE = re.compile(r"deck|floor|stair|step|walk|grat|helipad|landing|ramp|ground", re.IGNORECASE)


def converter_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    for name, suffixes in CONVERTERS.items():
        if suffix in suffixes:
            return name
    raise ValueError(f"unsupported model format: {suffix}")


def unitless(path: str) -> bool:
    """Formats whose files carry no unit, so converter metadata is a guess."""
    return Path(path).suffix.lower() in {".stl", ".obj", ".ply"}


def guess_scale(extent: float, meters_per_unit: float, units: str = "auto", trust_metadata: bool = True) -> float:
    """Metres per source unit, given the model's largest bounding-box extent in source units."""
    if units != "auto":
        return UNITS[units]
    if trust_metadata and SANE_EXTENT_M[0] <= extent * meters_per_unit <= SANE_EXTENT_M[1]:
        return meters_per_unit
    low, high = PLAUSIBLE_EXTENT_M
    plausible = [s for s in SCALE_CANDIDATES if low <= extent * s <= high]
    pool = plausible or list(SCALE_CANDIDATES)
    return min(pool, key=lambda s: abs(math.log(max(extent * s, 1e-9) / TYPICAL_EXTENT_M)))


def guess_up_axis(meta_up: str, flat_area: dict[str, float], forced: str = "auto", ratio: float = 2.0) -> str:
    """Up axis from metadata, overridden when far more surface area faces along another axis."""
    if forced != "auto":
        return forced
    dominant = max(flat_area, key=flat_area.get)
    if dominant != meta_up and flat_area[dominant] >= ratio * flat_area.get(meta_up, 0.0):
        return dominant
    return meta_up


def deck_level(samples: list[tuple[float, float]], bin_size: float = 0.5, skip_bottom: float = 0.05) -> float | None:
    """Height of the band with the most upward-facing area, ignoring sea or seabed planes at the very bottom."""
    if not samples:
        return None
    zs = [z for z, _ in samples]
    bottom = min(zs) + skip_bottom * (max(zs) - min(zs))
    kept = [(z, a) for z, a in samples if z > bottom] or samples
    bins: dict[int, list[float]] = {}
    for z, area in kept:
        acc = bins.setdefault(math.floor(z / bin_size), [0.0, 0.0])
        acc[0] += area
        acc[1] += area * z
    area, weighted = max(bins.values(), key=lambda acc: acc[0])
    return weighted / area if area else None


def is_walkable(name: str, up_fraction: float, min_up_area: float = 4.0, up_area: float = 0.0, threshold: float = 0.5) -> bool:
    if WALKABLE.search(name):
        return True
    return up_fraction >= threshold and up_area >= min_up_area


def preview_cameras(lo: list[float], hi: list[float]) -> dict[str, dict]:
    """A wide three-quarter shot of the whole model and an eye-level shot from just off a deck corner (deck at z = 0)."""
    centre = [(a + b) / 2.0 for a, b in zip(lo, hi)]
    size = [b - a for a, b in zip(lo, hi)]
    distance = 2.2 * math.dist(lo, hi)
    az, el = math.radians(-50.0), math.radians(16.0)
    wide_eye = [
        centre[0] + distance * math.cos(el) * math.cos(az),
        centre[1] + distance * math.cos(el) * math.sin(az),
        centre[2] + distance * math.sin(el),
    ]
    deck_eye = [centre[0] + 0.62 * size[0], centre[1] - 0.62 * size[1], 1.7]
    deck_target = [centre[0], centre[1], 1.7 + 0.12 * max(size[0], size[1])]
    return {
        "wide": {"eye": wide_eye, "target": centre, "focal": 24.0},
        "deck": {"eye": deck_eye, "target": deck_target, "focal": 16.0},
    }


class TextureMap:
    """Copies textures next to the output USD and hands back relative paths."""

    def __init__(self, out_dir: Path, search_dirs: list[Path], subdir: str = "textures"):
        self.dest = Path(out_dir) / subdir
        self.subdir = subdir
        self.search_dirs = [Path(d) for d in search_dirs]
        self.copied: dict[Path, str] = {}
        self.missing: list[str] = []
        self._index: dict[str, Path] | None = None

    def map(self, asset: str, layer_dir: Path) -> str | None:
        source = self._resolve(asset, Path(layer_dir))
        if source is None:
            if asset.lower().endswith(".mdl") and "/" not in asset and "\\" not in asset:
                return asset
            if asset not in self.missing:
                self.missing.append(asset)
            return None
        if source not in self.copied:
            self.copied[source] = self._copy(source)
        return self.copied[source]

    def _resolve(self, asset: str, layer_dir: Path) -> Path | None:
        normal = asset.replace("\\", "/")
        for candidate in (Path(normal), layer_dir / normal):
            if candidate.is_file():
                return candidate.resolve()
        return self._by_name().get(PureWindowsPath(asset).name.lower())

    def _by_name(self) -> dict[str, Path]:
        if self._index is None:
            self._index = {}
            for root in self.search_dirs:
                if root.is_dir():
                    for path in sorted(root.rglob("*")):
                        if path.is_file():
                            self._index.setdefault(path.name.lower(), path.resolve())
        return self._index

    def _copy(self, source: Path) -> str:
        self.dest.mkdir(parents=True, exist_ok=True)
        taken = set(self.copied.values())
        name, n = source.name, 0
        while f"./{self.subdir}/{name}" in taken:
            n += 1
            name = f"{source.stem}_{n}{source.suffix}"
        target = self.dest / name
        if target.resolve() != source:
            shutil.copy2(source, target)
        return f"./{self.subdir}/{name}"

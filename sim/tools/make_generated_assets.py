"""Create the project's own sky, sea and gauge textures in sim/assets/generated/.

    C:\\isaacsim\\python.bat sim\\tools\\make_generated_assets.py

Run this only when changing these files. They're committed to the repo. The sky
starts from Poly Haven's CC0 aristea_wreck HDRI with the coastline removed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIM_DIR = Path(__file__).resolve().parents[1]
SOURCE_SKY = SIM_DIR / "assets" / "third_party" / "hdris" / "aristea_wreck.hdr"
GENERATED = SIM_DIR / "assets" / "generated"
LAND_LIMIT_DEG = 4.0
BLEND_DEG = 4.0
HORIZON_GLOW = 1.4  # brightens the haze so the sky meets the bright far sea
GLOW_TOP_DEG = 14.0


def read_hdr(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header_end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", header_end)
    parts = data[header_end:line_end].split()
    height, width = int(parts[1]), int(parts[3])
    position = line_end + 1
    rgbe = np.zeros((height, width, 4), np.uint8)
    for row in range(height):
        if data[position] != 2 or data[position + 1] != 2:
            raise ValueError("Only run-length encoded HDR files are supported.")
        position += 4
        for channel in range(4):
            column = 0
            while column < width:
                count = data[position]
                position += 1
                if count > 128:
                    count -= 128
                    rgbe[row, column : column + count, channel] = data[position]
                    position += 1
                else:
                    rgbe[row, column : column + count, channel] = np.frombuffer(data, np.uint8, count, position)
                    position += count
                column += count
    exponent = rgbe[:, :, 3:4].astype(np.float64)
    scale = np.where(exponent > 0, np.power(2.0, exponent - 136.0), 0.0)
    return rgbe[:, :, :3] * scale


def write_hdr(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    peak = np.maximum(image.max(axis=2), 1e-32)
    mantissa, exponent = np.frexp(peak)
    scale = mantissa * 256.0 / peak
    rgbe = np.zeros((height, width, 4), np.uint8)
    rgbe[:, :, :3] = np.clip(image * scale[:, :, None], 0, 255).astype(np.uint8)
    rgbe[:, :, 3] = np.clip(exponent + 128, 0, 255).astype(np.uint8)
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode()
    path.write_bytes(header + rgbe.tobytes())


def make_sky() -> None:
    image = read_hdr(SOURCE_SKY)
    height, width = image.shape[:2]
    elevation = 90.0 - (np.arange(height) + 0.5) / height * 180.0
    limit = int(np.searchsorted(-elevation, -LAND_LIMIT_DEG))
    band = image[limit - 6 : limit].mean(axis=0)

    blur = width // 24
    wrapped = np.concatenate((band[-blur:], band, band[:blur]))
    kernel = np.ones(2 * blur + 1) / (2 * blur + 1)
    haze = np.stack(
        [np.convolve(wrapped[:, channel], kernel, mode="same")[blur:-blur] for channel in range(3)], axis=1
    )
    weight = np.clip((LAND_LIMIT_DEG + BLEND_DEG - elevation) / BLEND_DEG, 0, 1)[:, None, None]
    result = image * (1 - weight) + haze[None, :, :] * weight
    fade = np.clip((GLOW_TOP_DEG - elevation) / GLOW_TOP_DEG, 0, 1)[:, None, None] ** 2
    result *= 1 + (HORIZON_GLOW - 1) * fade
    write_hdr(GENERATED / "ocean_sky.hdr", result)


def make_sea_normal(size: int = 1024) -> None:
    rng = np.random.default_rng(11)
    axis = np.arange(size) / size * math.tau
    x, y = np.meshgrid(axis, axis)
    slope_x, slope_y = np.zeros((size, size)), np.zeros((size, size))
    for _ in range(90):
        radius = rng.uniform(4, 70)
        heading = rng.normal(0.5, 0.7)
        kx, ky = round(radius * math.cos(heading)), round(radius * math.sin(heading))
        length = math.hypot(kx, ky)
        if length == 0:
            continue
        amplitude = 0.05 / length**0.75
        phase = rng.uniform(0, math.tau)
        wave = np.cos(kx * x + ky * y + phase) * amplitude
        slope_x += wave * kx / 12
        slope_y += wave * ky / 12
    normal = np.dstack((-slope_x, -slope_y, np.ones_like(slope_x)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    Image.fromarray(((normal * 0.5 + 0.5) * 255).astype(np.uint8)).save(GENERATED / "sea_normal.png")


def make_gauge_dial(size: int = 512) -> None:
    image = Image.new("RGB", (size, size), (238, 238, 232))
    draw = ImageDraw.Draw(image)
    center = size / 2
    outer = size * 0.42
    draw.arc((center - outer, center - outer, center + outer, center + outer), 45 - 54, 45, fill=(190, 30, 25), width=18)
    for tick in range(0, 51):
        angle = math.radians(135 + 270 * tick / 50)
        major = tick % 10 == 0
        inner = outer - (size * 0.09 if major else size * 0.05)
        draw.line(
            (
                center + math.cos(angle) * inner,
                center + math.sin(angle) * inner,
                center + math.cos(angle) * outer,
                center + math.sin(angle) * outer,
            ),
            fill=(20, 20, 20),
            width=7 if major else 3,
        )
    image.save(GENERATED / "gauge_dial.png")


GENERATED.mkdir(parents=True, exist_ok=True)
make_sky()
make_sea_normal()
make_gauge_dial()
print(f"GENERATED_OK {GENERATED}")

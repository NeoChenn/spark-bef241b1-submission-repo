"""Grade Poly Haven's CC0 pure-sky HDRIs into the cinematic sky presets.

    C:\\isaacsim\\python.bat sim\\tools\\make_preset_skies.py

Writes sky_night.hdr, sky_storm.hdr and foam.png to sim/assets/generated/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIM_DIR = Path(__file__).resolve().parents[1]
HDRIS = SIM_DIR / "assets" / "third_party" / "hdris"
GENERATED = SIM_DIR / "assets" / "generated"
LUMA = np.array((0.2126, 0.7152, 0.0722))


def read_hdr(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header_end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", header_end)
    parts = data[header_end:line_end].split()
    height, width = int(parts[1]), int(parts[3])
    position = line_end + 1
    rgbe = np.zeros((height, width, 4), np.uint8)
    for row in range(height):
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
    return rgbe[:, :, :3] * np.where(exponent > 0, np.power(2.0, exponent - 136.0), 0.0)


def write_hdr(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    peak = np.maximum(image.max(axis=2), 1e-32)
    mantissa, exponent = np.frexp(peak)
    rgbe = np.zeros((height, width, 4), np.uint8)
    rgbe[:, :, :3] = np.clip(image * (mantissa * 256.0 / peak)[:, :, None], 0, 255).astype(np.uint8)
    rgbe[:, :, 3] = np.clip(exponent + 128, 0, 255).astype(np.uint8)
    path.write_bytes(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode() + rgbe.tobytes())


def grade(image: np.ndarray, tint, gain: float, contrast: float, saturation: float) -> np.ndarray:
    """Push clouds toward a tint, deepening the darks. Sky above and below the horizon stays smooth."""
    luminance = np.maximum(image @ LUMA, 1e-6)
    middle = np.exp(np.log(luminance).mean())
    shaped = np.power(luminance / middle, contrast) * middle
    colored = image / luminance[:, :, None] * saturation + (1 - saturation)
    return colored * np.array(tint) * shaped[:, :, None] * gain


def make_foam(size: int = 512) -> None:
    rng = np.random.default_rng(5)
    noise = np.zeros((size, size))
    for cells, weight in ((6, 1.0), (14, 0.7), (32, 0.45), (70, 0.25)):
        layer = rng.random((cells, cells))
        layer = np.asarray(Image.fromarray((layer * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
        noise += layer * weight
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    alpha = np.clip((noise - 0.56) * 9.0, 0, 1)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[:, :, :3] = 235
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(GENERATED / "foam.png")


source = read_hdr(HDRIS / "overcast_soil_puresky.hdr")
write_hdr(GENERATED / "sky_night.hdr", grade(source, (0.16, 0.62, 0.66), 0.42, 1.75, 0.0))
write_hdr(GENERATED / "sky_storm.hdr", grade(source, (0.46, 0.53, 0.58), 0.34, 1.9, 0.0))
make_foam()
print("PRESET_SKIES_OK")

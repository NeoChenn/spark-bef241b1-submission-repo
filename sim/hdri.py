from __future__ import annotations

from pathlib import Path

import numpy as np


def read_hdr(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    head_end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", head_end)
    dims = data[head_end:line_end].split()
    h, w = int(dims[1]), int(dims[3])
    buf = np.frombuffer(data, np.uint8, offset=line_end + 1)
    out = np.empty((h, w, 4), np.uint8)
    pos = 0
    for y in range(h):
        pos += 4
        for c in range(4):
            x = 0
            row = out[y, :, c]
            while x < w:
                n = int(buf[pos])
                pos += 1
                if n > 128:
                    n -= 128
                    row[x : x + n] = buf[pos]
                    pos += 1
                else:
                    row[x : x + n] = buf[pos : pos + n]
                    pos += n
                x += n
    e = out[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return (out[..., :3] * scale[..., None]).astype(np.float32)


def write_hdr(path: str | Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    m = img.max(axis=2)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, -128).astype(np.int32)
    mant = np.where(m[..., None] > 1e-32, img / np.ldexp(1.0, e)[..., None] * 256, 0)
    rgbe = np.concatenate([np.clip(mant, 0, 255).astype(np.uint8), np.clip(e + 128, 0, 255).astype(np.uint8)[..., None]], 2)
    body = bytearray()
    for y in range(h):
        body += bytes([2, 2, w >> 8, w & 255])
        for c in range(4):
            row = rgbe[y, :, c]
            for x in range(0, w, 128):
                chunk = row[x : x + 128]
                body += bytes([len(chunk)]) + chunk.tobytes()
    Path(path).write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode() + bytes(body))


def horizon_color(img: np.ndarray, max_elevation_deg: float = 4.0) -> np.ndarray:
    h = img.shape[0]
    rows = max(1, int(round(h * max_elevation_deg / 180.0)))
    return img[h // 2 - rows : h // 2].reshape(-1, 3).mean(0)

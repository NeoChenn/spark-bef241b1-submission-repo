import numpy as np


def rust_score(rgb) -> np.ndarray:
    p = np.asarray(rgb, np.float32)[..., :3]
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    rr = np.maximum(r, 1.0)
    hue = (r - g) / rr
    sat = (r - b) / rr
    return np.clip((hue - 0.22) / 0.15, 0.0, 1.0) * (hue < 0.8) * (sat > 0.4) * (r > 40) * (g > b)


def rust_heat(rgb, cell: int = 16) -> np.ndarray:
    s = rust_score(rgb)
    h, w = s.shape[0] // cell, s.shape[1] // cell
    frac = s[: h * cell, : w * cell].reshape(h, cell, w, cell).mean((1, 3))
    return 1.0 / (1.0 + np.exp(-(frac - 0.3) * 12.0))


def boxes(heat, thr: float, cell: int) -> list[tuple[int, int, int, int, float]]:
    hot = np.asarray(heat) >= thr
    seen = np.zeros_like(hot)
    out = []
    for i0, j0 in zip(*np.nonzero(hot)):
        if seen[i0, j0]:
            continue
        stack, comp = [(i0, j0)], []
        seen[i0, j0] = True
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < hot.shape[0] and 0 <= b < hot.shape[1] and hot[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        ii, jj = np.array(comp).T
        out.append((int(jj.min() * cell), int(ii.min() * cell), int((jj.max() + 1) * cell), int((ii.max() + 1) * cell),
                    float(np.asarray(heat)[ii, jj].mean())))
    return sorted(out, key=lambda b: -b[4])

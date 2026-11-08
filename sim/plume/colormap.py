import numpy as np

STOPS = np.array([0.0, 10.0, 50.0, 100.0])
COLORS = np.array([[38, 84, 214], [0, 206, 230], [250, 214, 50], [235, 45, 40]], np.float64)
VISIBLE_PCT = 1.0


def lel_rgb(pct) -> np.ndarray:
    p = np.clip(np.asarray(pct, np.float64), 0.0, 100.0)
    return np.stack([np.interp(p, STOPS, COLORS[:, c]) for c in range(3)], -1) / 255.0


def lel_alpha(pct, floor: float = VISIBLE_PCT, gain: float = 0.55) -> np.ndarray:
    p = np.asarray(pct, np.float64)
    a = np.log10(np.maximum(p, 1e-9) / floor) / np.log10(100.0 / floor)
    return np.clip(a, 0.0, 1.0) ** 0.8 * gain * (p >= floor)


def colorbar(width: int, height: int) -> np.ndarray:
    p = np.linspace(100.0, 0.0, height)
    return np.repeat((lel_rgb(p) * 255).astype(np.uint8)[:, None, :], width, axis=1)

import base64
import io

import numpy as np
from PIL import Image

from rig.plume import Field
from rig.sim.cli import parse_args
from sim.plume.isaac import BANDS, cloud
from sim.plume.pip import SIZE, compose, jpeg_b64, level


def blob() -> Field:
    g = np.indices((20, 16, 12)).astype(float)
    r2 = ((g - np.array([10, 8, 6])[:, None, None, None]) ** 2).sum(0)
    return Field(0.05 * np.exp(-r2 / 8.0), (5.0, -3.0, 42.0), 0.2)


def test_cloud_bands_partition_cells_and_stay_in_cells():
    f = blob()
    total = 0
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        pts, pct = cloud(f, lo, hi)
        assert np.all((pct >= lo) & (pct < hi))
        total += len(pts)
        cell = np.floor((pts - f.origin) / f.dx).astype(int)
        assert np.all(f.fraction[tuple(cell.T)] / f.lel * 100 >= lo - 1e-6)
    assert total == int(np.sum(f.fraction / f.lel * 100 >= BANDS[0]))
    a, _ = cloud(f)
    b, _ = cloud(f)
    assert np.array_equal(a, b)
    few, pct = cloud(f, max_points=10)
    assert len(few) == 10 and pct.min() >= np.sort(f.fraction.ravel())[-10] / f.lel * 100 - 1e-6


def test_pip_compose_size_and_levels():
    rgb = np.full((540, 960, 3), 90, np.uint8)
    stats = {"t": 3.2, "vol_10lel_m3": 2.1, "vol_50lel_m3": 0.01, "max_lel_pct": 64.0}
    im = compose(rgb, stats, {"kg_s": 0.0159, "sm3_h": 84.5}, {"speed_m_s": 5.0, "heading_deg": 90.0}, "GAS PLUME F5")
    assert im.size == SIZE
    arr = np.asarray(im)
    assert arr[:22].mean() < 90
    top, bottom = arr[36, 443].astype(int), arr[150, 443].astype(int)
    assert top[0] > top[2] + 80 and bottom[2] > bottom[0] + 80
    back = Image.open(io.BytesIO(base64.b64decode(jpeg_b64(im))))
    assert back.size == SIZE
    assert level(64)[0] == "STOP" and level(20)[0] == "WARN" and level(2)[0] == "OK"


def test_cli_leak_options():
    a = parse_args(["--leak", "F5", "--leak-mm", "4", "--leak-bar", "30", "--pip-size", "640x360"])
    assert a.leak == "F5" and a.leak_mm == 4.0 and a.leak_bar == 30.0 and a.pip_size == (640, 360)
    assert parse_args([]).leak == "" and parse_args([]).pip_size == (480, 270)

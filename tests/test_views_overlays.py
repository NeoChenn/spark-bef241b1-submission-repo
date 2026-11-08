import base64
import io

import numpy as np
from PIL import Image

from rig.camera import look_at
from rig.plume import Field
from rig.views.draw import depth_rgb, encode, sparkline
from rig.views.ogi import gas_column, ogi_image
from rig.views.rust import boxes, rust_heat


def test_rust_heat_finds_rust_patch_not_grey_or_yellow():
    img = np.full((96, 128, 3), 90, np.uint8)
    img[:, :40] = (200, 190, 60)
    img[32:64, 64:112] = (150, 70, 35)
    heat = rust_heat(img, cell=16)
    assert heat.shape == (6, 8)
    assert heat[2:4, 4:7].min() > 0.7
    assert heat[:, :2].max() < 0.3
    assert heat[:, 7].max() < 0.3
    bs = boxes(heat, 0.5, 16)
    assert len(bs) == 1
    x0, y0, x1, y1, s = bs[0]
    assert (x0, y0, x1, y1) == (64, 32, 112, 64) and s > 0.7


def test_boxes_splits_components():
    h = np.zeros((4, 6))
    h[0, 0] = h[3, 5] = 0.9
    assert len(boxes(h, 0.5, 10)) == 2


def plume():
    c = np.zeros((20, 20, 10), np.float32)
    c[8:12, 8:12, 2:6] = 0.04
    return Field(c, (0.0, -5.0, 0.0), 0.5)


def test_gas_column_hits_plume_only_on_crossing_rays():
    cam = look_at((5.0, -15.0, 2.0), (5.0, 0.0, 2.0), focal=12.0, w=64, h=48)
    col = gas_column(plume(), cam, stride=4)
    assert col.shape == (12, 16)
    assert col[6, 8] > 10
    assert col[0, 0] == 0


def test_gas_column_stops_at_depth():
    cam = look_at((5.0, -15.0, 2.0), (5.0, 0.0, 2.0), focal=12.0, w=64, h=48)
    depth = np.full((48, 64), 3.0, np.float32)
    assert gas_column(plume(), cam, depth=depth, stride=4).max() == 0


def test_ogi_image_colours_gas():
    rgb = np.full((48, 64, 3), 80, np.uint8)
    col = np.zeros((12, 16))
    col[4:8, 4:8] = 200.0
    out = np.asarray(ogi_image(rgb, col))
    assert out.shape == (48, 64, 3)
    gas, bg = out[24, 24].astype(int), out[2, 60].astype(int)
    assert abs(gas[0] - gas[2]) > 60
    assert abs(bg[0] - bg[2]) < 10


def test_depth_rgb_and_sparkline_and_encode():
    d = np.linspace(0.5, 30, 64 * 48).reshape(48, 64).astype(np.float32)
    d[0, 0] = np.inf
    im = depth_rgb(d)
    assert im.size == (64, 48)
    sparkline(im, [10, 20, 45, 30], (2, 30, 60, 46), 0, 60, ref=40)
    v = encode(im, "depth", 7)
    assert set(v) == {"jpg", "w", "h", "seq", "caption"}
    assert v["w"] == 64 and v["seq"] == 7
    assert Image.open(io.BytesIO(base64.b64decode(v["jpg"]))).size == (64, 48)
    assert len(v["jpg"]) < 25_000

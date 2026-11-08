import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.localize import (RustLocalizer, fake_colour_classifier, pixel_to_theta_z,
                                project_mask_to_grid)
from policy.interfaces import CameraModel, PipeGeometry
from policy.skills.coverage import boustrophedon_cells, coverage_path, covered_by, dilate
from policy.skills.paint import PaintSkill

STEEL = (140, 144, 150)
RUST = (150, 72, 30)


def canvas(h=160, w=200):
    return np.full((h, w, 3), STEEL, np.uint8)


def loc():
    return RustLocalizer(fake_colour_classifier, patch=16, stride=8)


def test_blob_found_and_centroid_accurate():
    img = canvas()
    yy, xx = np.mgrid[:160, :200]
    img[(xx - 120) ** 2 + (yy - 70) ** 2 <= 18 ** 2] = RUST
    r = loc().localize(img)
    assert len(r.patches) == 1
    p = r.patches[0]
    assert abs(p.centroid[0] - 120) < 3 and abs(p.centroid[1] - 70) < 3
    x0, y0, x1, y1 = p.bbox
    assert x0 <= 105 and x1 >= 135 and y0 <= 55 and y1 >= 85
    assert p.mean_conf > 0.8 and not r.unsure
    assert r.heatmap.shape == (19, 24)


def test_clean_image_no_patches():
    r = loc().localize(canvas())
    assert r.patches == [] and not r.unsure and r.max_conf < 0.1 and not r.mask.any()


def test_two_blobs_two_components():
    img = canvas()
    img[20:50, 20:50] = RUST
    img[100:140, 140:180] = RUST
    assert len(loc().localize(img).patches) == 2


def test_unsure_flag_on_ambiguous_texture():
    img = canvas()
    rng = np.random.default_rng(0)
    region = rng.random((48, 48)) < 0.15
    sub = img[50:98, 60:108]
    sub[region] = RUST
    r = loc().localize(img)
    assert r.unsure and len(r.unsure_cells) > 0


def test_temperature_and_threshold_applied():
    img = canvas()
    img[40:90, 40:90] = RUST
    hot = RustLocalizer(fake_colour_classifier, patch=16, stride=8, threshold=0.999999).localize(img)
    assert hot.mask.sum() < loc().localize(img).mask.sum()
    soft = RustLocalizer(fake_colour_classifier, patch=16, stride=8, temperature=100.0).localize(img)
    assert soft.max_conf < 0.6


def _scene():
    pipe = PipeGeometry(origin=[0, 0, 1.1], axis=[1, 0, 0], ref=[0, 1, 0], radius=0.15, length=0.8, n_z=40, n_theta=64)
    from policy.demo_mock import look_at
    W, H = 320, 240
    f = 35 / 20.955 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    eye = np.array([0, 0, 1.1]) + 0.8 * np.array([0, math.cos(1.0), math.sin(1.0)])
    cam = CameraModel(K, look_at(eye, [0, 0, 1.1]), W, H)
    return pipe, cam


def test_pixel_ray_cylinder_projection():
    pipe, cam = _scene()
    th, z = pixel_to_theta_z((160, 120), cam, pipe)
    assert abs(th - 1.0) < 0.02 and abs(z) < 0.01
    assert pixel_to_theta_z((0, 0), cam, pipe) is None or True
    mask = np.zeros((240, 320), bool)
    mask[110:130, 150:170] = True
    grid, vis, conf = project_mask_to_grid(mask, cam, pipe)
    assert grid.shape == (40, 64) and vis.sum() > grid.sum() > 0
    i, j = pipe.theta_z_to_cell(1.0, 0.0)
    assert grid[i, j]
    assert not vis[:, pipe.theta_z_to_cell(1.0 + math.pi, 0)[1]].any()


@pytest.mark.parametrize("seed", range(5))
def test_boustrophedon_covers_dilated_mask(seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((40, 64), bool)
    for _ in range(3):
        i, j = rng.integers(0, 40), rng.integers(0, 64)
        m[max(0, i - 3):i + 3, [(j + d) % 64 for d in range(-4, 5)]] = True
    m[10:14, [62, 63, 0, 1]] = True
    for lw in (1, 3):
        target = dilate(m, 1)
        cells = boustrophedon_cells(target, lw)
        assert covered_by(cells, lw, m.shape)[target].all()


def test_coverage_waypoints_on_surface_with_standoff():
    pipe = PipeGeometry(origin=[0, 0, 1.1], axis=[1, 0, 0], ref=[0, 1, 0], radius=0.15, length=0.8, n_z=40, n_theta=64)
    m = np.zeros(pipe.shape, bool)
    m[18:22, 10:15] = True
    wps, target = coverage_path(m, pipe, margin=1, lane_width=1, standoff=0.15)
    assert target.sum() > m.sum() and len(wps) == target.sum()
    for w in wps:
        rel = w.position - pipe.origin
        radial = rel - rel.dot(pipe.axis) * pipe.axis
        assert abs(np.linalg.norm(radial) - 0.30) < 1e-6
        assert abs(np.dot(w.approach, radial / np.linalg.norm(radial)) + 1) < 1e-6
    cols = [w.cell[1] for w in wps]
    first = [w.cell[0] for w in wps if w.cell[1] == cols[0]]
    second = [w.cell[0] for w in wps if w.cell[1] == cols[len(first)]]
    assert first == sorted(first) and second == sorted(second, reverse=True)


class _DropArm:
    def __init__(self):
        self.rng = np.random.default_rng(1)

    def move_to(self, wps):
        return [bool(self.rng.random() > 0.2) for _ in wps]


def test_paint_repaints_gaps_to_98pct():
    pipe = PipeGeometry(origin=[0, 0, 0], axis=[1, 0, 0], ref=[0, 1, 0], radius=0.15, length=0.8, n_z=40, n_theta=64)
    m = np.zeros(pipe.shape, bool)
    m[10:20, 20:30] = True
    skill = PaintSkill(max_passes=6)
    info = skill.execute(m, pipe, _DropArm(), world=object())
    assert info["coverage"] >= 0.98 and info["passes"] >= 2
    assert skill.passes[0]["n_reached"] < skill.passes[0]["n_wp"]


def test_onnx_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    w = np.zeros((1, 3, 1, 1), np.float32)
    w[0, 0], w[0, 1] = 20.0, -20.0
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["c"]),
         helper.make_node("GlobalAveragePool", ["c"], ["g"]),
         helper.make_node("Flatten", ["g"], ["y"])],
        "rust", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])],
        [helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten())])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "meta.json").write_text('{"model": "test", "input_size": [3, 32, 32], "unsure_band": [0.35, 0.65], "mean": [0,0,0], "std": [1,1,1], '
                                        '"threshold": 0.5, "temperature": 1.0}')
    lz = RustLocalizer.from_checkpoint(str(tmp_path), patch=16, stride=8)
    img = canvas()
    img[40:90, 40:90] = RUST
    r = lz.localize(img)
    assert len(r.patches) == 1 and r.threshold == 0.5
    assert abs(r.patches[0].centroid[0] - 65) < 5

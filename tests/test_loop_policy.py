import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from integration.onnx_mlp import OnnxMLP
from integration.walk_obs import WALK_OBS, leg_targets, scan_value, speed_profile, walk_obs


def _model(path, w0, b0, w1, b1, mean, std):
    inits = [numpy_helper.from_array(a.astype(np.float32), n) for a, n in
             ((w0, "w0"), (b0, "b0"), (w1, "w1"), (b1, "b1"), (mean, "m"), (std, "s"))]
    nodes = [helper.make_node("Sub", ["obs", "m"], ["x0"]), helper.make_node("Div", ["x0", "s"], ["x1"]),
             helper.make_node("Gemm", ["x1", "w0", "b0"], ["h"], transB=1), helper.make_node("Elu", ["h"], ["e"], alpha=1.0),
             helper.make_node("Gemm", ["e", "w1", "b1"], ["actions"], transB=1)]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 3])],
                          [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])], inits)
    onnx.save(helper.make_model(g), path)


def test_onnx_mlp_matches_numpy(tmp_path):
    rng = np.random.default_rng(0)
    w0, b0, w1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=(2, 4)), rng.normal(size=2)
    mean, std = rng.normal(size=(1, 3)), rng.uniform(0.5, 2, size=(1, 3))
    p = str(tmp_path / "m.onnx")
    _model(p, w0, b0, w1, b1, mean, std)
    x = rng.normal(size=(5, 3)).astype(np.float32)
    h = ((x - mean) / std) @ w0.T + b0
    h = np.where(h > 0, h, np.exp(h) - 1)
    want = h @ w1.T + b1
    net = OnnxMLP(p)
    assert net.in_dim == 3
    assert np.allclose(net(x), want, atol=1e-5)
    assert net(x[0]).shape == (1, 2)


def test_walk_obs_layout():
    o = walk_obs(cmd=(0.5, 0.0, 0.1), lin_vel=(0.5, 0, 0), ang_vel=(0, 0, 0.1), gravity=(0, 0, -1),
                 joint_pos_rel=np.ones(12), joint_vel=np.zeros(12), last_action=np.full(12, 2.0), base_height=0.32,
                 payload=0.0)
    assert o.shape == (WALK_OBS,) == (236,)
    assert np.allclose(o[0:3], [0.5, 0, 0]) and np.allclose(o[6:9], [0, 0, -1]) and np.allclose(o[9:12], [0.5, 0, 0.1])
    assert np.allclose(o[12:24], 1) and np.allclose(o[36:48], 2)
    assert np.allclose(o[48:235], scan_value(0.32)) and o[235] == 0.0


def test_scan_value_is_clipped_height_offset():
    assert scan_value(0.32) == pytest.approx(0.32 - 0.5)
    assert scan_value(5.0) == 1.0


def test_leg_targets_scale_and_order():
    d = np.arange(12, dtype=float)
    assert np.allclose(leg_targets(d, np.ones(12)), d + 0.25)


def test_speed_profile_ramps_and_stops_at_distance():
    v = [speed_profile(s, 2.0, vmax=0.6, ramp=0.4) for s in (0.0, 0.2, 1.0, 1.9, 2.0, 2.5)]
    assert v[0] > 0 and v[1] < v[2] == pytest.approx(0.6)
    assert v[3] < 0.6 and v[4] == 0.0 and v[5] == 0.0

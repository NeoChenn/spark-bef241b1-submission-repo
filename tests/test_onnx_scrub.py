"""OnnxScrubPolicy on CPU. Uses a small random MLP exported to ONNX with the policy's 327 -> 4 signature, so the
wiring is tested without the trained checkpoint (runs/scrub_v2/policy.onnx is not in git)."""
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
for _p in (r"C:\isaacsim\exts\isaacsim.pip.onnx\pip_prebundle",):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
pytest.importorskip("onnxruntime")

from policy.demo_mock import scene_geometry  # noqa: E402
from policy.scenario_runner import run_scenario  # noqa: E402
from policy.skills.onnx_scrub import OBS_DIM, OnnxScrubPolicy  # noqa: E402


@pytest.fixture(scope="module")
def dummy_onnx(tmp_path_factory):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(OBS_DIM, 32), torch.nn.ELU(), torch.nn.Linear(32, 4))
    path = tmp_path_factory.mktemp("ppo") / "policy.onnx"
    kw = dict(input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False)
    try:
        import onnx  # noqa: F401
    except ImportError:
        # The TorchScript exporter only needs the `onnx` package to splice in onnxscript functions; this MLP has
        # none, so pass the serialised model through (the Isaac Sim Python ships onnxruntime but not onnx).
        from torch.onnx._internal.torchscript_exporter import onnx_proto_utils
        mp = pytest.MonkeyPatch()
        mp.setattr(onnx_proto_utils, "_add_onnxscript_fn", lambda model_bytes, custom_opsets: model_bytes)
        try:
            torch.onnx.export(net, (torch.zeros(1, OBS_DIM),), str(path), **kw)
        finally:
            mp.undo()
        return str(path)
    torch.onnx.export(net, (torch.zeros(1, OBS_DIM),), str(path), **kw)
    return str(path)


def test_missing_onnx_message(tmp_path):
    pipe, _ = scene_geometry()
    with pytest.raises(FileNotFoundError, match="policy.onnx|not found"):
        OnnxScrubPolicy(pipe, tmp_path / "policy.onnx")


def test_obs_shape_and_action_range(dummy_onnx):
    pipe, theta = scene_geometry()
    pol = OnnxScrubPolicy(pipe, dummy_onnx)
    rust = np.zeros((pipe.n_z, pipe.n_theta))
    i, j = pipe.theta_z_to_cell(theta, 0.0)
    rust[i - 2:i + 3, j - 2:j + 3] = 0.8
    obs = {"ee_theta_z": np.array([theta, 0.0]), "target_mask": rust > 0, "perceived_rust": rust, "force": 0.0}
    for _ in range(10):
        a = pol.step(obs)
        assert a.shape == (3,) and np.all(np.isfinite(a))
        assert -math.pi <= a[0] <= math.pi and 0.0 <= a[2] <= 1.0
        obs["ee_theta_z"] = np.array([theta + a[0], a[1]])
    o = pol.policy_obs()
    assert o.shape == (1, OBS_DIM) and o.dtype == np.float32 and np.all(np.abs(o) <= 10.0)
    assert float(pol.sim.rust.rust.sum()) > 0
    assert pol.stats["steps"] == 10 and pol.stats["policy_calls"] == 10 * pol.substeps
    pol.reset()
    assert pol.sim is None


def test_scenario_completes_with_ppo(dummy_onnx):
    sc = {"id": "t-ppo", "seed": 3, "new_patch": False, "force_spike_step": 0, "occlusion_frames": 0,
          "person_duration_s": 0.0, "glare": False}
    r = run_scenario(sc, scrub_policy="ppo", onnx_path=dummy_onnx)
    assert r["scrub_policy"] == "ppo"
    assert r["outcome"] in ("DONE", "ESCALATE_TO_HUMAN", "TIMEOUT") and r["label"]
    assert r["scrub_policy_stats"]["steps"] > 0
    g = run_scenario(sc)
    assert g["scrub_policy"] == "greedy" and g["scrub_policy_stats"] is None


def test_numpy_fk_matches_torch_fk():
    from bench.scrub_sim import ArmFK
    from policy.skills.onnx_scrub import NumpyArmFK
    ref, fast = ArmFK(), NumpyArmFK()
    g = torch.Generator().manual_seed(0)
    q = ref.lower + (ref.upper - ref.lower) * torch.rand(3, 6, generator=g)
    base = torch.eye(4).repeat(3, 1, 1)
    base[:, 2, 3] = 0.32
    lift = torch.tensor([0.38, 0.41, 0.45])
    for a, b in zip(ref.forward(base, q, lift), fast.forward(base, q, lift)):
        assert a.shape == b.shape and torch.allclose(a, b, atol=1e-5)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        run_scenario({"seed": 1}, scrub_policy="nope")

import math

import pytest
import torch

from policy.walk.rma import (EXT_DIM, EXT_SCALE, PROPRIO, Adapter, DeckMotion, History, expand_first_layer,
                             r2_score, sample_payload)


def test_proprio_indices_skip_height_scan():
    assert len(PROPRIO) == 49
    assert PROPRIO[:48] == list(range(48)) and PROPRIO[-1] == 235


def test_history_push_keeps_newest_last_and_resets():
    h = History(3, 4, 2)
    for k in range(6):
        h.push(torch.full((3, 2), float(k)))
    assert h.data.shape == (3, 4, 2)
    assert h.data[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    h.reset(torch.tensor([1]))
    assert h.data[1].abs().sum() == 0 and h.data[0, -1, 0] == 5.0


def test_history_reset_fills_with_first_obs_after_reset():
    h = History(2, 3, 1)
    h.push(torch.ones(2, 1))
    h.reset(torch.tensor([0]))
    h.push(torch.full((2, 1), 7.0))
    assert h.data[0, :, 0].tolist() == [7.0, 7.0, 7.0]
    assert h.data[1, :, 0].tolist() == [1.0, 1.0, 7.0]


def test_adapter_shapes():
    a = Adapter(len(PROPRIO), 50, EXT_DIM)
    out = a(torch.randn(5, 50, len(PROPRIO)))
    assert out.shape == (5, EXT_DIM)


def test_adapter_can_fit_simple_target():
    torch.manual_seed(0)
    a = Adapter(3, 20, 1)
    opt = torch.optim.Adam(a.parameters(), 3e-3)
    x = torch.randn(256, 20, 3)
    y = x[:, :, 0].mean(1, keepdim=True) * 3
    for _ in range(300):
        opt.zero_grad()
        loss = ((a(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert r2_score(a(x).detach(), y)[0] > 0.9


def test_deck_motion_seeded_and_bounded():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    d1, d2 = DeckMotion.sample(64, g1), DeckMotion.sample(64, g2)
    t = torch.linspace(0, 30, 64)
    a1, a2 = d1.accel(t), d2.accel(t)
    assert torch.equal(a1, a2)
    assert a1.shape == (64, 3)
    lim = 9.81 * math.sin(d1.max_tilt)
    assert a1[:, :2].abs().max() <= lim + 1e-5
    assert a1[:, 2].abs().max() <= d1.max_heave + 1e-5


def test_deck_motion_zero_amplitude_is_static():
    d = DeckMotion.sample(4, torch.Generator().manual_seed(0), max_tilt=0.0, max_heave=0.0)
    assert d.accel(torch.full((4,), 5.0)).abs().max() < 1e-6


def test_deck_motion_index_and_assign():
    d = DeckMotion.sample(6, torch.Generator().manual_seed(1))
    e = DeckMotion.sample(2, torch.Generator().manual_seed(9))
    d.assign(torch.tensor([0, 3]), e)
    t = torch.zeros(6)
    assert torch.allclose(d.accel(t)[[0, 3]], e.accel(torch.zeros(2)))


def test_sample_payload_ranges():
    m, c = sample_payload(1000, torch.Generator().manual_seed(0), mass=(-1.0, 5.0), com=(0.05, 0.05, 0.02))
    assert m.shape == (1000,) and c.shape == (1000, 3)
    assert m.min() >= -1.0 and m.max() <= 5.0
    assert c[:, 0].abs().max() <= 0.05 and c[:, 2].abs().max() <= 0.02


def test_ext_scale_matches_dim():
    assert EXT_SCALE.shape == (EXT_DIM,)


def test_expand_first_layer_preserves_outputs():
    sd = {"actor.0.weight": torch.randn(8, 5), "actor.0.bias": torch.randn(8), "critic.0.weight": torch.randn(8, 5),
          "std": torch.ones(2)}
    out = expand_first_layer(sd, 3)
    assert out["actor.0.weight"].shape == (8, 8) and out["critic.0.weight"].shape == (8, 8)
    x = torch.randn(4, 5)
    xe = torch.cat([x, torch.randn(4, 3)], 1)
    assert torch.allclose(x @ sd["actor.0.weight"].T, xe @ out["actor.0.weight"].T)
    assert torch.equal(out["std"], sd["std"])


def test_r2_score_perfect_and_mean():
    y = torch.randn(100, 2)
    assert torch.allclose(r2_score(y, y), torch.ones(2))
    assert torch.allclose(r2_score(y.mean(0).expand_as(y), y), torch.zeros(2), atol=1e-6)


def test_r2_score_constant_target_is_nan_free():
    y = torch.ones(10, 1)
    assert not torch.isnan(r2_score(y, y)).any()


def test_mlp_from_state_matches_manual():
    from policy.walk.rma import mlp_from_state

    sd = {"actor.0.weight": torch.randn(4, 3), "actor.0.bias": torch.randn(4), "actor.2.weight": torch.randn(2, 4),
          "actor.2.bias": torch.randn(2), "critic.0.weight": torch.randn(1, 3), "critic.0.bias": torch.randn(1)}
    m = mlp_from_state(sd)
    x = torch.randn(5, 3)
    ref = torch.nn.functional.elu(x @ sd["actor.0.weight"].T + sd["actor.0.bias"]) @ sd["actor.2.weight"].T + sd["actor.2.bias"]
    assert torch.allclose(m(x), ref, atol=1e-6)


def test_rma_runtime_runs_exported_onnx(tmp_path):
    import numpy as np

    from policy.walk.rma import BASE_OBS, HIST, RMARuntime

    torch.manual_seed(0)
    adapter = Adapter(len(PROPRIO), HIST, EXT_DIM)
    actor = torch.nn.Linear(BASE_OBS + EXT_DIM, 12)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor, self.adapter = actor, adapter

        def forward(self, obs, hist):
            return self.actor(torch.cat([obs, self.adapter(hist)], 1))

    path = str(tmp_path / "p.onnx")
    torch.onnx.export(M(), (torch.zeros(1, BASE_OBS), torch.zeros(1, HIST, len(PROPRIO))), path,
                      input_names=["obs", "history"], output_names=["actions"], opset_version=17, dynamo=False,
                      dynamic_axes={"obs": {0: "n"}, "history": {0: "n"}, "actions": {0: "n"}})
    rt = RMARuntime(path, n=2)
    obs = np.random.default_rng(0).standard_normal((2, BASE_OBS)).astype(np.float32)
    a = rt.act(obs)
    assert a.shape == (2, 12)
    h = torch.from_numpy(obs[:, PROPRIO])[:, None].expand(-1, HIST, -1)
    ref = M()(torch.from_numpy(obs), h).detach().numpy()
    assert np.allclose(a, ref, atol=1e-4)
    rt.act(obs * 0)
    assert np.allclose(rt.history[:, -1], 0) and np.allclose(rt.history[:, 0], obs[:, PROPRIO])
    rt.reset([0])
    rt.act(obs)
    assert np.allclose(rt.history[0], obs[0, PROPRIO])


def test_gait_period_and_loop_trim():
    import numpy as np

    from policy.walk.gait import gait_period, loop_segment

    dt = 0.02
    t = np.arange(400) * dt
    q = np.stack([np.sin(2 * np.pi * t / 0.5), np.cos(2 * np.pi * t / 0.5 + 0.3)], 1)
    n = gait_period(q, dt, lo_s=0.2, hi_s=1.5)
    assert abs(n * dt - 0.5) < dt + 1e-9
    seg = loop_segment(q, n, cycles=3)
    assert len(seg) == 3 * n
    assert np.abs(seg[0] - q[len(q) - 3 * n]).max() < 1e-12
    assert np.abs(seg[-1] + (seg[1] - seg[0]) - seg[0]).max() < 0.15

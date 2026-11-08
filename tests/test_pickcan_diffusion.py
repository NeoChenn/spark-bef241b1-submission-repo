import pytest

torch = pytest.importorskip("torch")

from policy.pickcan.diffusion import CHUNK, DiffusionPolicy, MLPPolicy, chunk_targets, n_params


def test_chunk_targets_pad_with_last_valid_action():
    T, n = 5, 2
    a = torch.arange(T * n * 5, dtype=torch.float32).view(T, n, 5)
    valid = torch.ones(T, n, dtype=torch.bool)
    valid[3:, 1] = False
    c = chunk_targets(a, valid, k=3)
    assert c.shape == (T, n, 3, 5)
    assert torch.equal(c[0, 0, 1], a[1, 0])
    assert torch.equal(c[4, 0, 2], a[4, 0])
    assert torch.equal(c[2, 1, 2], a[2, 1])


def test_diffusion_learns_obs_conditioned_chunks():
    torch.manual_seed(0)
    obs = torch.randn(512, 6)
    target = torch.tanh(obs[:, :5])[:, None, :].expand(-1, CHUNK, -1).contiguous()
    m = DiffusionPolicy(6, width=128, depth=2, steps=20)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(600):
        i = torch.randint(0, 512, (128,))
        loss = m.loss(obs[i], target[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = m.sample(obs[:64], n_steps=6)
    assert out.shape == (64, CHUNK, 5)
    assert (out - target[:64]).abs().mean() < 0.25
    assert n_params(m) > 0


def test_mlp_policy_shapes():
    m = MLPPolicy(7, width=32, depth=1)
    assert m.sample(torch.zeros(3, 7)).shape == (3, CHUNK, 5)

import numpy as np
import pytest

from sim.plume import Field, sample_ppm


def linear_field():
    ix, iy, iz = np.meshgrid(np.arange(6), np.arange(5), np.arange(4), indexing="ij")
    return Field((0.001 * ix + 0.002 * iy + 0.003 * iz).astype(np.float32), (1.0, 2.0, 3.0), 0.5, meta={"flange": "F5"})


def test_trilinear_is_exact_on_linear_field():
    f = linear_field()
    p = np.array([[1.0 + 0.5 * 2.3 + 0.25, 2.0 + 0.5 * 1.7 + 0.25, 3.0 + 0.5 * 2.2 + 0.25]])
    assert f.sample(p)[0] == pytest.approx(1e6 * (0.001 * 2.3 + 0.002 * 1.7 + 0.003 * 2.2), rel=1e-5)
    assert sample_ppm(f, *p[0]) == pytest.approx(f.sample(p)[0])


def test_outside_is_zero_and_edges_clamp():
    f = linear_field()
    assert f.sample_ppm(-5, 0, 0) == 0.0
    lo, hi = f.bounds()
    assert np.allclose(hi, [4.0, 4.5, 5.0])
    assert f.sample_ppm(*(lo + 0.01)) == pytest.approx(0.0, abs=1e-3)
    assert f.lel_percent([lo + 0.25])[0] == pytest.approx(0.0)


def test_save_load_roundtrip(tmp_path):
    f = linear_field()
    g = Field.load(f.save(tmp_path / "snap.npz"))
    assert g.meta == {"flange": "F5"} and g.dx == 0.5
    assert np.allclose(g.fraction, f.fraction, atol=1e-9)
    j = f.to_json(stride=2)
    assert j["shape"] == [3, 3, 2] and j["dx"] == 1.0

import numpy as np

from sim.hdri import horizon_color, read_hdr, write_hdr


def test_roundtrip_and_horizon(tmp_path):
    img = np.zeros((90, 180, 3), np.float32)
    img[:45] = (0.2, 0.4, 0.8)
    img[43:45] = (1.0, 0.9, 0.7)
    img[10, 10] = (5000.0, 4000.0, 3000.0)
    path = tmp_path / "sky.hdr"
    write_hdr(path, img)
    back = read_hdr(path)
    assert back.shape == img.shape
    assert np.allclose(back, img, rtol=0.02)
    assert np.allclose(horizon_color(back, 4.0), (1.0, 0.9, 0.7), rtol=0.02)

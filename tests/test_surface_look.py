import time

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from integration.surface_look import EPS, SurfaceMasks, region_of, sleeve
from policy.interfaces import PipeGeometry

PIPE = PipeGeometry([0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], 0.15, 0.8, 40, 64)
GROW = np.ones((3, 3), bool)


def disk(cz, ct, r, value=1.0):
    i, j = np.mgrid[:40, :64]
    return np.where((i - cz) ** 2 + (j - ct) ** 2 <= r * r, value, 0.0).astype(np.float32)


def grow(cells, n=1):
    c = np.pad(np.asarray(cells, bool), ((0, 0), (n, n)), mode="wrap")
    for _ in range(n):
        c = binary_dilation(c, GROW)
    return c[:, n:-n]


@pytest.fixture(scope="module")
def masks():
    return SurfaceMasks(PIPE, (8, 32, 10, 40))


def scene():
    rust = disk(20, 24, 4.5, 0.6) * (np.random.default_rng(0).random((40, 64)) * 0.5 + 0.5)
    rust[20:23, 22:26] = 0.0
    sprayed = disk(20, 25, 7) > 0
    painted = (disk(19, 30, 5) > 0) & sprayed
    return rust, sprayed, painted


def test_masks_are_zero_beyond_one_cell_of_their_grid(masks):
    rust, sprayed, painted = scene()
    r, wet, primer = masks(rust, sprayed, painted)
    assert not (r[..., 3] > 0)[~masks.near(grow(rust > 0))].any()
    assert not (wet > 0)[~masks.near(grow(sprayed & ~painted))].any()
    assert not (primer > 0)[~masks.near(grow(painted))].any()
    z = np.zeros((40, 64))
    r0, w0, p0 = masks(z, z, z)
    assert r0[..., 3].max() == 0 and w0.max() == 0 and p0.max() == 0


def test_rust_never_shows_where_the_grid_and_its_neighbours_are_clean(masks):
    rust, sprayed, painted = scene()
    alpha = masks(rust, sprayed, painted)[0][..., 3]
    clean_core = ~grow(rust > 0)
    assert alpha[masks.near(clean_core)].max() == 0
    assert alpha[masks.near(rust > 0.2)].mean() > 100


def test_primer_never_covers_a_rusty_cell(masks):
    rust = disk(20, 21, 3, 0.4)
    painted = disk(20, 26, 7) > 0
    primer = masks(rust, painted, painted)[2]
    assert primer[masks.near(rust > EPS)].max() == 0
    assert primer[masks.near(painted & ~grow(~painted) & ~grow(rust > EPS, 2))].min() > 0.5


def test_masks_are_deterministic_and_fixed_in_surface_space(masks):
    rust, sprayed, painted = scene()
    a = masks(rust, sprayed, painted)
    b = SurfaceMasks(PIPE, (8, 32, 10, 40))(rust, sprayed, painted)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    rust2 = rust.copy()
    rust2[18, 20] = 0.0
    r2 = masks(rust2, sprayed, painted)[0][..., 3]
    changed = r2 != a[0][..., 3]
    assert changed.any() and not changed[~masks.near(grow(rust2 != rust, 2))].any()


def boundary_alignment(m, alpha):
    on = alpha > 0.5 * alpha.max()
    rows = on[1:] != on[:-1]
    cols = on[:, 1:] != on[:, :-1]
    row_cut = m.rows_cell[1:] != m.rows_cell[:-1]
    col_cut = m.cols_cell[1:] != m.cols_cell[:-1]
    aligned = rows[row_cut].sum() + cols[:, col_cut].sum()
    return aligned / max(rows.sum() + cols.sum(), 1)


def iso_tv(a):
    gy, gx = np.gradient(a.astype(np.float32))
    return float(np.hypot(gx, gy).sum())


def test_edges_have_no_cell_stair_steps(masks):
    rust, sprayed, painted = scene()
    r, wet, primer = masks(rust, sprayed, painted)
    nn_wet = masks.near(sprayed & ~painted).astype(np.float32)
    assert boundary_alignment(masks, nn_wet) == 1.0
    for a in (r[..., 3].astype(np.float32), wet, primer):
        assert boundary_alignment(masks, a) < 0.35
    i, j = np.mgrid[:40, :64]
    diag = ((i - 8) * 1.3 + (j - 10) < 22).astype(np.float32)
    w = masks(np.zeros((40, 64)), diag > 0, np.zeros((40, 64), bool))[1]
    assert iso_tv(w) < 0.85 * iso_tv(masks.near(diag > 0).astype(np.float32))


def test_masks_run_under_budget(masks):
    rust, sprayed, painted = scene()
    masks(rust, sprayed, painted)
    t0 = time.perf_counter()
    for _ in range(5):
        r, wet, primer = masks(rust, sprayed, painted)
        masks.wet_rgba(wet), masks.primer_rgba(primer)
    assert (time.perf_counter() - t0) / 5 < 0.2
    assert 400 <= max(masks.H, masks.W) <= 1200


def test_region_and_sleeve_cover_the_patch_smoothly():
    rust, sprayed, painted = scene()
    i0, i1, j0, j1 = region_of(rust, sprayed, painted, margin=3)
    on = np.argwhere(sprayed)
    assert i0 <= on[:, 0].min() - 3 and i1 >= on[:, 0].max() + 4
    assert j0 <= on[:, 1].min() - 3 and j1 >= on[:, 1].max() + 4
    pts, nrm, uv, idx = sleeve(PIPE, (i0, i1, j0, j1), 0.1528)
    assert np.allclose(np.linalg.norm(pts[:, 1:], axis=1), 0.1528)
    assert uv.min() == 0.0 and uv.max() == 1.0 and idx.max() == len(pts) - 1
    facet = 2 * np.pi / 64 / 6
    assert np.degrees(facet) < 1.0
    seam = np.zeros((40, 64))
    seam[20, [62, 63, 0, 1]] = 1
    a0, a1, b0, b1 = region_of(seam, margin=1)
    assert b1 - b0 == 6 and b0 % 64 == 61

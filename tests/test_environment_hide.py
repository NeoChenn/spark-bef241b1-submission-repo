from rig.sim.envhide import is_builtin_environment


def test_matches_rig_file_sea_sky_sun():
    assert is_builtin_environment("sea", "Mesh")
    assert is_builtin_environment("sea_164", "Mesh")
    assert is_builtin_environment("sky", "DomeLight")
    assert is_builtin_environment("sun", "DistantLight")


def test_ignores_other_prims():
    assert not is_builtin_environment("deck", "Mesh")
    assert not is_builtin_environment("seal_ring", "Mesh")
    assert not is_builtin_environment("lamp", "SphereLight")

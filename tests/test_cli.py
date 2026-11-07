from rig.sim.cli import DEFAULT_CAMERA, DEFAULT_SCENE, parse_args


def test_defaults():
    a = parse_args([])
    assert a.scene == DEFAULT_SCENE == "/rig/sim/output/rig_deck.usda"
    assert a.camera == DEFAULT_CAMERA == "/World/Cameras/hero"
    assert (a.width, a.height) == (1280, 720)
    assert a.renderer == "RealTimePathTracing"
    assert a.orbit == 0.0
    assert a.env == ""


def test_overrides():
    a = parse_args(["--scene", "test", "--orbit", "90", "--env", "noon", "--spp", "4", "--subframes", "2", "--width", "640"])
    assert a.scene == "test"
    assert a.orbit == 90.0
    assert a.env == "noon"
    assert (a.spp, a.subframes, a.width) == (4, 2, 640)


def test_parse_setting_types():
    from rig.sim.cli import parse_setting

    assert parse_setting("/rtx/a=true") == ("/rtx/a", True)
    assert parse_setting("/rtx/b=3") == ("/rtx/b", 3)
    assert parse_setting("/rtx/c=0.5") == ("/rtx/c", 0.5)
    assert parse_setting("/rtx/d=PathTracing") == ("/rtx/d", "PathTracing")

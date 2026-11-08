from rig.sim.cli import DEFAULT_CAMERA, DEFAULT_SCENE, parse_args


def test_defaults():
    a = parse_args([])
    assert a.scene == DEFAULT_SCENE == "/rig/sim/output/full_rig.usda"
    assert a.camera == DEFAULT_CAMERA == "/World/Cameras/wide"
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


def test_orbit_target_and_vec_parsing():
    a = parse_args(["--orbit-target", "0,-5,30", "--orbit-radius", "120", "--orbit-eye-z", "40", "--sea-level", "-20"])
    assert a.orbit_target == (0.0, -5.0, 30.0)
    assert (a.orbit_radius, a.orbit_eye_z, a.sea_level) == (120.0, 40.0, -20.0)
    b = parse_args([])
    assert b.orbit_target is None and b.orbit_radius == 0.0 and b.orbit_eye_z is None


def test_cinematic_preset_flag():
    assert parse_args(["--cinematic", "night"]).cinematic == "night"
    assert parse_args([]).cinematic == ""


def test_merged_flags():
    a = parse_args(["--cinematic", "night", "--merged", "--merged-hold", "0", "--merged-start", "2.5"])
    assert a.merged and a.merged_hold == 0.0 and a.merged_start == 2.5
    assert not parse_args([]).merged and parse_args([]).merged_hold == 8.0


def test_merged_from_env(monkeypatch):
    monkeypatch.setenv("RIG_MERGED", "1")
    assert parse_args([]).merged
    assert not parse_args(["--no-merged"]).merged


def test_atmosphere_flags():
    a = parse_args([])
    assert a.atmos == "" and a.weather == "" and a.wind_dir is None and a.gust is None and not a.lens
    b = parse_args(["--atmos", "live", "--weather", "calm@0,storm@20", "--wind-dir", "120", "--gust", "30", "--lens"])
    assert (b.atmos, b.weather, b.wind_dir, b.gust, b.lens) == ("live", "calm@0,storm@20", 120.0, 30.0, True)

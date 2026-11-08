import json

import numpy as np
import pytest

from integration.gas_detectors import LEAK_META, STATES, GasDetectors, load_layout

META = json.loads(LEAK_META.read_text())


def test_one_detector_per_flange_beside_the_spool():
    dets = load_layout()
    assert [d.flange for d in dets] == [f["id"] for f in META["flanges"]]
    for d, f in zip(dets, META["flanges"]):
        c, n = np.asarray(f["center_m"]), np.asarray(f["normal"])
        assert 0.2 < np.linalg.norm(d.sample - c) < 0.6
        rel = d.sample - c
        radial = rel - (rel @ n) * n
        assert np.linalg.norm(radial) > f["pipe_radius_m"] + 0.05
        assert d.sample[2] < d.frame[2, 3]


def test_translation_moves_every_sample():
    t = np.array([-18.25, 5.25, -42.5])
    for a, b in zip(load_layout(), load_layout(translation=t)):
        assert np.allclose(b.sample - a.sample, t)


def test_usd_detectors_switch_state():
    from pxr import Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    gd = GasDetectors(stage)
    pos = gd.positions()
    assert set(pos) == {f"GD-{f['id']}" for f in META["flanges"]}
    for state in STATES:
        gd.set_detector_state("F1", state)
        led = stage.GetPrimAtPath("/World/GasDetectors/GD_F1/led")
        bound = UsdShade.MaterialBindingAPI(led).GetDirectBinding().GetMaterialPath()
        assert str(bound).endswith(f"led_{state}") and gd.state("GD-F1") == state
        assert stage.GetPrimAtPath("/World/GasDetectors/GD_F1").GetAttribute("rig:state").Get() == state
    assert gd.state("F2") == "ok"
    with pytest.raises(ValueError):
        gd.set_detector_state("F1", "purple")

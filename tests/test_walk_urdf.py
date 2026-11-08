import xml.etree.ElementTree as ET

from policy.walk.urdf import strip_collisions

URDF = "sim/assets/go2_ur3/go2_ur3_lift.urdf"
PAYLOAD = ("ur3_", "lift_", "scrubber", "brush")


def test_strip_collisions_keeps_legs_and_drops_payload():
    root = ET.fromstring(strip_collisions(open(URDF).read(), PAYLOAD))
    has = {l.get("name"): l.find("collision") is not None for l in root.findall("link")}
    assert has["base"] and has["FL_foot"] and has["RR_calf"]
    assert not any(v for k, v in has.items() if k.startswith(PAYLOAD))
    masses = {l.get("name"): l.find("inertial/mass") for l in root.findall("link")}
    assert float(masses["ur3_upper_arm_link"].get("value")) == 3.42


def test_mesh_paths_made_absolute():
    root = ET.fromstring(strip_collisions(open(URDF).read(), PAYLOAD, "/abs/dir"))
    files = [m.get("filename") for m in root.iter("mesh")]
    assert files and all(f.startswith("/abs/dir/") for f in files)

import json

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from rig.render.usdsync import StageApplier, StageCapture, decode_value, encode_value

BASE = """#usda 1.0
(
    upAxis = "Z"
)
def Xform "World"
{
    def Xform "Robot"
    {
        double3 xformOp:translate = (0, 0, 0)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        def Xform "link"
        {
            double3 xformOp:translate = (0, 0, 0.5)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Mesh "ocean"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
    def Sphere "lamp"
    {
        float radius = 1
    }
}
"""


def pair(tmp_path):
    path = tmp_path / "scene.usda"
    path.write_text(BASE)
    return Usd.Stage.Open(str(path)), Usd.Stage.Open(str(path))


def wire(msg):
    return json.loads(json.dumps(msg))


@pytest.mark.parametrize("kind,value", [
    ("double3", Gf.Vec3d(1.5, -2, 3)),
    ("float3", Gf.Vec3f(0.25, 0.5, 1)),
    ("quatf", Gf.Quatf(0.5, 0.5, -0.5, 0.5)),
    ("quatd", Gf.Quatd(1, 0, 0, 0)),
    ("matrix4d", Gf.Matrix4d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 5, 6, 1)),
    ("float", 2.5),
    ("bool", True),
    ("int", 7),
    ("token", "inherited"),
    ("asset", Sdf.AssetPath("/rig/sim/tex.png")),
    ("token[]", Vt.TokenArray(["xformOp:translate", "xformOp:orient"])),
    ("float3[]", Vt.Vec3fArray([Gf.Vec3f(1, 2, 3), Gf.Vec3f(4, 5, 6)])),
    ("float[]", Vt.FloatArray([0.5, 0.25])),
    ("color3f", Gf.Vec3f(1, 0.5, 0)),
])
def test_value_roundtrip(kind, value):
    out = decode_value(kind, wire(encode_value(value)))
    if kind == "asset":
        assert out.path == value.path
    else:
        assert out == value


def test_capture_sends_changed_transforms_and_skips_heavy_arrays(tmp_path):
    src, _ = pair(tmp_path)
    cap = StageCapture(src)
    src.GetPrimAtPath("/World/Robot").GetAttribute("xformOp:translate").Set(Gf.Vec3d(1, 2, 0))
    src.GetPrimAtPath("/World/Robot/link").GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0.75))
    src.GetPrimAtPath("/World/ocean").GetAttribute("points").Set(Vt.Vec3fArray([Gf.Vec3f(9, 9, 9)]))
    delta = cap.collect()
    names = {(p, n) for p, n, _, _ in delta["attrs"]}
    assert names == {("/World/Robot", "xformOp:translate"), ("/World/Robot/link", "xformOp:translate")}
    assert cap.collect() == {"specs": [], "attrs": []}


def test_applier_mirrors_capture_through_json(tmp_path):
    src, dst = pair(tmp_path)
    cap, app = StageCapture(src), StageApplier(dst)
    robot = src.GetPrimAtPath("/World/Robot")
    robot.GetAttribute("xformOp:orient").Set(Gf.Quatf(0.7071068, 0, 0, 0.7071068))
    robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(3, 0, 0))
    src.GetPrimAtPath("/World/lamp").GetAttribute("radius").Set(4.0)
    src.GetPrimAtPath("/World/lamp").CreateAttribute("spark:level", Sdf.ValueTypeNames.Float).Set(0.5)
    assert app.apply(wire(cap.collect())) >= 4
    a = UsdGeom.Xformable(src.GetPrimAtPath("/World/Robot/link")).ComputeLocalToWorldTransform(0)
    b = UsdGeom.Xformable(dst.GetPrimAtPath("/World/Robot/link")).ComputeLocalToWorldTransform(0)
    assert Gf.IsClose(a, b, 1e-5)
    assert dst.GetPrimAtPath("/World/lamp").GetAttribute("radius").Get() == 4.0
    assert dst.GetPrimAtPath("/World/lamp").GetAttribute("spark:level").Get() == 0.5
    assert app.failed == 0


def test_new_and_removed_prims_travel_as_specs(tmp_path):
    src, dst = pair(tmp_path)
    cap, app = StageCapture(src), StageApplier(dst)
    cube = UsdGeom.Cube.Define(src, "/World/Patch")
    cube.GetSizeAttr().Set(0.3)
    cube.AddTranslateOp().Set(Gf.Vec3d(1, 1, 1))
    src.RemovePrim("/World/lamp")
    delta = wire(cap.collect())
    assert {p for p, _ in delta["specs"]} == {"/World/Patch", "/World/lamp"}
    app.apply(delta)
    patch = dst.GetPrimAtPath("/World/Patch")
    assert patch.IsA(UsdGeom.Cube) and patch.GetAttribute("size").Get() == pytest.approx(0.3)
    assert not dst.GetPrimAtPath("/World/lamp").IsActive()
    UsdGeom.Cube(src.GetPrimAtPath("/World/Patch")).GetSizeAttr().Set(0.6)
    app.apply(wire(cap.collect()))
    assert patch.GetAttribute("size").Get() == pytest.approx(0.6)


def test_keyframe_catches_up_a_fresh_stage(tmp_path):
    src, dst = pair(tmp_path)
    cap = StageCapture(src)
    for x in range(5):
        src.GetPrimAtPath("/World/Robot").GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, 0, 0))
        cap.collect()
    key = wire(cap.keyframe())
    assert key["key"] is True and len(key["attrs"]) == 1
    StageApplier(dst).apply(key)
    assert dst.GetPrimAtPath("/World/Robot").GetAttribute("xformOp:translate").Get() == Gf.Vec3d(4, 0, 0)


def test_relationship_targets_sync(tmp_path):
    src, dst = pair(tmp_path)
    cap, app = StageCapture(src), StageApplier(dst)
    src.GetPrimAtPath("/World/lamp").CreateRelationship("material:binding").SetTargets(["/World/Robot"])
    app.apply(wire(cap.collect()))
    assert dst.GetPrimAtPath("/World/lamp").GetRelationship("material:binding").GetTargets() == [Sdf.Path("/World/Robot")]

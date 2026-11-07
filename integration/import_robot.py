"""Convert go2_ur3_lift.urdf to USD with Isaac Sim's URDF importer.

    C:\\isaacsim\\python.bat integration\\import_robot.py

Writes sim/output/robot/ (the bridge references the .usd inside). Prints the link prims it finds,
because the bridge poses each link by name.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from integration.kinematics import LIFT_URDF, RobotModel  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "sim/output/robot"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    config = URDFImporterConfig(urdf_path=str(LIFT_URDF), usd_path=str(OUT), merge_fixed_joints=False,
                                fix_base=True, collision_from_visuals=False)
    usd = URDFImporter(config).import_urdf()
    print(f"ROBOT_USD {usd}", flush=True)
    stage = Usd.Stage.Open(usd)
    links = set(RobotModel.from_urdf().link_poses(__import__("numpy").eye(4), {}).keys())
    found, meshes = {}, 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            meshes += 1
        if prim.GetName() in links and prim.IsA(UsdGeom.Xformable):
            found.setdefault(prim.GetName(), str(prim.GetPath()))
    print(f"default prim {stage.GetDefaultPrim().GetPath()}  link prims {len(found)}/{len(links)}  geoms {meshes}")
    for name in ("base", "lift_carriage", "ur3_wrist_3_link", "brush_tip", "FL_foot"):
        print(f"  {name}: {found.get(name)}")
    missing = sorted(links - set(found))
    print("missing", missing[:20])
    print("files", [str(p.relative_to(OUT)) for p in sorted(OUT.rglob("*")) if p.is_file()][:30])


main()
app.close()
os._exit(0)

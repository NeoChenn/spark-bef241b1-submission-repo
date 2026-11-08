from __future__ import annotations

import os
import shutil
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from pxr import Usd, UsdGeom

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "sim/assets/go2_ur3/go2_ur3_fixer.urdf"
OUT = REPO / "sim/output/robot"
KEY = ("base", "lift_carriage", "ur3_wrist_3_link", "grasp_frame", "socket_link", "socket_tip", "patrol_lidar",
       "unit2_label", "FL_foot")


def main() -> None:
    dst = OUT / "go2_ur3_fixer"
    if dst.exists():
        shutil.rmtree(dst)
    OUT.mkdir(parents=True, exist_ok=True)
    config = URDFImporterConfig(urdf_path=str(URDF), usd_path=str(OUT), merge_fixed_joints=False, fix_base=True,
                                collision_from_visuals=False)
    usd = URDFImporter(config).import_urdf()
    print(f"ROBOT_USD {usd}", flush=True)
    stage = Usd.Stage.Open(usd)
    found = {}
    for prim in stage.Traverse():
        if prim.GetName() in KEY and prim.IsA(UsdGeom.Xformable):
            found.setdefault(prim.GetName(), str(prim.GetPath()))
    for name in KEY:
        print(f"  {name}: {found.get(name)}")
    print("missing", [k for k in KEY if k not in found])


main()
app.close()
os._exit(0)

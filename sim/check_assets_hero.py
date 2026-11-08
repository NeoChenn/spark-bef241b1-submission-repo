import argparse
import json
import os
import sys
import time

sys.path[:0] = ["/rig", "/rig/sim"]

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--out", default="")
args = ap.parse_args()

start = time.monotonic()
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": 640, "height": 360})
boot = time.monotonic() - start

import argparse as _ap

import numpy as np
import omni.timeline
import omni.usd
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Usd, UsdGeom, UsdPhysics

from rig.sim.main import load_cinematic

t0 = time.monotonic()
ctx = omni.usd.get_context()
load_cinematic(app, ctx, "night", _ap.Namespace(merged=True, merged_hold=0, merged_start=0.0))
for _ in range(30):
    app.update()
load = time.monotonic() - t0
stage = ctx.get_stage()
root = stage.GetPrimAtPath("/World/AssetsHero")
prims = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Xform)]
bodies = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
          if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute("physics:rigidBodyEnabled").Get() is not False]
joints = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint)]


def poses():
    cache = UsdGeom.XformCache()
    return np.array([np.array(cache.GetLocalToWorldTransform(p)).reshape(-1) for p in prims])


before = poses()
omni.timeline.get_timeline_interface().play()
app.update()
sim = get_physx_simulation_interface()
physx = get_physx_interface()
simulated = [b for b in bodies if physx.get_rigidbody_transformation(b).get("ret_val")]
dt, t = 1.0 / 60.0, 0.0
while t < args.seconds:
    sim.simulate(dt, t)
    sim.fetch_results()
    t += dt
for _ in range(5):
    app.update()
moved = float(np.abs(poses() - before).max()) if prims else 0.0
report = {"boot_s": round(boot, 1), "load_s": round(load, 1), "remote_active": root.GetChild("Remote").IsActive()
          if root.GetChild("Remote") else None, "prims": len(prims), "rigid_bodies": len(bodies),
          "simulated_bodies": len(simulated), "joints": len(joints), "sim_seconds": round(t, 1), "max_move_m": moved}
print("CHECK " + json.dumps(report), flush=True)
if args.out:
    with open(args.out, "w") as fh:
        json.dump(report, fh)
os._exit(0)

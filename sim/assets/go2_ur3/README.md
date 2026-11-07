# Unitree Go2 + UR3 (URDF package)

One URDF with a UR3 arm mounted on the back of a Unitree Go2, plus the meshes it needs.

| File | What it is |
|---|---|
| `go2_ur3.urdf` | Go2 + UR3 with the real masses (arm 10.63 kg, Go2 16.09 kg) |
| `meshes/go2/*.dae` | Go2 visual meshes (collision uses primitives defined in the URDF) |
| `meshes/ur3/...` | UR3 visual and collision meshes |
| `preview/go2_ur3_forward.png` | Side/top stick figure of the arm and centre of mass |
| `tools/build_combined.py` | Regenerates the URDF from the two upstream packages |
| `tools/validate_urdf.py` | Structure, mass, forward-kinematics, tip-over and static-torque checks |

Mesh paths are relative to the URDF, so keep `meshes/` next to it.

## Where it comes from

- Go2: `robots/go2_description` in unitreerobotics/unitree_ros (commit `41dec58`), BSD-3-Clause.
- UR3: `ur3.urdf` and meshes from the `robotics-toolbox/xacro_generated/ur_description` folder of Daniella1/urdf_files_dataset (that folder is MIT-licensed). I did not verify the licence of the UR meshes upstream, so check before redistributing them.

## How the arm is attached

Fixed joint `go2_ur3_mount`, parent Go2 `base`, child `ur3_base_link`, origin `xyz = (-0.01, 0, 0.064)`, no rotation. The Go2 shell top is flat (z about 0.058 to 0.063) for x between -0.09 and 0.07, measured from `base.dae`; the UR3 base (about 0.13 m wide, 0.086 m tall) sits on it just behind the Go2's centre of mass. Arm links, joints and transmissions are prefixed `ur3_` because both robots have a link called `base`. Go2 Chinese material names were replaced with ASCII ones. `<gazebo>` and `world` entries were dropped.

## What was checked

Standing pose (hips +/-0.1, thighs 0.8 front / 1.0 rear, calves -1.5). Margin is the distance from the combined centre of mass to the edge of the feet polygon (positive = inside). Torque is the worst leg's static torque to hold the pose, versus the Go2 URDF limits (hip 23.7, thigh 23.7, calf 45.4 N m).

| Arm pose | CoM margin | hip / thigh / calf torque (N m) |
|---|---|---|
| Go2 only | +0.172 m | 5.9 / 2.4 / 6.4 |
| fully extended forward (`zero`) | +0.118 m | 13.4 / 3.4 / 14.5 |
| straight up | +0.147 m | 11.0 / 4.9 / 11.9 |
| L-shape forward | +0.147 m | 11.9 / 4.3 / 12.9 |
| folded back | +0.147 m | 10.4 / 5.2 / 11.2 |

Also checked: unique names, one root, connected tree, all meshes present, no `package://` paths.

Not checked: the Isaac Sim import itself (I could not run it here), self-collision between arm, trunk and legs, and any dynamics. The torque numbers are static, vertical-force only, so walking, acceleration and arm motion will need more. The arm's links sit about 0.19 m to the left of the Go2's centreline, which shifts the CoM about 2.6 cm left.

## Be aware

The arm is 0.66x the Go2's mass. The real Go2's payload rating is far below 10.6 kg, but in simulation it holds still within torque limits. A locomotion policy trained on the bare Go2 will still need retraining with the extra mass. Combined mass is about 26.7 kg.

## Importing into Isaac Sim 6.2

1. Enable `isaacsim.asset.importer.urdf` (Window > Extensions) if needed.
2. File > Import `go2_ur3.urdf`; choose an output folder you can write to.
3. Start with defaults. Merge Mesh and Allow Self-Collision are the GUI options.
4. Legs: torque control (no drive) for a learned policy; arm joints: position drives with stiffness.
5. Make sure the base is floating, not fixed (check the articulation root, or use the Python importer).

## Rebuilding

```bash
python3 -I tools/build_combined.py \
  --go2 path/to/unitree_ros/robots/go2_description \
  --arm-dir path/to/urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/ur_description \
  --arm ur3 --out . [--mount-xyz X Y Z] [--mount-yaw RAD] [--arm-mass-scale S]
python3 -I tools/validate_urdf.py go2_ur3.urdf --arm ur3 --plot preview/check.png --pose forward
```

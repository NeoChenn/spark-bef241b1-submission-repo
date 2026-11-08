"""Build go2_ur3_gripper.urdf: go2_ur3_lift.urdf with the scrub brush swapped for a parallel-jaw gripper
and an open-top container on the Go2 trunk, for the patrol pick-and-carry task.

Changes to go2_ur3_lift.urdf (the original is left untouched):
  removes scrubber_link, brush_tip (+ ur3_scrubber_mount, scrubber_tip_joint)
  ur3_ee_link --fixed--> gripper_base (palm, along ur3_ee_link +x)
      --prismatic +y 0..opening/2--> gripper_finger_left
      --prismatic -y 0..opening/2 (mimic left)--> gripper_finger_right
      --fixed--> grasp_frame (between the finger pads, IK target)
  base --fixed--> container (open-top bin behind the lift mast) --fixed--> container_drop (release point)

Usage: python3 -I integration/make_go2_ur3_gripper.py [--opening 0.12]
"""
import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_go2_ur3_lift import fixed, geom, inertial  # noqa: E402

SRC = HERE.parent / "sim/assets/go2_ur3/go2_ur3_lift.urdf"
DST = HERE.parent / "sim/assets/go2_ur3/go2_ur3_gripper.urdf"

TRUNK_TOP_Z = 0.057          # base collision box 0.3762 x 0.0935 x 0.114, centred at the base origin
HOUSING_REAR_X = -0.055      # lift housing 0.09 wide centred at the mount x = -0.01
BIN_INNER = (0.26, 0.20, 0.12)
WALL = 0.005
PALM = (0.02, 0.16, 0.04)    # x (along flange axis), y (jaw travel), z
FINGER = (0.06, 0.01, 0.02)


def box(link, size, xyz, material=None, collide=True):
    s = {"size": " ".join(f"{v:.4g}" for v in size)}
    geom(link, "visual", ("box", s), xyz=xyz, material=material)
    if collide:
        geom(link, "collision", ("box", s), xyz=xyz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opening", type=float, default=0.12, help="max jaw opening in metres")
    args = ap.parse_args()
    half = args.opening / 2

    tree = ET.parse(SRC)
    robot = tree.getroot()
    robot.set("name", "go2_ur3_gripper")
    for j in list(robot.findall("joint")):
        if j.get("name") in ("ur3_scrubber_mount", "scrubber_tip_joint"):
            robot.remove(j)
    for l in list(robot.findall("link")):
        if l.get("name") in ("scrubber_link", "brush_tip"):
            robot.remove(l)

    for name, rgba in (("gripper_dark", "0.15 0.15 0.17 1"), ("bin_yellow", "0.95 0.75 0.1 1")):
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=rgba)

    palm = ET.SubElement(robot, "link", name="gripper_base")
    box(palm, PALM, f"{PALM[0]/2} 0 0", "gripper_dark")
    inertial(palm, 0.3, *PALM, xyz=f"{PALM[0]/2} 0 0")
    fixed(robot, "ur3_gripper_mount", "ur3_ee_link", "gripper_base")

    for side, sign in (("left", 1), ("right", -1)):
        f = ET.SubElement(robot, "link", name=f"gripper_finger_{side}")
        xyz = f"{FINGER[0]/2} {sign * FINGER[1]/2} 0"
        box(f, FINGER, xyz, "gripper_dark")
        inertial(f, 0.05, *FINGER, xyz=xyz)
        j = ET.SubElement(robot, "joint", name=f"gripper_finger_{side}_joint", type="prismatic")
        ET.SubElement(j, "origin", xyz=f"{PALM[0]} 0 0", rpy="0 0 0")
        ET.SubElement(j, "parent", link="gripper_base")
        ET.SubElement(j, "child", link=f"gripper_finger_{side}")
        ET.SubElement(j, "axis", xyz=f"0 {sign} 0")
        ET.SubElement(j, "limit", lower="0.0", upper=f"{half:.4g}", effort="40.0", velocity="0.1")
        ET.SubElement(j, "dynamics", damping="5.0", friction="0.5")
        if side == "right":
            ET.SubElement(j, "mimic", joint="gripper_finger_left_joint", multiplier="1", offset="0")

    grasp = ET.SubElement(robot, "link", name="grasp_frame")
    inertial(grasp, 0.001, 0.001, 0.001, 0.001)
    grasp_x = PALM[0] + FINGER[0] - 0.01
    fixed(robot, "gripper_grasp_joint", "gripper_base", "grasp_frame", xyz=f"{grasp_x:.4g} 0 0")

    lx, ly, lz = BIN_INNER
    ox, oy = lx + 2 * WALL, ly + 2 * WALL
    bin_x = HOUSING_REAR_X - 0.01 - ox / 2
    container = ET.SubElement(robot, "link", name="container")
    box(container, (ox, oy, WALL), f"0 0 {WALL/2}", "bin_yellow")
    for sx in (-1, 1):
        box(container, (WALL, oy, lz), f"{sx * (lx/2 + WALL/2):.4g} 0 {WALL + lz/2:.4g}", "bin_yellow")
    for sy in (-1, 1):
        box(container, (lx, WALL, lz), f"0 {sy * (ly/2 + WALL/2):.4g} {WALL + lz/2:.4g}", "bin_yellow")
    inertial(container, 0.8, ox, oy, lz, xyz=f"0 0 {lz/2:.4g}")
    fixed(robot, "base_container_mount", "base", "container", xyz=f"{bin_x:.4g} 0 {TRUNK_TOP_Z}")

    drop = ET.SubElement(robot, "link", name="container_drop")
    inertial(drop, 0.001, 0.001, 0.001, 0.001)
    fixed(robot, "container_drop_joint", "container", "container_drop", xyz=f"0 0 {WALL + lz + 0.08:.4g}")

    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=True)
    print(f"wrote {DST.relative_to(HERE.parent)}  (opening {args.opening} m, grasp_frame {grasp_x:.3f} m from flange, "
          f"bin inner {lx}x{ly}x{lz} m at base x={bin_x:.3f} z={TRUNK_TOP_Z})")


if __name__ == "__main__":
    main()

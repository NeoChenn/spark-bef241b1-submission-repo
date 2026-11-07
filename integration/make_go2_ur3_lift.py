"""Build go2_ur3_lift.urdf: the sim team's Go2 + UR3 with a telescoping lift mast and a scrub brush.

Why: with the UR3 mounted straight on the Go2, the end effector tops out at ~1.1 m (arm fully
vertical), but the rust on the rig's main pipe sits at ~1.22 m. An IK sweep over the rust patch
(brush held normal to the pipe) found no solution below ~0.35 m of lift, and better dexterity at
0.45 m. So the mast strokes 0-0.45 m: retracted while walking (low CoM), extended to the rust
height before scrubbing.

Changes to go2_ur3.urdf (the original is left untouched):
  base --fixed(go2 mount pose)--> lift_housing --prismatic z 0..0.45--> lift_carriage --fixed--> ur3_base_link
  ur3_ee_link --fixed--> scrubber_link (brush body)  --fixed--> brush_tip (contact point, 0.12 m out)

The output goes next to the original so the relative mesh paths keep working.

Usage: python3 -I integration/make_go2_ur3_lift.py [--stroke 0.45] [--tool 0.12]
"""
import argparse
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "sim/assets/go2_ur3/go2_ur3.urdf"
DST = HERE.parent / "sim/assets/go2_ur3/go2_ur3_lift.urdf"


def inertial(parent, mass, sx, sy, sz, xyz="0 0 0"):
    """Solid-box inertia; good enough for a mast and a brush."""
    i = ET.SubElement(parent, "inertial")
    ET.SubElement(i, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(i, "mass", value=f"{mass}")
    ixx = mass / 12 * (sy**2 + sz**2)
    iyy = mass / 12 * (sx**2 + sz**2)
    izz = mass / 12 * (sx**2 + sy**2)
    ET.SubElement(i, "inertia", ixx=f"{ixx:.6g}", ixy="0", ixz="0", iyy=f"{iyy:.6g}", iyz="0", izz=f"{izz:.6g}")


def geom(parent, tag, shape, xyz="0 0 0", rpy="0 0 0", material=None):
    e = ET.SubElement(parent, tag)
    ET.SubElement(e, "origin", xyz=xyz, rpy=rpy)
    g = ET.SubElement(e, "geometry")
    kind, attrs = shape
    ET.SubElement(g, kind, **attrs)
    if material and tag == "visual":
        ET.SubElement(e, "material", name=material)
    return e


def fixed(robot, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
    j = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(j, "origin", xyz=xyz, rpy=rpy)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke", type=float, default=0.45, help="lift travel in metres")
    ap.add_argument("--tool", type=float, default=0.12, help="flange to brush tip, metres")
    args = ap.parse_args()

    tree = ET.parse(SRC)
    robot = tree.getroot()
    robot.set("name", "go2_ur3_lift")

    mount = next(j for j in robot.findall("joint") if j.get("name") == "go2_ur3_mount")
    m_origin = mount.find("origin")
    mount_xyz, mount_rpy = m_origin.get("xyz"), m_origin.get("rpy", "0 0 0")
    robot.remove(mount)

    for name, rgba in (("lift_grey", "0.55 0.57 0.6 1"), ("brush_orange", "0.9 0.45 0.1 1")):
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=rgba)

    # Housing: a short square sleeve bolted where the UR3 base used to sit.
    housing = ET.SubElement(robot, "link", name="lift_housing")
    hz = 0.10
    geom(housing, "visual", ("box", {"size": f"0.09 0.09 {hz}"}), xyz=f"0 0 {hz/2}", material="lift_grey")
    geom(housing, "collision", ("box", {"size": f"0.09 0.09 {hz}"}), xyz=f"0 0 {hz/2}")
    inertial(housing, 1.2, 0.09, 0.09, hz, xyz=f"0 0 {hz/2}")
    fixed(robot, "go2_lift_mount", "base", "lift_housing", xyz=mount_xyz, rpy=mount_rpy)

    # Carriage: the inner column. Its origin is the top plate the UR3 bolts to. The column hangs
    # below the plate (visual only) so it looks telescoping; the plate keeps a small collision box.
    carriage = ET.SubElement(robot, "link", name="lift_carriage")
    col = args.stroke
    geom(carriage, "visual", ("box", {"size": f"0.06 0.06 {col}"}), xyz=f"0 0 {-col/2}", material="lift_grey")
    geom(carriage, "visual", ("box", {"size": "0.13 0.13 0.012"}), xyz="0 0 -0.006", material="lift_grey")
    geom(carriage, "collision", ("box", {"size": "0.13 0.13 0.012"}), xyz="0 0 -0.006")
    inertial(carriage, 0.9, 0.06, 0.06, col, xyz=f"0 0 {-col/2}")

    lift = ET.SubElement(robot, "joint", name="lift_joint", type="prismatic")
    ET.SubElement(lift, "origin", xyz=f"0 0 {hz}", rpy="0 0 0")
    ET.SubElement(lift, "parent", link="lift_housing")
    ET.SubElement(lift, "child", link="lift_carriage")
    ET.SubElement(lift, "axis", xyz="0 0 1")
    # Electric linear actuator class: 1 kN, 0.15 m/s.
    ET.SubElement(lift, "limit", lower="0.0", upper=f"{args.stroke}", effort="1000.0", velocity="0.15")
    ET.SubElement(lift, "dynamics", damping="50.0", friction="5.0")

    fixed(robot, "lift_ur3_mount", "lift_carriage", "ur3_base_link")

    # Brush: a cylinder along ur3_ee_link's +x (the flange's outward axis).
    tool = args.tool
    brush = ET.SubElement(robot, "link", name="scrubber_link")
    body_len, head_len = tool - 0.03, 0.03
    geom(brush, "visual", ("cylinder", {"radius": "0.015", "length": f"{body_len}"}),
         xyz=f"{body_len/2} 0 0", rpy=f"0 {math.pi/2} 0", material="lift_grey")
    geom(brush, "visual", ("cylinder", {"radius": "0.035", "length": f"{head_len}"}),
         xyz=f"{body_len + head_len/2} 0 0", rpy=f"0 {math.pi/2} 0", material="brush_orange")
    geom(brush, "collision", ("cylinder", {"radius": "0.035", "length": f"{head_len}"}),
         xyz=f"{body_len + head_len/2} 0 0", rpy=f"0 {math.pi/2} 0")
    inertial(brush, 0.35, tool, 0.07, 0.07, xyz=f"{tool/2} 0 0")
    fixed(robot, "ur3_scrubber_mount", "ur3_ee_link", "scrubber_link")

    tip = ET.SubElement(robot, "link", name="brush_tip")
    inertial(tip, 0.001, 0.001, 0.001, 0.001)
    fixed(robot, "scrubber_tip_joint", "scrubber_link", "brush_tip", xyz=f"{tool} 0 0")

    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=True)
    print(f"wrote {DST.relative_to(HERE.parent)}  (stroke {args.stroke} m, tool {tool} m)")


if __name__ == "__main__":
    main()

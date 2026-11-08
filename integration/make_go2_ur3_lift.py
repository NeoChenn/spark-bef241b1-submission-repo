import argparse
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "sim/assets/go2_ur3/go2_ur3.urdf"
DST = HERE.parent / "sim/assets/go2_ur3/go2_ur3_lift.urdf"
GRIP_DST = HERE.parent / "sim/assets/go2_ur3/go2_ur3_lift_grip.urdf"
TOTE = {"x": -0.2, "y": 0.0, "z": 0.062, "lx": 0.26, "ly": 0.22, "h": 0.09, "t": 0.008}


def inertial(parent, mass, sx, sy, sz, xyz="0 0 0"):
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


def add_suction(robot, tool):
    link = ET.SubElement(robot, "link", name="gripper_link")
    cup_len, lip = 0.028, 0.006
    shaft = tool - cup_len
    geom(link, "visual", ("cylinder", {"radius": "0.032", "length": "0.02"}), xyz="0.01 0 0",
         rpy=f"0 {math.pi/2} 0", material="lift_grey")
    geom(link, "visual", ("cylinder", {"radius": "0.012", "length": f"{shaft}"}), xyz=f"{shaft/2} 0 0",
         rpy=f"0 {math.pi/2} 0", material="lift_grey")
    geom(link, "visual", ("cylinder", {"radius": "0.016", "length": f"{cup_len - lip}"}),
         xyz=f"{shaft + (cup_len - lip)/2} 0 0", rpy=f"0 {math.pi/2} 0", material="cup_black")
    geom(link, "visual", ("cylinder", {"radius": "0.022", "length": f"{lip}"}), xyz=f"{tool - lip/2} 0 0",
         rpy=f"0 {math.pi/2} 0", material="cup_black")
    geom(link, "collision", ("cylinder", {"radius": "0.02", "length": f"{cup_len}"}),
         xyz=f"{tool - cup_len/2} 0 0", rpy=f"0 {math.pi/2} 0")
    inertial(link, 0.3, tool, 0.05, 0.05, xyz=f"{tool/2} 0 0")
    fixed(robot, "ur3_gripper_mount", "ur3_ee_link", "gripper_link")
    tip = ET.SubElement(robot, "link", name="grip_tip")
    inertial(tip, 0.001, 0.001, 0.001, 0.001)
    fixed(robot, "gripper_tip_joint", "gripper_link", "grip_tip", xyz=f"{tool} 0 0")


def add_tote(robot):
    T = TOTE
    link = ET.SubElement(robot, "link", name="tote_link")
    lx, ly, h, t = T["lx"], T["ly"], T["h"], T["t"]
    walls = [((lx, ly, t), (0, 0, t / 2)), ((t, ly, h), (lx / 2 - t / 2, 0, h / 2)),
             ((t, ly, h), (-lx / 2 + t / 2, 0, h / 2)), ((lx, t, h), (0, ly / 2 - t / 2, h / 2)),
             ((lx, t, h), (0, -ly / 2 + t / 2, h / 2))]
    for size, xyz in walls:
        sz = " ".join(f"{v:.4f}" for v in size)
        at = " ".join(f"{v:.4f}" for v in xyz)
        geom(link, "visual", ("box", {"size": sz}), xyz=at, material="tote_yellow")
        geom(link, "collision", ("box", {"size": sz}), xyz=at)
    inertial(link, 0.6, lx, ly, h, xyz=f"0 0 {h/2}")
    fixed(robot, "go2_tote_mount", "base", "tote_link", xyz=f"{T['x']} {T['y']} {T['z']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke", type=float, default=0.45, help="lift travel in metres")
    ap.add_argument("--tool", type=float, default=None, help="flange to tool tip, metres")
    ap.add_argument("--grip", action="store_true", help="suction gripper + tote instead of the scrub brush")
    args = ap.parse_args()
    if args.tool is None:
        args.tool = 0.20 if args.grip else 0.12

    tree = ET.parse(SRC)
    robot = tree.getroot()
    robot.set("name", "go2_ur3_lift_grip" if args.grip else "go2_ur3_lift")

    mount = next(j for j in robot.findall("joint") if j.get("name") == "go2_ur3_mount")
    m_origin = mount.find("origin")
    mount_xyz, mount_rpy = m_origin.get("xyz"), m_origin.get("rpy", "0 0 0")
    robot.remove(mount)

    mats = (("lift_grey", "0.55 0.57 0.6 1"), ("brush_orange", "0.9 0.45 0.1 1"))
    if args.grip:
        mats += (("cup_black", "0.06 0.06 0.07 1"), ("tote_yellow", "0.85 0.62 0.08 1"))
    for name, rgba in mats:
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=rgba)

    housing = ET.SubElement(robot, "link", name="lift_housing")
    hz = 0.10
    geom(housing, "visual", ("box", {"size": f"0.09 0.09 {hz}"}), xyz=f"0 0 {hz/2}", material="lift_grey")
    geom(housing, "collision", ("box", {"size": f"0.09 0.09 {hz}"}), xyz=f"0 0 {hz/2}")
    inertial(housing, 1.2, 0.09, 0.09, hz, xyz=f"0 0 {hz/2}")
    fixed(robot, "go2_lift_mount", "base", "lift_housing", xyz=mount_xyz, rpy=mount_rpy)

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
    ET.SubElement(lift, "limit", lower="0.0", upper=f"{args.stroke}", effort="1000.0", velocity="0.15")
    ET.SubElement(lift, "dynamics", damping="50.0", friction="5.0")

    fixed(robot, "lift_ur3_mount", "lift_carriage", "ur3_base_link")

    tool = args.tool
    if args.grip:
        add_suction(robot, tool)
        add_tote(robot)
        dst = GRIP_DST
    else:
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

        dst = DST

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"wrote {dst.relative_to(HERE.parent)}  (stroke {args.stroke} m, tool {tool} m)")


if __name__ == "__main__":
    main()

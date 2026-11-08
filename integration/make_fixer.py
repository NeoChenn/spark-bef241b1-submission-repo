import argparse
import math
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_go2_ur3_lift import fixed, geom, inertial  # noqa: E402

ASSETS = HERE.parent / "sim/assets/go2_ur3"
SRC = ASSETS / "go2_ur3_gripper.urdf"
DST = ASSETS / "go2_ur3_fixer.urdf"

TEAL = "0.180 0.769 0.714 1"
SHANK_R, SHANK_BACK, SHANK_FRONT = 0.012, 0.05, 0.03
HEAD_R, HEAD_LEN, BORE_R = 0.022, 0.035, 0.0165
TOOL_LINK = "socket_tip"
GRASP_LINK = "grasp_frame"
SOCKET_JOINT = "socket_drive_joint"
FINGER_JOINTS = ("gripper_finger_left_joint", "gripper_finger_right_joint")
FINGER_CLOSED_ON_SOCKET = SHANK_R
RETORQUE_LIFT = 0.40
RETORQUE_STANDOFF = 0.6
IK_SEEDS = ((0.0, -1.2, 1.4, -0.2, 1.57, 0.0), (0.0, -1.57, 1.57, 0.0, -1.57, 0.0))
LIDAR_XYZ = (0.345, 0.0, -0.04)
LABEL_Y, LABEL_X, LABEL_Z = 0.061, 0.02, 0.015
GLYPHS = {
    "U": [(0, 2, 0, 0), (0, 0, 1, 0), (1, 0, 1, 2)],
    "N": [(0, 0, 0, 2), (0, 2, 1, 0), (1, 0, 1, 2)],
    "I": [(0.5, 0, 0.5, 2)],
    "T": [(0, 2, 1, 2), (0.5, 0, 0.5, 2)],
    "2": [(0, 2, 1, 2), (1, 2, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0)],
    " ": [],
}


def stand_pose(center, normal, standoff=RETORQUE_STANDOFF, trunk_z=0.32, side_on=False):
    from integration.kinematics import pose
    n = [normal[0], normal[1]]
    face = math.atan2(-n[1], -n[0]) + (math.pi / 2 if side_on else 0.0)
    return pose(center[0] + standoff * n[0], center[1] + standoff * n[1], trunk_z, face)


def reach(model, base, target, approach, lift=RETORQUE_LIFT, side_on=False, iterations=200):
    from integration.kinematics import ARM_JOINTS
    best = None
    for seed in IK_SEEDS:
        q = model.default_q()
        q.update(dict(zip(ARM_JOINTS, seed)))
        q["lift_joint"] = lift
        q.update({j: FINGER_CLOSED_ON_SOCKET for j in FINGER_JOINTS})
        if side_on:
            q["ur3_shoulder_pan_joint"] -= math.pi / 2
        r = model.solve_ik(base, q, target, approach, iterations=iterations, tip=TOOL_LINK)
        if best is None or r.position_error + r.axis_error < best.position_error + best.axis_error:
            best = r
    return best


def material(robot, name, rgba):
    m = ET.SubElement(robot, "material", name=name)
    ET.SubElement(m, "color", rgba=rgba)


def cyl(link, r, length, x, material, collide=False):
    shape = ("cylinder", {"radius": f"{r:.4g}", "length": f"{length:.4g}"})
    geom(link, "visual", shape, xyz=f"{x:.4g} 0 0", rpy=f"0 {math.pi/2:.6f} 0", material=material)
    if collide:
        geom(link, "collision", shape, xyz=f"{x:.4g} 0 0", rpy=f"0 {math.pi/2:.6f} 0")


def box(link, size, xyz, material, rpy="0 0 0"):
    geom(link, "visual", ("box", {"size": " ".join(f"{v:.4g}" for v in size)}), xyz=xyz, rpy=rpy, material=material)


def label(link, text, side, cw=0.022, ch=0.036, gap=0.012, stroke=0.006):
    step = cw + gap
    width = len(text) * step - gap
    y = side * (LABEL_Y + 0.0025 + 0.0005)
    box(link, (width + 0.04, 0.004, ch + 0.022), f"{LABEL_X} {side * (LABEL_Y + 0.0005):.4g} {LABEL_Z}", "fixer_teal")
    for i, c in enumerate(text):
        for u0, v0, u1, v1 in GLYPHS[c]:
            pts = []
            for u, v in ((u0, v0), (u1, v1)):
                s = i * step + u * cw - width / 2
                pts.append((LABEL_X - side * s, LABEL_Z + (v / 2 - 0.5) * ch))
            (xa, za), (xb, zb) = pts
            length = math.hypot(xb - xa, zb - za) + stroke
            pitch = math.atan2(-(zb - za), xb - xa)
            box(link, (length, 0.001, stroke), f"{(xa + xb) / 2:.4g} {y:.4g} {(za + zb) / 2:.4g}", "label_white",
                rpy=f"0 {pitch:.6f} 0")


def build(drop_container: bool = True) -> pathlib.Path:
    if not SRC.exists():
        subprocess.run([sys.executable, str(HERE / "make_go2_ur3_gripper.py")], check=True)
    tree = ET.parse(SRC)
    robot = tree.getroot()
    robot.set("name", "go2_ur3_fixer")
    if drop_container:
        for j in list(robot.findall("joint")):
            if j.get("name") in ("base_container_mount", "container_drop_joint"):
                robot.remove(j)
        for l in list(robot.findall("link")):
            if l.get("name") in ("container", "container_drop"):
                robot.remove(l)

    for name, rgba in (("fixer_teal", TEAL), ("label_white", "0.96 0.97 0.97 1"),
                       ("socket_steel", "0.66 0.68 0.71 1"), ("socket_bore", "0.04 0.04 0.05 1"),
                       ("lidar_black", "0.06 0.06 0.07 1")):
        material(robot, name, rgba)

    links = {l.get("name"): l for l in robot.findall("link")}
    box(links["lift_housing"], (0.096, 0.096, 0.03), "0 0 0.05", "fixer_teal")
    box(links["lift_carriage"], (0.066, 0.066, 0.04), "0 0 -0.05", "fixer_teal")
    box(links["gripper_base"], (0.021, 0.162, 0.012), "0.01 0 0", "fixer_teal")

    sock = ET.SubElement(robot, "link", name="socket_link")
    shank = SHANK_BACK + SHANK_FRONT
    cyl(sock, SHANK_R, shank, (SHANK_FRONT - SHANK_BACK) / 2, "socket_steel", collide=True)
    cyl(sock, SHANK_R + 0.0015, 0.012, -0.02, "fixer_teal")
    cyl(sock, HEAD_R, HEAD_LEN, SHANK_FRONT + HEAD_LEN / 2, "socket_steel", collide=True)
    cyl(sock, BORE_R, 0.002, SHANK_FRONT + HEAD_LEN + 0.0005, "socket_bore")
    box(sock, (HEAD_LEN * 0.8, 0.004, 0.003), f"{SHANK_FRONT + HEAD_LEN / 2:.4g} 0 {HEAD_R:.4g}", "label_white")
    inertial(sock, 0.25, shank + HEAD_LEN, 2 * HEAD_R, 2 * HEAD_R, xyz=f"{SHANK_FRONT:.4g} 0 0")
    drive = ET.SubElement(robot, "joint", name="socket_drive_joint", type="continuous")
    ET.SubElement(drive, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(drive, "parent", link="grasp_frame")
    ET.SubElement(drive, "child", link="socket_link")
    ET.SubElement(drive, "axis", xyz="1 0 0")
    ET.SubElement(drive, "limit", effort="60.0", velocity="6.0")
    ET.SubElement(drive, "dynamics", damping="0.5", friction="0.1")

    tip = ET.SubElement(robot, "link", name=TOOL_LINK)
    inertial(tip, 0.001, 0.001, 0.001, 0.001)
    fixed(robot, "socket_tip_joint", "socket_link", TOOL_LINK, xyz=f"{SHANK_FRONT + HEAD_LEN:.4g} 0 0")

    lidar = ET.SubElement(robot, "link", name="patrol_lidar")
    shape = ("cylinder", {"radius": "0.024", "length": "0.032"})
    geom(lidar, "visual", shape, material="lidar_black")
    geom(lidar, "visual", ("cylinder", {"radius": "0.0245", "length": "0.008"}), xyz="0 0 0.01", material="fixer_teal")
    inertial(lidar, 0.12, 0.048, 0.048, 0.032)
    fixed(robot, "patrol_lidar_mount", "base", "patrol_lidar", xyz=" ".join(f"{v:.4g}" for v in LIDAR_XYZ))

    plate = ET.SubElement(robot, "link", name="unit2_label")
    for side in (1, -1):
        label(plate, "UNIT 2", side)
    inertial(plate, 0.05, 0.2, 0.13, 0.06)
    fixed(robot, "unit2_label_mount", "base", "unit2_label")

    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=True)
    return DST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-container", action="store_true")
    args = ap.parse_args()
    out = build(drop_container=not args.keep_container)
    print(f"wrote {out.relative_to(HERE.parent)}  tool {TOOL_LINK}, lidar patrol_lidar at {LIDAR_XYZ}")


if __name__ == "__main__":
    main()

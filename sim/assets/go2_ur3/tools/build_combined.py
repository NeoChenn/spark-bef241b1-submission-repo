#!/usr/bin/env python3
"""Mount a Universal Robots arm on top of a Unitree Go2 and write one self-contained URDF package.

Inputs are the two upstream description folders; nothing in them is modified.

  python3 -I tools/build_combined.py \
      --go2     <unitree_ros>/robots/go2_description \
      --arm-dir <path>/ur_description        (contains urdf/<arm>.urdf and meshes/<arm>/...) \
      --arm ur3 --out . [--mount-xyz X Y Z] [--mount-yaw RAD] [--arm-mass-scale S] [--name go2_ur3]

Output (under --out):
  <name>.urdf            combined robot, mesh paths relative to the URDF file
  meshes/go2/*.dae       Go2 visual meshes (only the ones the URDF uses)
  meshes/<arm>/...       arm visual (.dae) and collision (.stl) meshes

What it changes in the arm URDF: prefixes every link/joint/transmission name with '<arm>_' (both robots
have a link called 'base'), drops the Gazebo-only <gazebo> blocks and the 'world' link/joint, and rewrites
package:// mesh paths to relative ones. The arm is attached with one fixed joint to the Go2 'base' link.
"""
import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

GO2_ROOT_LINK = "base"          # Go2 trunk link


def die(msg):
    sys.exit(f"error: {msg}")


def rename_go2_materials(root):
    """Go2 material names contain Chinese characters; map them to stable ASCII names."""
    names = sorted({m.get("name") for m in root.iter("material")
                    if m.get("name") and not m.get("name").isascii()})
    mapping = {n: f"go2_mat_{i:02d}" for i, n in enumerate(names)}
    for m in root.iter("material"):
        if m.get("name") in mapping:
            m.set("name", mapping[m.get("name")])
    return mapping


def retarget_meshes(root, package_prefix, source_subdir, new_prefix):
    """Rewrite package:// mesh URIs to paths relative to the URDF.

    package_prefix  e.g. 'package://go2_description/'
    source_subdir   folder (inside the upstream package) the meshes live in, e.g. 'dae/'
    new_prefix      where they end up next to the URDF, e.g. 'meshes/go2/'
    Returns the mesh paths relative to source_subdir (for copying).
    """
    needed = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if not fn.startswith(package_prefix + source_subdir):
            die(f"unexpected mesh path {fn!r} (expected {package_prefix + source_subdir!r}...)")
        rel = fn[len(package_prefix + source_subdir):]
        needed.append(rel)
        mesh.set("filename", new_prefix + rel)
    return needed


def drop_links(root, names):
    """Remove links (and any joints touching them), e.g. the arm's 'world' anchor."""
    names = set(names)
    for j in list(root.findall("joint")):
        if j.find("parent").get("link") in names or j.find("child").get("link") in names:
            root.remove(j)
    for l in list(root.findall("link")):
        if l.get("name") in names:
            root.remove(l)


def prefix_arm(root, prefix):
    """Prefix every arm link / joint / transmission name so nothing clashes with the Go2."""
    for link in root.findall("link"):
        link.set("name", prefix + link.get("name"))
    for joint in root.findall("joint"):
        joint.set("name", prefix + joint.get("name"))
        for tag in ("parent", "child"):
            el = joint.find(tag)
            el.set("link", prefix + el.get("link"))
    for tr in root.findall("transmission"):
        if tr.get("name"):
            tr.set("name", prefix + tr.get("name"))
        for j in tr.findall("joint"):
            j.set("name", prefix + j.get("name"))
        for a in tr.findall("actuator"):
            a.set("name", prefix + a.get("name"))


def scale_arm_mass(root, s):
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        m = inertial.find("mass")
        m.set("value", repr(float(m.get("value")) * s))
        inertia = inertial.find("inertia")
        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            inertia.set(k, repr(float(inertia.get(k)) * s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go2", required=True, type=Path)
    ap.add_argument("--arm-dir", required=True, type=Path, help="the ur_description folder")
    ap.add_argument("--arm", default="ur3", help="arm model name: urdf/<arm>.urdf and meshes/<arm>/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default=None, help="output name (default go2_<arm>)")
    # Default mount: the Go2 shell top is flat (z ~0.058-0.063) for x in [-0.09, 0.07] and rises toward
    # the front, measured from base.dae. x=-0.01, z=0.064 rests a small arm base on the shell without
    # sinking in, just behind the Go2's trunk centre of mass.
    ap.add_argument("--mount-xyz", nargs=3, type=float, default=[-0.01, 0.0, 0.064],
                    metavar=("X", "Y", "Z"), help="arm base origin in the Go2 'base' frame (m)")
    ap.add_argument("--mount-yaw", type=float, default=0.0, help="arm yaw about +Z in the Go2 base frame (rad)")
    ap.add_argument("--arm-mass-scale", type=float, default=1.0,
                    help="multiply all arm masses and inertias (simulation-only weight change)")
    ap.add_argument("--arm-root-link", default="base_link", help="arm link that is bolted to the Go2")
    ap.add_argument("--drop-links", nargs="*", default=["world"], help="arm links to remove (anchors)")
    a = ap.parse_args()

    prefix = a.arm + "_"
    name = a.name or f"go2_{a.arm}"
    go2_urdf = a.go2 / "urdf" / "go2_description.urdf"
    arm_urdf = a.arm_dir / "urdf" / f"{a.arm}.urdf"
    for p in (go2_urdf, arm_urdf):
        if not p.is_file():
            die(f"missing {p}")

    go2 = ET.parse(go2_urdf).getroot()
    arm = ET.parse(arm_urdf).getroot()

    mat_map = rename_go2_materials(go2)

    # Go2:  package://go2_description/dae/X.dae                 -> meshes/go2/X.dae
    # Arm:  package://ur_description/meshes/<arm>/visual/X.dae  -> meshes/<arm>/visual/X.dae
    go2_files = retarget_meshes(go2, "package://go2_description/", "dae/", "meshes/go2/")
    sub = f"meshes/{a.arm}/"
    arm_files = retarget_meshes(arm, "package://ur_description/", sub, sub)

    for g in arm.findall("gazebo"):          # Gazebo-only plugin/material blocks
        arm.remove(g)
    drop_links(arm, a.drop_links)
    if a.arm_root_link not in {l.get("name") for l in arm.findall("link")}:
        die(f"arm root link {a.arm_root_link!r} not found")
    children = {j.find("child").get("link") for j in arm.findall("joint")}
    if a.arm_root_link in children:
        die(f"{a.arm_root_link!r} still has a parent joint in the arm URDF; pass --drop-links for its anchor")

    prefix_arm(arm, prefix)
    if a.arm_mass_scale != 1.0:
        scale_arm_mass(arm, a.arm_mass_scale)

    for child in list(arm):                   # Go2 root + all arm links/joints/transmissions
        go2.append(child)

    mount = ET.SubElement(go2, "joint", {"name": f"go2_{a.arm}_mount", "type": "fixed"})
    ET.SubElement(mount, "origin", {"xyz": " ".join(repr(v) for v in a.mount_xyz),
                                    "rpy": f"0 0 {a.mount_yaw!r}"})
    ET.SubElement(mount, "parent", {"link": GO2_ROOT_LINK})
    ET.SubElement(mount, "child", {"link": prefix + a.arm_root_link})

    go2.set("name", name)
    ET.indent(go2, space="  ")

    out = a.out
    for rel in sorted(set(go2_files)):
        dst = out / "meshes" / "go2" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.go2 / "dae" / rel, dst)
    for rel in sorted(set(arm_files)):
        dst = out / "meshes" / a.arm / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.arm_dir / "meshes" / a.arm / rel, dst)

    header = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- Generated by tools/build_combined.py. Do not edit by hand; re-run the script.\n"
        "     Unitree Go2 (BSD-3-Clause, github.com/unitreerobotics/unitree_ros) with a\n"
        f"     Universal Robots {a.arm.upper()} on its back.\n"
        f"     Arm mount: xyz={a.mount_xyz} yaw={a.mount_yaw} rad in the Go2 'base' frame; "
        f"arm mass scale={a.arm_mass_scale}. -->\n"
    )
    (out / f"{name}.urdf").write_text(header + ET.tostring(go2, encoding="unicode") + "\n", encoding="utf-8")
    print(f"wrote {out / (name + '.urdf')}  ({len(mat_map)} Go2 material names made ASCII)")


if __name__ == "__main__":
    main()

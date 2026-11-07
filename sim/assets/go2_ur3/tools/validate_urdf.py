#!/usr/bin/env python3
"""Check a combined Go2 + UR arm URDF and report mass, centre of mass, tip-over margin and static joint torques.

  python3 -I tools/validate_urdf.py go2_ur3.urdf [--arm ur3] [--plot preview.png] [--pose NAME]

Checks: unique names, single root, connected tree, all links/meshes exist, no package:// URIs, ASCII materials.
Reports: mass split, forward kinematics of tool0, the combined centre of mass relative to the feet support
polygon, and a rough static estimate of the hip/thigh/knee torque needed just to stand, against the Go2's
URDF torque limits, for several arm poses. Needs numpy (and matplotlib for --plot).
This is a static geometry/mass check (vertical foot forces, minimum-norm load split). It is not a
dynamics, walking or self-collision test, so treat the torque numbers as orders of magnitude.
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
P2 = math.pi / 2

# Go2 standing pose: hips +/-0.1, thighs 0.8 front / 1.0 rear, calves -1.5
STAND = {}
for leg, hip, thigh in (("FL", 0.1, 0.8), ("FR", -0.1, 0.8), ("RL", 0.1, 1.0), ("RR", -0.1, 1.0)):
    STAND[f"{leg}_hip_joint"] = hip
    STAND[f"{leg}_thigh_joint"] = thigh
    STAND[f"{leg}_calf_joint"] = -1.5

# Joint conventions differ between UR description packages, so poses are listed per arm file flavour.
# ur3: robotics-toolbox ur_description. ur10: ros-industrial ur_description.
ARM_POSES = {
    "ur3": {
        "zero":    [0, 0, 0, 0, 0, 0],               # extended horizontally forward
        "up":      [0, -P2, 0, -P2, 0, 0],           # straight up
        "forward": [0, -P2, P2, -P2, 0, 0],          # upper arm vertical, forearm forward (L shape)
        "folded":  [0, -P2, -P2, P2, 0, 0],          # forearm folded back over the trunk
    },
    "ur10": {
        "zero":    [0, 0, 0, 0, 0, 0],
        "up":      [0, -P2, 0, -P2, 0, 0],
        "forward": [0, -P2, P2, -P2, 0, 0],
    },
}


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_axis(axis, q):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    x, y, z = a / n
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def floats(s, n):
    return [float(v) for v in (s or " ".join(["0"] * n)).split()]


class Robot:
    def __init__(self, path):
        self.path = Path(path)
        self.root = ET.parse(path).getroot()
        self.links, self.joints, self.dup = {}, {}, []
        for l in self.root.findall("link"):
            if l.get("name") in self.links:
                self.dup.append(l.get("name"))
            self.links[l.get("name")] = l
        for j in self.root.findall("joint"):
            if j.get("name") in self.joints:
                self.dup.append(j.get("name"))
            self.joints[j.get("name")] = j
        self.parent_of = {}
        self.children = {n: [] for n in self.links}
        for name, j in self.joints.items():
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            self.parent_of[c] = (name, p)
            if p in self.children:
                self.children[p].append((name, c))
        roots = [n for n in self.links if n not in self.parent_of]
        self.root_link = roots[0] if roots else None
        self.roots = roots

    def checks(self):
        ok = True

        def bad(msg):
            nonlocal ok
            ok = False
            print("  FAIL:", msg)
        if self.dup:
            bad(f"duplicate names: {self.dup}")
        for name, j in self.joints.items():
            for tag in ("parent", "child"):
                if j.find(tag).get("link") not in self.links:
                    bad(f"joint {name} {tag} link {j.find(tag).get('link')!r} does not exist")
        if len(self.roots) != 1:
            bad(f"expected one root link, found {self.roots}")
        else:
            seen, stack = set(), [self.roots[0]]
            while stack:
                n = stack.pop()
                seen.add(n)
                stack.extend(c for _, c in self.children[n])
            if seen != set(self.links):
                bad(f"links not connected to root: {sorted(set(self.links) - seen)}")
        n_mesh = 0
        for m in self.root.iter("mesh"):
            fn = m.get("filename", "")
            n_mesh += 1
            if fn.startswith("package://") or fn.startswith("/"):
                bad(f"non-relative mesh path {fn}")
            elif not (self.path.parent / fn).is_file():
                bad(f"missing mesh file {fn}")
        nonascii = [m.get("name") for m in self.root.iter("material") if m.get("name") and not m.get("name").isascii()]
        if nonascii:
            bad(f"non-ASCII material names: {nonascii[:3]}")
        if self.root.find("gazebo") is not None:
            bad("leftover <gazebo> blocks")
        print(f"  {len(self.links)} links, {len(self.joints)} joints, {n_mesh} mesh refs, root='{self.root_link}'")
        return ok

    def fk(self, q):
        """Return {link: 4x4 world transform} for joint values q (dict name -> rad); missing = 0."""
        T = {self.root_link: np.eye(4)}
        stack = [self.root_link]
        while stack:
            p = stack.pop()
            for jname, c in self.children[p]:
                j = self.joints[jname]
                o = j.find("origin")
                A = np.eye(4)
                A[:3, :3] = rot_rpy(*floats(o.get("rpy") if o is not None else None, 3))
                A[:3, 3] = floats(o.get("xyz") if o is not None else None, 3)
                if j.get("type") in ("revolute", "continuous"):
                    ax = j.find("axis")
                    axis = floats(ax.get("xyz") if ax is not None else "1 0 0", 3)
                    J = np.eye(4)
                    J[:3, :3] = rot_axis(axis, q.get(jname, 0.0))
                    A = A @ J
                T[c] = T[p] @ A
                stack.append(c)
        return T

    def masses(self):
        out = {}
        for name, l in self.links.items():
            i = l.find("inertial")
            if i is None:
                continue
            o = i.find("origin")
            out[name] = (float(i.find("mass").get("value")),
                         np.array(floats(o.get("xyz") if o is not None else None, 3)))
        return out

    def com(self, T, select=lambda n: True):
        tot, acc = 0.0, np.zeros(3)
        for n, (m, c) in self.masses().items():
            if select(n):
                tot += m
                acc += m * (T[n] @ np.append(c, 1.0))[:3]
        return tot, acc / tot

    def joint_axis_world(self, T, jname):
        j = self.joints[jname]
        ax = j.find("axis")
        axis = np.array(floats(ax.get("xyz") if ax is not None else "1 0 0", 3))
        return T[j.find("child").get("link")][:3, :3] @ axis

    def effort_limit(self, jname):
        lim = self.joints[jname].find("limit")
        return float(lim.get("effort")) if lim is not None and lim.get("effort") else float("nan")


LEGS = ("FL", "FR", "RL", "RR")


def foot_loads(r, T, M, com):
    """Vertical foot forces (N) that balance weight and moments about the CoM; minimum-norm split."""
    P = np.array([T[f"{l}_foot"][:2, 3] for l in LEGS])
    A = np.vstack([np.ones(4), P[:, 0] - com[0], P[:, 1] - com[1]])
    return np.linalg.lstsq(A, np.array([M * 9.81, 0.0, 0.0]), rcond=None)[0]


def static_torques(r, T, F):
    """Worst |torque| (N m) per joint type needed to carry the foot loads F (vertical forces only)."""
    worst = {"hip": 0.0, "thigh": 0.0, "calf": 0.0}
    for leg, f in zip(LEGS, F):
        foot = T[f"{leg}_foot"][:3, 3]
        for jn in worst:
            name = f"{leg}_{jn}_joint"
            axis = r.joint_axis_world(T, name)
            jpos = T[r.joints[name].find("child").get("link")][:3, 3]
            tau = np.dot(axis, np.cross(foot - jpos, np.array([0.0, 0.0, f])))
            worst[jn] = max(worst[jn], abs(tau))
    return worst


def hull(points):
    pts = sorted(set(map(tuple, points)))

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def margin(poly, pt):
    """Signed distance (m) from pt to the polygon boundary: + inside, - outside (CCW polygon)."""
    best = 1e9
    for i in range(len(poly)):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        best = min(best, float(np.dot(np.array(pt) - a, n)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--arm", default="ur3", choices=list(ARM_POSES), help="arm name used as link prefix")
    ap.add_argument("--plot", help="write a side/top stick-figure PNG")
    ap.add_argument("--pose", default="forward", help="arm pose to draw")
    a = ap.parse_args()
    prefix = a.arm + "_"
    poses = ARM_POSES[a.arm]
    if a.pose not in poses:
        sys.exit(f"--pose must be one of {list(poses)}")

    r = Robot(a.urdf)
    print("Structure checks")
    ok = r.checks()
    if r.root_link is None or len(r.roots) != 1:
        sys.exit("cannot continue without a single root link")

    ms = r.masses()
    arm_m = sum(m for n, (m, _) in ms.items() if n.startswith(prefix))
    go2_m = sum(m for n, (m, _) in ms.items() if not n.startswith(prefix))
    print(f"\nMass: Go2 {go2_m:.2f} kg + arm {arm_m:.2f} kg = {go2_m + arm_m:.2f} kg  (arm is {arm_m / go2_m:.2f}x the Go2)")

    limits = {jn: r.effort_limit(f"FL_{jn}_joint") for jn in ("hip", "thigh", "calf")}
    print("Go2 torque limits from the URDF (N m): hip {hip:.1f}, thigh {thigh:.1f}, calf {calf:.1f}".format(**limits))

    feet = [f"{l}_foot" for l in LEGS]
    print("\nStanding pose, per arm pose. margin = distance of the combined CoM inside the feet polygon (m).")
    print("Torques are the worst leg's static torque (hip / thigh / calf, N m); '!' marks a value over its limit.")
    print(f"{'arm pose':9s} {'tool0 xyz in Go2 base frame (m)':32s} {'CoM xyz (m)':24s} {'margin':>7s}  {'static torque hip / thigh / calf':>34s}")
    results = {}

    def report(name, T, com, M, tool):
        poly = hull([T[f][:2, 3] for f in feet])
        mg = margin(poly, com[:2])
        F = foot_loads(r, T, M, com)
        tq = static_torques(r, T, F)
        flag = {k: ("!" if tq[k] > limits[k] else " ") for k in tq}
        tool_s = np.array2string(tool, precision=2, suppress_small=True) if tool is not None else "-"
        print(f"{name:9s} {tool_s:32s} {np.array2string(com, precision=3, suppress_small=True):24s} {mg:+7.3f}  "
              f"{tq['hip']:7.1f}{flag['hip']} / {tq['thigh']:5.1f}{flag['thigh']} / {tq['calf']:5.1f}{flag['calf']}"
              f"{'  (min foot load ' + format(F.min(), '.0f') + ' N' + (' LIFTS' if F.min() < 0 else '') + ')'}")
        return poly

    # Go2 alone baseline (arm links ignored)
    T0 = r.fk(dict(STAND))
    M0, com0 = r.com(T0, lambda n: not n.startswith(prefix))
    report("Go2 only", T0, com0, M0, None)
    for name, qv in poses.items():
        q = dict(STAND)
        q.update({prefix + j: v for j, v in zip(ARM_JOINTS, qv)})
        T = r.fk(q)
        M, com = r.com(T)
        tool = T[prefix + "tool0"][:3, 3]
        poly = report(name, T, com, M, tool)
        results[name] = (T, com, poly)
    print(f"\nLowest foot sits {-min(T0[f][2, 3] for f in feet):.3f} m below the Go2 base origin; "
          f"Go2-only CoM is {(-min(T0[f][2, 3] for f in feet) + com0[2]):.2f} m above the ground.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        T, com, poly = results[a.pose]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, (i, j, title) in zip(axs, ((0, 2, f"side view (x forward, z up), arm pose '{a.pose}'"),
                                          (0, 1, "top view (x forward, y left)"))):
            for jname, jel in r.joints.items():
                p, c = jel.find("parent").get("link"), jel.find("child").get("link")
                if p not in T or c not in T:
                    continue
                pa, ca = T[p][:3, 3], T[c][:3, 3]
                col = "tab:orange" if c.startswith(prefix) else "tab:blue"
                ax.plot([pa[i], ca[i]], [pa[j], ca[j]], color=col, lw=2 if col == "tab:orange" else 1.4)
            bx, bz, by = 0.3762 / 2, 0.114 / 2, 0.0935 / 2
            if (i, j) == (0, 2):
                ax.plot([-bx, bx, bx, -bx, -bx], [-bz, -bz, bz, bz, -bz], color="k", lw=1)
            else:
                ax.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], color="k", lw=1)
                pp = poly + [poly[0]]
                ax.plot([p[0] for p in pp], [p[1] for p in pp], "g--", lw=1, label="feet support polygon")
            ax.plot([com[i]], [com[j]], "r*", ms=13, label="combined CoM")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)" if j == 2 else "y (m)")
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{Path(a.urdf).name}  (orange = {a.arm.upper()}, blue = Go2; static check, no collision test)", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("wrote", a.plot)

    print("\nRESULT:", "structure OK" if ok else "STRUCTURE PROBLEMS (see FAIL lines)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

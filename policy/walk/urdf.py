from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

PAYLOAD_LINKS = ("ur3_", "lift_", "scrubber", "brush")


def strip_collisions(text: str, prefixes: tuple[str, ...], mesh_dir: str | None = None) -> str:
    root = ET.fromstring(text)
    for link in root.findall("link"):
        if link.get("name", "").startswith(prefixes):
            for c in link.findall("collision"):
                link.remove(c)
    if mesh_dir:
        for m in root.iter("mesh"):
            fn = m.get("filename", "")
            if fn and not os.path.isabs(fn) and "://" not in fn:
                m.set("filename", os.path.join(mesh_dir, fn))
    return ET.tostring(root, encoding="unicode")


def walk_urdf(src: str, out_dir: str | None = None) -> str:
    out_dir = out_dir or tempfile.gettempdir()
    dst = os.path.join(out_dir, "go2_ur3_lift_walk.urdf")
    with open(src) as f:
        text = strip_collisions(f.read(), PAYLOAD_LINKS, os.path.dirname(os.path.abspath(src)))
    with open(dst, "w") as f:
        f.write(text)
    return dst

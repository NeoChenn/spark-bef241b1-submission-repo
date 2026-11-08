from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

HOST = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"


def keys(prefix: str) -> list[str]:
    xml = urllib.request.urlopen(f"{HOST}/?prefix={prefix}").read().decode()
    return re.findall(r"<Key>([^<]*)</Key>", xml)


def root_usd(folder: str) -> str:
    rel = folder.removeprefix(HOST + "/")
    found = [k for k in keys(rel) if re.fullmatch(re.escape(rel) + r"[^/]+\.usd", k)]
    return f"{HOST}/{sorted(found)[0]}"


def mirror(url: str, cache: Path) -> Path:
    if url.endswith("/"):
        url = root_usd(url)
    rel = url.removeprefix(HOST + "/")
    folder = rel.rsplit("/", 1)[0] + "/"
    for key in keys(folder):
        if re.search(r"\.usd[ac]?$", key) and "/.thumbs/" not in key:
            dst = cache / key
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{HOST}/{key}", dst)
    return cache / rel, url


def bounds(path: Path) -> dict:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        vs = prim.GetVariantSets()
        if "PhysX" in vs.GetNames():
            vs.GetVariantSet("PhysX").SetVariantSelection("Disabled")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return {"min": [round(v, 4) for v in box.GetMin()], "max": [round(v, 4) for v in box.GetMax()],
            "mpu": UsdGeom.GetStageMetersPerUnit(stage), "up": UsdGeom.GetStageUpAxis(stage)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("items", nargs="+")
    args = ap.parse_args()
    data = json.loads(args.out.read_text()) if args.out.exists() else {}
    for item in args.items:
        name, url = item.split("=", 1)
        path, full = mirror(url, args.cache)
        data[name] = {**bounds(path), "url": full}
        print(name, data[name])
    args.out.write_text(json.dumps(data, indent=1) + "\n")

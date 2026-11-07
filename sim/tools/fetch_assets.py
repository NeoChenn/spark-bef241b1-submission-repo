"""Download the third-party assets listed in sim/assets/manifest.json.

Uses only the Python standard library, so any Python 3.9+ works:

    python sim/tools/fetch_assets.py

Files land in sim/assets/third_party/, which Git ignores. Each file is checked
against the MD5 published by the Poly Haven API, and files that already match
are skipped, so the script is safe to re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

API = "https://api.polyhaven.com/files/"
# Poly Haven asks API clients to send a unique User-Agent.
USER_AGENT = "spark-hackathon-asset-fetch/1.0"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
TEXTURE_MAPS = {"diff": "Diffuse", "nor_gl": "nor_gl", "rough": "Rough", "metal": "Metal", "ao": "AO"}


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def safe_target(root: Path, relative: str) -> Path:
    """Resolve a path from the API, refusing anything that leaves ``root``."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe path from API: {relative!r}")
    return root.joinpath(*parts)


def download(entry: dict, target: Path) -> int:
    """Fetch one file unless a verified copy exists. Returns bytes downloaded."""
    expected = entry["md5"]
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == expected:
        return 0
    data = get(entry["url"])
    actual = hashlib.md5(data).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {entry['url']}: {actual} != {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def fetch_model(asset_id: str, resolution: str, out: Path) -> int:
    files = json.loads(get(API + asset_id))
    usd = files["usd"][resolution]["usd"]
    root = out / "models" / asset_id
    total = download(usd, root / PurePosixPath(usd["url"]).name)
    for relative, entry in usd.get("include", {}).items():
        total += download(entry, safe_target(root, relative))
    # Some USD packages (barrel_03, for one) omit a roughness map they reference.
    rough = files.get("Rough", {}).get(resolution, {}).get("exr")
    if rough:
        total += download(rough, root / "textures" / PurePosixPath(rough["url"]).name)
    return total


def fetch_texture(asset_id: str, resolution: str, out: Path) -> int:
    files = json.loads(get(API + asset_id))
    total = 0
    for name, key in TEXTURE_MAPS.items():
        if key in files:
            total += download(
                files[key][resolution]["jpg"],
                out / "textures" / asset_id / f"{name}.jpg",
            )
    return total


def fetch_hdri(asset_id: str, resolution: str, out: Path) -> int:
    files = json.loads(get(API + asset_id))
    return download(files["hdri"][resolution]["hdr"], out / "hdris" / f"{asset_id}.hdr")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=ASSETS_DIR / "manifest.json")
    parser.add_argument("--out", type=Path, default=ASSETS_DIR / "third_party")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = (
        [(fetch_model, asset, manifest["resolution"]) for asset in manifest["models"]]
        + [(fetch_texture, asset, manifest["resolution"]) for asset in manifest["textures"]]
        + [(fetch_hdri, asset, manifest["hdri_resolution"]) for asset in manifest["hdris"]]
    )

    total = 0
    failures = []
    for fetcher, asset, resolution in jobs:
        try:
            size = fetcher(asset, resolution, args.out)
            total += size
            print(f"ok   {asset} ({size / 1e6:.1f} MB new)")
        except (OSError, KeyError, ValueError) as error:
            failures.append(asset)
            print(f"FAIL {asset}: {error}", file=sys.stderr)

    print(f"Downloaded {total / 1e6:.1f} MB into {args.out}")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

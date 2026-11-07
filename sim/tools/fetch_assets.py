"""Download the third-party assets listed in sim/assets/manifest.json.

Uses only the Python standard library, so any Python 3.9+ works:

    python sim/tools/fetch_assets.py                  # everything
    python sim/tools/fetch_assets.py --kind hdri      # one kind
    python sim/tools/fetch_assets.py --only Rust007   # one asset
    python sim/tools/fetch_assets.py --licences       # rewrite sim/assets/LICENSES.md

Files land in sim/assets/third_party/<path>. Poly Haven files are checked against
the API's MD5 and skipped when they match; ambientCG and direct downloads are skipped
when the target already exists. Entries marked "remote" (NVIDIA Omniverse content)
are loaded by URL at runtime and never downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from asset_manifest import (
    Asset,
    ambientcg_map_name,
    ambientcg_zip_url,
    licences_markdown,
    load_assets,
    polyhaven_hdri_file,
    polyhaven_model_files,
    polyhaven_texture_files,
    safe_target,
    select,
)

PH_API = "https://api.polyhaven.com/files/"
USER_AGENT = "spark-hackathon-asset-fetch/1.0"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download_md5(url: str, md5: str, target: Path) -> int:
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == md5:
        return 0
    data = get(url)
    actual = hashlib.md5(data).hexdigest()
    if actual != md5:
        raise ValueError(f"checksum mismatch for {url}: {actual} != {md5}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def fetch_polyhaven(asset: Asset, out: Path) -> int:
    files = json.loads(get(PH_API + asset.upstream_id))
    root = safe_target(out, asset.path)
    if asset.kind == "texture":
        jobs = polyhaven_texture_files(files, asset.resolution)
    elif asset.kind == "model":
        jobs = polyhaven_model_files(files, asset.resolution)
    else:
        url, md5, name = polyhaven_hdri_file(files, asset.resolution, asset.path)
        jobs, root = [(url, md5, name)], root.parent if asset.path.endswith(".hdr") else root
    return sum(download_md5(url, md5, safe_target(root, rel)) for url, md5, rel in jobs)


def fetch_ambientcg(asset: Asset, out: Path) -> int:
    root = safe_target(out, asset.path)
    if (root / "diff.jpg").exists():
        return 0
    data = get(asset.download or ambientcg_zip_url(asset.upstream_id, asset.resolution))
    written = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.namelist():
            name = ambientcg_map_name(member)
            if name:
                root.mkdir(parents=True, exist_ok=True)
                written += (root / name).write_bytes(archive.read(member))
    if not (root / "diff.jpg").exists():
        raise ValueError(f"no colour map in {asset.id} archive")
    return written


def fetch_direct(asset: Asset, out: Path) -> int:
    target = safe_target(out, asset.path)
    if target.exists():
        return 0
    data = get(asset.download or asset.url)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.write_bytes(data)


FETCHERS = {"polyhaven": fetch_polyhaven, "ambientcg": fetch_ambientcg}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=ASSETS_DIR / "manifest.json")
    parser.add_argument("--out", type=Path, default=ASSETS_DIR / "third_party")
    parser.add_argument("--kind", action="append", help="texture, hdri, model or decal (repeatable)")
    parser.add_argument("--only", action="append", help="asset id (repeatable)")
    parser.add_argument("--licences", action="store_true", help="write LICENSES.md and exit")
    args = parser.parse_args()

    assets = load_assets(json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.licences:
        target = args.manifest.parent / "LICENSES.md"
        target.write_text(licences_markdown(assets), encoding="utf-8")
        print(f"Wrote {target}")
        return 0

    total, failures = 0, []
    for asset in select(assets, set(args.kind or []), set(args.only or [])):
        try:
            size = FETCHERS.get(asset.source, fetch_direct)(asset, args.out)
            total += size
            print(f"ok   {asset.kind:7} {asset.id} ({size / 1e6:.1f} MB new)")
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            failures.append(asset.id)
            print(f"FAIL {asset.id}: {error}", file=sys.stderr)

    print(f"Downloaded {total / 1e6:.1f} MB into {args.out}")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

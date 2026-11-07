from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = json.loads((ROOT / "sim" / "assets" / "rig_model" / "candidates.json").read_text())


def get(url: str, headers: dict | None = None) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as response:
        return response.read()


def fetch(key: str) -> Path:
    entry = CANDIDATES[key]
    dest = ROOT / ".work" / "sources" / key
    dest.mkdir(parents=True, exist_ok=True)
    if "direct" in entry:
        target = dest / entry["direct"].rsplit("/", 1)[1]
        target.write_bytes(get(entry["direct"]))
        return target
    token = os.environ.get("SKETCHFAB_TOKEN")
    if not token:
        raise SystemExit(f"set SKETCHFAB_TOKEN, or download {entry['url']} by hand (glTF) into {dest}")
    api = f"https://api.sketchfab.com/v3/models/{entry['sketchfab_uid']}/download"
    links = json.loads(get(api, {"Authorization": f"Token {token}"}))
    archive = dest / "gltf.zip"
    archive.write_bytes(get(links["gltf"]["url"]))
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    return next(dest.rglob("*.gltf"))


if __name__ == "__main__":
    print(fetch(sys.argv[1]))

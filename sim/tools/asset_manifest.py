"""Pure manifest logic for sim/assets/manifest.json: parsing, URLs and file naming."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

KINDS = ("texture", "hdri", "model", "decal")
REQUIRED = ("id", "kind", "source", "url", "licence", "resolution", "path")
PH_TEXTURE_MAPS = {
    "Diffuse": "diff",
    "nor_gl": "nor_gl",
    "Rough": "rough",
    "Metal": "metal",
    "AO": "ao",
    "Displacement": "disp",
    "Opacity": "opacity",
}
ACG_MAPS = {
    "Color": "diff",
    "NormalGL": "nor_gl",
    "Roughness": "rough",
    "Metalness": "metal",
    "AmbientOcclusion": "ao",
    "Displacement": "disp",
    "Opacity": "opacity",
}
ACG_MEMBER = re.compile(r"^[A-Za-z0-9]+_\d+K-JPG_([A-Za-z]+)\.jpg$")


@dataclass(frozen=True)
class Asset:
    id: str
    kind: str
    source: str
    url: str
    licence: str
    resolution: str
    path: str
    category: str = ""
    download: str = ""
    remote: bool = False
    note: str = ""
    source_id: str = ""

    @property
    def upstream_id(self) -> str:
        return self.source_id or self.id


def _legacy(data: dict) -> list[dict]:
    def entry(asset_id: str, kind: str, resolution: str, path: str) -> dict:
        return {
            "id": asset_id,
            "kind": kind,
            "source": "polyhaven",
            "url": f"https://polyhaven.com/a/{asset_id}",
            "licence": "CC0-1.0",
            "resolution": resolution,
            "path": path,
        }

    return (
        [entry(a, "model", data["resolution"], f"models/{a}") for a in data.get("models", [])]
        + [entry(a, "texture", data["resolution"], f"textures/{a}") for a in data.get("textures", [])]
        + [entry(a, "hdri", data["hdri_resolution"], f"hdris/{a}.hdr") for a in data.get("hdris", [])]
    )


def load_assets(data: dict) -> list[Asset]:
    raw = data["assets"] if data.get("version") == 2 else _legacy(data)
    assets, seen = [], set()
    for item in raw:
        for field in REQUIRED:
            if not item.get(field):
                raise ValueError(f"asset {item.get('id', '?')}: missing {field}")
        if item["kind"] not in KINDS:
            raise ValueError(f"asset {item['id']}: unknown kind {item['kind']!r}")
        if item["id"] in seen:
            raise ValueError(f"duplicate asset id {item['id']}")
        seen.add(item["id"])
        assets.append(Asset(**{k: v for k, v in item.items() if k in Asset.__dataclass_fields__}))
    return assets


def select(assets: list[Asset], kinds: set[str] | None = None, ids: set[str] | None = None) -> list[Asset]:
    return [
        a
        for a in assets
        if not a.remote and (not kinds or a.kind in kinds) and (not ids or a.id in ids)
    ]


def safe_target(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe path: {relative!r}")
    return root.joinpath(*parts)


def ambientcg_zip_url(asset_id: str, resolution: str) -> str:
    return f"https://ambientcg.com/get?file={asset_id}_{resolution.upper()}-JPG.zip"


def ambientcg_map_name(member: str) -> str | None:
    match = ACG_MEMBER.match(member)
    if not match or match.group(1) not in ACG_MAPS:
        return None
    return ACG_MAPS[match.group(1)] + ".jpg"


def polyhaven_texture_files(files: dict, resolution: str) -> list[tuple[str, str, str]]:
    out = []
    for key, name in PH_TEXTURE_MAPS.items():
        entry = files.get(key, {}).get(resolution, {})
        fmt = "jpg" if "jpg" in entry else "png" if "png" in entry else None
        if fmt:
            out.append((entry[fmt]["url"], entry[fmt]["md5"], f"{name}.{fmt}"))
    return out


def polyhaven_hdri_file(files: dict, resolution: str, path: str) -> tuple[str, str, str]:
    entry = files["hdri"][resolution]["hdr"]
    name = PurePosixPath(path).name if path.endswith(".hdr") else PurePosixPath(entry["url"]).name
    return entry["url"], entry["md5"], name


def polyhaven_model_files(files: dict, resolution: str) -> list[tuple[str, str, str]]:
    usd = files["usd"][resolution]["usd"]
    out = [(usd["url"], usd["md5"], PurePosixPath(usd["url"]).name)]
    out += [(e["url"], e["md5"], rel) for rel, e in usd.get("include", {}).items()]
    rough = files.get("Rough", {}).get(resolution, {}).get("exr")
    if rough:
        target = f"textures/{PurePosixPath(rough['url']).name}"
        if target not in {t for _, _, t in out}:
            out.append((rough["url"], rough["md5"], target))
    return out


def licences_markdown(assets: list[Asset]) -> str:
    order = {k: i for i, k in enumerate(KINDS)}
    rows = sorted(assets, key=lambda a: (order[a.kind], a.id.lower()))
    lines = [
        "# Third-party asset licences",
        "",
        "Generated from `sim/assets/manifest.json` by `sim/tools/fetch_assets.py --licences`.",
        "",
        "| id | kind | source | licence | url |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {a.id} | {a.kind} | {a.source} | {a.licence} | {a.url} |" for a in rows]
    return "\n".join(lines) + "\n"

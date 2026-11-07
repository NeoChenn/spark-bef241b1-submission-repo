# Spark submission

A coordinated robot crew that repairs corroded pipe on an offshore rig, built in NVIDIA Isaac Sim 6.2.

## Open the rig deck

1. Clone this repo, or pull the latest changes.
2. Open Isaac Sim 6.2.
3. Choose **File > Open** and select `sim/output/rig_deck.usda`.

The first load can take a few minutes while shaders build.

## Rebuild the scene

Run this from the repo root. Changing the seed moves the pipe damage.

```bat
C:\isaacsim\python.bat sim\build_rig_deck.py --seed 7
```

This rewrites `sim/output/`, including the preview images. `sim/output/rig_deck_meta.json` gives the
damage location, work zone and cameras for the robotics and training teams.

## Assets

Everything in `sim/assets/third_party/` comes from [Poly Haven](https://polyhaven.com/license) under
CC0, so it's free to use and redistribute. To add an asset, put its Poly Haven ID in
`sim/assets/manifest.json`, then run `python sim/tools/fetch_assets.py`.

## Layout

| Path | Contents |
|---|---|
| `sim/build_rig_deck.py` | Builds the scene and renders previews |
| `sim/assets/` | Asset list and downloaded models, textures and skies |
| `sim/output/` | Generated scene, metadata and preview images |
| `sim/tools/` | Asset download and inspection scripts |

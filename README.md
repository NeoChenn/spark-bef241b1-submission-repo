# Offshore rig sim

Isaac Sim running headless in Docker on a GB10 (watson), streaming a camera view to a web page at
`https://rig.zeecloud.org`. This first milestone is a test world (ground, lights, a few shapes, one spinning red cube,
a fixed camera) to prove the plumbing. The rig itself, quadrupeds and policies come later.

## Architecture

```
sim (Isaac Sim 6.2.0, GPU)  --JPEG over TCP 127.0.0.1:9000-->  web (FastAPI, 127.0.0.1:8080)  <--  tunnel (cloudflared)  <--  browser
```

- `sim`: `rig/sim/main.py` builds the scene (`rig/sim/scene.py`), reads the camera's RGB annotator, JPEG-encodes and
  sends frames with `rig/sim/sender.py`. Isaac's WebRTC livestream does not work on aarch64, hence our own path.
- `web`: `rig/web/server.py` keeps only the latest frame and pushes it to browsers over `/ws/video`.
  `/api/health` and `/api/frame.jpg` are for checks.
- `tunnel`: compose profile `public`, started only when `.env` has `TUNNEL_TOKEN`.

All three use host networking and bind to loopback only.

## Local dev (Mac)

```bash
uv run pytest
uv run uvicorn rig.web.server:app --port 8080   # page at http://localhost:8080, waits for a sim
```

## Run on watson

Everything happens inside `~/Documents/spark-bef241b1-submission-repo`. Edit on the Mac, push, `git pull` on watson.

```bash
ssh watson
cd ~/Documents && git clone https://github.com/NeoChenn/spark-bef241b1-submission-repo.git   # first time
cd spark-bef241b1-submission-repo && git pull
scripts/remote.sh check    # driver, docker, GPU (CDI), image, ports, tunnel token. Reports only.
scripts/remote.sh up       # builds and starts web + sim (+ tunnel if .env has a token)
scripts/remote.sh status   # containers, /api/health, last sim fps line
scripts/remote.sh logs sim
scripts/remote.sh frame logs/frame.jpg
scripts/remote.sh down     # keeps the shader cache volumes
```

**First start:** Isaac compiles shaders for several minutes. Frames are black until then (`mean=` near 0 in the sim's
`fps=` log line). The `rig-isaac-*` volumes keep the cache, so later starts are quick.

**Without the tunnel:** `ssh -N -L 8080:127.0.0.1:8080 watson`, then open http://localhost:8080. Do not use
`brev port-forward` (it leaves a persistent binding on the node).

## Cloudflare setup (one time)

1. Zero Trust → Networks → Tunnels → create a tunnel (cloudflared), e.g. `rig-watson`. Copy its token.
2. Public hostname: `rig.zeecloud.org` → service `http://localhost:8080`.
3. Access → Applications → self-hosted app for `rig.zeecloud.org` with an email allow policy.
4. On watson: `cp .env.example .env` and set `TUNNEL_TOKEN=...`. Never commit `.env`.
5. `scripts/remote.sh up`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unresolvable CDI devices nvidia.com/gpu=all` | Host needs `nvidia-ctk cdi generate` (sudo). Ask before changing the host. |
| Black frames after 10+ minutes | Try the faster renderer: set the sim `command:` to add `--renderer RaytracedLighting`, then `up`. |
| Page says "waiting for sim" | `scripts/remote.sh logs sim`: look for a Traceback, or `connected=False` (web not up). |
| Low fps | Lower `--fps`, or use `RaytracedLighting`. |
| `Permission denied` under `/logs` | `chmod 777 logs` (Isaac runs as uid 1234). `remote.sh up` does this. |

## Rig deck scene

A first offshore rig deck for Isaac Sim 6.2, separate from the Docker test world above.

1. Open Isaac Sim 6.2.
2. Choose **File > Open** and select `sim/output/rig_deck.usda`.

The first load can take a few minutes while shaders build.

To rebuild the scene, run this from the repo root. Changing the seed moves the pipe damage.

```bat
C:\isaacsim\python.bat sim\build_rig_deck.py --seed 7
```

This rewrites `sim/output/`, including the preview images. `sim/output/rig_deck_meta.json` gives the damage location,
work zone and cameras for the robotics and training teams.

Everything in `sim/assets/third_party/` comes from [Poly Haven](https://polyhaven.com/license) under CC0, so it's free
to use and redistribute. To add an asset, put its Poly Haven ID in `sim/assets/manifest.json`, then run
`python sim/tools/fetch_assets.py`.

| Path | Contents |
|---|---|
| `sim/build_rig_deck.py` | Builds the scene and renders previews |
| `sim/assets/` | Asset list and downloaded models, textures and skies |
| `sim/output/` | Generated scene, metadata and preview images |
| `sim/tools/` | Asset download and inspection scripts |

## Full oil rig scene

A textured, full-scale rig for Isaac Sim 6.2. Open `sim/output/full_rig.usda` with **File > Open**, or rebuild it from the
repo root:

```bat
C:\isaacsim\python.bat sim\build_full_rig.py
```

The rig model is [Oil Rig](https://sketchfab.com/3d-models/oil-rig-2a9fadae585f4bbfa030772e54e64ace) by AnsysLearn under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), split, rescaled, textured and placed at sea for this project.
See `sim/assets/third_party/oil_rig_ansyslearn/CREDITS.md`. The source gives no real-world size, so the 0.25 m per unit
scale is an estimate. `sim/output/full_rig_meta.json` lists the rig's bounds, deck heights, parts, valves, gauges,
true gauge readings and a patrol route checked against the geometry.

`sim/rig_details.py` adds the process pipes, valves, gauges, props, helideck markings and windows.
`sim/tools/make_generated_assets.py` makes the files in `sim/assets/generated/`: an ocean sky with the coastline removed
from Poly Haven's CC0 `aristea_wreck`, a sea ripple map and a gauge dial.

## Rust detection and repair (branch `rust-detection`)

A simulated quadruped with a scrubber arm finds rust on pipework and fixes it:
detect → spray → wait 30 s → scrub (PPO) → re-detect → paint → verify. Training runs on Modal (L40S).

| Path | What |
|---|---|
| `detection/` | Rust classifier (LoRA fine-tune), patch heatmap localizer, ONNX export. See `detection/DATA.md`. |
| `policy/envs/` | Isaac Lab scrub environment + rust-state model (pure torch). |
| `policy/skills/` | Classical spray / paint coverage skills. |
| `policy/orchestrator.py` | Perception-driven state machine that sequences the repair. |
| `infra/` | Shared Modal image + volumes for training. |
| `integration/INTERFACE.md` | Contract with the rig scene (prim paths, rust texture, robot). |

Setup: `uv sync` (tests: `uv run pytest`); add `--group train` for Modal training and data prep tools.

scale is an estimate. `sim/output/full_rig_meta.json` lists the rig's bounds, deck heights and parts.

## Importing a full rig model

`scripts/import_model.sh` (on watson) turns any model into `sim/assets/rig_model/<name>/<name>.usd`. The output is in metres,
Z up, with the main deck at z = 0 and the topside centred. It also writes `textures/` with relative paths,
UsdPreviewSurface materials, static triangle-mesh collision on walkable meshes, and `report.json`. The report has bounds,
triangle, mesh and material counts, and lists missing textures.

```bash
scripts/import_model.sh path/in/repo/rig.glb --name rig                 # glb gltf fbx obj stl ply usd usdz step stp iges
scripts/import_model.sh path/in/repo/rig.blend --name rig               # blend dae 3ds x3d abc: Blender (rig-import-blender) -> GLB first
scripts/import_model.sh rig.fbx --name rig --units cm --up Y --deck-z 32 --drop 'sea|boat'   # overrides when detection is wrong
/isaac-sim/python.sh sim/tools/preview_model.py sim/assets/rig_model/rig/rig.usd --sea-z -20   # in rig-sim:6.2.0; wide + deck PNGs
```

**Converters on aarch64 (Isaac 6.2):**
- `omni.kit.asset_converter` (assimp) handles glTF/GLB, FBX, OBJ, STL, PLY, LXO, MD5, BVH and USD.
- `omni.kit.converter.hoops_core` handles STEP and IGES.
- `.blend` is rejected by both, so it goes through Blender 4.0.2 (Ubuntu arm64, `docker/blender.Dockerfile`).

**Detection:**
- Units come from the stage's metadata unless that gives a size under 1 m or over 2 km, or the format is unitless (STL, OBJ, PLY). Then the importer picks
  the m/cm/mm/in/ft scale that gives a platform-sized result.
- The up axis is the metadata axis unless twice as much flat area faces another axis.
- The deck is the height band with the most upward-facing area, ignoring sea or seabed planes at the very bottom.
- Pass `--units`, `--up` or `--deck-z` (in metres, after the fix) when a model fools these.

Candidate platforms are listed with their licences in `sim/assets/rig_model/candidates.json`. To download one, run
`SKETCHFAB_TOKEN=... python3 sim/tools/fetch_model.py <key>`; this needs a free Sketchfab account.

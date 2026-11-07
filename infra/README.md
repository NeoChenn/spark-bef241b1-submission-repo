# infra: Isaac Sim 5.1 + Isaac Lab 2.3.2 on Modal L40S

## Which Modal profile to use

The default `team` profile (workspace `agentverse`) has **no payment method**, so it refuses
L40S (`InvalidError: Please add a payment method to use L40S GPU functions`).
Run all GPU jobs on the personal workspace:

```bash
export MODAL_PROFILE=jacobyuan99      # or prefix every command with it
```

Volumes (`rust-data`, `rust-ckpt`, `rig-assets`) are `create_if_missing`, so they exist in
whichever workspace you run in. **Everyone must use the same profile**, or you'll end up with two
separate sets of volumes.

## Files

| File | What it does |
|---|---|
| `isaac_image.py` | `isaac_image`, `ISAAC_GPU="L40S"`, `ISAACLAB_PY`, `ISAACLAB_DIR`, `isaac_env()` |
| `smoke_test.py` | nvidia-smi, then headless AppLauncher with a Go2 on a ground plane for 100 steps, then 50 iters of stock Go2 PPO with 512 envs (logs in `/ckpt/smoke/`) |
| `stream.py` | `record_video()` renders a headless camera to an mp4 on the volume (works). `stream()` is WebRTC and does not work on Modal (see below) |

## Commands

```bash
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach infra/smoke_test.py               # full smoke test
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach infra/smoke_test.py --iters 0     # sim only
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run infra/stream.py                            # demo mp4 -> ./demo.mp4
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run infra/stream.py --usd-path /assets/rig.usd \
    --seconds 10 --eye 6,6,4 --target 0,0,1 --name rig
```

Use `--detach` for anything long. Without it, a dropped local connection (we've already hit
`ECONNRESET`) kills the app.

## Image (what works)

Isaac Sim is installed from pip, not from the NGC container. The base is
`nvidia/cuda:12.8.1-runtime-ubuntu22.04` with Modal's `add_python="3.11"`, plus:

1. apt: GL/X/Vulkan userspace libs, git, cmake, build-essential, ffmpeg
2. `torch==2.7.0 torchvision==0.22.0` from the cu128 index
3. `isaacsim[all,extscache]==5.1.0` from `https://pypi.nvidia.com`
4. `git clone IsaacLab v2.3.2` into `/opt/IsaacLab`, then `pip install -e` of `isaaclab`,
   `isaaclab_assets`, `isaaclab_tasks` and `isaaclab_rl[rsl_rl]` (rsl-rl-lib 3.1.2)
5. `.add_local_python_source("policy", "infra")`: repo code is mounted at `/root` when the
   container starts, so code edits don't trigger a rebuild. `PYTHONPATH=/root` is set so
   subprocesses can import it too.

Env vars: `ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1`.

Modal injects the NVIDIA graphics driver into the container (`NVIDIA_DRIVER_CAPABILITIES=all`;
`libGLX_nvidia`, `libnvidia-rtcore`, `libnvoptix` and `/etc/vulkan/icd.d/nvidia_icd.json` are all
present, driver 610), so RTX rendering works without any extra setup.

## How to launch an Isaac Lab script (for the scrub-env / PPO code)

Use **`ISAACLAB_PY` (`/usr/local/bin/python`)**. You don't need `isaaclab.sh` or `python.sh`.
Run the script as a **subprocess** (Kit wants to own the process, and `simulation_app.close()`
may hard-exit):

```python
from infra.isaac_image import isaac_image, ISAAC_GPU, ISAACLAB_PY, ISAACLAB_DIR, isaac_env
from infra.common import CKPT_VOL, CKPT_DIR, ASSET_VOL, ASSET_DIR

@app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=6*3600,
              volumes={CKPT_DIR: CKPT_VOL, ASSET_DIR: ASSET_VOL})
def train():
    subprocess.run([ISAACLAB_PY, "/root/policy/train.py", "--task", "Rig-Scrub-v0",
                    "--num_envs", "1024", "--headless"],
                   cwd=f"{CKPT_DIR}/scrub", env=isaac_env(), check=True)
    CKPT_VOL.commit()
```

* Your script must create the `AppLauncher` **before** it imports `isaaclab.sim`,
  `isaaclab.envs`, `omni.*` or `pxr`, and it must import your env module (for example
  `import policy.envs.scrub_env`) so that `gym.register` runs. The easiest route is to copy
  `/opt/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py` and add that import.
* The stock `train.py` writes logs to `./logs/rsl_rl/<experiment>/<timestamp>/` relative to
  **cwd**, so set `cwd` to a directory on the volume and call `CKPT_VOL.commit()` afterwards.
* Cameras: pass `--enable_cameras`. Video of a trained policy:
  `play.py --task T --headless --enable_cameras --video --video_length 600`.
* Give the function at least `cpu=8`. Modal's default is 1 core, which is very slow for Kit startup.

## Timings (L40S, measured)

FILLED_BY_SMOKE

## Streaming

Isaac Sim 5.1 WebRTC livestreaming (`LIVESTREAM=1` / `--livestream 1` in AppLauncher) needs
**TCP 49100** for signalling and **UDP 47998** for media. `modal.forward()` only tunnels TCP,
so the WebRTC client can complete signalling but never gets any video. Live viewing on Modal is
therefore not possible. The fallback is `record_video()`, which renders a headless RTX camera
through an Isaac Lab `Camera` sensor, encodes with imageio/ffmpeg (libx264), and writes to
`rust-ckpt:/videos/<name>.mp4`. That same mp4 is what we use for the demo video.

## Gotchas

* The first launch in a fresh container compiles RTX shaders and loads extensions, which takes
  a few minutes. Later launches in the **same** container are faster. Ground-plane and robot
  USDs are fetched from NVIDIA's S3 asset server, so the container needs internet.
* The `team` profile cannot use L40S (see top).
* Kit prints a lot of harmless warnings (IOMMU, `omni.kit.registry`, missing display); you can
  ignore them.

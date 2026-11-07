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
| `stream.py` | `record_video()` renders a headless camera to an mp4 on the volume. It **crashes on Modal** (RTX doesn't work under gVisor, see below) |

## Commands

```bash
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach infra/smoke_test.py               # full smoke test
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach infra/smoke_test.py --iters 0     # sim only
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

Measured with `smoke_test.py` on 2026-11-07, image `im-Qn4Tzv1YCcVmLkFbtCtmOW`:

| Step | Time |
|---|---|
| First image build (torch + isaacsim pip + Isaac Lab) | about 12 min (torch 75 s, isaacsim 230 s, Isaac Lab 310 s). Later builds hit the cache. |
| Container cold start (image already built) | 7 to 90 s (90 s the first time the image is pulled to a worker) |
| Kit / AppLauncher headless startup | 9 to 15 s |
| Go2 on a ground plane, single robot, 100 steps | about 150 steps/s |
| go2_ur3 URDF import (UrdfFileCfg, fix_base) plus 100 steps | 24 s wall: 18 joints, 52 bodies, base held at 0.50 m |
| **PPO**: Isaac-Velocity-Flat-Unitree-Go2-v0, 512 envs, 50 iters | **about 22,000 env-steps/s**, 51 s wall including startup. Checkpoints go to `/ckpt/smoke/logs/rsl_rl/unitree_go2_flat/<ts>/model_49.pt` |
| Full smoke test, total container time | about 100 s |

## Rendering and video: BLOCKED on Modal

* **RTX rendering crashes on Modal.** Anything that creates an RTX Hydra engine segfaults
  inside `librtx.scenedb.plugin.so` at startup: `--enable_cameras`, Camera/TiledCamera sensors,
  replicator annotators, and gym RecordVideo all hit it. The log line just before the crash is
  `rtx.neuraylib: It appears that the "nvidia" kernel module is not loaded`. Modal containers
  run under gVisor (`Kernel: 4.19.0-gvisor`), which proxies CUDA but not the full Vulkan/RTX
  path. Disabling driver checks, multi-GPU and texture streaming via kit args didn't help.
  **Physics plus PPO without cameras works fine.**
* Consequences:
  * The scrub env must not use camera observations. Use privileged state (rust mask from the
    pure-torch rust model).
  * `stream.py:record_video` is implemented but crashes on Modal. It should work on a machine
    where RTX works (for example the sim team's box), with `--usd-path` pointing at a local path.
  * For demo or eval clips: log rollouts on Modal (root pose plus joint positions per frame, and
    rust state) to `/ckpt/...` as npz, then replay and render them on the sim team's Isaac Sim
    6.2 machine, which already has a camera to JPEG to websocket path (`rig/sim/sender.py`).
* Streaming: dropped. The sim team streams through their own path, and Isaac Sim WebRTC needs
  UDP 47998, which `modal.forward` (TCP only) can't carry anyway.

## Gotchas

* **`simulation_app.close()` hangs forever on Modal.** End your own scripts with
  `sys.stdout.flush(); os._exit(0)`. The stock `train.py` and `play.py` also call `close()`
  after saving, so launch them with `infra.isaac_image.run_isaac(cmd, cwd=..., done_marker=...,
  grace=...)`, which kills the process after the marker line appears. For rsl_rl, use
  `done_marker=f"Learning iteration {N-1}/{N}"`. Commit the volume afterwards.
* Dependency pins that matter: `torch==2.7.0+cu128` (Isaac Sim 5.1),
  `tensordict==0.9.1` (0.10 and later need torch 2.8 and fail with `is_structseq_instance`), and
  `setuptools<80` plus `flatdict==4.0.1 --no-build-isolation` (flatdict's sdist needs
  `pkg_resources`).
* The URDF importer rejects default joint positions that fall outside the joint limits. On
  go2_ur3 the calf range is [-2.72, -0.84], so use a standing pose (thigh 0.8, calf -1.5).
* **Assets must be on the `rig-assets` volume in the jacobyuan99 workspace.** The lead uploaded
  go2_ur3 to the *team* workspace's volume, so I re-uploaded `sim/assets/go2_ur3` there:
  `MODAL_PROFILE=jacobyuan99 .venv/bin/modal volume put rig-assets sim/assets/go2_ur3 /go2_ur3`.
* Ground-plane and Go2 USDs are fetched from NVIDIA's S3 asset server, so the container needs
  internet.
* The `team` profile cannot use L40S (see top).
* Kit prints a lot of harmless warnings (IOMMU, `omni.kit.registry`, missing display); you can
  ignore them.

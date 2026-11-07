"""Modal image with Isaac Sim 5.1 (pip) + Isaac Lab v2.3.2 + rsl_rl, for L40S.

Usage from another Modal app:

    from infra.isaac_image import isaac_image, ISAAC_GPU, ISAACLAB_PY, ISAACLAB_DIR, isaac_env
    from infra.common import CKPT_VOL, CKPT_DIR

    @app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=3*3600,
                  volumes={CKPT_DIR: CKPT_VOL})
    def train():
        subprocess.run([ISAACLAB_PY, f"{ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl/train.py",
                        "--task", "...", "--headless"], check=True, env=isaac_env())

Why pip and not nvcr.io/nvidia/isaac-sim:5.1.0: the NGC container is ~20 GB, has no
python on PATH for Modal, and runs everything through /isaac-sim/python.sh. The pip wheels
(`isaacsim[all,extscache]==5.1.0` from pypi.nvidia.com) on a plain Ubuntu 22.04 + Python 3.11
image are the officially supported path for Isaac Lab and work with Modal's injected NVIDIA
driver (Modal mounts libGLX_nvidia / libnvidia-rtcore / Vulkan ICD with
NVIDIA_DRIVER_CAPABILITIES=all, which RTX rendering needs).
"""
import os

import modal

ISAAC_GPU = "L40S"
ISAACLAB_TAG = "v2.3.2"            # latest Isaac Lab release for Isaac Sim 5.1
ISAACLAB_DIR = "/opt/IsaacLab"
ISAACLAB_PY = "/usr/local/bin/python"  # the python that has isaacsim + isaaclab (no isaaclab.sh needed)
REPO_ROOT = "/root"                # where add_local_python_source puts policy/ and infra/

_ENV = {
    "ACCEPT_EULA": "Y",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "PRIVACY_CONSENT": "Y",
    "OMNI_KIT_ALLOW_ROOT": "1",
    "ISAACLAB_PATH": ISAACLAB_DIR,
    "PYTHONPATH": REPO_ROOT,       # so subprocess scripts can `import policy...` / `import infra...`
    "PYTHONUNBUFFERED": "1",
    "PIP_NO_CACHE_DIR": "1",
}

isaac_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "build-essential", "cmake", "wget", "curl", "ffmpeg",
        # X / GL / Vulkan userspace libs Kit dlopens even when headless
        "libglu1-mesa", "libgl1", "libegl1", "libglib2.0-0", "libsm6", "libice6", "libxt6",
        "libxext6", "libxrender1", "libxi6", "libxrandr2", "libxinerama1", "libxcursor1",
        "libx11-6", "libxkbcommon0", "libvulkan1", "libgomp1", "libnss3",
    )
    .env(_ENV)
    # heavy, rarely-changing layers first
    .pip_install(
        "torch==2.7.0", "torchvision==0.22.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "isaacsim[all,extscache]==5.1.0",
        extra_index_url="https://pypi.nvidia.com",
    )
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_TAG} https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_DIR}",
        # flatdict's sdist needs pkg_resources (removed in setuptools>=81) -> pin + no build isolation
        "python -m pip install 'setuptools<80' wheel && python -m pip install --no-build-isolation flatdict==4.0.1",
        # same as `./isaaclab.sh --install rsl_rl` but explicit (editable, like isaaclab.sh does)
        f"cd {ISAACLAB_DIR} && python -m pip install "
        "-e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks "
        "-e 'source/isaaclab_rl[rsl_rl]' "
        "--extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com",
        # isaaclab deps sometimes bump torch; force back the cu128 build Isaac Sim 5.1 expects
        "python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("imageio[ffmpeg]", "tensorboard")
    # repo code last: mounted at container start, no rebuild when it changes
    .add_local_python_source("policy", "infra")
)


def isaac_env(**extra) -> dict:
    """os.environ + the Isaac vars, for subprocess.run(..., env=isaac_env())."""
    env = dict(os.environ)
    env.update(_ENV)
    env.update({k: str(v) for k, v in extra.items()})
    return env

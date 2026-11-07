"""Run Isaac Sim / Isaac Lab on a Modal GPU and get a video back.

The container image and the GPU choice are adapted from the organisers' working example,
https://github.com/TheFloatingString/isaac-lab-on-modal (Isaac Sim 5.1.0, Isaac Lab v2.3.0,
L40S). That repo shows no LICENSE file, so credit it and check with its author before
redistributing. Do not change the pinned versions without a reason: the first build is slow
(10+ minutes) and these versions are known to work.

STATUS: written but NOT yet run. Run the smoke test below first.

    pip install modal
    modal setup                      # one-time login (you do this, not the agent)
    modal run isaac/modal_app.py     # ANYmal-D smoke test -> renders/smoke.mp4

Notes:
- Use the L40S. A100 and H100 are not supported because they lack RT cores.
- There is no live GUI over Modal. We save frames and encode them to video.
- Each run costs credits. Stop apps when finished and keep an eye on the balance.
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("flange-robot-isaac")

ISAACLAB_REF = "v2.3.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"})
    .apt_install(
        "git", "build-essential", "cmake", "libglu1-mesa", "libgl1", "libegl1", "libxext6",
        "libx11-6", "libxrandr2", "libxinerama1", "libxcursor1", "libxi6", "libsm6",
        "libice6", "libxt6", "libvulkan1", "vulkan-tools", "libglib2.0-0", "libxkbcommon0",
        "ffmpeg",
    )
    .pip_install("torch==2.7.0", "torchvision==0.22.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("isaacsim[all,extscache]==5.1.0", extra_index_url="https://pypi.nvidia.com")
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_REF} https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab",
        "pip install 'setuptools<70' wheel && pip install --no-build-isolation flatdict==4.0.1",
        "cd /opt/IsaacLab/source && ls && pip install -e isaaclab -e isaaclab_assets -e isaaclab_tasks"
        " --extra-index-url https://pypi.nvidia.com",
        "pip install click==8.1.7 idna==3.10 h5py && pip show isaaclab isaaclab_tasks",
    )
    .add_local_file(str(HERE / "render_scene.py"), "/root/render_scene.py")
)


@app.function(image=image, gpu="L40S", timeout=3000)
def render(task: str, eye: list, target: list, steps: int, every: int) -> bytes:
    """Run one scene inside the container and return an MP4."""
    import glob
    import subprocess

    frames = "/tmp/frames"
    cmd = ["python", "-u", "/root/render_scene.py", "--task", task, "--out-dir", frames,
           "--steps", str(steps), "--every", str(every),
           "--eye", *map(str, eye), "--target", *map(str, target)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/IsaacLab")
    print(r.stderr[-4000:])
    print("RETURNCODE", r.returncode)
    print("STDOUT_TAIL", r.stdout[-3000:])
    if "RENDER_OK" not in r.stdout:
        raise RuntimeError("render failed: see logs above")

    n = len(glob.glob(f"{frames}/frame_*.png"))
    if n == 0:
        raise RuntimeError("no frames were written")
    mp4 = "/tmp/out.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "15", "-i", f"{frames}/frame_%04d.png",
         "-pix_fmt", "yuv420p", mp4],
        check=True, capture_output=True,
    )
    return open(mp4, "rb").read()


@app.local_entrypoint()
def main(
    task: str = "Isaac-Velocity-Flat-Anymal-D-v0",
    steps: int = 120,
    every: int = 2,
    out: str = "renders/smoke.mp4",
):
    """Smoke test: a quadruped task, third-person camera, saved as a video."""
    eye = [-3.0, 0.0, 1.8]
    target = [0.0, 0.0, 0.4]
    data = render.remote(task, eye, target, steps, every)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print("saved", path, len(data), "bytes")

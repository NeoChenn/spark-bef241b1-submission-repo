"""Headless video recording.

STATUS: crashes on Modal L40S. The RTX Hydra engine segfaults under gVisor (see infra/README.md).
Physics and PPO work there; cameras don't. Use this where RTX works, or log rollouts and replay them.

1. record_video: headless Isaac Sim renders a camera to an mp4 on the
   rust-ckpt volume (/ckpt/videos/<name>.mp4) and the local entrypoint downloads it.

       MODAL_PROFILE=jacobyuan99 .venv/bin/modal run infra/stream.py                       # demo: Go2 on ground
       MODAL_PROFILE=jacobyuan99 .venv/bin/modal run infra/stream.py --usd-path /assets/rig.usd --seconds 10 \
           --eye 6,6,4 --target 0,0,1

   For a *trained policy* video use Isaac Lab's built-in recorder instead (gym RecordVideo):
       python $ISAACLAB/scripts/reinforcement_learning/rsl_rl/play.py --task <T> --headless \
           --enable_cameras --video --video_length 600 --checkpoint <ckpt.pt>
   (writes <log_dir>/videos/play/*.mp4).

Live viewing is handled by the sim team (rig/sim/sender.py: RGB annotator -> JPEG over TCP ->
   websocket page); WebRTC is not used (Modal tunnels are TCP-only, WebRTC media needs UDP 47998).
"""
import os
import subprocess
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra.common import ASSET_DIR, ASSET_VOL, CKPT_DIR, CKPT_VOL  # noqa: E402
from infra.isaac_image import ISAAC_GPU, ISAACLAB_PY, isaac_image, run_isaac  # noqa: E402

app = modal.App("rig-isaac-record")
VIDEO_DIR = f"{CKPT_DIR}/videos"

RECORD_SCRIPT = r'''
import argparse, json, time
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app
cfg = json.loads(args.cfg)

import imageio
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

if cfg["usd_path"]:
    # open the teammate's stage first; SimulationContext then attaches to it (adds physicsScene if missing)
    import omni.usd
    omni.usd.get_context().open_stage(cfg["usd_path"])
    for _ in range(20):
        simulation_app.update()

fps = cfg["fps"]
substeps = 4
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / (fps * substeps), device="cuda:0",
                                                          render_interval=substeps))
if not cfg["usd_path"]:
    from isaaclab.assets import Articulation
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
    g = sim_utils.GroundPlaneCfg(); g.func("/World/ground", g)
    l = sim_utils.DomeLightCfg(intensity=2000.0); l.func("/World/DomeLight", l)
    robot = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Robot"))
else:
    robot = None
    import omni.usd
    from pxr import UsdLux
    stage = omni.usd.get_context().get_stage()
    if not any(p.HasAPI(UsdLux.LightAPI) or p.GetTypeName().endswith("Light") for p in stage.Traverse()):
        l = sim_utils.DomeLightCfg(intensity=2000.0); l.func("/World/DomeLight", l)

cam = Camera(CameraCfg(
    prim_path="/World/RecCam", update_period=0, height=cfg["height"], width=cfg["width"],
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=cfg["focal_length"], clipping_range=(0.05, 1000.0)),
))
sim.reset()
cam.set_world_poses_from_view(torch.tensor([cfg["eye"]], device="cuda:0"),
                              torch.tensor([cfg["target"]], device="cuda:0"))
t0 = time.time()
for _ in range(cfg["warmup_frames"]):  # RTX needs a few frames (and shader compile on first run)
    sim.render()
print(f"[rec] warmup {time.time() - t0:.1f}s", flush=True)

writer = imageio.get_writer(cfg["out"], fps=fps, codec="libx264", quality=8, macro_block_size=1)
n = int(cfg["seconds"] * fps)
t0 = time.time()
for i in range(n):
    for _ in range(substeps):
        if robot is not None:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
        sim.step(render=False)
        if robot is not None:
            robot.update(sim.get_physics_dt())
    sim.render()
    cam.update(dt=1.0 / fps)
    frame = cam.data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
    writer.append_data(frame)
writer.close()
print(f"[rec] {n} frames in {time.time() - t0:.1f}s -> {cfg['out']}", flush=True)
print("[rec] VIDEO_OK", flush=True)
import os, sys
sys.stdout.flush()
os._exit(0)  # simulation_app.close() hangs on Modal
'''


@app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=3600,
              volumes={CKPT_DIR: CKPT_VOL, ASSET_DIR: ASSET_VOL})
def record_video(usd_path: str | None = None, seconds: float = 10.0, name: str = "demo",
                 eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 0.3), width: int = 1280, height: int = 720,
                 fps: int = 30, focal_length: float = 18.0, warmup_frames: int = 30, kit_args: str = "") -> bytes:
    """Render `seconds` of the stage at usd_path (None = demo Go2 scene) to /ckpt/videos/<name>.mp4."""
    import json
    os.makedirs(VIDEO_DIR, exist_ok=True)
    out = f"{VIDEO_DIR}/{name}.mp4"
    cfg = dict(usd_path=usd_path, seconds=seconds, eye=list(eye), target=list(target), width=width,
               height=height, fps=fps, focal_length=focal_length, warmup_frames=warmup_frames, out=out)
    with open("/tmp/record.py", "w") as f:
        f.write(RECORD_SCRIPT)
    t = time.time()
    extra = ["--kit_args", kit_args] if kit_args else []
    run_isaac([ISAACLAB_PY, "/tmp/record.py", "--headless", "--enable_cameras", "--cfg", json.dumps(cfg)] + extra,
              timeout=3000, done_marker="VIDEO_OK", grace=10)
    print(f"[rec] wall {time.time() - t:.1f}s", flush=True)
    if not os.path.exists(out):
        raise RuntimeError("video was not written; see logs above")
    CKPT_VOL.commit()
    with open(out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(usd_path: str = "", seconds: float = 10.0, name: str = "demo", eye: str = "3,3,2",
         target: str = "0,0,0.3", out: str = "", kit_args: str = ""):
    data = record_video.remote(usd_path or None, seconds, name,
                               tuple(float(x) for x in eye.split(",")),
                               tuple(float(x) for x in target.split(",")), kit_args=kit_args)
    out = out or f"{name}.mp4"
    with open(out, "wb") as f:
        f.write(data)
    print(f"saved {out} ({len(data) / 1e6:.1f} MB); also on volume rust-ckpt:/videos/{name}.mp4")

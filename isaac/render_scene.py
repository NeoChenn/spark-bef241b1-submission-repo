"""Render a short clip of an Isaac Lab task. Runs inside the Modal container.

Adapted from `smoke.py` in https://github.com/TheFloatingString/isaac-lab-on-modal.
Isaac Lab modules can only be imported after AppLauncher has started Kit, so every
isaaclab import lives inside run().

STATUS: written but NOT yet run. Zero actions are applied, so the robot just holds its
default pose. Its purpose is to prove the render pipeline before we build our own scene.

    python render_scene.py --task Isaac-Velocity-Flat-Anymal-D-v0 --out-dir /tmp/frames \
        --steps 120 --every 2 --eye -3 0 1.8 --target 0 0 0.4
"""

import argparse
import os

from isaaclab.app import AppLauncher


def run(task: str, out_dir: str, eye: list[float], target: list[float], steps: int, every: int) -> None:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    simulation_app = AppLauncher(parser.parse_args(["--headless", "--enable_cameras"])).app

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401
    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    os.makedirs(out_dir, exist_ok=True)

    cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    cfg.scene.tp_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/tp_cam",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    env = gym.make(task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    cam = u.scene["tp_cam"]
    origin = u.scene.env_origins[0]
    eyes = (origin + torch.tensor(eye, device=u.device)).unsqueeze(0)
    targets = (origin + torch.tensor(target, device=u.device)).unsqueeze(0)
    act = torch.zeros(u.action_space.shape, device=u.device)

    saved = 0
    last_std = 0.0
    for step in range(steps):
        cam.set_world_poses_from_view(eyes, targets)
        env.step(act)
        if step % every == 0:
            rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            last_std = float(rgb.std())
            Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{saved:04d}.png"))
            saved += 1

    print("task", task, "frames", saved, "last frame std", last_std)
    assert saved > 0 and last_std > 1.0, "frames look blank"
    print("RENDER_OK")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eye", type=float, nargs=3, required=True)
    p.add_argument("--target", type=float, nargs=3, required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--every", type=int, default=2)
    a = p.parse_args()
    run(a.task, a.out_dir, a.eye, a.target, a.steps, a.every)

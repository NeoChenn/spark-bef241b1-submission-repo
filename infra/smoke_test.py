"""Smoke test: Isaac Sim 5.1 + Isaac Lab PPO on a Modal L40S.

    MODAL_PROFILE=jacobyuan99 .venv/bin/modal run infra/smoke_test.py            # full test
    MODAL_PROFILE=jacobyuan99 .venv/bin/modal run infra/smoke_test.py --iters 0  # skip PPO

(a) nvidia-smi  (b) headless AppLauncher  (c) ground + Unitree Go2, 100 physics steps
(d) 50 iters of stock Isaac-Velocity-Flat-Unitree-Go2-v0 rsl_rl PPO, 512 envs, logs -> /ckpt/smoke/
"""
import os
import re
import subprocess
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra.common import CKPT_DIR, CKPT_VOL  # noqa: E402
from infra.isaac_image import ISAAC_GPU, ISAACLAB_DIR, ISAACLAB_PY, isaac_env, isaac_image  # noqa: E402

app = modal.App("rig-isaac-smoke")

SCENE_SCRIPT = r'''
import argparse, time
t0 = time.time()
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app
print(f"[smoke] SIM_STARTUP_S {time.time() - t0:.1f}", flush=True)

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/ground", ground)
light = sim_utils.DomeLightCfg(intensity=2000.0)
light.func("/World/Light", light)
robot = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Robot"))
t1 = time.time()
sim.reset()
print(f"[smoke] SCENE_RESET_S {time.time() - t1:.1f}", flush=True)
t2 = time.time()
for _ in range(100):
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
dt = time.time() - t2
print(f"[smoke] SINGLE_ROBOT_STEPS_PER_S {100 / dt:.1f}", flush=True)
print(f"[smoke] go2 base z after 100 steps = {robot.data.root_pos_w[0, 2].item():.3f} m", flush=True)
print("[smoke] SCENE_OK", flush=True)
simulation_app.close()
'''


def _run(cmd, cwd=None, timeout=None):
    """Stream a subprocess to stdout and return its full output."""
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=isaac_env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    t0 = time.time()
    for line in p.stdout:
        out.append(line)
        sys.stdout.write(line)
        if timeout and time.time() - t0 > timeout:
            p.kill()
            break
    p.wait()
    return p.returncode, "".join(out)


@app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=3600,
              volumes={CKPT_DIR: CKPT_VOL})
def smoke(t_submit: float, iters: int = 50, num_envs: int = 512):
    t_start = time.time()
    timings = {"container_cold_start_s": round(t_start - t_submit, 1)}
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)

    # (b)+(c) headless sim + Go2 on a ground plane
    with open("/tmp/scene_smoke.py", "w") as f:
        f.write(SCENE_SCRIPT)
    t = time.time()
    rc, out = _run([ISAACLAB_PY, "/tmp/scene_smoke.py", "--headless"], timeout=1200)
    timings["scene_test_wall_s"] = round(time.time() - t, 1)
    for key in ("SIM_STARTUP_S", "SCENE_RESET_S", "SINGLE_ROBOT_STEPS_PER_S"):
        m = re.search(rf"{key} ([\d.]+)", out)
        if m:
            timings[key.lower()] = float(m.group(1))
    timings["scene_ok"] = "SCENE_OK" in out
    print("[smoke] timings so far:", timings, flush=True)

    # (d) stock PPO
    if iters > 0:
        log_root = f"{CKPT_DIR}/smoke"
        os.makedirs(log_root, exist_ok=True)
        t = time.time()
        rc, out = _run([
            ISAACLAB_PY, f"{ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl/train.py",
            "--task", "Isaac-Velocity-Flat-Unitree-Go2-v0", "--num_envs", str(num_envs),
            "--max_iterations", str(iters), "--headless",
        ], cwd=log_root, timeout=2400)  # train.py writes ./logs/rsl_rl/<exp>/<date>/
        timings["ppo_wall_s"] = round(time.time() - t, 1)
        timings["ppo_rc"] = rc
        sps = [float(x) for x in re.findall(r"Computation:\s*([\d.]+)\s*steps/s", out)]
        if sps:
            timings["ppo_steps_per_s_last"] = sps[-1]
            timings["ppo_steps_per_s_mean"] = round(sum(sps[1:] or sps) / len(sps[1:] or sps), 1)
        rew = re.findall(r"Mean reward:\s*([-\d.]+)", out)
        if rew:
            timings["ppo_mean_reward_first_last"] = (float(rew[0]), float(rew[-1]))
        timings["ppo_iters_done"] = len(sps)
        CKPT_VOL.commit()
        print(subprocess.run(["bash", "-lc", f"find {log_root} -maxdepth 5 | head -30"],
                             capture_output=True, text=True).stdout)

    timings["total_container_s"] = round(time.time() - t_start, 1)
    print("[smoke] RESULT", timings, flush=True)
    return timings


@app.local_entrypoint()
def main(iters: int = 50, num_envs: int = 512):
    res = smoke.remote(time.time(), iters, num_envs)
    print("\n=== SMOKE RESULT ===")
    for k, v in res.items():
        print(f"{k:32s} {v}")

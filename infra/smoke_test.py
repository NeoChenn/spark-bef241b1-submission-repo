import os
import re
import subprocess
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra.common import ASSET_DIR, ASSET_VOL, CKPT_DIR, CKPT_VOL
from infra.isaac_image import ISAAC_GPU, ISAACLAB_DIR, ISAACLAB_PY, isaac_image, run_isaac

app = modal.App("rig-isaac-smoke")
URDF_DEFAULT = f"{ASSET_DIR}/go2_ur3/go2_ur3.urdf"

SCENE_SCRIPT = r'''
import argparse, time
t0 = time.time()
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="")
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
if args.urdf:
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    UC = sim_utils.UrdfConverterCfg
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=args.urdf, fix_base=True, merge_fixed_joints=False,
            joint_drive=UC.JointDriveCfg(gains=UC.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=5.0))),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={
            ".*_thigh_joint": 0.8, ".*_calf_joint": -1.5, ".*_hip_joint": 0.0, "ur3_.*": 0.0}),  # calf limits [-2.72,-0.84]
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    ))
else:
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
if args.urdf:
    print(f"[smoke] URDF joints={robot.num_joints} bodies={robot.num_bodies} names={robot.joint_names}", flush=True)
print("[smoke] SCENE_OK", flush=True)
import os, sys
sys.stdout.flush()
os._exit(0)  # simulation_app.close() hangs on Modal (gVisor); hard-exit instead
'''


@app.function(image=isaac_image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=3600,
              volumes={CKPT_DIR: CKPT_VOL, ASSET_DIR: ASSET_VOL})
def smoke(t_submit: float, iters: int = 50, num_envs: int = 512, urdf: str = URDF_DEFAULT):
    t_start = time.time()
    timings = {"container_cold_start_s": round(t_start - t_submit, 1)}
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)

    with open("/tmp/scene_smoke.py", "w") as f:
        f.write(SCENE_SCRIPT)
    t = time.time()
    rc, out = run_isaac([ISAACLAB_PY, "/tmp/scene_smoke.py", "--headless"], timeout=900,
                   done_marker="SCENE_OK", grace=10)
    timings["scene_test_wall_s"] = round(time.time() - t, 1)
    for key in ("SIM_STARTUP_S", "SCENE_RESET_S", "SINGLE_ROBOT_STEPS_PER_S"):
        m = re.search(rf"{key} ([\d.]+)", out)
        if m:
            timings[key.lower()] = float(m.group(1))
    timings["scene_ok"] = "SCENE_OK" in out

    if urdf and os.path.exists(urdf):
        t = time.time()
        rc, out = run_isaac([ISAACLAB_PY, "/tmp/scene_smoke.py", "--headless", "--urdf", urdf], timeout=900,
                            done_marker="SCENE_OK", grace=10)
        timings["urdf_wall_s"] = round(time.time() - t, 1)
        timings["urdf_ok"] = "SCENE_OK" in out
        m = re.search(r"URDF joints=(\d+) bodies=(\d+)", out)
        if m:
            timings["urdf_joints_bodies"] = (int(m.group(1)), int(m.group(2)))
    elif urdf:
        timings["urdf_ok"] = f"missing {urdf}"
    print("[smoke] timings so far:", timings, flush=True)

    if iters > 0:
        log_root = f"{CKPT_DIR}/smoke"
        os.makedirs(log_root, exist_ok=True)
        t = time.time()
        rc, out = run_isaac([
            ISAACLAB_PY, f"{ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl/train.py",
            "--task", "Isaac-Velocity-Flat-Unitree-Go2-v0", "--num_envs", str(num_envs),
            "--max_iterations", str(iters), "--headless",
        ], cwd=log_root, timeout=2400, done_marker=f"Learning iteration {iters - 1}/{iters}", grace=90)
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
def main(iters: int = 50, num_envs: int = 512, urdf: str = URDF_DEFAULT):
    res = smoke.remote(time.time(), iters, num_envs, urdf)
    print("\n=== SMOKE RESULT ===")
    for k, v in res.items():
        print(f"{k:32s} {v}")

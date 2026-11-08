import argparse
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ARM = ["ur3_shoulder_pan_joint", "ur3_shoulder_lift_joint", "ur3_elbow_joint", "ur3_wrist_1_joint",
       "ur3_wrist_2_joint", "ur3_wrist_3_joint", "lift_joint"]


def reader(robot, names):
    ids = [robot.joint_names.index(n) for n in names]
    return lambda: [round(float(v), 4) for v in robot.data.joint_pos[0, ids].tolist()]


def gait(args):
    import torch

    from policy.walk.walk_env_cfg import LEGS, WalkEnvCfg
    from policy.walk.walk_sim import deck_cfg, load_policy, make_env, set_cmd

    cfg = deck_cfg(WalkEnvCfg(), 1)
    cfg.terminations.base_contact = None
    cfg.terminations.bad_orientation = None
    env, policy = load_policy(make_env(cfg), args.ckpt)
    base = env.unwrapped
    r = base.scene["robot"]
    q = reader(r, LEGS)
    with torch.inference_mode():
        env.reset()
    dt = base.step_dt
    rows = []
    for k in range(int(args.seconds / dt)):
        set_cmd(base, [(args.speed, 0.0, 0.0)])
        with torch.inference_mode():
            obs = env.get_observations()
            env.step(policy(obs))
        rows.append({"t": round(k * dt, 4), "q": q(), "z": round(float(r.data.root_pos_w[0, 2]), 4),
                     "vx": round(float(r.data.root_lin_vel_b[0, 0]), 3)})
    return {"source": "walk policy rollout (PPO, Rig-Walk-Go2UR3Lift-v0, recorded)", "hz": 1.0 / dt,
            "joints": LEGS, "speed_cmd": args.speed, "rows": rows}


def scrub(args):
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from policy.envs import scrub_env_cfg as C
    from policy.envs.scrub_env import TASKS

    task, name = TASKS["lift_v2"]
    cfg = getattr(C, name)()
    cfg.scene.num_envs = 1
    cfg.seed = 3
    for k in ("radius_range", "height_range", "standoff_range", "axis_dist_range"):
        lo, hi = getattr(cfg, k)
        setattr(cfg, k, ((lo + hi) / 2,) * 2)
    cfg.axial_range = cfg.yaw_range = (0.0, 0.0)
    agent = C.ScrubPPORunnerCfg()
    agent.device = "cuda:0"
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=base.device)
    q = reader(base._robot, ARM)
    obs = env.get_observations()
    rows = []
    dt = base.step_dt
    for k in range(int(args.seconds / dt)):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if bool(dones[0]):
            break
        rows.append({"t": round(k * dt, 4), "q": q()})
    return {"source": "scrub rollout (PPO v2, Rig-Scrub-Go2UR3Lift-v2, recorded)", "hz": 1.0 / dt, "joints": ARM,
            "rows": rows}


def pickcan(args):
    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    importlib.import_module("policy.envs.pickcan_env")
    from policy.envs.pickcan_env_cfg import PickCanEnvCfg
    from policy.pickcan.script import NAMES, ScriptedPicker

    cfg = PickCanEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 7
    env = RslRlVecEnvWrapper(gym.make("Rig-PickCan-Go2UR3Lift-v0", cfg=cfg), clip_actions=1.0)
    u = env.unwrapped
    picker = ScriptedPicker(1, u.device, dt=u.ctrl_dt, scale=cfg.action_scale)
    q = reader(u._robot, ARM)
    rows, done = [], 0
    for k in range(int(args.seconds / u.ctrl_dt)):
        with torch.inference_mode():
            act = picker.act(u._tip()[0], u.p_tgt, u.geom_grasp(), u.sealed, u._robot.data.joint_pos[:, u.lift_id])
            env.step(act)
        ph = NAMES[int(picker.phase[0])]
        rows.append({"t": round(k * u.ctrl_dt, 4), "q": q(), "phase": ph, "sealed": bool(u.sealed[0])})
        done = done + 1 if ph == "DONE" else 0
        if done * u.ctrl_dt > 1.0:
            break
    return {"source": "pick-can rollout (scripted picker, Rig-PickCan-Go2UR3Lift-v0, recorded)", "hz": 1.0 / u.ctrl_dt,
            "joints": ARM, "rows": rows}


def export(raw_dir: str, out_dir: str) -> None:
    from rig.sim.gait import gait_from_raw
    from rig.sim.taskmotion import export_clip

    os.makedirs(out_dir, exist_ok=True)
    load = lambda n: json.load(open(os.path.join(raw_dir, f"{n}_raw.json")))
    for name, data in (("gait_loop", gait_from_raw(load("gait"))), ("scrub", export_clip(load("scrub"))),
                       ("pick_can", export_clip(load("pickcan")))):
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"[export] {name} -> {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["gait", "scrub", "pickcan", "export"])
    p.add_argument("--ckpt", default="")
    p.add_argument("--out", default="")
    p.add_argument("--raw", default="runs/motion")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--speed", type=float, default=0.8)
    if sys.argv[1:2] == ["export"]:
        args = p.parse_args()
        export(args.raw, args.out or os.path.join(ROOT, "rig/sim/motion"))
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    AppLauncher(args)
    rec = {"gait": gait, "scrub": scrub, "pickcan": pickcan}[args.mode](args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f)
    print(f"[record] {args.mode} rows={len(rec['rows'])} hz={rec['hz']:.1f} -> {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

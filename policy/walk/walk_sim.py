from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def deck_cfg(cfg, num_envs: int, scene: str = "", deck_z: float = 0.0, size: float = 140.0):
    import isaaclab.sim as sim_utils
    import isaaclab.terrains as tg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.terrains import TerrainGeneratorCfg

    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(size, size), border_width=0.0, num_rows=1, num_cols=1, curriculum=False,
        sub_terrains={"deck": tg.MeshPlaneTerrainCfg(proportion=1.0)})
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    if scene:
        cfg.scene.sky_light = None
        cfg.scene.backdrop = AssetBaseCfg(prim_path="/World/Backdrop", spawn=sim_utils.UsdFileCfg(usd_path=scene),
                                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -deck_z)))
    cfg.events.push_robot = None
    cfg.events.payload = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    cfg.events.base_com = None
    cfg.observations.policy.enable_corruption = False
    c = cfg.commands.base_velocity
    c.resampling_time_range = (1.0e9, 1.0e9)
    c.heading_command, c.rel_heading_envs, c.rel_standing_envs, c.debug_vis = False, 0.0, 0.0, False
    cfg.episode_length_s = 1.0e5
    return cfg


def make_env(cfg, render: bool = False):
    import gymnasium as gym

    from policy.walk.walk_env_cfg import TASK

    return gym.make(TASK, cfg=cfg, render_mode="rgb_array" if render else None)


def load_policy(env, ckpt: str):
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from policy.walk.walk_env_cfg import WalkPPORunnerCfg

    agent = WalkPPORunnerCfg()
    agent.device = "cuda:0"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(ckpt)
    return wrapped, runner.get_inference_policy(device=env.unwrapped.device)


def place(base, xy, yaw, z=0.36):
    import torch

    r = base.scene["robot"]
    n = base.num_envs
    st = r.data.default_root_state.clone()
    st[:, 0], st[:, 1], st[:, 2] = xy[0], xy[1], z
    st[:, 3], st[:, 4], st[:, 5], st[:, 6] = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    st[:, 7:] = 0.0
    ids = torch.arange(n, device=base.device)
    r.write_root_state_to_sim(st, ids)
    r.write_joint_state_to_sim(r.data.default_joint_pos, r.data.default_joint_vel, env_ids=ids)


def pose2d(base, i=0):
    r = base.scene["robot"]
    p = r.data.root_pos_w[i]
    q = r.data.root_quat_w[i]
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return float(p[0]), float(p[1]), float(p[2]), float(yaw)


def set_cmd(base, cmd):
    import torch

    base.command_manager.get_term("base_velocity").vel_command_b[:] = torch.as_tensor(cmd, device=base.device,
                                                                                      dtype=torch.float32)


def fallen(base):
    from isaaclab.managers import SceneEntityCfg

    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

    cs = SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", "lift_.*", "ur3_.*"])
    cs.resolve(base.scene)
    return mdp.illegal_contact(base, 1.0, cs) | mdp.bad_orientation(base, 1.0)


def eval_tracking(args):
    import torch

    from policy.walk.route import falls_per_km
    from policy.walk.walk_env_cfg import WalkEnvCfg

    cfg = WalkEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.curriculum.terrain_levels = None
    cfg.events.push_robot = None
    cfg.events.payload.params["p_deployed"] = 0.0
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (5.0, 5.0)
    env, policy = load_policy(make_env(cfg), args.ckpt)
    base = env.unwrapped
    r = base.scene["robot"]
    names = list(cfg.scene.terrain.terrain_generator.sub_terrains)
    props = torch.tensor([s.proportion for s in cfg.scene.terrain.terrain_generator.sub_terrains.values()])
    cum = torch.cumsum(props / props.sum(), 0)
    ncol = cfg.scene.terrain.terrain_generator.num_cols
    kind = torch.tensor([int(torch.nonzero(j / ncol + 0.001 < cum)[0]) for j in range(ncol)], device=base.device)
    obs = env.get_observations()
    stats = {n: {"err": 0.0, "werr": 0.0, "steps": 0, "dist": 0.0, "falls": 0} for n in names}
    age = torch.zeros(base.num_envs, device=base.device)
    dt = base.step_dt
    for _ in range(int(args.track_s / dt)):
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))
        cmd = base.command_manager.get_command("base_velocity")
        v = r.data.root_lin_vel_b[:, :2]
        err = torch.linalg.norm(cmd[:, :2] - v, dim=1)
        werr = (cmd[:, 2] - r.data.root_ang_vel_b[:, 2]).abs()
        k = kind[base.scene.terrain.terrain_types]
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
        fell = dones.bool() & ~timeouts
        ok = age > 0.5
        for i, n in enumerate(names):
            m = (k == i)
            mo = m & ok
            s = stats[n]
            s["err"] += float(err[mo].sum())
            s["werr"] += float(werr[mo].sum())
            s["steps"] += int(mo.sum())
            s["dist"] += float((torch.linalg.norm(v, dim=1)[m] * dt).sum())
            s["falls"] += int(fell[m].sum())
        age = torch.where(dones.bool(), torch.zeros_like(age), age + dt)
        age = torch.where(base.command_manager.get_term("base_velocity").time_left > 4.5, torch.zeros_like(age), age)
    out = {}
    for n, s in stats.items():
        out[n] = {"vel_err_xy_mps": s["err"] / max(1, s["steps"]), "yaw_rate_err_radps": s["werr"] / max(1, s["steps"]),
                  "dist_km": s["dist"] / 1000.0, "falls": s["falls"], "falls_per_km": falls_per_km(s["falls"], s["dist"])}
    tot = {k: sum(s[k] for s in stats.values()) for k in ("err", "werr", "steps", "dist", "falls")}
    out["all"] = {"vel_err_xy_mps": tot["err"] / tot["steps"], "yaw_rate_err_radps": tot["werr"] / tot["steps"],
                  "dist_km": tot["dist"] / 1000.0, "falls": tot["falls"],
                  "falls_per_km": falls_per_km(tot["falls"], tot["dist"])}
    return out


def eval_patrol(args, meta, full):
    import torch

    from policy.walk.route import WaypointFollower, polyline_length, route_to_goal, stance_from_meta

    from policy.walk.walk_env_cfg import WalkEnvCfg

    cfg = deck_cfg(WalkEnvCfg(), args.num_envs)
    cfg.terminations.base_contact = None
    cfg.terminations.bad_orientation = None
    env, policy = load_policy(make_env(cfg), args.ckpt)
    base = env.unwrapped
    gx, gy, gyaw = stance_from_meta(meta)
    route = full["patrol_route_m"]
    path = route_to_goal(route, (gx, gy))
    start = path[0]
    yaw0 = math.atan2(path[1][1] - start[1], path[1][0] - start[0])
    with torch.inference_mode():
        env.reset()
        place(base, start, yaw0)
    followers = [WaypointFollower(path, gyaw, v_max=args.v_max, pos_tol=args.pos_tol, yaw_tol=0.06, v_min=args.v_min, hold_tol=0.08, hold_yaw=0.08) for _ in range(base.num_envs)]
    obs = env.get_observations()
    dt = base.step_dt
    t, t_done = 0.0, [None] * base.num_envs
    falls = [False] * base.num_envs
    track = []
    while t < args.patrol_s and any(d is None for d in t_done):
        cmds = []
        for i, f in enumerate(followers):
            x, y, _, yaw = pose2d(base, i)
            vx, vy, wz, done = f.command(x, y, yaw)
            cmds.append((vx, vy, wz))
            if done and t_done[i] is None:
                t_done[i] = t
        set_cmd(base, cmds)
        with torch.inference_mode():
            obs = env.get_observations()
            obs, _, _, _ = env.step(policy(obs))
        v = base.scene["robot"].data.root_lin_vel_b[:, :2]
        c = torch.as_tensor(cmds, device=base.device)[:, :2]
        track.append(float(torch.linalg.norm(c - v, dim=1).mean()))
        fl = fallen(base)
        falls = [a or bool(b) for a, b in zip(falls, fl.tolist())]
        t += dt
    finals = [pose2d(base, i) for i in range(base.num_envs)]
    pos_err = [math.hypot(x - gx, y - gy) for x, y, _, _ in finals]
    from policy.walk.route import wrap

    return {"route_m": polyline_length(path), "start": [float(start[0]), float(start[1])], "goal": [gx, gy, gyaw],
            "time_to_pipe_s": [None if d is None else round(d, 2) for d in t_done],
            "final_pos_err_m": pos_err, "final_axial_err_m": [x - gx for x, _, _, _ in finals],
            "final_lateral_err_m": [y - gy for _, y, _, _ in finals], "final_yaw_err_rad": [abs(wrap(f[3] - gyaw)) for f in finals],
            "final_trunk_z_m": [f[2] for f in finals], "falls": sum(falls),
            "mean_cmd_tracking_err_mps": sum(track) / max(1, len(track))}


def backdrop_layer(scene: str, out: str):
    from pxr import Sdf, Usd, UsdPhysics

    src = Usd.Stage.Open(scene)
    layer = Sdf.Layer.CreateNew(out) if not os.path.exists(out) else Sdf.Layer.FindOrOpen(out)
    layer.Clear()
    layer.subLayerPaths.append(scene)
    st = Usd.Stage.Open(layer)
    st.SetEditTarget(layer)
    st.SetDefaultPrim(st.GetPrimAtPath("/World"))
    n = 0
    for prim in src.Traverse():
        if str(prim.GetPath()).startswith("/World/Robots"):
            continue
        if prim.IsA(UsdPhysics.Scene):
            st.OverridePrim(prim.GetPath()).SetActive(False)
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(st.OverridePrim(prim.GetPath())).CreateCollisionEnabledAttr(False)
            n += 1
    if src.GetPrimAtPath("/World/Robots").IsValid():
        st.OverridePrim("/World/Robots").SetActive(False)
    layer.Save()
    return out, n


def dress(stage, work, args):
    from pxr import UsdGeom

    UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground")).MakeInvisible()
    try:
        from policy import night_look
    except ImportError:
        return {}
    return night_look.apply(stage, "/World/Backdrop/RepairZone", work, full=True, dome=args.dome,
                            lamp_gain=args.lamp_gain, rig_gain=args.rig_gain, robot_root="/World/envs/env_0/Robot",
                            robot_color=args.robot_color)


def lead_camera(p, yaw, front=2.6, side=1.35, height=1.75):
    import numpy as np

    c, s = math.cos(yaw), math.sin(yaw)
    eye = np.array([p[0] + c * front - s * side, p[1] + s * front + c * side, height])
    look = np.array([p[0] + c * 0.3, p[1] + s * 0.3, 0.45])
    return eye, look


def render(args, meta, full):
    import carb
    import imageio
    import numpy as np
    import omni.usd
    import torch
    from isaaclab.envs import ViewerCfg
    from isaaclab.sim import RenderCfg

    from policy.walk.route import WaypointFollower, polyline_length, resample, route_to_goal, stance_from_meta
    from policy.walk.walk_env_cfg import WalkEnvCfg, apply_payload

    deck_z = float(meta.get("world", {}).get("deck_z_m", 0.0))
    os.makedirs(args.out_dir, exist_ok=True)
    layer, n_off = backdrop_layer(os.path.abspath(args.scene), os.path.join(os.path.abspath(args.out_dir), "backdrop.usda"))
    cfg = deck_cfg(WalkEnvCfg(), 1, scene=layer, deck_z=deck_z)
    cfg.terminations.base_contact = None
    cfg.terminations.bad_orientation = None
    cfg.scene.robot.spawn.make_instanceable = False
    cfg.viewer = ViewerCfg(eye=(0, 0, 3), lookat=(1, 0, 0), resolution=(1280, 720), origin_type="world")
    cfg.sim.render = RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", carb_settings={
        "/rtx/rendermode": args.renderer, "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False,
        "/rtx/post/histogram/enabled": False})
    gx, gy, gyaw = stance_from_meta(meta)
    work = (float(meta["damage"]["position_m"][0]), float(meta["main_pipe"]["y_m"]) + 0.08, 1.2)
    look_info = {}

    from isaaclab.scene import InteractiveScene

    scene_init = InteractiveScene.__init__

    def dressed_init(self, scene_cfg):
        scene_init(self, scene_cfg)
        look_info.update(dress(omni.usd.get_context().get_stage(), work, args))

    InteractiveScene.__init__ = dressed_init
    try:
        from policy.night_look import grade
    except ImportError:
        def grade(x):
            return x

    env, policy = load_policy(make_env(cfg, render=True), args.ckpt)
    base = env.unwrapped
    stage = omni.usd.get_context().get_stage()
    print("[walk] colliders off", n_off, "look", look_info, "origin", base.scene.env_origins[0].tolist(), flush=True)
    settings = carb.settings.get_settings()
    for key, val in cfg.sim.render.carb_settings.items():
        settings.set(key, val)
    path = route_to_goal(full["patrol_route_m"], (gx, gy))
    dense = resample(path, 0.05)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    k0 = int(np.searchsorted(s, s[-1] - args.start_back))
    sub = np.vstack([dense[k0:]])
    yaw0 = math.atan2(sub[min(10, len(sub) - 1)][1] - sub[0][1], sub[min(10, len(sub) - 1)][0] - sub[0][0])
    follower = WaypointFollower(sub, gyaw, v_max=args.v_max, pos_tol=args.pos_tol, yaw_tol=0.06, v_min=args.v_min, hold_tol=0.08, hold_yaw=0.08)
    with torch.inference_mode():
        env.reset()
        place(base, sub[0], yaw0)
        apply_payload(base, torch.zeros(1, device=base.device))
    dt = base.step_dt
    obs = env.get_observations()
    hero = np.array([gx + 2.6, gy + 2.1, 1.75]), np.array([gx, gy - 0.25, 0.8])
    stage.GetPrimAtPath(cfg.viewer.cam_prim_path).GetAttribute("focalLength").Set(args.focal)
    frames, log, t, t_stop, alpha = [], [], 0.0, None, 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    t_wall = time.time()
    cam_eye, cam_look = lead_camera((sub[0][0], sub[0][1], 0.0), yaw0)
    base.sim.set_camera_view(cam_eye.tolist(), cam_look.tolist())
    for _ in range(40):
        base.sim.render()
    while t < args.max_s:
        x, y, z, yaw = pose2d(base)
        vx, vy, wz, done = follower.command(x, y, yaw)
        if done and t_stop is None:
            t_stop = t
        state = "WALK"
        if t_stop is not None:
            state = "STOP"
            if t > t_stop + args.settle_s:
                alpha = min(1.0, alpha + dt / args.deploy_s)
                state = "LIFT" if alpha < 1.0 else "READY"
            with torch.inference_mode():
                apply_payload(base, torch.full((1,), alpha, device=base.device))
        set_cmd(base, [(vx, vy, wz)])
        with torch.inference_mode():
            obs = env.get_observations()
            obs, _, _, _ = env.step(policy(obs))
        p = np.array([x, y, z])
        eye, look = lead_camera(p, yaw)
        rem = max(0.0, follower.total - follower.progress)
        w = float(np.clip((args.blend_m - rem) / args.blend_m, 0.0, 1.0)) if t_stop is None else 1.0
        w = w * w * (3 - 2 * w)
        eye, look = (1 - w) * eye + w * hero[0], (1 - w) * look + w * hero[1]
        a = 1.0 - math.exp(-dt / 0.35)
        cam_eye, cam_look = cam_eye + a * (eye - cam_eye), cam_look + a * (look - cam_look)
        base.sim.set_camera_view(cam_eye.tolist(), cam_look.tolist())
        rgb = base.render()
        if rgb is not None and rgb.size and rgb[..., :3].mean() > 1 and t > 0.2:
            frames.append(grade(np.ascontiguousarray(rgb[..., :3])))
        v = base.scene["robot"].data.root_lin_vel_b[0]
        log.append({"t": round(t, 3), "state": state, "x": x, "y": y, "z": z, "yaw": yaw, "cmd": [vx, vy, wz],
                    "v": [float(v[0]), float(v[1])], "alpha": alpha})
        t += dt
        if state == "READY" and t > t_stop + args.settle_s + args.deploy_s + args.hold_s:
            break
    fps = int(round(1 / dt))
    out = os.path.join(args.out_dir, "walk_to_pipe.mp4")
    imageio.mimsave(out, frames, fps=fps, quality=8, macro_block_size=8)
    for q in (0.05, 0.3, 0.6, 0.85, 1.0):
        imageio.imwrite(os.path.join(args.out_dir, f"key_{int(q * 100):03d}.png"), frames[int(q * (len(frames) - 1))])
    x, y, z, yaw = pose2d(base)
    q = base.scene["robot"].data.joint_pos[0].tolist()
    names = base.scene["robot"].joint_names
    from policy.walk.route import wrap

    hand = {"stance_goal": [gx, gy, gyaw], "final": [x, y, z, yaw], "pos_err_m": math.hypot(x - gx, y - gy),
            "yaw_err_rad": abs(wrap(yaw - gyaw)), "trunk_height_m": z, "t_stop_s": t_stop,
            "walk_m": polyline_length(sub), "q": dict(zip(names, q))}
    with open(os.path.join(args.out_dir, "handoff.json"), "w") as f:
        json.dump(hand, f, indent=1)
    with open(os.path.join(args.out_dir, "walk_to_pipe.json"), "w") as f:
        json.dump({"frames": len(frames), "fps": fps, "wall_s": round(time.time() - t_wall), "log": log}, f)
    print("[walk] HANDOFF", json.dumps({k: v for k, v in hand.items() if k != "q"}), flush=True)
    print(f"[walk] {len(frames)} frames -> {out}", flush=True)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="track", choices=["track", "patrol", "render"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--meta", default=os.path.join(ROOT, "sim/output/rig_merged_meta.json"))
    p.add_argument("--full_meta", default=os.path.join(ROOT, "sim/output/full_rig_meta.json"))
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_merged.usda"))
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--track_s", type=float, default=60.0)
    p.add_argument("--patrol_s", type=float, default=150.0)
    p.add_argument("--v_max", type=float, default=1.0)
    p.add_argument("--start_back", type=float, default=8.0)
    p.add_argument("--focal", type=float, default=24.0)
    p.add_argument("--pos_tol", type=float, default=0.06)
    p.add_argument("--v_min", type=float, default=0.25)
    p.add_argument("--blend_m", type=float, default=4.0)
    p.add_argument("--settle_s", type=float, default=0.6)
    p.add_argument("--deploy_s", type=float, default=4.0)
    p.add_argument("--hold_s", type=float, default=1.2)
    p.add_argument("--max_s", type=float, default=30.0)
    p.add_argument("--renderer", default="RaytracedLighting")
    p.add_argument("--robot_color", default="graphite")
    p.add_argument("--dome", type=float, default=120.0)
    p.add_argument("--rig_gain", type=float, default=0.25)
    p.add_argument("--lamp_gain", type=float, default=1.0)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    args.enable_cameras = args.mode == "render"
    AppLauncher(args)
    with open(args.meta) as f:
        meta = json.load(f)
    with open(args.full_meta) as f:
        full = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "render":
        render(args, meta, full)
    else:
        res = eval_tracking(args) if args.mode == "track" else eval_patrol(args, meta, full)
        print(f"[walk] {args.mode.upper()}", json.dumps(res), flush=True)
        with open(os.path.join(args.out_dir, f"{args.mode}.json"), "w") as f:
            json.dump({"ckpt": args.ckpt, **res}, f, indent=1)
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

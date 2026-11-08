from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make(args, eval_mode=False):
    import gymnasium as gym

    from policy.walk.rma_env_cfg import TASK_RMA, WalkRMAEnvCfg

    cfg = WalkRMAEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if getattr(args, "blind", False):
        cfg.observations.policy.extrinsics.scale = 0.0
    if eval_mode:
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.curriculum.terrain_levels = None
        cfg.events.push_robot = None
        cfg.observations.policy.enable_corruption = False
        c = cfg.commands.base_velocity
        c.heading_command, c.rel_standing_envs, c.resampling_time_range = False, 0.0, (5.0, 5.0)
        cfg.events.rma_payload.params["mass"] = (0.0, args.max_mass)
        cfg.events.rma_deck_reset.params.update({"max_tilt": args.max_tilt, "max_heave": args.max_heave})
    return cfg, gym.make(TASK_RMA, cfg=cfg)


def teacher(args):
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from policy.walk.rma import EXT_DIM, expand_first_layer
    from policy.walk.rma_env_cfg import WalkRMAPPORunnerCfg

    cfg, raw = make(args)
    agent = WalkRMAPPORunnerCfg()
    agent.seed, agent.max_iterations, agent.device = args.seed, args.iters, "cuda:0"
    agent.policy.noise_std_type = "log"
    if args.lr:
        agent.algorithm.learning_rate, agent.algorithm.schedule = args.lr, "fixed"
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    os.makedirs(args.run_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=args.run_dir, device=agent.device)
    sd = dict(torch.load(args.base, map_location="cuda:0", weights_only=False)["model_state_dict"])
    sd["log_std"] = torch.log(sd.pop("std").clamp_min(1e-3))
    runner.alg.policy.load_state_dict(expand_first_layer(sd, EXT_DIM))
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    runner.save(os.path.join(args.run_dir, "teacher_blind.pt" if args.blind else "teacher.pt"))
    print(f"[rma] TEACHER_DONE {time.time() - t0:.0f}s", flush=True)


def load_actor(path, device):
    import torch

    from policy.walk.rma import mlp_from_state

    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    return mlp_from_state(sd).to(device).eval()


def student(args):
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from policy.walk.rma import BASE_OBS, EXT_DIM, HIST, PROPRIO, Adapter, History, r2_score

    cfg, raw = make(args)
    env = RslRlVecEnvWrapper(raw)
    dev = raw.unwrapped.device
    actor = load_actor(os.path.join(args.run_dir, "teacher.pt"), dev)
    adapter = Adapter(len(PROPRIO), HIST, EXT_DIM).to(dev)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    hist = History(raw.unwrapped.num_envs, HIST, len(PROPRIO), dev)
    obs = env.get_observations()["policy"]
    log = []
    for k in range(args.steps):
        hist.push(obs[:, PROPRIO])
        e = obs[:, BASE_OBS:]
        pred = adapter(hist.data)
        loss = ((pred - e) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        lat = e if k < args.steps // 3 else pred.detach()
        with torch.no_grad():
            act = actor(torch.cat([obs[:, :BASE_OBS], lat], 1))
        nxt, _, dones, _ = env.step(act)
        obs = nxt["policy"]
        ids = dones.nonzero().flatten()
        if len(ids):
            hist.reset(ids)
        if k % 100 == 0:
            r2 = r2_score(pred.detach(), e).tolist()
            log.append({"step": k, "loss": float(loss), "r2": r2})
            print(f"[rma] student {k} loss {float(loss):.4f} r2 {[round(x, 2) for x in r2]}", flush=True)
    torch.save(adapter.state_dict(), os.path.join(args.run_dir, "adapter.pt"))
    with open(os.path.join(args.run_dir, "student_log.json"), "w") as f:
        json.dump(log, f)
    export(args, actor, adapter, dev)
    print("[rma] STUDENT_DONE", flush=True)


def export(args, actor, adapter, dev):
    import torch
    from torch import nn

    from policy.walk.rma import BASE_OBS, HIST, PROPRIO

    class RMAPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor, self.adapter = actor, adapter

        def forward(self, obs, hist):
            return self.actor(torch.cat([obs, self.adapter(hist)], 1))

    m = RMAPolicy().cpu().eval()
    torch.onnx.export(m, (torch.zeros(1, BASE_OBS), torch.zeros(1, HIST, len(PROPRIO))),
                      os.path.join(args.run_dir, "policy_rma.onnx"), input_names=["obs", "history"],
                      output_names=["actions"], opset_version=17, dynamo=False,
                      dynamic_axes={"obs": {0: "n"}, "history": {0: "n"}, "actions": {0: "n"}})
    m.to(dev)


def evaluate(args):
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from policy.walk.rma import BASE_OBS, EXT_DIM, EXT_NAMES, HIST, PROPRIO, Adapter, History, r2_score
    from policy.walk.route import falls_per_km

    cfg, raw = make(args, eval_mode=True)
    env = RslRlVecEnvWrapper(raw)
    base = raw.unwrapped
    dev = base.device
    r = base.scene["robot"]
    old = load_actor(args.base, dev)
    actor = load_actor(os.path.join(args.run_dir, "teacher.pt"), dev)
    adapter = Adapter(len(PROPRIO), HIST, EXT_DIM).to(dev)
    adapter.load_state_dict(torch.load(os.path.join(args.run_dir, "adapter.pt"), map_location=dev))
    adapter.eval()
    zeros = torch.zeros(base.num_envs, EXT_DIM, device=dev)
    pols = {
        "base": lambda o, h: old(o[:, :BASE_OBS]),
        "rma": lambda o, h: actor(torch.cat([o[:, :BASE_OBS], adapter(h)], 1)),
        "oracle": lambda o, h: actor(o),
        "no_latent": lambda o, h: actor(torch.cat([o[:, :BASE_OBS], zeros], 1)),
    }
    bp = os.path.join(args.run_dir, "teacher_blind.pt")
    if os.path.exists(bp):
        blind = load_actor(bp, dev)
        pols["finetune_blind"] = lambda o, h: blind(torch.cat([o[:, :BASE_OBS], zeros], 1))
    dt = base.step_dt
    out = {}
    for name in [n for n in args.policies.split(",") if n in pols]:
        with torch.inference_mode():
            torch.manual_seed(args.seed)
            base._rma_gen.manual_seed(args.seed + 7)
            env.reset()
            obs = env.get_observations()["policy"]
            hist = History(base.num_envs, HIST, len(PROPRIO), dev)
            age = torch.zeros(base.num_envs, device=dev)
            s = {"err": 0.0, "werr": 0.0, "steps": 0, "dist": 0.0, "falls": 0}
            preds, tgts = [], []
            for k in range(int(args.track_s / dt)):
                hist.push(obs[:, PROPRIO])
                if name == "rma" and k % 5 == 0:
                    preds.append(adapter(hist.data).cpu())
                    tgts.append(obs[:, BASE_OBS:].cpu())
                nxt, _, dones, extras = env.step(pols[name](obs, hist.data))
                obs = nxt["policy"]
                cmd = base.command_manager.get_command("base_velocity")
                v = r.data.root_lin_vel_b[:, :2]
                ok = age > 0.5
                s["err"] += float(torch.linalg.norm(cmd[:, :2] - v, dim=1)[ok].sum())
                s["werr"] += float((cmd[:, 2] - r.data.root_ang_vel_b[:, 2]).abs()[ok].sum())
                s["steps"] += int(ok.sum())
                s["dist"] += float((torch.linalg.norm(v, dim=1) * dt).sum())
                to = extras.get("time_outs", torch.zeros_like(dones)).bool()
                s["falls"] += int((dones.bool() & ~to).sum())
                ids = dones.nonzero().flatten()
                if len(ids):
                    hist.reset(ids)
                age = torch.where(dones.bool(), torch.zeros_like(age), age + dt)
                age = torch.where(base.command_manager.get_term("base_velocity").time_left > 4.5, torch.zeros_like(age), age)
        out[name] = {"vel_err_xy_mps": s["err"] / max(1, s["steps"]), "yaw_rate_err_radps": s["werr"] / max(1, s["steps"]),
                     "dist_km": s["dist"] / 1000.0, "falls": s["falls"], "falls_per_km": falls_per_km(s["falls"], s["dist"])}
        if preds:
            r2 = r2_score(torch.cat(preds), torch.cat(tgts)).tolist()
            out[name]["latent_r2"] = dict(zip(EXT_NAMES, r2))
        print(f"[rma] eval {name} {json.dumps(out[name])}", flush=True)
    res = {"benchmark": {"num_envs": base.num_envs, "track_s": args.track_s, "seed": args.seed,
                         "payload_mass_kg": [0.0, args.max_mass], "arm_deployed_p": 0.5,
                         "deck_max_tilt_rad": args.max_tilt, "deck_max_heave_mps2": args.max_heave},
           "policies": out}
    with open(os.path.join(args.run_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[rma] EVAL_DONE", flush=True)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["teacher", "student", "eval"], required=True)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--track_s", type=float, default=40.0)
    p.add_argument("--max_mass", type=float, default=5.0)
    p.add_argument("--max_tilt", type=float, default=0.1)
    p.add_argument("--max_heave", type=float, default=1.5)
    p.add_argument("--policies", default="base,rma,oracle,no_latent")
    p.add_argument("--base", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--lr", type=float, default=0.0)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    {"teacher": teacher, "student": student, "eval": evaluate}[args.mode](args)
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

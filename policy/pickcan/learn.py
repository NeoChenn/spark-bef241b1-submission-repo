from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_env(n, seed):
    import gymnasium as gym

    importlib.import_module("policy.envs.pickcan_env")
    from policy.envs.pickcan_env_cfg import PickCanEnvCfg

    cfg = PickCanEnvCfg()
    cfg.scene.num_envs = n
    cfg.seed = seed
    env = gym.make("Rig-PickCan-Go2UR3Lift-v0", cfg=cfg)
    env.reset(seed=seed)
    return env, env.unwrapped, cfg


def collect(u, env, steps, log):
    import torch

    from policy.pickcan.script import ScriptedPicker

    N, dev = u.num_envs, u.device
    picker = ScriptedPicker(N, dev, dt=u.ctrl_dt, scale=u.cfg.action_scale)
    u.sense_geometry()
    O, A, D, S = [], [], [], []
    noise = (torch.rand(N, device=dev) < 0.5).float() * 0.3
    t0 = time.time()
    for step in range(steps):
        if step % 100 == 0:
            log(f"collect step {step} {time.time() - t0:.0f}s")
        with torch.inference_mode():
            o = u.geo_obs()
            a = picker.act(u._tip()[0], u.p_tgt, u.geom_grasp(), u.sealed, u._robot.data.joint_pos[:, u.lift_id])
            noisy = a.clone()
            noisy[:, :3] += noise[:, None] * torch.randn_like(a[:, :3])
            _, _, term, trunc, _ = env.step(noisy.clamp(-1.0, 1.0))
        done = (term | trunc).bool()
        succ = torch.zeros(N, dtype=torch.bool, device=dev)
        if done.any():
            st = u.last_episode_stats
            succ[st["env_ids"]] = st["success"]
            picker.reset(done.nonzero(as_tuple=False).squeeze(-1))
        O.append(o.clone())
        A.append(a.clone())
        D.append(done)
        S.append(succ)
    O, A, D, S = (torch.stack(x) for x in (O, A, D, S))
    T = O.shape[0]
    end = torch.full((N,), -1, dtype=torch.long, device=dev)
    ok = torch.zeros(N, dtype=torch.bool, device=dev)
    ends, keep = torch.empty(T, N, dtype=torch.long, device=dev), torch.empty(T, N, dtype=torch.bool, device=dev)
    for t in range(T - 1, -1, -1):
        end = torch.where(D[t], torch.full_like(end, t), end)
        ok = torch.where(D[t], S[t], ok)
        ends[t], keep[t] = end, ok & (end >= 0)
    from policy.pickcan.diffusion import CHUNK

    idx = torch.minimum(torch.arange(T, device=dev)[:, None, None] + torch.arange(CHUNK, device=dev)[None, :, None],
                        ends[:, None, :].clamp_min(0))
    chunks = A[idx, torch.arange(N, device=dev)[None, None, :]].permute(0, 2, 1, 3)
    n_eps, n_ok = int(D.sum()), int(S.sum())
    log(f"collect {T} steps x {N} envs in {time.time() - t0:.0f}s: {n_eps} episodes, {n_ok} successful "
        f"({n_ok / max(n_eps, 1):.0%}), {int(keep.sum())} samples kept")
    return O[keep], chunks[keep], {"episodes": n_eps, "successful": n_ok, "samples": int(keep.sum()),
                                   "collect_s": round(time.time() - t0, 1)}


def train(model, obs, act, minutes, log, name, bs=2048, lr=3e-4):
    import torch

    model.norm.fit(obs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0, it, n = time.time(), 0, obs.shape[0]
    total = None
    while time.time() - t0 < minutes * 60:
        i = torch.randint(0, n, (bs,), device=obs.device)
        loss = model.loss(obs[i], act[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        it += 1
        total = loss.item() if total is None else 0.99 * total + 0.01 * loss.item()
        if it % 2000 == 0:
            log(f"{name} it {it} loss {total:.4f} {time.time() - t0:.0f}s")
    return {"iters": it, "loss": round(total, 5), "train_s": round(time.time() - t0, 1)}


def evaluate(u, env, policy_fn, replan, max_steps, log, name):
    import torch

    from policy.pickcan.run import summarize

    N, dev = u.num_envs, u.device
    with torch.inference_mode():
        env.reset(seed=u.cfg.seed)
        u.sense_geometry()
    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    eps, last, plan, k = [], getattr(u, "last_episode_stats", None), None, 0
    lat = []
    for step in range(max_steps):
        with torch.inference_mode():
            if plan is None or k >= replan:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                plan = policy_fn(u)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
                k = 0
            a = plan[:, k] if plan.dim() == 3 else plan
            k += 1
            _, _, term, trunc, _ = env.step(a)
        d = (term | trunc).bool()
        st = getattr(u, "last_episode_stats", None)
        if st is not None and st is not last:
            last = st
            ids = st["env_ids"]
            for j in (~done_once[ids]).nonzero(as_tuple=False).squeeze(-1).tolist():
                eps.append({kk: (v[j].item() if hasattr(v, "dim") else v) for kk, v in st.items() if kk != "env_ids"})
            done_once[ids] = True
        if d.any() and hasattr(policy_fn, "reset"):
            policy_fn.reset(d.nonzero(as_tuple=False).squeeze(-1))
        if d.any() and plan is not None and plan.dim() == 3:
            plan = None
        if bool(done_once.all()):
            break
    res = summarize(eps, u.ctrl_dt)
    res["batch_inference_ms"] = round(1000 * sum(lat) / max(len(lat), 1), 2)
    log(f"eval {name}: " + json.dumps(res))
    return res


def latency(model, obs_dim, dev, n_steps):
    import torch

    x = torch.zeros(1, obs_dim, device=dev)
    for _ in range(5):
        model.sample(x, n_steps=n_steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        model.sample(x, n_steps=n_steps)
    torch.cuda.synchronize()
    return round(1000 * (time.perf_counter() - t0) / 50, 2)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--collect_envs", type=int, default=768)
    p.add_argument("--collect_steps", type=int, default=1000)
    p.add_argument("--diff_min", type=float, default=15.0)
    p.add_argument("--mlp_min", type=float, default=5.0)
    p.add_argument("--ddim", type=int, default=8)
    p.add_argument("--replan", type=int, default=4)
    p.add_argument("--out", default="runs/pickup/learn")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--load", default="")
    p.add_argument("--eval_envs", type=int, default=256)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless = True
    AppLauncher(args)

    import torch

    from policy.pickcan.diffusion import DiffusionPolicy, MLPPolicy, load_weights, n_params
    from policy.pickcan.script import ScriptedPicker

    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "learn.log"), "a")

    def log(msg):
        print(f"[learn] {msg}", flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    if args.load:
        env, u, cfg = make_env(args.eval_envs, args.seed)
        obs_dim = u.geo_obs().shape[1]
        dev = u.device
        diff, mlp = DiffusionPolicy(obs_dim).to(dev), MLPPolicy(obs_dim).to(dev)
        ext = ".npz" if os.path.exists(os.path.join(args.load, "mlp.npz")) else ".pt"
        load_weights(diff, os.path.join(args.load, "diffusion" + ext), dev)
        load_weights(mlp, os.path.join(args.load, "mlp" + ext), dev)
        report = {"loaded": args.load, "diffusion": {"params": n_params(diff)}, "mlp": {"params": n_params(mlp)}}
    else:
        env, u, cfg = make_env(args.collect_envs, args.seed + 100)
        obs, act, cinfo = collect(u, env, args.collect_steps, log)
        torch.save({"obs": obs.cpu(), "act": act.cpu()}, os.path.join(args.out, "demos.pt"))
        obs_dim = obs.shape[1]
        dev = obs.device
        report = {"collect": cinfo, "obs_dim": obs_dim}
        diff = DiffusionPolicy(obs_dim).to(dev)
        report["diffusion"] = train(diff, obs, act, args.diff_min, log, "diffusion")
        report["diffusion"]["params"] = n_params(diff)
        torch.save(diff.state_dict(), os.path.join(args.out, "diffusion.pt"))
        mlp = MLPPolicy(obs_dim).to(dev)
        report["mlp"] = train(mlp, obs, act, args.mlp_min, log, "mlp", lr=1e-3)
        report["mlp"]["params"] = n_params(mlp)
        torch.save(mlp.state_dict(), os.path.join(args.out, "mlp.pt"))
        del obs, act
    diff.eval()
    mlp.eval()
    report["diffusion"]["latency_ms_b1"] = latency(diff, obs_dim, dev, args.ddim)
    report["mlp"]["latency_ms_b1"] = latency(mlp, obs_dim, dev, 0)
    log("train done " + json.dumps(report))
    u.cfg.seed = args.seed

    picker = ScriptedPicker(u.num_envs, u.device, dt=u.ctrl_dt, scale=u.cfg.action_scale)

    class Scripted:
        def __call__(self, uu):
            return picker.act(uu._tip()[0], uu.p_tgt, uu.geom_grasp(), uu.sealed, uu._robot.data.joint_pos[:, uu.lift_id])

        def reset(self, ids):
            picker.reset(ids)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    results = {}
    for name, fn, rp in (("scripted", Scripted(), 1),
                         ("diffusion", lambda uu: diff.sample(uu.geo_obs(), n_steps=args.ddim, generator=g), args.replan),
                         ("mlp_bc", lambda uu: mlp.sample(uu.geo_obs()), args.replan)):
        picker.reset()
        results[name] = evaluate(u, env, fn, rp, 2000, log, name)
    report["eval"] = results
    report["eval_note"] = f"first episode of each of {u.num_envs} envs, seed {args.seed}, same scenarios for every policy"
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("REPORT " + json.dumps(report))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)

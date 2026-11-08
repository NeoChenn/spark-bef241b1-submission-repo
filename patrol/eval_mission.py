"""Closed-loop mission eval in the 2D sim: shuttle legs, pick-ups, false picks, collisions, blocked events.

  python -m patrol.eval_mission --policy pure_pursuit --episodes 30
  python -m patrol.eval_mission --policy runs/patrol/nav_v1/policy.onnx --episodes 30 --legs 4 --beams 360
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from patrol.controller import PatrolController, cmd_to_action
from patrol.mission import MissionCfg, PatrolMission
from patrol.sim2d import PatrolSim, SimCfg

PICKABLE_DIAM = 0.11  # gripper opens 0.12 m; 1 cm grasp margin, same as DetectCfg.pick_max_width
IN_PATH_TUBE = 0.32


def pickable_in_path(sim: PatrolSim, ref) -> list[int]:
    out = []
    for k in range(sim.cfg.n_obstacles):
        if not bool(sim.obs_on[0, k]) or 2 * float(sim.obs_r[0, k]) > PICKABLE_DIAM:
            continue
        _, lat, _ = ref.project(sim.obs_c[0, k].numpy()[None])
        if abs(float(lat[0])) - float(sim.obs_r[0, k]) < IN_PATH_TUBE:
            out.append(k)
    return out


def run_episode(policy: str, seed: int, legs: int, beams: int, max_steps: int, mcfg: MissionCfg) -> dict:
    sim = PatrolSim(SimCfg(num_envs=1, seed=seed, start_frac=0.0, obstacle_r=(0.03, 0.2), p_obstacle=0.8,
                           max_steps=10**9))
    angles_t = -math.pi + (torch.arange(beams) + 0.5) * (2 * math.pi / beams)
    angles = angles_t.numpy()
    ctl = PatrolController(policy, sim.P[0].numpy(), ds=sim.cfg.ds)
    mission = PatrolMission(ctl, mcfg)
    targets = set(pickable_in_path(sim, ctl.ref))
    n_targets = len(targets)
    picked_ok, false_picks, collision, steps = 0, 0, False, 0
    false_info, collide_state = [], None
    with torch.no_grad():
        while mission.legs < legs and steps < max_steps:
            ranges = sim.lidar(beams=angles_t)[0].numpy()
            pose = (float(sim.pos[0, 0]), float(sim.pos[0, 1]), float(sim.yaw[0]))
            n_rev = mission.reversed_count
            cmd = mission.step(steps * sim.cfg.dt, ranges, angles, pose, float(sim.v[0]), float(sim.w[0]))
            if mission.reversed_count != n_rev:
                sim.reverse(torch.tensor([0]))
            if cmd.event == "picked":
                hit = None
                for k in range(sim.cfg.n_obstacles):
                    if (bool(sim.obs_on[0, k]) and 2 * float(sim.obs_r[0, k]) <= PICKABLE_DIAM
                            and np.linalg.norm(sim.obs_c[0, k].numpy() - cmd.arm_target_xy) < 0.2):
                        hit = k
                        break
                if hit is None:
                    false_picks += 1
                    d = (sim.obs_c[0] - torch.from_numpy(cmd.arm_target_xy).float()).norm(dim=-1)
                    d = torch.where(sim.obs_on[0], d, torch.full_like(d, float("inf")))
                    k = int(d.argmin())
                    false_info.append({"nearest_m": round(float(d[k]), 3), "nearest_diam": round(2 * float(sim.obs_r[0, k]), 3)})
                else:
                    sim.remove_obstacle(0, hit)
                    picked_ok += 1
                    targets.discard(hit)
            _, _, _, info = sim.step(torch.from_numpy(cmd_to_action(cmd.vx, cmd.wz))[None], auto_reset=False)
            steps += 1
            if bool(info["collision"][0]):
                collision, collide_state = True, mission.state
                dob = (sim.pos[0] - sim.obs_c[0]).norm(dim=-1) - sim.obs_r[0] - sim.cfg.robot_r
                dob = torch.where(sim.obs_on[0], dob, torch.full_like(dob, 9.0))
                k = int(dob.argmin())
                hit = ("small_obj" if 2 * float(sim.obs_r[0, k]) <= PICKABLE_DIAM else "big_obj") if float(dob[k]) < 0.02 else "wall"
                break
    return {"seed": seed, "legs": mission.legs, "targets": n_targets, "picked": picked_ok,
            "missed": len(targets), "false_picks": false_picks, "false_pick_info": false_info,
            "collision": collision, "collision_state": collide_state, "collision_hit": hit if collision else None,
            "reports": len(mission.reports),
            "blocked": mission.blocked_events, "sim_time_s": round(steps * sim.cfg.dt, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="pure_pursuit")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--legs", type=int, default=4)
    ap.add_argument("--beams", type=int, default=360)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--out", default="runs/patrol/mission_eval.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    eps = [run_episode(a.policy, 1000 + e, a.legs, a.beams, a.max_steps, MissionCfg()) for e in range(a.episodes)]
    n = len(eps)
    tgt = sum(e["targets"] for e in eps)
    summary = {
        "policy": a.policy, "episodes": n, "legs_target": a.legs, "beams": a.beams,
        "all_legs_completed_rate": sum(e["legs"] >= a.legs for e in eps) / n,
        "mean_legs": sum(e["legs"] for e in eps) / n,
        "collision_rate": sum(e["collision"] for e in eps) / n,
        "pickable_in_path": tgt, "picked": sum(e["picked"] for e in eps),
        "pick_rate": sum(e["picked"] for e in eps) / max(tgt, 1),
        "false_picks": sum(e["false_picks"] for e in eps),
        "blocked_events": sum(e["blocked"] for e in eps),
        "reports": sum(e["reports"] for e in eps),
        "collisions_by_cause": {f"{s}:{h}": sum(1 for e in eps if e["collision_state"] == s and e["collision_hit"] == h)
                                for s, h in {(e["collision_state"], e["collision_hit"]) for e in eps if e["collision"]}},
        "mean_sim_time_s": sum(e["sim_time_s"] for e in eps) / n,
        "wall_sec": round(time.time() - t0, 1),
        "per_episode": eps,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    return summary


if __name__ == "__main__":
    main()

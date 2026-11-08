# patrol: path following + pick-up on a predefined path

A Go2 walks back and forth along a predefined path between two bins. A PPO policy keeps it on the path
and away from the walls using a 2D lidar, and a rule-based lidar detector finds objects in the path. It
picks up what fits in its container and dumps the container at the bins. Training happens in a 2D sim;
the sim team tests it in Isaac Sim (see `integration/PATROL_INTERFACE.md`).

| File | What |
|---|---|
| `features.py` | obs/action spec shared by training and deployment (53-D obs, (vx, wz) action) |
| `sim2d.py` | batched torch 2D sim: random paths, pipe walls, objects, lidar, lagged unicycle base, pushes |
| `ppo.py` | PPO trainer, pure-pursuit baseline, held-out evaluation, ONNX export |
| `modal_app.py` | trains on Modal CPU containers (workspace `neochenn`, volume `rust-ckpt:/patrol/<run>`) |
| `controller.py` | deploy-side `PatrolController`: scan + pose + path → (vx, wz), numpy + onnxruntime |
| `detect.py` | lidar object detector (`detect_frame`, `Tracker`) with size → pickable |
| `mission.py` | `PatrolMission` state machine: FOLLOW / APPROACH / PICK / DUMP / BLOCKED, shuttles S ↔ E |
| `eval_detect.py` | detector threshold sweep against sim ground truth → `DETECT_RESULTS.md` |
| `eval_mission.py` | full shuttle missions in the 2D sim: picks, collisions, legs completed |

## Commands

```bash
env/Scripts/python.exe -m patrol.ppo --iters 300 --envs 1024 --out runs/patrol/nav_v1   # local train (CPU)
modal run --detach patrol/modal_app.py --runs nav_v1:0,nav_v1b:1 --iters 250            # Modal CPU train
modal volume get rust-ckpt patrol/nav_v1 runs/patrol/                                   # fetch results
env/Scripts/python.exe -m patrol.eval_detect                                            # detector sweep
env/Scripts/python.exe -m patrol.eval_mission                                           # mission eval
env/Scripts/python.exe -m pytest -q tests/test_patrol_sim.py                            # tests
```

Outputs: `runs/patrol/<run>/{policy.onnx, policy_meta.json, deploy.pt, model.pt, eval.json, progress.jsonl}`.

## Results

| | goal rate | collision rate | mean lateral error | mean speed |
|---|---|---|---|---|
| PPO (`nav_v1b`, 250 iters, 16.4M steps, 17 min on 16 CPU) | 0.810 | 0.189 | 0.098 m | 0.98 m/s |
| PPO (`nav_v1`) | 0.798 | 0.202 | 0.086 m | 0.99 m/s |
| pure-pursuit baseline | 0.730 | 0.218 | 0.095 m | 0.60 m/s |

1000 held-out 20 m paths, with random pushes, narrowings and objects on the path. PPO beats the baseline but
is well short of the target (goal >= 95%, collision <= 1%): it drives at full speed and gets pushed into walls.
`nav_v2` (fine-tuned from `nav_v1b` with collision -30, clearance band 0.3 m, w_clear 2) is the fix in progress.

Detector (see `DETECT_RESULTS.md`): precision 0.998, recall 0.988, F1 0.993 at 360 beams; pickable decision 96.2%.

Mission eval (`eval_mission`, 30 episodes x 4 legs, 360 beams, `nav_v1b`): 53% complete all legs, 47% end in a
collision, 6/16 pickable objects picked, 0 false picks. Collisions dominate; the nav policy is the bottleneck.

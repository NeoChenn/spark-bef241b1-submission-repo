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
| **PPO (`nav_v3a`, from v2, 180 beams, 4 objects, 400 iters x 4096 envs, 14 min on L4)** | **0.832** | **0.096** | 0.106 m | 0.83 m/s |
| PPO (`nav_v3b`, same, other seed) | 0.800 | 0.083 | 0.109 m | 0.75 m/s |
| PPO (`nav_v2`, from v1b, collision -30, clearance band 0.3 m) | 0.828 | 0.159 | 0.109 m | 0.91 m/s |
| PPO (`nav_v1b`, 250 iters, 16.4M steps, 17 min on 16 CPU) | 0.810 | 0.189 | 0.098 m | 0.98 m/s |
| PPO (`nav_v1`) | 0.798 | 0.202 | 0.086 m | 0.99 m/s |
| pure-pursuit baseline | 0.730 | 0.218 | 0.095 m | 0.60 m/s |

1000 held-out 20 m paths, with random pushes, narrowings and objects on the path (v3 rows are evaluated
with the 180-beam lidar they were trained on). Deploy `nav_v3a`. Still short of the target (goal >= 95%,
collision <= 1%). Ablation on `nav_v2`: with no objects it reaches 99.8% goal / 0.3% collision even with
pushes, so path following is solved and object avoidance is the remaining gap (small objects: 44% collision
at 72 beams, 26% at 360; large objects: 20%). v3 cut collisions 40% but now sometimes stalls behind an
object (timeouts 7%), which the mission handles as BLOCKED and turns back.

Detector (see `DETECT_RESULTS.md`): precision 0.998, recall 0.988, F1 0.993 at 360 beams; pickable decision 96.2%.

Mission eval (`eval_mission`, 30 episodes x 4 legs = ~80 m each, 360 beams):

| policy | all 4 legs done | collision | picked / pickable | false picks | blocked events |
|---|---|---|---|---|---|
| `nav_v3a` | 63% | 37% | 6 / 16 | 0 | 17 |
| `nav_v1b` | 53% | 47% | 6 / 16 | 0 | 0 |

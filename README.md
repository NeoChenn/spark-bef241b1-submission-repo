# Rig Swarm: pipe rust detection + repair (Agentverse 2026, Robotics track)

A simulated quadruped with a scrubber arm finds rust on pipework and fixes it:
detect → spray → wait 30 s → scrub (PPO) → re-detect → paint → verify.

| Path | What |
|---|---|
| `detection/` | Rust classifier (LoRA fine-tune), patch heatmap localizer, ONNX export. Runs on Modal. |
| `policy/envs/` | Isaac Lab scrub environment + rust-state model (pure torch). |
| `policy/skills/` | Classical spray / paint coverage skills. |
| `policy/orchestrator.py` | Perception-driven state machine that sequences the repair. |
| `infra/` | Shared Modal image + volumes (Isaac Sim 5.1 / Isaac Lab on L40S). |
| `integration/INTERFACE.md` | Contract with the Isaac Sim scene (prim paths, rust texture, robot). |

Setup: `uv venv --python 3.11 .venv && uv pip install --python .venv modal torch numpy pytest`

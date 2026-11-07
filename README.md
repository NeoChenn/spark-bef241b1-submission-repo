# Flange-integrity robot (working title)

Agentverse 2026, Robotics Track. An embodied agent that patrols an offshore platform, reads bolt
torque stripes and gas levels, decides which flange bolts need re-tightening, and acts behind a gas
interlock and a human permit-to-work gate.

The concept is not final. Read [docs/problem-statement.md](docs/problem-statement.md) for the problem,
sources and the items still to verify.

## Status (be honest in the submission)

| Part | State |
|---|---|
| Safety supervisor (`flange_robot/safety.py`) | Done, unit-tested. Deterministic rules, not learned. |
| Decision log / perception receipts (`decision_log.py`) | Done. JSONL, one event per decision and per Gemini call. |
| Gemini wrapper (`gemini_client.py`) | Written and unit-tested with a fake backend. **Live call path untested: no key yet.** |
| MuJoCo environment (`env.py`) | **Skeleton.** One flange, 8 bolts, a point-mass robot, scene-change hooks. No arm, no reward. |
| Policy (PPO) | Not started. Needs a reward, a training env and a greedy baseline to beat. |
| Perception model | Not started. Currently the env returns noisy readings directly. |
| Console UI, video, technical summary | Not started. |

All limits in `config.py` are illustrative placeholders. Replace them with sourced values.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
```

Run a scene by hand:

```python
from flange_robot.env import Action, FlangeEnv

env = FlangeEnv(seed=0)
env.loosen_bolt(0, 0.4)               # a bolt turns
for _ in range(300):
    obs = env.step(Action("move_to", 0))
print(obs["bolts"])                    # noisy stripe readings, with confidence
```

## Layout

```
flange_robot/
  config.py         model names, safety limits, edge budget, usage guard (one place)
  safety.py         deterministic safety verdicts, overrides the policy
  decision_log.py   append-only JSONL evidence log
  gemini_client.py  slow-path reasoning with retries, rate guard, logging, local fallback
  env.py            MuJoCo skeleton with live scene-change hooks
tests/              24 tests, all passing
docs/               problem statement
```

## Design rules

- **Perception drives action.** Every decision is logged with its evidence.
- **Safety is deterministic.** Gas, people, confidence and the permit gate are rules that override any policy.
- **Gemini is the slow path only.** It never sits in the real-time loop. If it fails, the robot falls back to flagging a human.
- **Honest commits.** No replayed paths, no hardcoded model output standing in for real calls.

## API keys: read this

1. Copy `.env.example` to `.env` and put the Gemini key there. `.env` is gitignored.
2. **Never commit a key.** A leaked key gets every key on the project revoked. If one leaks, delete and recreate it in AI Studio. A new commit does not remove it from git history.
3. The hackathon account is deleted about a day after the event. Export everything you want to keep.
4. Do not loop model calls. The rate guard caps calls per minute, but keep usage modest.

## Owners (fill in)

| Area | Owner |
|---|---|
| World and sim | TODO |
| Policy and models | TODO |
| Console / product UI | TODO |
| Technical summary, video, pitch | TODO |

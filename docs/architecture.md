# Architecture (draft)

Working title: flange-integrity robot. See [problem-statement.md](problem-statement.md) for the problem and sources.
Items marked **[verify]** are unconfirmed and must be checked before they appear in the technical summary.

## One-line description

A quadruped with a gripper patrols an offshore platform. It reads bolt torque stripes and gas levels, decides which flange bolts need re-tightening, routes around hazards that appear mid-run, and re-tightens loose bolts on its own (no approval step) behind a deterministic safety layer, then reports to the supervisor.

## Layers

| Layer | Runs | Does | Fails to |
|---|---|---|---|
| Task planner | Gemini, slow path, off the control loop | Chooses the next zone, interprets findings, writes the operator report | Flag a human (never a silent guess) |
| Navigation | Local | Costmap from sensor data, global route, local avoidance, replans when the scene changes | Stop and flag a human |
| Recon skill | Local | Approach a flange, read torque stripes and gas, record findings with confidence | Report "unsure", never "fine" |
| Fine-action skill | Local | Align the tool, re-torque a bolt, re-read the stripe to verify | Abort and hand back |
| Safety supervisor | Local, rules, **already built** (`flange_robot/safety.py`) | Overrides every layer above | Fail safe: stop and retreat |

Rules that hold across all layers:
- **Perception drives action.** Every decision is logged with its evidence (`decision_log.py`).
- **Gemini is never in the real-time loop.** If it is unavailable, the robot flags a human.
- **Safety is rules, not learning.** Gas, people and low confidence override any policy. There is no approval gate by default; it can be switched on with `PolicyConfig.require_permit`.

## Skills (what the planner can choose)

Each skill has a precondition it must check and a postcondition it must verify.

| Skill | Precondition | Postcondition |
|---|---|---|
| `navigate_to(zone)` | Safety verdict is ALLOW | Arrived, or replanned, or flagged |
| `inspect(flange)` | Within sensor range, gas below limit | Findings list with a confidence per bolt |
| `retorque(bolt)` | In reach, confidence above the minimum, gas below the limit, diagnosis is a loose bolt | Stripe re-read and aligned |
| `retreat()` | Always allowed | At the safe point |
| `request_human(reason)` | Always allowed | Operator has the evidence |
| ~~`weld()`~~ | Not implemented on purpose | The agent refuses and alerts the supervisor (hot-work rules) |

Refusing hot work is a deliberate "what it cannot do" moment for the pitch.

## Live scene changes (the unscripted change in the video)

| Trigger | What the robot must do | Already in `env.py`? |
|---|---|---|
| **Oil cans fall in its path** (several; some roll) | Detect them through its sensor, replan around them | No, next build step |
| A can lands on the target bolt | Pick another bolt or flag a human | No |
| Gas leak starts | Safety layer stops it and it retreats | Yes: `trigger_gas_leak()` |
| A bolt loosens mid-run | Notice the stripe shift and re-prioritise | Yes: `loosen_bolt()` |
| Dust or spray degrades the camera | Confidence drops, so it slows, re-reads or flags | Yes: `set_dust()` |

The cans must be detected through the simulated sensor. Reading object positions from the physics engine would break "every decision traces to what it perceived."

## Simulator decision

The team chose **Isaac Sim on Modal** (L40S) for the demo and video world. See `isaac/README.md`. Trade-offs to accept knowingly:
- Slow iteration: the first build takes 10+ minutes, and there is no live GUI, only saved frames turned into video.
- The brief asks for a video "captured from the stream" with one unscripted change in a single continuous take. **Ask the organisers whether stitched frames qualify.** A live scene-change trigger may be hard over Modal.
- The MuJoCo environment stays as a fast twin for tests and policy training. The policy's observation and action layout must be identical in both worlds, and the contract should be written down once.

## Embodiment decision

- **Target:** a quadruped with an arm and gripper. MuJoCo Menagerie lists Boston Dynamics Spot with 19 degrees of freedom, consistent with 12 leg joints plus a 7-joint arm and gripper **[verify by opening the model]**. Unitree Go2 and ANYmal are also there. Unitree Z1 is a separate arm that could be mounted on Go2 with extra work.
- **Biggest risk: locomotion.** Menagerie provides models, not gait controllers. Do not spend a day training a walking policy.
  - Plan A: a pretrained quadruped policy from MuJoCo Playground **[verify it exists for the chosen robot]**.
  - Plan B: the agent sends velocity commands to a simplified base controller. State this plainly in the technical summary.
- The current `env.py` uses a point-mass robot so work can proceed before the embodiment is settled.

## Assets

- Sources found: Sketchfab, CGTrader, Meshy, Tripo, Free3D, TurboSquid.
- **Check every licence** (attribution rules). Keep a credits list.
- Convert to glTF or OBJ. Use the mesh for appearance only and simple boxes and cylinders for collision.
- Keep polygon counts low so the browser demo stays smooth.

## Navigation: Dimensional (DimOS) spike

DimOS is an open-source agentic-robotics framework. Its landing page lists navigation (SLAM, dynamic obstacle avoidance, route planning, exploration), manipulation, perception, spatial memory and LLM agents via MCP. I read only the landing page **[verify in the docs and repo]**.

- **Time-box: 30 to 45 minutes.** Install it, run the sim quickstart, and see whether it can drive our MuJoCo scene.
- If it works, reuse it and credit it clearly. If not, write a simple costmap with A* and replanning. Check the licence either way.
- The technical summary must say what is ours and what is DimOS.

## Edge budget (to fill with measured numbers)

| Component | Hardware | Latency |
|---|---|---|
| Detector | TODO | TODO |
| Navigation and policy | TODO | TODO |
| Safety supervisor | CPU | TODO |
| Gemini reasoning | Remote, off the control loop | Not in the loop |

Slide 07 says decisions that depend on a remote service score lowest on autonomy, so the robot must work with Gemini unavailable.

## Honest limits to state

- Simulation only; perception uses synthetic data.
- Locomotion and tool alignment are simplified (see above).
- Safety limits in `config.py` are placeholders until sourced.
- Fatality statistics are US and mostly onshore.

## Build order

1. Falling oil cans, a costmap and A* with replanning in `env.py`, with tests.
2. Perception that turns sensor readings into obstacle and bolt detections.
3. A greedy baseline planner, then a learned policy to compare against it.
4. Embodiment swap (Spot, Go2 or ANYmal) and the rig assets.
5. Console UI showing the decision trace, then the 60-second video script.

## Owners (fill in)

| Area | Owner |
|---|---|
| World, assets, embodiment | TODO |
| Navigation and policy | TODO |
| Perception and Gemini planner | TODO |
| Console UI, video, pitch, technical summary | TODO |

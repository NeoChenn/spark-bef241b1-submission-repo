# Isaac Sim on Modal

Renders Isaac Lab scenes on a Modal GPU and returns a video. **Written but not yet run.** Nobody has confirmed the first build works.

## Run the smoke test (you do this; it needs your Modal login)

```bash
pip install modal
modal setup                          # one-time login
modal run isaac/modal_app.py         # ANYmal-D quadruped -> renders/smoke.mp4
```

Expect the first run to take 10+ minutes while the image downloads and compiles. Later runs reuse the cached image.

If it works, you get `renders/smoke.mp4` (gitignored). Tell us. If it fails, copy the log lines printed after `RETURNCODE` and `STDOUT_TAIL`.

## Rules for this setup

- **L40S only.** A100 and H100 lack RT cores and are unsupported.
- **No live GUI.** We save frames and encode a video. Check with the organisers whether frames stitched into a video count as "captured from the stream" for the submission video.
- **Cost.** Each run uses credits. Stop Modal apps when finished. Never loop runs in a script.
- **Pinned versions** (Isaac Sim 5.1.0, Isaac Lab v2.3.0) come from the organisers' example. Change them only deliberately.
- **Credit and licence.** The image and render script are adapted from https://github.com/TheFloatingString/isaac-lab-on-modal, which shows no LICENSE file. Credit it in the technical summary.

## Files

- `modal_app.py`: Modal image, GPU, `render()` function and a `main` entrypoint
- `render_scene.py`: runs inside the container; steps a task and saves frames

## Next steps (not built)

1. Confirm the smoke test works.
2. Check which quadruped tasks and robot configs exist in Isaac Lab v2.3.0 (an ANYmal-D task works in the reference; whether a Spot-with-arm config exists is **unverified**).
3. Build the rig scene: platform and pipework assets (USD), flange and bolts, oil cans dropped mid-run.
4. Load the trained policy (`.onnx`) and drive the robot from our control loop.

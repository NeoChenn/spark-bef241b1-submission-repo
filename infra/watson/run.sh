#!/usr/bin/env bash
# Isaac Lab 2.3.2 / Isaac Sim 5.1 on the GB10 (aarch64) box.
#   docker build -t rig-scrub-isaaclab:5.1 -f infra/watson/Dockerfile.isaaclab infra/watson
#   infra/watson/run.sh policy.train_scrub --worker --num_envs 1024 --iters 150 --run_dir /repo/runs/scrub_watson
#   infra/watson/run.sh policy.play_scrub --worker --ckpt /repo/runs/scrub_watson/model_150.pt --out_dir /repo/runs/scrub_watson --meta /repo/sim/output/rig_deck_meta.json --force_patch
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/runs" && chmod 777 "$REPO/runs"
exec docker run --rm --name "rig-scrub-$(date +%s)" --device nvidia.com/gpu=all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONPATH=/repo -e PYTORCH_JIT="${PYTORCH_JIT:-0}" \
  -v "$REPO:/repo" -v rig-scrub-kitcache:/isaac-sim/kit/cache -v rig-scrub-ovcache:/isaac-sim/.cache \
  -w /repo --entrypoint /isaac-sim/python.sh rig-scrub-isaaclab:5.1 -m "$@" --headless

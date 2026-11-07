#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/runs" && chmod 777 "$REPO/runs"
exec docker run --rm --name "${NAME:-rig-scrub}-$(date +%s)" --device nvidia.com/gpu=all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONPATH=/repo -e PYTORCH_JIT="${PYTORCH_JIT:-0}" \
  -v "$REPO:/repo" -v rig-scrub-kitcache:/isaac-sim/kit/cache -v rig-scrub-ovcache:/isaac-sim/.cache \
  -w /repo --entrypoint /isaac-sim/python.sh rig-scrub-isaaclab:5.1 -m "$@" --headless

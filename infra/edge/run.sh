#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${EDGE_WORK:-$REPO/runs/edge}"
TAG="${1:-contended}"
shift || true
IMAGE="${EDGE_IMAGE:-nvcr.io/nvidia/tensorrt:25.10-py3}"
NAME="rig-edge-bench-$TAG"
test -f "$WORK/bundle/detector.onnx" || { echo "missing $WORK/bundle; run: uv run python -m infra.edge.prep"; exit 1; }
mkdir -p "$WORK/engines" "$WORK/out" "$WORK/pip"
chmod 777 "$WORK/engines" "$WORK/out" "$WORK/pip"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --name "$NAME" --device nvidia.com/gpu=all \
  -v "$WORK/bundle:/work/bundle:ro" -v "$WORK/engines:/work/engines" -v "$WORK/out:/work/out" \
  -v "$WORK/pip:/work/pip" -v "$REPO/infra/edge:/work/edge:ro" "$IMAGE" bash -c "
    pip install -q --target /work/pip onnxruntime==1.31.1 2>/dev/null || true
    PYTHONPATH=/work/pip python3 /work/edge/bench.py --tag $TAG $*"

#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${EDGE_WORK:-$REPO/runs/edge}"
TAG="${1:-contended}"
shift || true
IMAGE="${EDGE_IMAGE:-nvcr.io/nvidia/tensorrt:25.10-py3}"
test -f "$WORK/bundle/detector.onnx" || { echo "missing $WORK/bundle; run: uv run python -m infra.edge.prep"; exit 1; }
mkdir -p "$WORK/engines" "$WORK/out" "$WORK/pip"
chmod 777 "$WORK/engines" "$WORK/out" "$WORK/pip"
docker run -d --init --rm --memory 16g --name "rig-edge-bench-$TAG" --device nvidia.com/gpu=all -e PYTHONUNBUFFERED=1 \
  -v "$WORK/bundle:/work/bundle:ro" -v "$WORK/engines:/work/engines" -v "$WORK/out:/work/out" \
  -v "$WORK/pip:/work/pip" -v "$REPO/infra/edge:/work/edge:ro" "$IMAGE" bash -lc "
    (test -d /work/pip/onnxruntime || pip install -q --root-user-action=ignore --target /work/pip --no-deps \
      onnxruntime==1.31.1 flatbuffers protobuf coloredlogs humanfriendly packaging sympy mpmath) &&
    PYTHONPATH=/work/pip python3 /work/edge/bench.py --tag $TAG $* 2>&1 | tee /work/out/$TAG.log"
echo "started rig-edge-bench-$TAG; follow with: docker logs --tail 20 rig-edge-bench-$TAG"

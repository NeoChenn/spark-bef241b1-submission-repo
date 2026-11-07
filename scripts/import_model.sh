#!/usr/bin/env bash
# Import a 3D model into sim/assets/rig_model/<name>/ using the Isaac container (and Blender for .blend/.dae/.3ds/.x3d).
#   scripts/import_model.sh path/inside/repo/model.glb --name rig [import_model.py options]
set -euo pipefail
cd "$(dirname "$0")/.."
input="$1"; shift
[[ -f "$input" ]] || { echo "no such file: $input" >&2; exit 1; }
case "$(realpath "$input")" in "$PWD"/*) ;; *) echo "input must be inside $PWD" >&2; exit 1;; esac
rel="$(realpath --relative-to="$PWD" "$input")"
tag="rig-import"
gpu_args=(--device nvidia.com/gpu=all --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y)
mkdir -p .work/import_tmp sim/assets/rig_model
chmod a+rwX .work/import_tmp sim/assets/rig_model

ext="${input##*.}"; ext="${ext,,}"
extra=()
if [[ " blend dae 3ds x3d wrl abc " == *" $ext "* ]]; then
  docker image inspect "$tag-blender:latest" >/dev/null 2>&1 || docker build -t "$tag-blender:latest" -f docker/blender.Dockerfile docker
  glb=".work/import_tmp/$(basename "${input%.*}").glb"
  docker run --rm --name "$tag-blender-$$" --user "$(id -u):$(id -g)" -v "$PWD":/work -w /work "$tag-blender:latest" \
    --python sim/tools/blender_to_glb.py -- "/work/$rel" "/work/$glb"
  extra=(--source-name "$(basename "$input")")
  rel="$glb"
fi

docker run --rm --name "$tag-run-$$" "${gpu_args[@]}" -e PYTHONPATH=/work \
  -v "$PWD":/work -v "$tag-cache":/isaac-sim/.cache -v "$tag-kit":/isaac-sim/kit/cache \
  -v "$tag-nv":/isaac-sim/.nv/ComputeCache -w /work rig-sim:6.2.0 \
  /work/sim/tools/import_model.py "/work/$rel" "${extra[@]}" "$@"

#!/usr/bin/env bash
# Build image-patch/perlayer/: the image's EXL3 loader and DeepSeek V4 model file with per-layer-experts.diff
# applied (plans/PLAN_LOADER_PATCH.md §4.4). The output is not in git; compose-perlayer.yml mounts it.
# The script stops if the diff does not apply, for example after a change of the image.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
FILES="vllm/model_executor/layers/quantization/exl3.py vllm/models/deepseek_v4/nvidia/model.py"
TREE=$(mktemp -d)
trap 'rm -rf "$TREE"' EXIT
for f in $FILES; do
  mkdir -p "$TREE/$(dirname "$f")"
  docker run --rm --pull=never --network none --entrypoint cat "$IMAGE" "/opt/vllm/$f" > "$TREE/$f"
done
patch -s -p1 -d "$TREE" < per-layer-experts.diff
mkdir -p perlayer
cp "$TREE/vllm/model_executor/layers/quantization/exl3.py" perlayer/exl3.py
cp "$TREE/vllm/models/deepseek_v4/nvidia/model.py" perlayer/deepseek_v4_model.py
python3 -m py_compile perlayer/exl3.py perlayer/deepseek_v4_model.py
echo "built $(pwd)/perlayer"

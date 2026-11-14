#!/usr/bin/env bash
# Run the tests of the layer skip patch in the server image (plans/PLAN_FEWER_LAYERS.md §6.5): test_layer_skip.py
# (CPU, image-patch/layer_skip.py) and, when it is there, test_layer_skip_gpu.py (the patched layer with the real mHC
# kernels; build image-patch/skip/ first with image-patch/build_skip.sh). The GPU tests need about 1 GB of memory.
# Run them only when the bot is idle and MemAvailable is at least 8 GiB, and not while a test window runs.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
[ "$avail" -ge 8 ] || { echo "only $avail GiB available; the tests need 8" >&2; exit 2; }
files=(test_layer_skip.py)
mounts=()
if [ -f test_layer_skip_gpu.py ] && [ -s image-patch/skip/deepseek_v4_model.py ]; then
  files+=(test_layer_skip_gpu.py)
  mounts=(-v "$PWD/image-patch/skip/deepseek_v4_model.py:/opt/vllm/vllm/models/deepseek_v4/nvidia/model.py:ro"
          -v "$PWD/image-patch/layer_skip.py:/opt/vllm/vllm/models/deepseek_v4/nvidia/layer_skip.py:ro")
fi
docker run --rm --pull=never --gpus all --ipc host --network none --entrypoint python -v "$PWD:/s:ro" -w /s \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache "${mounts[@]}" \
  "$IMAGE" -m pytest -q -p no:cacheprovider "${files[@]}" "$@"

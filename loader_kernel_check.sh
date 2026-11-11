#!/usr/bin/env bash
# Run loader_kernel_check.py in the server image on the GPU (plans/PLAN_LOADER_PATCH.md §5.3): the patched exl3.py
# and the SparkInfer files of the live server are mounted, with this repo's ./cache, which the test boots also use.
# CAUTION: the check uses about 6 GiB of GPU memory beside the live server. Run it only when the bot is idle and no
# test window runs. The script stops before the load if less than 12 GiB is available.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
SERVER=/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark
SI=/opt/sparkinfer/sparkinfer/moe
docker run --rm --pull=never --network none --gpus all --ipc host --name loader-kernel-check --entrypoint python \
  -w "$PWD" \
  -v "$PWD:$PWD:ro" -v "$SERVER:$SERVER:ro" -v "$PWD/cache:/cache" -e XDG_CACHE_HOME=/cache \
  -e TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi -e TRITON_CACHE_DIR=/cache/triton -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/image-patch/perlayer/exl3.py:/opt/vllm/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -v "$SERVER/image-patch/sparkinfer/moe/fused_moe/_impl.py:$SI/fused_moe/_impl.py:ro" \
  -v "$SERVER/image-patch/sparkinfer/moe/_shared/kernels/tiny_decode.py:$SI/_shared/kernels/tiny_decode.py:ro" \
  "$IMAGE" loader_kernel_check.py --uniform tp1-prune5-k160 --perlayer tp1-prunemd5-k149 "$@"

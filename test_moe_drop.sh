#!/usr/bin/env bash
# Run test_moe_drop.py in the server image, with the patched router mounted (plans/PLAN_FEWER_EXPERTS.md §6).
# Build the router file first: image-patch/build_router.sh. The CUDA graph test needs a few MB of GPU memory.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
R=/opt/vllm/vllm/model_executor/layers/fused_moe/router
[ -f image-patch/fused_topk_bias_router.py ] || image-patch/build_router.sh
docker run --rm --gpus all --entrypoint python -v "$PWD:/s:ro" -w /s -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/image-patch/moe_drop.py:$R/moe_drop.py:ro" \
  -v "$PWD/image-patch/fused_topk_bias_router.py:$R/fused_topk_bias_router.py:ro" \
  "$IMAGE" -m pytest -q -p no:cacheprovider test_moe_drop.py "$@"

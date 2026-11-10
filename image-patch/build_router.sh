#!/usr/bin/env bash
# Build fused_topk_bias_router.py in this folder: the image's router file with router-drop.diff applied
# (plans/PLAN_FEWER_EXPERTS.md §6). The output is not in git; compose-mask.yml mounts it with moe_drop.py.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
docker run --rm --entrypoint cat "$IMAGE" /opt/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py \
  > fused_topk_bias_router.orig.py
patch -s -o fused_topk_bias_router.py fused_topk_bias_router.orig.py router-drop.diff
rm fused_topk_bias_router.orig.py
echo "built $(pwd)/fused_topk_bias_router.py"

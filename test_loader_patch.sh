#!/usr/bin/env bash
# Run test_loader_patch.py in the server image with the patched files of image-patch/perlayer/ mounted
# (plans/PLAN_LOADER_PATCH.md §5.2). Build them first with image-patch/build_perlayer.sh. The tests use no GPU memory.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
P=$PWD/image-patch/perlayer
docker run --rm --pull=never --network none --gpus all --entrypoint python -v "$PWD:/s:ro" -w /s \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$P/exl3.py:/opt/vllm/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -v "$P/deepseek_v4_model.py:/opt/vllm/vllm/models/deepseek_v4/nvidia/model.py:ro" \
  "$IMAGE" -m pytest -q -p no:cacheprovider test_loader_patch.py "$@"

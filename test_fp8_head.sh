#!/usr/bin/env bash
# Run test_fp8_head.py in the server image, with fp8_target_head.py mounted (plans/PLAN_SERVE_TUNE.md §7.2).
# The tests need about 0.3 GB of GPU memory. Do not run them while a test window runs.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
docker run --rm --gpus all --ipc host --entrypoint python -v "$PWD:/s:ro" -w /s -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/image-patch/fp8_target_head.py:/opt/vllm/vllm/model_executor/layers/fp8_target_head.py:ro" \
  "$IMAGE" -m pytest -q -p no:cacheprovider test_fp8_head.py "$@"

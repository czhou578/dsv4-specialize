#!/usr/bin/env bash
# Build deepseek_v4_model.py in this folder: the image's DeepSeek V4 model file with head-fp8.diff applied
# (plans/PLAN_SERVE_TUNE.md §7.2). The output is not in git; compose-head.yml mounts it with fp8_target_head.py.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
docker run --rm --entrypoint cat "$IMAGE" /opt/vllm/vllm/models/deepseek_v4/nvidia/model.py > deepseek_v4_model.orig.py
patch -s -o deepseek_v4_model.py deepseek_v4_model.orig.py head-fp8.diff
rm deepseek_v4_model.orig.py
python3 -m py_compile deepseek_v4_model.py
echo "built $(pwd)/deepseek_v4_model.py"

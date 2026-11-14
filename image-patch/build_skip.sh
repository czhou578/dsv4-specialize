#!/usr/bin/env bash
# Build image-patch/skip/deepseek_v4_model.py: the image's DeepSeek V4 model file with per-layer-experts.diff, then
# layer-skip.diff applied (plans/PLAN_FEWER_LAYERS.md §0 and §6.1). The output is not in git; compose-skip.yml mounts
# it with layer_skip.py. It runs build_perlayer.sh first, and stops if a diff does not apply (a change of the image).
set -euo pipefail
cd "$(dirname "$0")"
./build_perlayer.sh
mkdir -p skip
patch -s -o skip/deepseek_v4_model.py perlayer/deepseek_v4_model.py layer-skip.diff
python3 -m py_compile skip/deepseek_v4_model.py layer_skip.py
echo "built $(pwd)/skip/deepseek_v4_model.py"

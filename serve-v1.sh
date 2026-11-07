#!/usr/bin/env bash
# Capture boots only (compose-capture.yml): run the server's serve script with Model Runner V1.
# The serve script exports VLLM_USE_V2_MODEL_RUNNER=1 for every mode, and vLLM rejects
# --enable-return-routed-experts on V2 ("Model Runner V2 does not yet support: routed experts
# capture", 2026-11-02). V1 cannot run DSpark, so compose-capture.yml sets MODE=mtp0.
# The original script is mounted read-only at SERVE_SRC; this wrapper changes that one line in a copy.
set -euo pipefail
src=${SERVE_SRC:-/opt/vllm/serve-ds4-flash.orig.sh}
out=${SERVE_OUT:-/tmp/serve-ds4-flash-v1.sh}
line='export VLLM_USE_V2_MODEL_RUNNER=1'
n=$(grep -cxF "$line" "$src" || true)
if [ "$n" != 1 ]; then
  echo "serve-v1.sh: expected one line '$line' in $src, found $n" >&2
  exit 2
fi
sed "s/^${line}\$/export VLLM_USE_V2_MODEL_RUNNER=0/" "$src" > "$out"
echo "serve-v1.sh: Model Runner V1 ($out)"
[ "${SERVE_DRY_RUN:-0}" = 1 ] && exit 0
exec bash "$out" "$@"

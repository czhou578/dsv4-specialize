#!/usr/bin/env bash
# Build the pruned folder in a root container (PLAN_PHYSICAL_PRUNE.md §4). The main server stays up.
# The 43 expert files of data/tp1 are root:600, so only root can read them.
set -euo pipefail
cd "$(dirname "$0")"
SRC=${SRC:-tp1-botmask5-k160}
OUT=${OUT:-tp1-prune5-k160}
IMAGE=$(sed -n 's/^ *image: //p' compose-mask.yml)
SERVER=$HOME/Projects/DeepSeek-v4-Flash-One-DGX-Spark
HERE=$PWD
OWNER="$(id -u):$(id -g)"
echo "$(date '+%F %T') build $SRC -> $OUT"
docker run --rm --pull=never --network none --memory 8g --name prune-build \
  -e HOME="$HOME" -e PYTHONDONTWRITEBYTECODE=1 -w "$HERE" \
  -v "$SERVER:$SERVER:ro" -v "$HERE:$HERE" \
  --entrypoint bash "$IMAGE" -c "
    python3 -m unittest -q test_prune_experts &&
    python3 prune_experts.py --source $SRC --out $OUT --verify /opt/recipe/scripts/verify_tp1_manifest.py
    rc=\$?; [ -e $OUT ] && chown -hR $OWNER $OUT; exit \$rc"
echo "$(date '+%F %T') build done: exit $?"

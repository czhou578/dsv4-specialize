#!/usr/bin/env bash
# Build the three mask folders of PLAN_BETTER_RANK.md §6 from the stock model (data/tp1, 216 experts):
#   tp1-wmask5-k160   the weighted rank (expert_rank_weight.json), K 160, top-k 5
#   tp1-wmask5-k144   the weighted rank, K 144, top-k 5
#   tp1-botmask5-k144 the count rank of the same capture (expert_rank_count.json), K 144, top-k 5: the control
# No downtime: the build reads data/tp1 and writes about 16 GB per point. Run it when no speed test runs.
set -u
cd "$(dirname "$0")"
PY=/home/colin-spark/.hermes/stock-research/.venv/bin/python
R=/home/colin-spark/Projects/dsv4-specialize/results
VERIFY=/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark/image-patch/verify_tp1_manifest.py
LOG=${LOG:-/home/colin-spark/.hermes-e2e/rank-masks.log}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
build() {  # $1 rank file, $2 K, $3 top-k 6 folder, $4 top-k 5 folder
  [ -d "$4" ] && { log "$4 exists; skipped"; return 0; }
  log "build $3 from $1 (K $2)"
  $PY mask_checkpoint.py --keep "$2" --rank "$1" --out "$3" >> "$LOG" 2>&1 || { log "ERROR: mask build failed for $3"; return 1; }
  log "build $4 (top-k 5)"
  $PY topk_checkpoint.py --source "$3" --top-k 5 --out "$4" >> "$LOG" 2>&1 || { log "ERROR: top-k build failed for $4"; return 1; }
  python3 "$VERIFY" "$4" --skip-checksums >> "$LOG" 2>&1 && log "$4 ok" || { log "ERROR: manifest check failed for $4"; return 1; }
}
for f in ${RANK_NEW:-$R/expert_rank_skill_floor.json} $R/expert_rank_count.json; do [ -f "$f" ] || { log "ERROR: no $f"; exit 1; }; done
rc=0
RANK_NEW=${RANK_NEW:-$R/expert_rank_skill_floor.json}
log "the new rank: $RANK_NEW"
build $RANK_NEW 160 tp1-wmask-k160 tp1-wmask5-k160 || rc=1
build $RANK_NEW 144 tp1-wmask-k144 tp1-wmask5-k144 || rc=1
build $R/expert_rank_count.json 144 tp1-cmask-k144 tp1-botmask5-k144 || rc=1
log "rank masks done (rc $rc)"; exit $rc

#!/usr/bin/env bash
# The wide-draft window of plans/PLAN_SERVE_TUNE.md §8.2 (#7) on port 8890, then the restore of the live server.
# Each point boots the live folder with compose-tune.yml at the live settings, with one draft and one depth, and
# runs the graph check, bot decode x 5 and the code test:
#   D0  the live draft (dspark-draft-k64-bot) at depth 5: the reference
#   W1, W2  dspark-draft-k96-bot at depths 5 and 6;  W3, W4  dspark-draft-k128-bot at depths 5 and 6
# The depth cannot go below 5, the block size of the drafts (§3). Build the wide drafts first (§8.2, result).
# CAUTION: this stops the bot's model server for about 3 h 15 min. Start it only with the user's go, not
# 05:00-10:00 on weekdays, and not during another test:
#   tmux new -d -s draft "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/tune_draft.sh"
# POINTS (default "d0:k64:5 w1:k96:5 w2:k96:6 w3:k128:5 w4:k128:6") lists the points as name:draft:depth.
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/tune-draft.log}
RESTORE_TAG=draft-wide
TEST_COMPOSE="-f compose-mask.yml -f compose-tune.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-d0:k64:5 w1:k96:5 w2:k96:6 w3:k128:5 w4:k128:6}
NOISE=0.042                        # the noise floor of the bot decode test
export MASK_NAME=prune5b MASK_K=160 TUNE_LABEL=tune-d0   # compose-tune.yml needs TUNE_LABEL, also for a down
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
FAILS=0
draft() { [ $1 = k64 ] && echo $SERVER/data/dspark-draft-k64-bot || echo $S/dspark-draft-$1-bot; }
rate() { $PY -c 'import json,sys; print(json.loads(sys.argv[1])["bot"]["rate"])' \
  "$($PY $S/speed_result.py --results $R $1)" 2>/dev/null || echo 0; }

dpoint() {  # $1 name, $2 draft, $3 depth; returns 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local label=tune-$1 s=$(( $3 + 1 )) f=$R/boot-tune-$1.log rc ok=0
  export TUNE_LABEL=$label TUNE_DEPTH=$3 DRAFT_DIR=$(draft $2) TUNE_GRAPH_SIZES=$s,$((2 * s)),$((4 * s)) TUNE_GRAPH_MAX=$((4 * s))
  [ -f $DRAFT_DIR/model.safetensors.index.json ] || { log "ERROR: no draft $DRAFT_DIR"; return 2; }
  [ $(( $(now) + 2400 )) -gt $RESTORE_S ] && { log "$label: needs 40 min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  stop_main || return 9
  log "$label: draft $DRAFT_DIR, depth $3, graph sizes $TUNE_GRAPH_SIZES"
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  grep 'Initializing a V1 LLM engine' $f | grep -qF "'cudagraph_capture_sizes': [${TUNE_GRAPH_SIZES//,/, }]" \
    && grep -q 'Capturing CUDA graphs (FULL)' $f && grep -q 'Capturing dspark CUDA graphs (FULL)' $f && ok=1
  log "$label graph check: $ok; $(avail_gib) GiB available"
  BOT_RUNS=5 speed $label
  down || return 9
}

log "draft window start (points $POINTS; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
for p in $POINTS; do
  IFS=: read -r name dr depth <<< "$p"
  dpoint $name $dr $depth; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
done
restore
B=$(rate tune-d0)
for p in $POINTS; do
  IFS=: read -r name _ _ <<< "$p"; r=$(rate tune-$name)
  log "tune-$name: bot decode $r tok/s; above the noise against D0 ($B): $(awk -v a="$r" -v b="$B" -v n=$NOISE \
    'BEGIN {print (b > 0 && a > b * (1 + n)) ? "yes" : "no"}')"
done
log "draft window done"

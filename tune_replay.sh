#!/usr/bin/env bash
# The replay window of plans/PLAN_SERVE_TUNE.md §7 (Step 2) on port 8890, then the restore of the live server.
#   H1  the FP8 target head (#4, compose-head.yml) on the live folder: head check, replay, bot decode x 5, code test
#   N1  a K per layer, rule global, mean 144 (#5, tp1-klg5-k144): replay
#   N2  a K per layer, rule even, mean 144 (tp1-kle5-k144): replay
#   N3  mean 152 with the rule of N1 or N2 that has the lower NLL (tp1-klg5-k152 or tp1-kle5-k152): replay
#   H0  the live settings (T0 of §6.2): the speed reference of H1. Bot decode x 5, code test
# Every point boots with compose-tune.yml at its defaults (the live settings) and the bot draft. The gate points run
# first; a point that cannot end before RESTORE_AT is skipped. Build first: image-patch/build_head.sh and the four
# folders of §7.3 (alloc_k.py, mask_checkpoint.py --keep-per-layer, topk_checkpoint.py --top-k 5).
# CAUTION: this stops the bot's model server for about 4 h. Start it only with the user's go, not 05:00-10:00 on
# weekdays, and not during another test:
#   tmux new -d -s replay "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/tune_replay.sh"
# POINTS (default "h1 n1 n2 n3 h0") selects the points; N3_RULE (klg5 or kle5) sets the rule of N3.
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/tune-replay.log}
RESTORE_TAG=replay
TUNE="-f compose-mask.yml -f compose-tune.yml"
TEST_COMPOSE=$TUNE
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-h1 n1 n2 n3 h0}
export MASK_NAME=prune5b MASK_K=160 TUNE_LABEL=tune-h0   # compose-tune.yml needs TUNE_LABEL, also for a down
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
FAILS=0
on() { [[ " $POINTS " == *" $1 "* ]]; }
nll() { awk '$1 == "nll" {print $3}' $R/tune-$1-compare.txt 2>/dev/null; }

# One point: $1 name, $2 MASK_NAME, $3 MASK_K, $4 head (0|1), $5 replay (0|1), $6 bot runs, $7 minutes it needs.
# Returns 0 done, 1 boot failed, 2 skipped, 9 safety failure.
hpoint() {
  local label=tune-$1 f=$R/boot-tune-$1.log rc ok=1
  export TUNE_LABEL=$label MASK_NAME=$2 MASK_K=$3
  TEST_COMPOSE="$TUNE$([ $4 = 1 ] && echo ' -f compose-head.yml')"
  [ $(( $(now) + 60 * $7 )) -gt $RESTORE_S ] && { log "$label: needs $7 min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  [ -f $S/tp1-$2-k$3/rank-sliced-tp1-manifest.json ] || { log "ERROR: no folder $S/tp1-$2-k$3"; return 2; }
  [ $4 = 0 ] || [ -s $S/image-patch/deepseek_v4_model.py ] || { log "ERROR: no deepseek_v4_model.py"; return 2; }
  stop_main || return 9
  log "$label: folder tp1-$2-k$3, FP8 target head $4, compose $TEST_COMPOSE"
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  log "$label: $(avail_gib) GiB available; $(grep -oE 'FP8 target head: [^;]*; [^;]*' $f | head -1)"
  if [ $4 = 1 ] && ! grep -q 'FP8 target head: .* GiB BF16 freed' $f; then
    ok=0; log "ERROR: $label has no FP8 target head line; no replay, no speed"
  fi
  if [ $ok = 1 ] && [ $5 = 1 ]; then
    replay $label
    [ $4 = 1 ] && (cd $W && $PY -m model_eval.replay --compare $R/prune5-k160.jsonl $R/$label.jsonl \
      > $R/$label-vs-prune5-k160.txt 2>&1) && log "$label against the live replay: $(grep -E '^nll' $R/$label-vs-prune5-k160.txt)"
  fi
  [ $ok = 1 ] && [ "$6" -gt 0 ] && BOT_RUNS=$6 speed $label
  down || return 9
}

run() {  # hpoint with the rules of run_lib.sh: two failed boots in a row, or a safety failure, restore and stop
  hpoint "$@"; local rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
  return $rc
}

log "replay window start (points $POINTS; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
for need in $S/compose-tune.yml $S/compose-head.yml $S/tp1-prune5b-k160/rank-sliced-tp1-manifest.json \
  $R/baseline-k216.jsonl $R/prune5-k160.jsonl; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

on h1 && run h1 prune5b 160 1 1 5 80
on n1 && run n1 klg5 144 0 1 0 42
on n2 && run n2 kle5 144 0 1 0 42
if on n3; then
  G=$(nll n1) E=$(nll n2)
  RULE=${N3_RULE:-$(awk -v g="${G:-9}" -v e="${E:-9}" 'BEGIN {print (e < g ? "kle5" : "klg5")}')}
  log "N3: NLL global ${G:-none}, even ${E:-none}; rule $RULE"
  run n3 $RULE 152 0 1 0 42
fi
on h0 && run h0 prune5b 160 0 0 5 40
restore
log "replay window done"

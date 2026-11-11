#!/usr/bin/env bash
# The replay window of plans/PLAN_COLD_EXPERTS.md §7 on port 8890, then the restore of the live server.
# Four masks of data/tp1 (cold_sets.py, mask_checkpoint.py --drop-file, topk_checkpoint.py --top-k 5). Each drops the
# live drops plus a set of cold experts:
#   R0  nothing more: the live set as a mask (tp1-coldr05-k160), the reference
#   M1  the mandatory experts below 0.1% of the weight (tp1-coldm15-k154)
#   D1  the coldest ordinary experts, up to 1% of the weight per layer (tp1-coldd15-k155)
#   MD  M1 and D1 (tp1-coldmd5-k149)
# Each point: boot, the smoke requests (prefill_probe.py --smoke), the replay, the comparisons against baseline-k216
# (run_lib.sh) and against R0 (cold_sets.py --report), stop. The R0 check: R0 must match botmask5-k160 within 0.5% of
# the NLL, or the window restores and stops. A point that cannot end before RESTORE_AT is skipped.
# The test boots pin the KV cache at 7.25 GiB, as the live server since 2026-11-10: a mask of 216 experts leaves little
# free memory, and the smaller pin adds 0.75 GiB.
# CAUTION: this stops the bot's model server for about 3 h 15 min. Start it only with the user's go, not 05:00-10:00
# on weekdays unless the cron jobs are paused, and not during another test:
#   tmux new -d -s cold "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/cold_replay.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/cold-replay.log}
RESTORE_TAG=cold
TEST_COMPOSE="-f compose-mask.yml -f serve-tune/compose-tune.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-r0 m1 d1 md}
declare -A FOLDER=([r0]=coldr05:160 [m1]=coldm15:154 [d1]=coldd15:155 [md]=coldmd5:149)
export TUNE_KV_BYTES=7784628224 TUNE_LABEL=cold-r0 MASK_NAME=coldr05 MASK_K=160   # compose-tune.yml needs TUNE_LABEL
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
FAILS=0
pct() { awk '$1 == "nll" {printf "%+.2f", 100 * ($3 / $2 - 1)}' "$1" 2>/dev/null; }

cpoint() {  # $1 point; returns 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 label=cold-$1 name=${FOLDER[$1]%:*} k=${FOLDER[$1]#*:} rc
  export TUNE_LABEL=$label MASK_NAME=$name MASK_K=$k
  [ $(( $(now) + 44 * 60 )) -gt $RESTORE_S ] && { log "$label: needs 44 min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  [ -f $S/tp1-$name-k$k/rank-sliced-tp1-manifest.json ] || { log "ERROR: no folder $S/tp1-$name-k$k"; return 2; }
  stop_main || return 9
  log "$label: folder tp1-$name-k$k"
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  log "$label: $(avail_gib) GiB available after the boot"
  timeout 300 $PY $S/serve-tune/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$label --smoke \
    --out $R/smoke-$label.json > /dev/null 2> $R/smoke-$label.err; rc=$?
  log "$label smoke: exit $rc, $(docker logs $MASK 2>&1 | grep -c AssertionError) assertion lines; $(tail -1 $R/smoke-$label.err | cut -c1-200)"
  replay $label
  if [ $p != r0 ] && [ -s $R/cold-r0.jsonl ]; then
    (cd $W && $PY -m model_eval.replay --compare $R/cold-r0.jsonl $R/$label.jsonl > $R/$label-vs-cold-r0.txt 2>&1)
    $PY $S/cold_sets.py --report $R/cold-r0.jsonl $R/$label.jsonl > $R/$label-report.txt 2>&1
    log "$label against R0: $(head -1 $R/$label-report.txt)"
  fi
  down || return 9
}

log "cold window start (points $POINTS; boot by $BOOT_BY; restore at $RESTORE_AT)"
for need in $S/serve-tune/compose-tune.yml $S/serve-tune/prefill_probe.py $S/cold_sets.py $R/baseline-k216.jsonl \
  $R/botmask5-k160.jsonl; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

for p in $POINTS; do
  cpoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
  if [ $p = r0 ] && [ $rc = 0 ]; then   # the R0 check of §5.2
    (cd $W && $PY -m model_eval.replay --compare $R/botmask5-k160.jsonl $R/cold-r0.jsonl > $R/cold-r0-vs-botmask5-k160.txt 2>&1)
    d=$(pct $R/cold-r0-vs-botmask5-k160.txt)
    log "R0 check: NLL ${d:-none}% against botmask5-k160 (the limit is 0.5%)"
    awk -v d="${d:-9}" 'BEGIN {exit !(d >= -0.5 && d <= 0.5)}' || { log "R0 check fails: restore and stop"; restore; exit 1; }
  fi
done
restore
for p in $POINTS; do
  log "cold-$p: NLL $(pct $R/cold-$p-compare.txt)% against baseline-k216; $(grep -E '^(PASS|FAIL)' $R/cold-$p-compare.txt 2>/dev/null | cut -c1-6 | tr '\n' ' ')"
done
log "cold window done"

#!/usr/bin/env bash
# The lossless window of plans/PLAN_SERVE_TUNE.md §6 (Step 1): the live folder (tp1-prune5b-k160) and the bot draft
# on port 8890 at five points, then the restore of the live server.
#   T0  the live settings (depth 5): the reference. Graph check, prefill probe, bot decode x 5, code test
#   T1  depth 3 (graph sizes 4,8,16): graph check, smoke, bot decode x 5, code test
#   T2  depth 4 (graph sizes 5,10,20): as T1
#   T3  the fastest valid depth of T0-T2, with the FP8 draft head: graph and head checks, bot decode x 5, code test
#   T4  that depth, the head only if T3 beat its base by more than the noise, KV pin 7.25 GiB, prefill chunk 8224:
#       graph check, smoke, prefill probe, bot decode x 3, code test. If T4 does not boot, T4b has the 8 GiB pin.
# run_lib.sh runs docker compose in ~/Projects/dsv4-specialize, so compose-tune.yml and prefill_probe.py must be there.
# CAUTION: this stops the bot's model server for about 3.5 h. Start it only with the user's go, not 05:00-10:00 on
# weekdays, and not during another test:
#   tmux new -d -s tune "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/tune_night.sh"
# POINTS (default "t0 t3 t4") selects the points. Without T0-T3, set BEST_DEPTH and T4_HEAD for T4.
# T1 and T2 are not in the default: the image refuses a depth below the draft's dspark_block_size (5 for the live
# draft), because a smaller depth gives incorrect output (vllm/config/speculative.py). Use them with a draft of a
# smaller block size only.
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/tune-night.log}
RESTORE_TAG=tune
TEST_COMPOSE="-f compose-mask.yml -f compose-tune.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-t0 t3 t4}
NOISE=0.042                        # the noise floor of the bot decode test
KV8=8589934592 KV725=7784628224    # the KV pins: 8 GiB (live) and 7.25 GiB
export MASK_NAME=prune5b MASK_K=160 TUNE_LABEL=tune-t0   # compose-tune.yml needs TUNE_LABEL, also for a down
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
declare -A RATE VALID DEPTH=([t1]=3 [t2]=4)
HEAD_OK=0 FAILS=0
on() { [[ " $POINTS " == *" $1 "* ]]; }
faster() { awk -v a="$1" -v b="$2" -v n="${3:-0}" 'BEGIN {exit !(b > 0 && a > b * (1 + n))}'; }   # a beats b by n
rate() { $PY -c 'import json,sys; print(json.loads(sys.argv[1])["bot"]["rate"])' \
  "$($PY $S/speed_result.py --results $R $1)" 2>/dev/null || echo 0; }

probe() {  # $1 label, $2 kind (probe|smoke), the rest: prefill_probe.py arguments; logs the summary line
  local label=$1 kind=$2 left=$(( RESTORE_S - $(now) )) rc n; shift 2
  [ $left -le $([ $kind = probe ] && echo 900 || echo 120) ] && { log "$label: no time for the $kind"; return; }
  memwatch_start "$label-$kind"
  timeout $left $PY $S/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$label \
    --out $R/$kind-$label.json "$@" > /dev/null 2> $R/$kind-$label.err; rc=$?
  memwatch_stop
  n=$(docker logs $MASK 2>&1 | grep -cE 'AssertionError|Workspace is locked')
  log "$label $kind: exit $rc, $n assertion lines; $(tail -1 $R/$kind-$label.err | cut -c1-400)"
}

# One point: $1 name, $2 depth, $3 FP8 draft head (0|1), $4 KV bytes, $5 prefill chunk, $6 bot runs, $7 checks
# (graph, head, smoke, probe). Returns 0 done, 1 boot failed, 2 skipped (deadline), 9 safety failure.
tpoint() {
  local label=tune-$1 s=$(( $2 + 1 )) f=$R/boot-tune-$1.log rc
  export TUNE_LABEL=$label TUNE_DEPTH=$2 TUNE_FP8_DRAFT_HEAD=$3 TUNE_KV_BYTES=$4 TUNE_PREFILL=$5
  export TUNE_GRAPH_SIZES=$s,$((2 * s)),$((4 * s)) TUNE_GRAPH_MAX=$((4 * s))
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  stop_main || return 9
  log "$label: depth $2, graph sizes $TUNE_GRAPH_SIZES, FP8 draft head $3, KV pin $4 bytes, prefill chunk $5"
  boot $label; rc=$?
  if [ $rc != 0 ]; then log "$label KV need: $(grep -oE '[0-9.]+ GiB KV cache is needed' $f | head -1)"; return $rc; fi
  log "$label: $(grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $f | head -1)"
  if [[ $7 == *graph* ]]; then   # the engine took the sizes, and the target and the draft captured full graphs
    VALID[$1]=0
    grep 'Initializing a V1 LLM engine' $f | grep -qF "'cudagraph_capture_sizes': [${TUNE_GRAPH_SIZES//,/, }]" \
      && grep -q 'Capturing CUDA graphs (FULL)' $f && grep -q 'Capturing dspark CUDA graphs (FULL)' $f && VALID[$1]=1
    log "$label graph check: ${VALID[$1]}"
  fi
  if [[ $7 == *head* ]]; then
    grep -q 'draft-time only; verify pass untouched' $f && HEAD_OK=1 || HEAD_OK=0
    log "$label FP8 draft head loaded: $HEAD_OK"
  fi
  [[ $7 == *smoke* ]] && probe $label smoke --smoke
  [[ $7 == *probe* ]] && probe $label probe --cold 8000 32000 128000 --warm 200000 8000
  if [ "$6" -gt 0 ]; then BOT_RUNS=$6 speed $label; RATE[$1]=$(rate $label); fi
  down || return 9
}

run() {  # tpoint with the rules of run_lib.sh: two failed boots in a row, or a safety failure, restore and stop
  tpoint "$@"; local rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
  return $rc
}

log "tune night start (points $POINTS; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
for need in $S/compose-tune.yml $S/prefill_probe.py $S/tp1-prune5b-k160/rank-sliced-tp1-manifest.json; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

on t0 && run t0 5 0 $KV8 1024 5 "graph probe"
on t0 && [ "${VALID[t0]:-0}" != 1 ] && log "WARNING: the graph check fails on the live settings; read boot-tune-t0.log"
on t1 && run t1 3 0 $KV8 1024 5 "graph smoke"
on t2 && run t2 4 0 $KV8 1024 5 "graph smoke"
D=5 B=${RATE[t0]:-0}
for n in t1 t2; do
  [ "${VALID[$n]:-0}" = 1 ] && faster "${RATE[$n]:-0}" "$B" && { D=${DEPTH[$n]}; B=${RATE[$n]}; }
done
D=${BEST_DEPTH:-$D}
log "bot decode: T0 ${RATE[t0]:-none}, T1 ${RATE[t1]:-none}, T2 ${RATE[t2]:-none} tok/s; depth $D; above the noise" \
  "against T0: $(faster "$B" "${RATE[t0]:-0}" $NOISE && echo yes || echo no)"
H=0
if on t3; then
  run t3 $D 1 $KV8 1024 5 "graph head"
  [ "$HEAD_OK" = 1 ] && faster "${RATE[t3]:-0}" "$B" $NOISE && H=1
  log "FP8 draft head: T3 ${RATE[t3]:-none} against $B tok/s; T4 takes the head: $H"
fi
H=${T4_HEAD:-$H}
if on t4; then
  run t4 $D $H $KV725 8224 3 "graph smoke probe"
  [ $? = 1 ] && run t4b $D $H $KV8 8224 3 "graph smoke probe"
fi
restore
log "tune night done"

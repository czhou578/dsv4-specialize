#!/usr/bin/env bash
# Phase 0, step 2 of the colinfer plan (colin-inference-engine docs/history/PLAN_QWEN35_MOE.md §5): the fixed memory of
# a colinfer server by part, with Qwen3.8-27B at its smallest settings, then the restore of the live server. The server
# runs as a plain process of the colinfer venv on port 8002, under the model lock of colinfer, with no restart.
# All points: 1 slot of 131,072 tokens, the checkpoint's own decode weights (no INT copies), no vision tower.
#   f1  no drafter, no prefix caching: no checkpoint ring and no session cache. The smallest engine
#   f2  f1 with prefix caching and no session cache: the checkpoint ring
#   f3  f2 with the session cache: its buffers. Boot and one snapshot only: a request would write into
#       ~/.cache/colinfer/sessions, the cache of the user's own conversations
#   f4  f2 with the MTP drafter: its weights, its state and the cycle graphs
# The tests of f1, f2 and f4: smoke (a short request and a tool call) and one prompt of LONG tokens (default 100,000).
# A snapshot follows the ready line and each test (colinfer_floor.py --snap): the fall of the available memory, the
# bytes of the torch allocator, the resident memory of the process and its CUDA context. The available memory is
# sampled every 2 s. The report (results/colinfer-floor.json, and the log) gives each snapshot in GB.
# CAUTION: this stops the bot's model server for about 20 min. Start it only with the user's go, not 05:00-10:00 on
# weekdays while the cron jobs are active, and not during another test:
#   tmux new -d -s colinfer-floor "START_AFTER=... BOOT_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/scripts/colinfer_floor.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/colinfer-floor.log}
RESTORE_TAG=colinfer-floor
SPEED_BY=${SPEED_BY:-${RESTORE_AT:-}}   # run_lib.sh needs it; this window runs no speed test
HERE=$(cd "$(dirname "$0")/.." && pwd)   # the repo root of this checkout
source "$HERE/scripts/run_lib.sh"

POINTS=${POINTS:-f1 f2 f3 f4}
C=${C:-$HOME/Projects/colin-inference-engine}   # the colinfer checkout; its venv has the engine as an editable install
CV=$C/.venv/bin CURL=http://127.0.0.1:8002 CNAME=colinfer-floor LONG=${LONG:-100000} NEED=${NEED:-5}
BASE="--slots 1 --max-seq-len 131072 --decode-weights checkpoint --no-vision"
declare -A FLAGS=([f1]="$BASE --spec none --no-prefix-caching" [f2]="$BASE --spec none --session-cache-gb 0" \
                  [f3]="$BASE --spec none" [f4]="$BASE --spec mtp --session-cache-gb 0") \
  TESTS=([f1]="smoke long" [f2]="smoke long" [f3]="" [f4]="smoke long")
CPID= SAMPLER=
CPAT='^[^ ]*python[^ ]* -m engine\.server( |$)'   # a colinfer server on any port; the gateway (engine.server.gateway) does not match
crunning() { pgrep -f "$CPAT" > /dev/null; }
cpid() { pgrep -f "$CPAT" | head -1; }
client() { $PY $HERE/tools/colinfer_floor.py --base-url $CURL --model $CNAME "$@"; }
sample_start() {  # $1 file: the available memory every 2 s
  ( while true; do echo "$(date +%s) $(avail_mb)" >> "$1"; sleep 2; done ) & SAMPLER=$!
}
sample_stop() { [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=; }
cstop() {  # stop the colinfer server and wait until it is gone
  [ -n "$CPID" ] && kill -TERM -- -"$CPID" 2>/dev/null
  local t0=$(now)
  while crunning && [ $(( $(now) - t0 )) -lt 60 ]; do sleep 2; done
  crunning && pkill -KILL -f "$CPAT"
  CPID=
}
launch() {  # $1 point, $2 server log: start the server under the model lock and wait for /health; 0 healthy, 1 failed
  local t0
  env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PATH=$CV:$PATH setsid flock -n $HOME/.cache/colinfer/gpu-model.lock \
    $CV/python -m engine.server --host 127.0.0.1 --port 8002 ${FLAGS[$1]} > "$2" 2>&1 &
  CPID=$!
  t0=$(now)
  while [ $(( $(now) - t0 )) -lt 300 ]; do
    curl -fsS --max-time 5 $CURL/health > /dev/null 2>&1 && return 0
    kill -0 "$CPID" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
snap() {  # $1 point, $2 when, $3 the available MB before the boot: one snapshot line
  client --snap --pid "$(cpid)" --when $2 --before $3 >> $R/colinfer-floor-$1-snap.jsonl 2>> "$LOG" || log "ERROR: colinfer-$1 has no snapshot at $2"
}
tests() {  # $1 point, $2 the available MB before the boot: each test of TESTS, with a snapshot after it
  local p=$1 t left o=$R/colinfer-floor-$1
  for t in ${TESTS[$p]}; do
    left=$(( RESTORE_S - $(now) - 300 ))
    case $t in
      smoke) $PY $HERE/tools/qwen_fit.py --base-url $CURL --model $CNAME --smoke > $o-smoke.json 2>&1
        log "colinfer-$p smoke: exit $?; $(tail -1 $o-smoke.json | cut -c1-200)";;
      long) timeout $left $PY $HERE/tools/colinfer_floor.py --base-url $CURL --model $CNAME --long $LONG > $o-long.json 2>&1
        log "colinfer-$p long prompt: exit $?; $(tail -1 $o-long.json | cut -c1-200)";;
    esac
    snap $p $t $2
  done
}
finish() { sample_stop; cstop; restore; }
[ -n "${COLINFER_STUB:-}" ] && source "$COLINFER_STUB"   # the dry run of test_colinfer_floor.py: stub boots and clients

fpoint() {  # $1 point; 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 before o=$R/colinfer-floor-$1
  [ -n "${FLAGS[$p]:-}" ] || { log "ERROR: no point $p"; return 2; }
  [ $(( $(now) + NEED * 60 )) -gt $RESTORE_S ] && { log "colinfer-$p: needs $NEED min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "colinfer-$p: after $BOOT_BY, not booted"; return 2; }
  stop_main || return 9
  crunning && { log "ERROR: a colinfer server runs"; return 9; }
  wait_mem 300 100 || { log "ERROR: less than 100 GiB available ($(avail_gib))"; return 9; }
  before=$(avail_mb)
  log "colinfer-$p: ${FLAGS[$p]}; $before MB available"
  rm -f $o-mem.log $o-snap.jsonl; sample_start $o-mem.log
  launch $p $o-server.log || { sample_stop; cstop; log "ERROR: colinfer-$p did not boot: $(tail -2 $o-server.log | tr '\n' ' ' | cut -c1-300)"; return 1; }
  log "colinfer-$p healthy: $(avail_mb) MB available; $(grep -oE '\[engine\] ready:.*' $o-server.log | tail -1 | cut -c1-200)"
  snap $p ready $before
  tests $p $before
  sample_stop; cstop
}

log "colinfer floor window start (points $POINTS; boot by $BOOT_BY; restore at $RESTORE_AT)"
for need in $CV/python $C/engine/server/api.py $HERE/tools/colinfer_floor.py $HERE/tools/qwen_fit.py; do [ -e $need ] || { log "ERROR: no $need"; exit 1; }; done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
[ "$(now)" -ge $BOOT_BY_S ] && { log "after $BOOT_BY: no boot"; exit 1; }

log "available beside the main server: $(avail_mb) MB"
FAILS=0
for p in $POINTS; do
  fpoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; finish; exit 1; };;
    9) log "safety failure: the safe action, restore"; finish; exit 1;;
  esac
done
finish
client --report $R --points "$POINTS" --out $R/colinfer-floor.json 2>&1 | while read -r line; do log "report: $line"; done
log "colinfer floor window done"

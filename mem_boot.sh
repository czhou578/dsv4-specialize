#!/usr/bin/env bash
# The memory test boot of PLAN_MODEL_ROUTER.md §3 (phase C): the deployed pruned model (tp1-prune5-k160 ->
# data/tp1-k160t5) on port 8890 at GPU_MEMORY_UTILIZATION 0.73 and 128k context (compose-mem073.yml). It records the
# boot lines, the free host memory at idle and during one bot decode run, then restores the live server.
# CAUTION: this stops the bot's model server. Start it only with the user's go:
#   tmux new -d -s mem-boot "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/mem_boot.sh"
# - a boot that is not healthy in 15 min: down, record, go on; two failed boots in a row: restore;
# - free memory every 30 s during each client run; below 1 GiB, stop the client;
# - deadlines: no boot with a replay after BOOT_BY, no speed test after SPEED_BY, and every client stops
#   at RESTORE_AT, when the restore starts. Any other failure: the safe action (restore).
LOG=${LOG:-/home/colin-spark/.hermes-e2e/mem-boot.log}
RESTORE_TAG=mem
TEST_COMPOSE="-f compose-mask.yml -f compose-mem073.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

log "memory boot start (boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
export MASK_NAME=prune5 MASK_K=160
L=prune5-k160-u073
if stop_main; then
  boot $L; rc=$?
  if [ $rc = 0 ]; then
    grep -oE 'Available KV cache memory: [0-9.]+ GiB|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|CUDA graph[^"]*took [0-9.]+ GiB' $R/boot-$L.log | sort -u | sed 's/^/    /' | tee -a "$LOG" > /dev/null
    sleep 60
    log "$L idle: $(free -m | awk '/^Mem:/ {print "used " $3 " MB, available " $7 " MB"}')"
    memwatch_start "$L"
    left=$(( RESTORE_S - $(now) ))
    log "$L bot decode test, 1 run ($left s left)"
    (cd $W && timeout $left $PY -m model_eval.speed --base-url $URL --out $R/speed-bot-$L.jsonl --runs 1 > $R/speed-bot-$L.log 2>&1)
    log "$L bot decode exit $?: $(tail -1 $R/speed-bot-$L.log | cut -c1-160)"
    memwatch_stop
    log "$L lowest available memory during the test: $(awk -F= '{print $2}' /home/colin-spark/.hermes-e2e/memory-$L.log | sort -n | head -1) MB"
    down
  fi
else
  log "safety failure before the boot"
fi
restore
log "memory boot done"

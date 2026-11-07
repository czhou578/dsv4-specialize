#!/usr/bin/env bash
# The test boot of PLAN_PHYSICAL_PRUNE.md §6: tp1-prune5-k160 on port 8890 with the current draft, the replay
# set (same routing as tp1-botmask5-k160), the speed tests, then the restore of the main server.
# CAUTION: this stops the bot's model server. Start it only with the user's go, in its own tmux session:
#   tmux new -d -s prune-boot ~/Projects/dsv4-specialize/prune_boot.sh
# The safety rules and the shared functions are in run_lib.sh (PLAN_OVERNIGHT_MASK_GRID.md §2).
# - a boot that is not healthy in 15 min: down, record, go on; two failed boots in a row: restore;
# - free memory every 30 s during each client run; below 1 GiB, stop the client;
# - deadlines: no boot with a replay after BOOT_BY, no speed test after SPEED_BY, and every client stops
#   at RESTORE_AT, when the restore starts. Any other failure: the safe action (restore).
LOG=${LOG:-/home/colin-spark/.hermes-e2e/prune-boot.log}
RESTORE_TAG=prune
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

log "prune boot start (start after $START_AFTER; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
grep -q '"self_check": {' $S/tp1-prune5-k160/PRUNE.json || { log "ERROR: tp1-prune5-k160 has no passed self-check"; exit 1; }
run_point prune5 160 full
if [ -f $R/prune5-k160.jsonl ]; then
  (cd $W && $PY -m model_eval.replay --compare $R/botmask5-k160.jsonl $R/prune5-k160.jsonl > $R/prune5-k160-vs-botmask5-k160.txt 2>&1)
  log "prune5-k160 against botmask5-k160 (same routing): $(grep -E '^(nll|tool_names|PASS|FAIL)' $R/prune5-k160-vs-botmask5-k160.txt | tr -s ' ' | tr '\n' ';')"
fi
restore
log "prune boot done"

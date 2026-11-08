#!/usr/bin/env bash
# The draft test of PLAN_PHYSICAL_PRUNE.md §7: the live pruned model (tp1-prune5b-k160 -> data/tp1-k160t5) on port 8890
# with another DSpark draft (DRAFT_DIR), the bot and code decode tests, then the restore. Speculative decoding does not
# change the output, so speed and acceptance are the only measures. Label: prune5b-k160 (the stock draft's files are
# speed-*-prune5-k160.*).
# CAUTION: this stops the bot's model server. Start it only with the user's go:
#   tmux new -d -s draft "DRAFT_DIR=... START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/draft_test.sh"
# - a boot that is not healthy in 15 min: down, record, go on; two failed boots in a row: restore;
# - free memory every 30 s during each client run; below 1 GiB, stop the client;
# - deadlines: no boot with a replay after BOOT_BY, no speed test after SPEED_BY, and every client stops
#   at RESTORE_AT, when the restore starts. Any other failure: the safe action (restore).
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/draft-test.log}
RESTORE_TAG=draft
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

: "${DRAFT_DIR:?set DRAFT_DIR}"; export DRAFT_DIR
log "draft test start (draft $DRAFT_DIR; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
export MASK_NAME=prune5b MASK_K=160
L=prune5b-k160
if stop_main; then
  boot $L; rc=$?
  if [ $rc = 0 ]; then
    grep -oE "draft[^\"]{0,80}experts[^\"]{0,40}|DSPARK[^\"]{0,100}" $R/boot-$L.log | sort -u | head -5 | sed 's/^/    /' | tee -a "$LOG" > /dev/null
    speed $L
    down
  fi
else
  log "safety failure before the boot"
fi
restore
log "draft test done"

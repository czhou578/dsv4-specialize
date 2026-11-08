#!/usr/bin/env bash
# The top-k test boots of PLAN_OVERNIGHT_TOPK.md §5, then the restore of the main server (§6).
# Start it in its own tmux session after the folders of §4 pass the self-check:
#   tmux new -d -s topk-run ~/Projects/dsv4-specialize/topk_run.sh
# The safety rules and the shared functions are in run_lib.sh. Set the four deadlines in the environment.
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/topk-run.log}
RESTORE_TAG=topk
SPEEDF=/home/colin-spark/Projects/dsv4-specialize/history/hermes-e2e/stepf-speed.log   # stepf-speed.sh (history) wrote it
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

log "top-k run start (start after $START_AFTER; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
until { grep -qE 'speed tests done|ERROR' "$SPEEDF"; } && [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
log "speed run: $(tail -1 "$SPEEDF")"

# §5.1 top-k 5 on bot K 160: boot, replay, speed tests, stop.
run_point botmask5 160 full; rc51=$?
# §5.2 the next point.
if [ $rc51 = 0 ] && gates_pass botmask5-k160; then
  log "§5.2: top-k 5 passes both gates: top-k 4 on bot K 160"
  run_point botmask4 160 full
else
  log "§5.2: top-k 5 does not pass (boot rc $rc51): bot K 144 at top-k 6, replay only"
  if [ ! -d $S/tp1-botmask-k144 ]; then
    log "build $S/tp1-botmask-k144"
    (cd $S && $PY mask_checkpoint.py --keep 144 --rank $R/expert_rank_bot.json --out $S/tp1-botmask-k144) >> "$LOG" 2>&1 \
      && python3 $SERVER/image-patch/verify_tp1_manifest.py $S/tp1-botmask-k144 --skip-checksums >> "$LOG" 2>&1 \
      || { log "ERROR: the K 144 build failed"; }
  fi
  [ -d $S/tp1-botmask-k144 ] && run_point botmask 144 replay
fi
# §5.3 optional: top-k 5 on K 216, replay only, if the replay can start before BOOT_BY (a boot takes up to 15 min).
if [ $rc51 = 1 ]; then
  log "§5.3 skipped: top-k 5 did not boot on K 160"
elif [ $(( $(now) + 900 )) -lt $BOOT_BY_S ]; then
  run_point top5 216 replay
else
  log "§5.3 skipped: the replay cannot start before $BOOT_BY"
fi
# §6
restore
log "top-k run done"

#!/usr/bin/env bash
# The test window of plans/PLAN_MD_REVIEW.md §5, then the judge of §6. The MD prune (tp1-prunemd5-k149) boots on the
# patched image (compose-perlayer.yml, port 8890) at the live settings: the KV pin of 7.25 GiB and the live draft.
# After the smoke test, the 19 held-out bot prompts run end to end once for each label of LABELS. Then the restore of
# the live server, and with the live server up (no downtime):
#   1. the check of each new run, and the compare against each live run (the five gates of step F);
#   2. the blind page of 20 pairs (results/review-md.html, key in results/review-md.key.json);
#   3. the judge: each new run against the live runs, then the controls (live against live, new against new);
#   4. the report (review_report.py), in this log.
# The window starts when review_live.sh logs "live runs done" and START_AFTER has passed. No boot after BOOT_BY.
# RESTORE_AT stops the runs; a rerun skips the done prompts.
# CAUTION: this stops the bot's model server for about 7.5 h. Start it only with the user's go, pause the Hermes cron
# jobs first, and start the overnight guard with GUARD_AT after RESTORE_AT:
#   tmux new -d -s window "START_AFTER=... BOOT_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/review_window.sh"
# Build image-patch/perlayer/ first (image-patch/build_perlayer.sh), or the boot fails.
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/review-window.log}
RESTORE_TAG=review
TEST_COMPOSE="-f compose-mask.yml -f serve-tune/compose-tune.yml -f compose-perlayer.yml"
SPEED_BY=${SPEED_BY:-${RESTORE_AT:-}}   # run_lib.sh needs it; this window runs no speed test
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

LABELS=${LABELS:-md-1 md-2}
LIVE=${LIVE:-live-1 live-2}
LIVE_LOG=${LIVE_LOG:-$S/logs/review-live.log}
export TUNE_KV_BYTES=7784628224 TUNE_LABEL=md-review MASK_NAME=prunemd5 MASK_K=149
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
SEED=${SEED:-13}
done_n() { ls /home/colin-spark/.hermes-e2e/$1/runs/*/done.json 2>/dev/null | wc -l; }

log "review window: wait for the live runs and $START_AFTER (boot by $BOOT_BY; restore at $RESTORE_AT; runs $LABELS)"
for need in $S/compose-perlayer.yml $S/image-patch/perlayer/exl3.py $S/image-patch/perlayer/deepseek_v4_model.py \
  $S/tp1-$MASK_NAME-k$MASK_K/rank-sliced-tp1-manifest.json $S/serve-tune/prefill_probe.py; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until grep -q "live runs done" "$LIVE_LOG" 2>/dev/null && [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do
  [ "$(now)" -ge $BOOT_BY_S ] && { log "the live runs did not end by $BOOT_BY: no window"; exit 1; }
  sleep 60
done
for l in $LIVE; do log "$l: $(done_n $l) of 19 prompts done"; done

stop_main || { log "safety failure before the boot: restore"; restore; exit 1; }
boot $TUNE_LABEL || { restore; exit 1; }
log "$TUNE_LABEL: $(avail_mb) MB available; $(grep -oE 'per-layer experts: .*' $R/boot-$TUNE_LABEL.log | head -1 | cut -c1-200)"
timeout 600 $PY $S/serve-tune/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$TUNE_LABEL \
  --out $R/smoke-$TUNE_LABEL.json --smoke > /dev/null 2> $R/smoke-$TUNE_LABEL.err
rc=$?
log "smoke: exit $rc; $(tail -1 $R/smoke-$TUNE_LABEL.err | cut -c1-200)"
[ $rc = 0 ] || { log "ERROR: the smoke test failed: no runs"; down; restore; exit 1; }

# The memory watch of final5.sh: below 1 GiB available, stop the runner and its sandbox container.
( while true; do m=$(avail_mb); echo "$(date +%T) available_mb=$m" >> $S/logs/memory-review-window.log
    if [ "$m" -lt 1024 ]; then log "MEMORY below 1 GiB ($m MB): stop the run"; pkill -f model_eval.e2e.run
      docker ps -q --filter name=e2e-md | xargs -r docker rm -f; fi
    sleep 30; done ) &
MW=$!
for label in $LABELS; do
  for attempt in $(seq 1 30); do   # a memory stop ends the runner; a rerun skips the done prompts
    left=$(( RESTORE_S - $(now) ))
    [ $left -le 60 ] && { log "restore time reached"; break; }
    log "$label attempt $attempt ($left s left, $(done_n $label) of 19 prompts done)"
    (cd $W && timeout $left $PY -m model_eval.e2e.run --base-url $URL/v1 --label $label --research 8) >> "$LOG" 2>&1 \
      && break
    docker ps -q --filter name=e2e-$label | xargs -r docker rm -f
    sleep 60
  done
  log "$label: $(done_n $label) of 19 prompts done"
done
kill $MW 2>/dev/null
down || { restore; exit 1; }
restore

for l in $LABELS; do
  (cd $W && $PY -m model_eval.e2e.check $l) > $R/review-$l-check.txt 2>&1
  for b in $LIVE; do
    (cd $W && $PY -m model_eval.e2e.check --compare $b $l) > $R/review-$l-vs-$b.txt 2>&1
    log "$l against $b: $(grep -E '^(PASS|FAIL)' $R/review-$l-vs-$b.txt | tr -s ' ' | cut -c1-60 | tr '\n' ';')"
  done
done
(cd $W && $PY -m model_eval.e2e.judge $LIVE $(echo $LABELS | tr ' ' '+') --page $R/review-md.html --sample 20 \
  --seed $SEED) > $R/review-md-page.log 2>&1
log "blind page: $(tail -1 $R/review-md-page.log)"
for l in $LABELS; do
  log "judge: $LIVE against $l"
  (cd $W && $PY -m model_eval.e2e.judge $LIVE $l) > $R/review-judge-$l.log 2>&1
  log "judge $l: exit $?; $(tail -1 $R/review-judge-$l.log | cut -c1-300)"
done
for pair in "$LIVE" "$LABELS"; do   # the controls: one model on both sides
  log "control judge: $pair"
  (cd $W && $PY -m model_eval.e2e.judge $pair) > "$R/review-judge-control-${pair// /-}.log" 2>&1
  log "control $pair: exit $?; $(tail -1 "$R/review-judge-control-${pair// /-}.log" | cut -c1-300)"
done
$PY $S/review_report.py --live $LIVE --new $LABELS 2>&1 | while read -r line; do log "$line"; done
log "review done"

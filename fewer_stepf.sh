#!/usr/bin/env bash
# Phase E of plans/PLAN_FEWER_EXPERTS.md (§9): step F (PLAN_E2E_RUNNER.md §15, bot repo) on the chosen point. The 19
# held-out bot prompts run end to end on the test server (compose-mask.yml with compose-drop.yml), then the restore of
# the live server, the checks against baseline-1 and baseline-2, the judge against the three base runs (on the live
# server, no downtime) and the blind page of 20 pairs (PLAN_FINAL_REVIEW.md §7).
# On the same boot, and before the full runs:
# - REPLAY=1: the replay screen of the point. Gate: both replay gates pass and the summed NLL is below MAX_NLL
#   (percent against the baseline); else no full runs.
# - SPEED=1: the bot decode test (5 runs) and the code test (Phase D, §8). SPEED_MIN: no full runs if the bot decode
#   is below it (tok/s). REPLAY=0 skips the replay.
# CAUTION: this stops the bot's model server for about 3.5 h (4.5 h with REPLAY and SPEED). Do not run it
# 05:00-10:00 on weekdays while the cron jobs are active: daily-followups writes into the run folders then
# (PLAN_E2E_RUNNER.md §15.9). The overnight guard restores the live server at 04:15: stop it or move it first.
#   tmux new -d -s stepf "POINT=fewer-c6 MASK_NAME=prune5 MOE_DROP=none MOE_NORM=anchor:6 REPLAY=1 SPEED=1 \
#     START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/fewer_stepf.sh"
# MASK_NAME/MASK_K pick the folder tp1-NAME-kK (default botmask, 160; prune5 is the live folder). DRAFT_DIR
# defaults to the live draft. RESTORE_AT ends the run (a rerun skips the done prompts). JUDGE=0 skips the judge.
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/fewer-stepf.log}
RESTORE_TAG=stepf
TEST_COMPOSE="-f compose-mask.yml -f compose-drop.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINT=${POINT:?set POINT, e.g. fewer-c6}
LABEL=${LABEL:-$POINT}   # the run folder ~/.hermes-e2e/$LABEL
export MASK_NAME=${MASK_NAME:-botmask} MASK_K=${MASK_K:-160} MOE_LABEL=$POINT MOE_DROP=${MOE_DROP:?} MOE_NORM=${MOE_NORM:?}
export DRAFT_DIR=${DRAFT_DIR:-$SERVER/data/dspark-draft-k64-bot}
MAX_NLL=${MAX_NLL:-2.04}   # the live rule (+2.27%) minus the run noise (0.23)
BOOT_MIN_GIB=110
BASES="baseline-1 baseline-2 baseline-3"
done_n() { ls /home/colin-spark/.hermes-e2e/$LABEL/runs/*/done.json 2>/dev/null | wc -l; }

log "step F start ($POINT on tp1-$MASK_NAME-k$MASK_K: MOE_DROP=$MOE_DROP MOE_NORM=$MOE_NORM; draft $DRAFT_DIR;" \
  "replay ${REPLAY:-0}, speed ${SPEED:-0}; boot by $BOOT_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
[ "$(now)" -ge $BOOT_BY_S ] && { log "after $BOOT_BY: not booted"; exit 1; }
stop_main || { log "safety failure before the boot: restore"; restore; exit 1; }
boot $POINT || { restore; exit 1; }
n=$(grep -oE '\[moe_drop\] layer [0-9]+ ' $R/boot-$POINT.log | sort -u | wc -l)
log "$POINT patch: $n target layers; hash: $(grep -E '\[moe_drop\].*hash=True' $R/boot-$POINT.log | grep -oE 'layer [0-9]+' | sort -u | tr '\n' ' ')"
[ "$n" = 43 ] || { log "ERROR: the patch did not reach the 43 target layers"; down; restore; exit 1; }

E2E=1
if [ "${REPLAY:-0}" = 1 ]; then
  replay $POINT
  verdict=$($PY -c "
import sys; sys.path.insert(0, '$S'); import fewer_report as f
p = f.point('$POINT'); print('pass' if f.passes(p) and p['nll_pct'] < $MAX_NLL else 'fail', round(p['nll_pct'], 2) if p else None)")
  log "$POINT replay gate (NLL below +$MAX_NLL%): $verdict"
  [ "${verdict%% *}" = pass ] || E2E=0
fi
if [ "${SPEED:-0}" = 1 ]; then
  BOT_RUNS=5 speed $POINT
  if [ -n "${SPEED_MIN:-}" ]; then   # Phase D gate for a same-speed point: not slower than the live server by the noise
    rate=$($PY -c "import json; print(json.loads(open('$R/speed-bot-$POINT.log').read().split('\n')[-2])['rate'])" 2>/dev/null || echo 0)
    gate=$($PY -c "print('pass' if float('$rate') >= float('$SPEED_MIN') else 'fail')")
    log "$POINT speed gate (bot decode at least $SPEED_MIN tok/s): $gate ($rate)"
    [ "$gate" = pass ] || E2E=0
  fi
fi

if [ $E2E = 1 ]; then
  # The memory watch of final5.sh: below 1 GiB available, stop the runner and its sandbox container.
  ( while true; do m=$(avail_mb); echo "$(date +%T) available_mb=$m" >> $S/logs/memory-$LABEL-e2e.log
      if [ "$m" -lt 1024 ]; then log "MEMORY below 1 GiB ($m MB): stop the run"; pkill -f model_eval.e2e.run
        docker ps -q --filter name=e2e-$LABEL | xargs -r docker rm -f; fi
      sleep 30; done ) &
  MW=$!
  for attempt in $(seq 1 30); do   # a guard stop or a memory stop ends the runner; a rerun skips the done prompts
    left=$(( RESTORE_S - $(now) ))
    [ $left -le 60 ] && { log "restore time reached"; break; }
    log "$LABEL attempt $attempt ($left s left, $(done_n) of 19 prompts done)"
    (cd $W && timeout $left $PY -m model_eval.e2e.run --base-url $URL/v1 --label $LABEL --research 8) >> "$LOG" 2>&1 && break
    docker ps -q --filter name=e2e-$LABEL | xargs -r docker rm -f
    sleep 60
  done
  kill $MW 2>/dev/null
  log "$LABEL: $(done_n) of 19 prompts done"
else
  log "no full runs: a gate failed"
fi
down || { restore; exit 1; }
restore
[ $E2E = 1 ] || exit 0

(cd $W && $PY -m model_eval.e2e.check $LABEL) > $R/stepf-$LABEL-check.txt 2>&1
for b in baseline-1 baseline-2; do
  (cd $W && $PY -m model_eval.e2e.check --compare $b $LABEL) > $R/stepf-$LABEL-vs-$b.txt 2>&1
  log "$LABEL against $b: $(grep -E '^(PASS|FAIL)' $R/stepf-$LABEL-vs-$b.txt | tr -s ' ' | tr '\n' ';')"
done
[ "${JUDGE:-1}" = 1 ] || { log "step F done (no judge)"; exit 0; }
log "judge: $BASES against $LABEL on the live server"
(cd $W && $PY -m model_eval.e2e.judge $BASES $LABEL) > $R/stepf-$LABEL-judge.log 2>&1
log "judge exit $?: $(tail -1 $R/stepf-$LABEL-judge.log | cut -c1-300)"
(cd $W && $PY -m model_eval.e2e.judge $BASES $LABEL --page $R/review-$LABEL.html --sample 20 --seed 11) >> $R/stepf-$LABEL-judge.log 2>&1
log "blind page: $R/review-$LABEL.html; step F done"

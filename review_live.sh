#!/usr/bin/env bash
# The base runs of plans/PLAN_MD_REVIEW.md §4: the 19 held-out bot prompts end to end on the live server (port 8888,
# no downtime), once for each label of LABELS, then the check of each run. The runs are clients of the live server:
# a bot message waits behind the current request. A memory watch stops a run below 1 GiB available; a rerun skips
# the done prompts. The last log line is "live runs done", which review_window.sh waits for.
#   tmux new -d -s live "LABELS='live-1 live-2' ~/Projects/dsv4-specialize/review_live.sh"
set -u
S=/home/colin-spark/Projects/dsv4-specialize
W=/home/colin-spark/Projects/stock-research-model-eval
PY=/home/colin-spark/.hermes/stock-research/.venv/bin/python
LOG=${LOG:-$S/logs/review-live.log}
LABELS=${LABELS:?set LABELS, e.g. "live-1 live-2"}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
done_n() { ls /home/colin-spark/.hermes-e2e/$1/runs/*/done.json 2>/dev/null | wc -l; }

log "live runs start ($LABELS) on the live server"
( while true; do m=$(avail_mb); echo "$(date +%T) available_mb=$m" >> $S/logs/memory-review-live.log
    if [ "$m" -lt 1024 ]; then log "MEMORY below 1 GiB ($m MB): stop the run"; pkill -f model_eval.e2e.run
      docker ps -q --filter name=e2e-live | xargs -r docker rm -f; fi
    sleep 30; done ) &
MW=$!
for label in $LABELS; do
  for attempt in 1 2 3 4 5 6; do   # a guard stop or a memory stop ends the runner; a rerun skips the done prompts
    log "$label attempt $attempt ($(done_n $label) of 19 prompts done)"
    (cd $W && $PY -m model_eval.e2e.run --base-url http://127.0.0.1:8888/v1 --label $label --research 8) >> "$LOG" 2>&1 \
      && break
    docker ps -q --filter name=e2e-$label | xargs -r docker rm -f
    sleep 60
  done
  log "$label: $(done_n $label) of 19 prompts done"
done
kill $MW 2>/dev/null
for label in $LABELS; do
  (cd $W && $PY -m model_eval.e2e.check $label) > $S/results/review-$label-check.txt 2>&1
  log "check $label: exit $?; $(grep -cE '^FAIL' $S/results/review-$label-check.txt) FAIL lines"
done
log "live runs done"

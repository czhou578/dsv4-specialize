#!/usr/bin/env bash
# The safety net of plans/PLAN_MD_REVIEW.md §6: it makes sure that the steps after the window are complete.
# It waits until review_window.sh has ended, for any reason, and until the live server (port 8888) is healthy. If the
# restore of the window failed, the overnight guard restores the live server, and this script waits for it. Then:
#   1. the check of each run, and the compare of each new run against each live run (seconds; always again);
#   2. the blind page, if it is missing;
#   3. each judge call, up to 3 attempts. A failed request ends a judge call, and the window script does not try
#      again. The judge skips the rows that are already in its file, so a complete call ends in seconds;
#   4. the report, in results/review-report.txt and in this log.
# It only sends requests to the live server: it never stops or starts a server.
#   tmux new -d -s finish ~/Projects/dsv4-specialize/review_finish.sh
set -u
S=/home/colin-spark/Projects/dsv4-specialize
W=/home/colin-spark/Projects/stock-research-model-eval
PY=${PY:-/home/colin-spark/.hermes/stock-research/.venv/bin/python}
R=${R:-$S/results}
LOG=${LOG:-$S/logs/review-finish.log}
LABELS=${LABELS:-md-1 md-2}
LIVE=${LIVE:-live-1 live-2}
SEED=${SEED:-13}
WINDOW=${WINDOW:-$S/review_window.sh}   # the script to wait for
RETRY_S=${RETRY_S:-300}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
done_n() { ls /home/colin-spark/.hermes-e2e/$1/runs/*/done.json 2>/dev/null | wc -l; }
healthy() { curl -fsS --max-time 5 http://127.0.0.1:8888/health > /dev/null 2>&1; }
wait_live() {  # $1: seconds; true when the live server is healthy within that time
  local end=$(( $(date +%s) + $1 ))
  until healthy; do [ "$(date +%s)" -ge $end ] && return 1; sleep 60; done
}
judge() {  # $1: the name of the log, then the judge arguments
  local name=$1 attempt; shift
  for attempt in 1 2 3; do
    wait_live 7200 || { log "ERROR: judge $name: the live server is not healthy for 2 h"; return 1; }
    if (cd $W && $PY -m model_eval.e2e.judge "$@") >> $R/review-judge-$name.log 2>&1; then
      log "judge $name: $(tail -1 $R/review-judge-$name.log | cut -c1-300)"; return 0
    fi
    log "judge $name: attempt $attempt failed: $(grep -E 'Error' $R/review-judge-$name.log | tail -1 | cut -c1-200)"
    sleep $RETRY_S
  done
  return 1
}

log "finisher: wait for review_window.sh to end"
while pgrep -f "^bash $WINDOW" > /dev/null; do sleep 60; done
log "$(basename $WINDOW) has ended; its last line: $(tail -1 $S/logs/review-window.log | cut -c1-200)"
NEW=""
for l in $LIVE $LABELS; do log "$l: $(done_n $l) of 19 prompts done"; done
for l in $LABELS; do [ "$(done_n $l)" -gt 0 ] && NEW="$NEW $l"; done
NEW=${NEW# }
[ -n "$NEW" ] || { log "no new runs: nothing to judge"; exit 1; }
wait_live 14400 || { log "ERROR: the live server is not healthy for 4 h: stop"; exit 1; }

for l in $LIVE $NEW; do (cd $W && $PY -m model_eval.e2e.check $l) > $R/review-$l-check.txt 2>&1; done
for l in $NEW; do
  for b in $LIVE; do
    (cd $W && $PY -m model_eval.e2e.check --compare $b $l) > $R/review-$l-vs-$b.txt 2>&1
    log "$l against $b: $(grep -E '^(PASS|FAIL)' $R/review-$l-vs-$b.txt | tr -s ' ' | cut -c1-60 | tr '\n' ';')"
  done
done
if [ ! -s $R/review-md.html ]; then
  (cd $W && $PY -m model_eval.e2e.judge $LIVE $(echo $NEW | tr ' ' '+') --page $R/review-md.html --sample 20 \
    --seed $SEED) > $R/review-md-page.log 2>&1
  log "blind page (made again): $(tail -1 $R/review-md-page.log)"
fi
for l in $NEW; do judge $l $LIVE $l; done
judge control-${LIVE// /-} $LIVE
[ "$(echo $NEW | wc -w)" -ge 2 ] && judge control-${NEW// /-} $NEW
$PY $S/review_report.py --live $LIVE --new $NEW > $R/review-report.txt 2>&1
while read -r line; do log "$line"; done < $R/review-report.txt
log "review finished: the report is in $R/review-report.txt, the blind page is $R/review-md.html"

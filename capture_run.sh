#!/usr/bin/env bash
# The capture boot of PLAN_BETTER_RANK.md §4: the stock model (data/tp1) on port 8890 from compose-capture.yml
# (Model Runner V1, no draft, --enable-return-routed-experts, the weights patch), the capture of the rank data, the
# three rank files, then the restore of the live server.
# CAUTION: this stops the bot's model server. Start it only with the user's go:
#   tmux new -d -s capture "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/capture_run.sh"
# (SPEED_BY is unused here; run_lib.sh needs it set.)
# - a boot that is not healthy in 15 min: down, record, go on; two failed boots in a row: restore;
# - free memory every 30 s during each client run; below 1 GiB, stop the client;
# - deadlines: no boot with a replay after BOOT_BY, no speed test after SPEED_BY, and every client stops
#   at RESTORE_AT, when the restore starts. Any other failure: the safe action (restore).
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/capture-run.log}
RESTORE_TAG=capture
TEST_COMPOSE="-f compose-capture.yml"
TEST_PROJECT=dsv4-capture
MASK=dsv4-capture-deepseek-v4-flash-capture-1
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

SUMS=$S/capture-out/routing_sums.npz
RANKDATA=$W/model_eval/data/rank
log "capture run start (boot by $BOOT_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
[ -f $RANKDATA/sessions.jsonl ] || { log "ERROR: no $RANKDATA/sessions.jsonl"; exit 1; }
rm -f "$SUMS" "$SUMS.tmp.npz"
export MASK_NAME=capture MASK_K=216
if stop_main; then
  boot capture-k216; rc=$?
  if [ $rc = 0 ]; then
    # Smoke test: one small request must move the sums file (the patch of §4.1 works), else no capture.
    curl -s --max-time 300 $URL/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash-0731-capture","messages":[{"role":"user","content":"Say OK."}],"max_tokens":1}' > /dev/null
    sleep 3
    if [ -f "$SUMS" ] && $PY -c "import numpy as np,sys; z=np.load('$SUMS'); sys.exit(0 if int(z['steps'])>0 and z['weight'].sum()>0 else 1)"; then
      log "smoke: the sums file moves ($($PY -c "import numpy as np; z=np.load('$SUMS'); print(int(z['steps']),'steps,',int(z['tokens'][0]),'tokens, weight sum per layer-0 token', round(float(z['weight'][0].sum()/max(z['tokens'][0],1)),3))"))"
      memwatch_start capture-k216
      left=$(( RESTORE_S - $(now) ))
      log "capture of $RANKDATA ($left s left)"
      (cd $W && timeout $left $PY $S/capture.py --base-url $URL --out $R/capture-rank.jsonl --data $RANKDATA \
        --sources discord,subagent,e2e,cron --sums "$SUMS" > $R/capture-rank.log 2>&1)
      log "capture exit $?: $(wc -l < $R/capture-rank.jsonl) rows, $(grep -c '"error"' $R/capture-rank.jsonl) errors"
      memwatch_stop
      for order in count weight skill_max; do
        (cd $W && $PY $S/capture.py --rank $R/capture-rank.jsonl $R/expert_rank_$order.json --order $order) 2>&1 | tail -1 | tee -a "$LOG"
      done
    else
      log "ERROR: the sums file did not move after the smoke request; no capture. Check the patch (§4.1)"
      docker logs $MASK 2>&1 | grep -iE "capture|routing_sums|Traceback" | tail -5 | tee -a "$LOG"
    fi
    down
  fi
else
  log "safety failure before the boot"
fi
restore
log "capture run done"

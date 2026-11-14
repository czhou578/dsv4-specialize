#!/usr/bin/env bash
# The probe window of plans/PLAN_FEWER_LAYERS.md §7 on port 8890, then the restore of the live server.
# One eager boot of the live folder (the MD prune, compose-perlayer.yml) with the layer probe (compose-skip.yml: the
# run-time skip file, the probe sums, no draft, the prefix cache off, the KV pin of 7.25 GiB). Then
# layer_screen.py --greedy: R, the BI pass, round 1, the Gram sums, rounds 2 and 3, R2. The client ends 15 min before
# RESTORE_AT, and the report goes to results/skip-probe-report.md. If the boot fails with Model Runner V2, one more
# boot uses Model Runner V1 (compose-skip-v1.yml).
# Before the start: build image-patch/skip/ (image-patch/build_skip.sh). The script picks the sets if
# results/skip-sets.json is missing. It refuses to start if results/skip-probe.jsonl exists (the client would resume
# from it): move the old file away, or set RESUME=1 to resume on purpose.
# CAUTION: this stops the bot's model server for about 3 h. Start it only with the user's go, pause the Hermes cron
# jobs first, and do not start it during another test:
#   tmux new -d -s probe "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/skip_probe.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/skip-probe.log}
RESTORE_TAG=skip-probe
TEST_COMPOSE="-f compose-mask.yml -f serve-tune/compose-tune.yml -f compose-perlayer.yml -f compose-skip.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

export MASK_NAME=prunemd5 MASK_K=149 TUNE_LABEL=skip-probe TUNE_KV_BYTES=7784628224
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
export SKIP_PROBE=1 SKIP_FILE=/skip/skip.json SKIP_MODE=mtp0 SKIP_PREFIX_CACHE=0 MASK_EXTRA_ARGS=--enforce-eager
export TUNE_GRAPH_SIZES=none       # eager: no CUDA graph sizes for vLLM (the serve script passes none)
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
probe_line() { grep -q 'layer skip: set none (run-time file /skip/skip.json); .*probe on' "$R/boot-$1.log"; }
client() {  # the greedy rounds of §7; the client ends 15 min before RESTORE_AT
  local left=$(( RESTORE_S - $(now) - 600 ))
  [ $left -le 600 ] && { log "no time for the client"; return 1; }
  memwatch_start skip-probe
  (cd $S && timeout $left $PY $S/layer_screen.py --greedy --base-url $URL --skip-dir $OUT --sets $R/skip-sets.json \
    --log $PLOG --deadline $(( RESTORE_S - 900 )) >> $R/skip-probe-client.log 2>&1)
  local rc=$?
  memwatch_stop
  return $rc
}
[ -n "${SKIP_STUB:-}" ] && source "$SKIP_STUB"   # the dry run of test_skip_scripts.py: stub boots and clients
OUT=${OUT:-$S/skip-out} PLOG=$R/skip-probe.jsonl

log "probe window start (boot by $BOOT_BY; restore at $RESTORE_AT)"
for need in $S/image-patch/skip/deepseek_v4_model.py $S/image-patch/layer_skip.py $S/compose-skip.yml \
  $S/compose-skip-v1.yml $S/compose-perlayer.yml $S/tp1-$MASK_NAME-k$MASK_K/rank-sliced-tp1-manifest.json; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
if [ -e $PLOG ] && [ "${RESUME:-0}" != 1 ]; then log "ERROR: $PLOG exists; move it away or set RESUME=1"; exit 1; fi
[ -s $R/skip-sets.json ] || $PY $S/layer_screen.py --pick --sets $R/skip-sets.json >> "$LOG" 2>&1 || { log "ERROR: --pick"; exit 1; }
mkdir -p $OUT && rm -f $OUT/skip.json $OUT/layer_probe.npz $OUT/layer_gram.npz
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
[ "$(now)" -ge $BOOT_BY_S ] && { log "after $BOOT_BY: no boot"; exit 1; }

stop_main || { log "safety failure: the safe action, restore"; restore; exit 1; }
label=skip-probe
boot $label; rc=$?
if [ $rc = 1 ]; then
  log "the Model Runner V2 boot failed: one boot with Model Runner V1"
  TEST_COMPOSE="$TEST_COMPOSE -f compose-skip-v1.yml" label=skip-probe-v1
  boot $label; rc=$?
fi
[ $rc != 0 ] && { log "no probe boot (rc $rc): restore"; restore; exit 1; }
if ! probe_line $label; then
  log "ERROR: the boot log has no probe line: restore"; down; restore; exit 1
fi
log "$(grep -o 'layer skip: set .*' $R/boot-$label.log | head -1); $(avail_gib) GiB available"
client; rc=$?
if [ $rc != 0 ] && [ $(( RESTORE_S - $(now) )) -gt 1800 ]; then
  log "the client ended with exit $rc: one more run (it resumes from its log)"
  client; rc=$?
fi
log "client exit $rc: $(wc -l < $PLOG 2>/dev/null) rows; $(tail -2 $R/skip-probe-client.log | tr '\n' ' ' | cut -c1-300)"
down || log "ERROR: a test container is left"
restore
$PY $S/layer_screen.py --report --log $PLOG > $R/skip-probe-report.md 2>&1
sed -n '/^## The decision/,$p' $R/skip-probe-report.md | sed -n 's/^- //p' | while read -r line; do log "decision: $line"; done
sed -n 's/^- \(R2*:\)/\1/p' $R/skip-probe-report.md | while read -r line; do log "$line"; done
log "probe window done"

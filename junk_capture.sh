#!/usr/bin/env bash
# The route capture of the junk prompts on the stock model: window 1 of plans/PLAN_JUNK_CAPTURE.md (§5).
# Steps: wait for START_AFTER; refuse a window that touches the cron window (05:00 to 10:00 on weekdays) unless
# CRON_OK=1; stop Qwen; stop DeepSeek (stop_main: an idle bot, then 100 GiB available); move the old sums file into
# OUT; boot data/tp1 on port 8890 from compose-capture.yml and compose-capture-junk.yml (Model Runner V1,
# --enable-return-routed-experts, KV pin 6.5 GiB, --max-model-len 131072) under the memory guard; check the server
# process and the KV size of the boot lines; junk_routes.py smoke; junk_routes.py capture of the probe prompts
# (ROUNDS rounds) and the pilot transcripts, stopped at RESTORE_AT; down; restore DeepSeek; after 5 min, start Qwen
# again with pair_run.sh if it ran before (QWEN_AFTER=1: also when it did not). A failed check skips the capture.
# CLIENT=rank runs capture.py on the calibration split of DATA (default qwen-data, SOURCES claude,repo) in place of
# junk_routes.py: the coding capture of plans/PLAN_CODE_RANK.md §4, into OUT/capture-code.jsonl.
# CAUTION: this stops both model servers for about 45 min. Start it only with the user's go, in tmux, with the
# deadlines (SPEED_BY is not used; it defaults to BOOT_BY):
#   tmux new -d -s junkcap "START_AFTER=... BOOT_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/junk_capture.sh"
D=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/junk-capture.log}
RESTORE_TAG=junkcap
TEST_COMPOSE="-f compose/compose-capture.yml -f compose/compose-capture-junk.yml"
TEST_PROJECT=dsv4-capture
MASK=dsv4-capture-deepseek-v4-flash-capture-1
export SPEED_BY=${SPEED_BY:-${BOOT_BY:-}}
source "$D/run_lib.sh"

OUT=${OUT:-$R/junk-capture} ROUNDS=${ROUNDS:-3}
PROMPTS=${PROMPTS:-$R/junk-bisect/prompts.json} PILOT=${PILOT:-$R/agent-eval/pilot}
CLIENT=${CLIENT:-junk} DATA=${DATA:-$S/qwen-data} SOURCES=${SOURCES:-claude,repo}
SUMS=${SUMS:-$S/capture-out/routing_sums.npz}
PIDFILE=${PIDFILE:-$R/pair-qwen.pid}
QWEN_SEQS=${QWEN_SEQS:-4} QWEN_KV_BYTES=${QWEN_KV_BYTES:-3575989770}   # the live Qwen (TAG=textkv) of 2026-11-15
KV_MAX=${KV_MAX:-150000}   # the 6.5 GiB pin gave 70,514 tokens on 2026-11-15; the 0.93 boot of 11-06 gave 473,251
export MASK_NAME=capture MASK_K=216
unset CAPTURE_DIR CAPTURE_UTIL CAPTURE_EXTRA_ARGS ROUTING_PROFILE   # the stock data/tp1, no routing profile
BOOT_MIN_GIB=110   # vLLM's boot check at util 0.90 needs 109.5 GiB free

proc_text() {  # the environment and the command line of the server process, then the host folder of /models/tp1
  local t
  t=$(docker exec $MASK sh -c 'tr "\0" "\n" < /proc/1/environ; tr "\0" " " < /proc/1/cmdline' 2>/dev/null)
  [ -n "$t" ] && printf '%s\nMODEL_DIR=%s\n' "$t" \
    "$(docker inspect $MASK --format '{{range .Mounts}}{{if eq .Destination "/models/tp1"}}{{.Source}}{{end}}{{end}}' 2>/dev/null)"
}

check_server() {  # true when the server runs data/tp1 with the settings of compose-capture-junk.yml
  local text s kv ok=0
  text=$(proc_text)
  [ -n "$text" ] || { log "ERROR: cannot read the server process"; return 1; }
  for s in --enable-return-routed-experts "--kv-cache-memory-bytes 6979321856" "max-model-len 131072"; do
    grep -qF -- "$s" <<< "$text" || { log "ERROR: the server lacks '$s'"; ok=1; }
  done
  grep -qx "MODEL_DIR=$SERVER/data/tp1" <<< "$text" || { log "ERROR: /models/tp1 is not $SERVER/data/tp1"; ok=1; }
  kv=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$R/boot-capture-junk.log" 2>/dev/null | tail -1 | tr -dc 0-9)
  if [ -z "$kv" ] || [ "$kv" -ge "$KV_MAX" ]; then
    log "ERROR: a KV cache of ${kv:-unknown} tokens is not the 6.5 GiB pin (below $KV_MAX)"; ok=1
  fi
  return $ok
}

smoke_test() {  # one short raw prompt: routing rows of 43 x 6 ids, and the sums file moves
  local out rc
  job_phase "smoke"
  out=$(cd $S && $PY $S/junk_routes.py smoke --base-url $URL --sums "$SUMS" 2>&1); rc=$?
  log "$(tail -1 <<< "$out")"
  return $rc
}

client() {  # the capture, stopped at RESTORE_AT
  local left=$(( RESTORE_S - $(now) )) rc
  [ $left -le 120 ] && { log "no time for the capture"; return 1; }
  job_phase "capture"
  memwatch_start junk-capture
  if [ "$CLIENT" = rank ]; then
    log "capture.py of $DATA ($SOURCES, calibration split; $left s left)"
    (cd $S && timeout $left $PY $S/capture.py --base-url $URL --out $OUT/capture-code.jsonl --data $DATA \
      --sources $SOURCES --sums "$SUMS" > $OUT/capture-code.log 2>&1); rc=$?
    log "capture exit $rc: $(cat $OUT/capture-code.jsonl 2>/dev/null | grep -c .) rows, $(cat $OUT/capture-code.jsonl \
      2>/dev/null | grep -c '"error"') errors; $(tail -1 $OUT/capture-code.log)"
  else
    log "capture of the probe prompts ($ROUNDS rounds) and the pilot transcripts ($left s left)"
    (cd $S && timeout $left $PY $S/junk_routes.py capture --base-url $URL --out $OUT/capture.jsonl --sums "$SUMS" \
      --prompts $PROMPTS --rounds $ROUNDS --pilot $PILOT > $OUT/capture.log 2>&1); rc=$?
    log "capture exit $rc: $(tail -1 $OUT/capture.log)"
  fi
  memwatch_stop
}

memguard_start() {  # kill the test container below 1.5 GiB available: a full memory can hard-reset the Spark
  ( while sleep 5; do
      m=$(avail_mb)
      [ "$m" -lt 1536 ] && { log "MEMORY GUARD: $m MB available: kill the test container"; docker kill $MASK > /dev/null 2>&1; }
    done ) &
  MG=$!
}
memguard_stop() { kill "$MG" 2>/dev/null; wait "$MG" 2>/dev/null; }

qwen_up() { [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
stop_qwen() {  # as pair_run.sh: the process group of the server
  local pid=$(cat "$PIDFILE") i
  log "stop Qwen (pid $pid; $(avail_mb) MB available)"
  kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid"
  for i in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
  kill -0 "$pid" 2>/dev/null && { kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid"; }
  rm -f "$PIDFILE"
  log "Qwen stopped ($(avail_mb) MB available)"
}
start_qwen() {
  log "start Qwen again (SEQS=$QWEN_SEQS KV_BYTES=$QWEN_KV_BYTES TESTS=idle) after 5 min for the memory drift"
  sleep 300
  local i                                  # the drift after a restore: wait up to 15 more min for pair_run.sh's NEED_MB
  for i in $(seq 1 30); do [ "$(avail_mb)" -ge "${NEED_MB:-28600}" ] && break; sleep 30; done
  (cd $S && TAG=junkcap SEQS=$QWEN_SEQS KV_BYTES=$QWEN_KV_BYTES TESTS=idle KEEP=1 ./pair_run.sh) >> "$LOG" 2>&1
  qwen_up && log "Qwen up again" || log "ERROR: Qwen did not start; see logs/pair-junkcap.log"
}

cron_clash() {  # $1 $2: epochs; true when [$1, $2] overlaps 05:00-10:00 of a weekday
  local d end a b
  d=$(date -d "@$1" +%F) end=$(date -d "@$2" +%F)
  while :; do
    if [ "$(date -d "$d" +%u)" -le 5 ]; then
      a=$(date -d "$d 05:00" +%s) b=$(date -d "$d 10:00" +%s)
      [ "$1" -lt "$b" ] && [ "$2" -gt "$a" ] && return 0
    fi
    [ "$d" = "$end" ] && return 1
    d=$(date -d "$d + 1 day" +%F)
  done
}

[ -n "${JUNKCAP_STUB:-}" ] && source "$JUNKCAP_STUB"   # the dry run of test_junk_capture.py: stub servers and client

log "junk capture: $ROUNDS rounds; start after $START_AFTER; boot by $BOOT_BY; restore at $RESTORE_AT"
case $CLIENT in
  junk) NEEDS="$PROMPTS $PILOT $S/junk_routes.py";;
  rank) NEEDS="$DATA/sessions.jsonl $DATA/tools.json $S/capture.py";;
  *) log "ERROR: unknown CLIENT $CLIENT (junk or rank)"; exit 1;;
esac
for need in $NEEDS $S/compose/compose-capture.yml $S/compose/compose-capture-junk.yml; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ] || [ "$(now)" -ge $BOOT_BY_S ]; do sleep 30; done
[ "$(now)" -ge $BOOT_BY_S ] && { log "after $BOOT_BY: no window, no server action"; exit 1; }
if [ "${CRON_OK:-0}" != 1 ] && cron_clash "$(now)" "$RESTORE_S"; then
  log "ERROR: the window up to $RESTORE_AT touches the cron window (05:00 to 10:00 on weekdays): no server action"
  exit 1
fi
mkdir -p $OUT

QWEN_WAS_UP=0
if qwen_up; then QWEN_WAS_UP=1; stop_qwen; fi
if stop_main; then
  [ -f "$SUMS" ] && mv "$SUMS" "$OUT/routing_sums-before-$(date +%Y%m%d-%H%M).npz"   # the client reads deltas
  rm -f "$SUMS.tmp.npz"
  memguard_start
  boot capture-junk; rc=$?
  if [ $rc = 0 ]; then
    if check_server && smoke_test; then client; else log "a check failed: no capture"; fi
    down
  fi
  memguard_stop
else
  log "safety failure before the boot"
fi
restore
{ [ $QWEN_WAS_UP = 1 ] || [ "${QWEN_AFTER:-0}" = 1 ]; } && start_qwen
log "junk capture done: $(cat $OUT/capture.jsonl 2>/dev/null | grep -c .) rows, $(cat $OUT/capture.jsonl 2>/dev/null \
  | grep -c '"error"') errors in $OUT/capture.jsonl"

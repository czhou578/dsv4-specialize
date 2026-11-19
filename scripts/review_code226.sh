#!/usr/bin/env bash
# The review window of plans/PLAN_CODE226_DEPLOY.md §5 on port 8890, then the restore of the live server.
# Both points boot the patched image (compose-perlayer.yml) with the bot draft:
#   md       tp1-prunemd5-k149, the live folder at its live settings (256k, KV pin 7.25 GiB), the reference of the
#            same window: boot, smoke, one replay, the speed test (3 runs).
#   code226  tp1-prunecode226t5-k154 at the settings of the deploy (160k, KV pin CAP_PIN, default 7,431,600,000 bytes):
#            boot, smoke, two replays, the speed test (5 runs), the prefill probe up to 128k tokens. If vLLM says the
#            pin is too small, one more boot at the need plus 4%, as cap_switch.sh planned.
# Steps: wait for START_AFTER; refuse the weekday cron window (05:00 to 10:00) unless CRON_OK=1; stop Qwen; the
# points; restore DeepSeek; start Qwen again if it ran before (QWEN_AFTER=1: also when it did not); log the gates.
# Built from loader_window.sh and the window rules of run_lib.sh (two failed boots in a row: restore).
# CAUTION: this stops both model servers for about 2 h. Start it only with the user's go, in tmux:
#   tmux new -d -s review "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... \
#     ~/Projects/dsv4-specialize/scripts/review_code226.sh"
D=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/review-code226.log}
RESTORE_TAG=review
TEST_COMPOSE="-f compose/compose-mask.yml -f serve-tune/compose-tune.yml -f compose/compose-perlayer.yml -f compose/compose-cap.yml"
source "$D/run_lib.sh"

POINTS=${POINTS:-md code226}
declare -A FOLDER=([md]=prunemd5:149 [code226]=prunecode226t5:154) LEN=([md]=256000 [code226]=160000) \
  PIN=([md]=7784628224 [code226]=${CAP_PIN:-7431600000}) REPLAYS=([md]="a" [code226]="a b") RUNS=([md]=3 [code226]=5) \
  NEED=([md]=45 [code226]=80) FREE_MB
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot TUNE_PREFILL=1024
export TUNE_LABEL=rv-md MASK_NAME=prunemd5 MASK_K=149 TUNE_KV_BYTES=7784628224   # compose-tune.yml needs them, also for a down
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
PIDFILE=${PIDFILE:-$R/pair-qwen.pid}
QWEN_SEQS=${QWEN_SEQS:-4} QWEN_KV_BYTES=${QWEN_KV_BYTES:-3575989770}   # the live Qwen before the deploy
FAILS=0
pct() { awk '$1 == "nll" {printf "%+.2f", 100 * ($3 / $2 - 1)}' "$1" 2>/dev/null; }

probe() {  # $1 label, $2 kind (probe|smoke), the rest: prefill_probe.py arguments
  local label=$1 kind=$2 left=$(( RESTORE_S - $(now) )) rc; shift 2
  [ $left -le $([ $kind = probe ] && echo 900 || echo 120) ] && { log "$label: no time for the $kind"; return; }
  memwatch_start "$label-$kind"
  timeout $left $PY $S/serve-tune/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$label \
    --out $R/$kind-$label.json "$@" > /dev/null 2> $R/$kind-$label.err; rc=$?
  memwatch_stop
  log "$label $kind: exit $rc; $(tail -1 $R/$kind-$label.err | cut -c1-300)"
}

rpoint() {  # $1 point; returns 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 label=rv-$1 name=${FOLDER[$1]%:*} k=${FOLDER[$1]#*:} rc need x
  export TUNE_LABEL=$label MASK_NAME=$name MASK_K=$k CAP_LEN=${LEN[$p]} TUNE_KV_BYTES=${PIN[$p]}
  [ $(( $(now) + ${NEED[$p]} * 60 )) -gt $RESTORE_S ] && { log "$label: needs ${NEED[$p]} min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  [ -f $S/tp1-$name-k$k/config.json ] || { log "ERROR: no folder $S/tp1-$name-k$k"; return 2; }
  stop_main || return 9
  log "$label: folder tp1-$name-k$k, max-model-len ${LEN[$p]}, KV pin ${PIN[$p]} bytes"
  boot $label; rc=$?
  need=$(grep -oE '[0-9.]+ GiB KV cache is needed' $R/boot-$label.log 2>/dev/null | head -1 | cut -d' ' -f1)
  if [ $rc = 1 ] && [ -n "$need" ]; then
    PIN[$p]=$(awk -v n=$need 'BEGIN {printf "%d", n * 1.04 * 2^30}')
    export TUNE_KV_BYTES=${PIN[$p]}
    log "$label: the KV pin was too small (the need is $need GiB): one more boot with ${PIN[$p]} bytes"
    boot $label; rc=$?
  fi
  [ $rc != 0 ] && return $rc
  FREE_MB[$p]=$(avail_mb)
  log "$label: ${FREE_MB[$p]} MB available after the boot; $(grep -oE 'per-layer experts: [0-9]+ counts' $R/boot-$label.log | head -1)"
  probe $label smoke --smoke
  for x in ${REPLAYS[$p]}; do replay $label-$x; done   # each replay needs its own label: a rerun skips its targets
  BOT_RUNS=${RUNS[$p]} speed $label
  [ $p = code226 ] && probe $label probe --cold 8000 32000 128000
  down || return 9
}

qwen_up() { [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
stop_qwen() {  # as pair_run.sh: the process group of the server
  local pid=$(cat "$PIDFILE") i
  log "stop Qwen (pid $pid; $(avail_mb) MB available)"
  kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid"
  for i in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
  kill -0 "$pid" 2>/dev/null && { kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid"; }
  rm -f "$PIDFILE"
}
start_qwen() {
  log "start Qwen again (SEQS=$QWEN_SEQS KV_BYTES=$QWEN_KV_BYTES TESTS=idle) after 5 min for the memory drift"
  sleep 300
  local i
  for i in $(seq 1 30); do [ "$(avail_mb)" -ge "${NEED_MB:-28600}" ] && break; sleep 30; done
  (cd $S && TAG=review SEQS=$QWEN_SEQS KV_BYTES=$QWEN_KV_BYTES TESTS=idle KEEP=1 ./scripts/pair_run.sh) >> "$LOG" 2>&1
  qwen_up && log "Qwen up again" || log "ERROR: Qwen did not start; see logs/pair-review.log"
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

report() {  # the gates of §5, from the files of the window
  local p
  for p in $POINTS; do
    log "rv-$p boot: $(grep -oE 'Model loading took [0-9.]+ GiB|per-layer experts: [0-9]+ counts|GPU KV cache size: [0-9,]+ tokens' \
      $R/boot-rv-$p.log 2>/dev/null | sort -u | tr '\n' ';'); KV pin ${PIN[$p]}; ${FREE_MB[$p]:-?} MB available"
    log "rv-$p smoke: $(tail -1 $R/smoke-rv-$p.err 2>/dev/null | cut -c1-200)"
  done
  for l in rv-md-a rv-code226-a rv-code226-b; do
    log "$l: NLL $(pct $R/$l-compare.txt)% against baseline-k216; $(grep -E '^(PASS|FAIL)' $R/$l-compare.txt 2>/dev/null | cut -c1-6 | tr '\n' ' ')"
  done
  $PY - "$R" <<'PYEOF' 2>&1 | while read -r line; do log "$line"; done
import json, subprocess, sys
from pathlib import Path
r = Path(sys.argv[1])
load = lambda p: {(x["session"], x["segment"], x["index"]): x for x in map(json.loads, p.read_text().splitlines())}
files = {"md": [r / "rv-md-a.jsonl"], "code226": [r / "rv-code226-a.jsonl", r / "rv-code226-b.jsonl"]}
if not all(p.exists() for ps in files.values() for p in ps):
    sys.exit("code226 against md: a replay file is missing")
rows = {k: [load(p) for p in ps] for k, ps in files.items()}
keys = set.intersection(*(set(d) for ds in rows.values() for d in ds))
nll = {k: sum(d[t]["reasoning_nll"] + d[t]["answer_nll"] for d in ds for t in keys) / len(ds) for k, ds in rows.items()}
print(f"code226 against md, the mean of the replays on {len(keys)} common targets: NLL "
      f"{100 * (nll['code226'] / nll['md'] - 1):+.2f}% (the gate: at most +0.5%)")
rate = {}
for p in ("md", "code226"):
    out = subprocess.run([sys.executable, str(r.parent / "tools/speed_result.py"), "--results", str(r), f"rv-{p}"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    rate[p] = json.loads(out[-1])["bot"]["rate"] if out else None
if all(rate.values()):
    print(f"bot decode: md {rate['md']:.2f} tok/s, code226 {rate['code226']:.2f} tok/s, "
          f"{100 * (rate['code226'] / rate['md'] - 1):+.1f}% (the gate: within 3%)")
PYEOF
}

[ -n "${REVIEW_STUB:-}" ] && source "$REVIEW_STUB"   # the dry run of test_review_code226.py: stub servers and clients

log "review window start (points $POINTS; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
for need in $S/compose/compose-cap.yml $S/compose/compose-perlayer.yml $S/image-patch/perlayer/exl3.py \
  $S/image-patch/perlayer/deepseek_v4_model.py $S/serve-tune/prefill_probe.py $R/baseline-k216.jsonl; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ] || [ "$(now)" -ge $BOOT_BY_S ]; do sleep 30; done
[ "$(now)" -ge $BOOT_BY_S ] && { log "after $BOOT_BY: no window, no server action"; exit 1; }
if [ "${CRON_OK:-0}" != 1 ] && cron_clash "$(now)" "$RESTORE_S"; then
  log "ERROR: the window up to $RESTORE_AT touches the cron window (05:00 to 10:00 on weekdays): no server action"
  exit 1
fi

QWEN_WAS_UP=0
if qwen_up; then QWEN_WAS_UP=1; stop_qwen; fi
for p in $POINTS; do
  rpoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; break; };;
    9) log "safety failure: the safe action, restore"; break;;
  esac
done
restore
{ [ $QWEN_WAS_UP = 1 ] || [ "${QWEN_AFTER:-0}" = 1 ]; } && start_qwen
report
log "review window done"

#!/usr/bin/env bash
# The shared functions and safety rules of the test-boot scripts (topk_run.sh, prune_boot.sh, mem_boot.sh,
# capture_run.sh, rank_night.sh, draft_test.sh). A script sets its own values, then sources this file:
#   LOG=...  RESTORE_TAG=...  [TEST_COMPOSE=...]  [TEST_PROJECT=...]  [MASK=...]
#   source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"
# The deadlines START_AFTER, BOOT_BY, SPEED_BY and RESTORE_AT come from the environment and have no defaults.
#
# Safety (PLAN_OVERNIGHT_MASK_GRID.md §2):
# - never two models in memory: before each boot, no server container and at least 100 GiB available;
# - a boot that is not healthy in 15 min: down, record, go on; two failed boots in a row: restore;
# - free memory every 30 s during each client run; below 1 GiB, stop the client;
# - deadlines: no boot with a replay after BOOT_BY, no speed test after SPEED_BY, and every client stops
#   at RESTORE_AT, when the restore starts. Any other failure: the safe action (restore).
# The live server is MAIN_COMPOSE with the container MAIN_CONTAINER (default: compose-md149t5.yml in the server folder,
# the MD prune since 2026-11-12). Change the defaults here and in overnight_guard.sh when the live server changes.
set -u
TEST_COMPOSE=${TEST_COMPOSE:--f compose-mask.yml}   # the compose files of the test server (word-split on purpose)
TEST_PROJECT=${TEST_PROJECT:-dsv4-mask}             # the compose project name of the test server
RESTORE_TAG=${RESTORE_TAG:?set RESTORE_TAG}          # names the main-restore-<tag>-<try>.log files
START_AFTER=${START_AFTER:?set START_AFTER, e.g. "2026-11-07 22:30"}
BOOT_BY=${BOOT_BY:?set BOOT_BY}
SPEED_BY=${SPEED_BY:?set SPEED_BY}
RESTORE_AT=${RESTORE_AT:?set RESTORE_AT}
SERVER=/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark
S=/home/colin-spark/Projects/dsv4-specialize
W=/home/colin-spark/Projects/stock-research-model-eval
PY=/home/colin-spark/.hermes/stock-research/.venv/bin/python
R=$S/results   # the result files; until 2026-11-08 they were in $W/model_eval/results
URL=http://127.0.0.1:8890
MAIN_COMPOSE=${MAIN_COMPOSE:-$SERVER/compose-md149t5.yml}
MAIN=${MAIN_CONTAINER:-deepseek-v4-flash-md149t5-deepseek-v4-flash-1}
MASK=${MASK:-dsv4-mask-deepseek-v4-flash-mask-1}
LOG=${LOG:?set LOG before you source run_lib.sh}
BOOT_BY_S=$(date -d "$BOOT_BY" +%s); SPEED_BY_S=$(date -d "$SPEED_BY" +%s); RESTORE_S=$(date -d "$RESTORE_AT" +%s)

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
now() { date +%s; }
avail_gib() { free -g | awk '/^Mem:/ {print $7}'; }
avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
main_running() { [ "$(docker inspect -f '{{.State.Running}}' $MAIN 2>/dev/null)" = true ]; }
servers() { docker ps --format '{{.Names}}' | grep -E 'deepseek|dsv4'; }
busy() { curl -s --max-time 5 http://127.0.0.1:8888/metrics | awk '/^vllm:num_requests_(running|waiting)/ {s+=$2} END {print (s>0)}'; }
wait_mem() {  # $1: seconds, $2: GiB (default 100); true when at least that much memory is available
  local end=$(( $(now) + ${1:-180} )) need=${2:-100}
  while [ "$(now)" -lt "$end" ]; do [ "$(avail_gib)" -ge "$need" ] && return 0; sleep 5; done
  [ "$(avail_gib)" -ge "$need" ]
}
memwatch_start() {  # $1: label
  ( while true; do m=$(avail_mb); echo "$(date +%T) available_mb=$m" >> $S/logs/memory-$1.log
      if [ "$m" -lt 1024 ]; then log "MEMORY below 1 GiB ($m MB): stop the client"
        pkill -f "model_eval.replay|model_eval.speed|acceptance_c1|prefill_probe"; fi
      sleep 30; done ) &
  MW=$!
}
memwatch_stop() { kill "$MW" 2>/dev/null; wait "$MW" 2>/dev/null; }

down() {  # stop the test server; true when no test container is left
  log "stop the test server ($MASK_NAME-k$MASK_K)"
  (cd $S && docker compose $TEST_COMPOSE down) >> "$LOG" 2>&1
  if docker ps -a --format '{{.Names}}' | grep -q "$TEST_PROJECT"; then log "ERROR: a test container is left"; return 1; fi
}

stop_main() {  # stop the main server when the bot is idle (up to 30 min); 9 on a safety failure
  if main_running; then
    local end=$(( $(now) + 1800 ))
    while [ "$(busy)" != 0 ] && [ "$(now)" -lt "$end" ]; do sleep 15; done
    [ "$(busy)" = 0 ] || { log "the bot stayed busy for 30 min"; return 9; }
    log "stop the main server"
    (cd $SERVER && docker compose -f "$MAIN_COMPOSE" stop) >> "$LOG" 2>&1
    main_running && { log "ERROR: the main server still runs"; return 9; }
  fi
  if servers > /dev/null; then log "ERROR: a server container runs: $(servers | tr '\n' ' ')"; return 9; fi
  wait_mem 300 || { log "ERROR: less than 100 GiB available ($(avail_gib))"; return 9; }
}

boot() {  # $1: label; 0 healthy, 1 boot failed (down done), 9 safety failure
  local label=$1 ok=0 t0 boot_log=$R/boot-$1.log
  if servers > /dev/null; then log "ERROR: a server container runs: $(servers | tr '\n' ' ')"; return 9; fi
  # BOOT_MIN_GIB: a test boot at util 0.90 needs 109.5 GiB free (vLLM's boot check); the restore keeps 100
  wait_mem 300 "${BOOT_MIN_GIB:-100}" || { log "ERROR: less than ${BOOT_MIN_GIB:-100} GiB available ($(avail_gib))"; return 9; }
  log "boot $label ($(avail_gib) GiB available)"
  (cd $S && docker compose $TEST_COMPOSE up -d) >> "$LOG" 2>&1
  t0=$(now)
  while [ $(( $(now) - t0 )) -lt 900 ]; do
    curl -fsS --max-time 5 $URL/health > /dev/null 2>&1 && { ok=1; break; }
    case "$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' $MASK 2>/dev/null || echo gone)" in
      exited\ [1-9]*|dead*|gone*) break;;
    esac
    sleep 15
  done
  docker logs $MASK > "$boot_log" 2>&1
  log "$label boot lines: $(grep -oE 'Model loading took [0-9.]+ GiB[^,]*|GPU KV cache size: [0-9,]+ tokens' "$boot_log" | sort -u | tr '\n' ';')"
  grep -iE 'scratch|top-?k[ =]|tid2eid|num_experts_per_tok' "$boot_log" | grep -v '^$' | tail -5 | sed 's/^/    /' | tee -a "$LOG" > /dev/null
  if [ $ok = 0 ]; then
    log "ERROR: $label not healthy after $(( $(now) - t0 )) s; last error lines:"
    grep -iE 'error|exception|mismatch' "$boot_log" | tail -8 | sed 's/^/    /' | tee -a "$LOG" > /dev/null
    down || return 9
    return 1
  fi
  log "$label healthy after $(( $(now) - t0 )) s"
}

replay() {  # $1: label
  local label=$1 left=$(( RESTORE_S - $(now) ))
  [ $left -le 60 ] && { log "$label: no time for the replay"; return; }
  memwatch_start "$label"
  log "$label replay ($left s left)"
  (cd $W && timeout $left $PY -m model_eval.replay --base-url $URL --out $R/$label.jsonl > $R/$label.log 2>&1)
  log "$label replay exit $?: $(wc -l < $R/$label.jsonl 2>/dev/null) rows"
  memwatch_stop
  (cd $W && $PY -m model_eval.replay --compare $R/baseline-k216.jsonl $R/$label.jsonl > $R/$label-compare.txt 2>&1)
  (cd $W && $PY -m model_eval.replay --compare $R/botmask-k160.jsonl $R/$label.jsonl > $R/$label-vs-botmask-k160.txt 2>&1)
  log "$label against the baseline: $(grep -E '^(nll|tool_names|PASS|FAIL)' $R/$label-compare.txt | tr -s ' ' | tr '\n' ';')"
}

gates_pass() {  # $1: label; both gates PASS on all 60 targets
  [ "$(grep -c '^PASS' $R/$1-compare.txt 2>/dev/null)" = 2 ] && [ "$(wc -l < $R/$1.jsonl 2>/dev/null)" -ge 60 ]
}

speed() {  # $1: label; BOT_RUNS sets the runs of the bot decode test (default 3)
  local label=$1 model=deepseek-v4-flash-0731-$1 left n
  [ "$(now)" -ge $SPEED_BY_S ] && { log "$label: after $SPEED_BY, no speed test"; return; }
  memwatch_start "$label-speed"
  left=$(( RESTORE_S - $(now) ))
  log "$label bot decode test ($left s left)"
  (cd $W && timeout $left $PY -m model_eval.speed --base-url $URL --out $R/speed-bot-$label.jsonl --runs ${BOT_RUNS:-3} > $R/speed-bot-$label.log 2>&1)
  log "$label bot decode test exit $?: $(tail -1 $R/speed-bot-$label.log | cut -c1-200)"
  if [ "$(now)" -ge $SPEED_BY_S ]; then
    log "$label: after $SPEED_BY, no code decode test"
  else
    for n in 0 1 2 3; do
      left=$(( RESTORE_S - $(now) ))
      [ $left -le 60 ] && { log "$label: restore time reached"; break; }
      [ $n = 1 ] && (cd $W && $PY -c "import json; from model_eval.speed import counters; print(json.dumps(counters('$URL')))" > $R/speed-code-$label-counters-before.json)
      (cd $W && timeout $left $PY $SERVER/image-patch/acceptance_c1.py --skip-correctness --skip-prefill --base-url $URL --model $model \
        > $R/speed-code-$label-$n.json 2> $R/speed-code-$label-$n.err)
      log "$label code decode run $n exit $?"
    done
    (cd $W && $PY -c "import json; from model_eval.speed import counters; print(json.dumps(counters('$URL')))" > $R/speed-code-$label-counters-after.json)
  fi
  memwatch_stop
  log "$label speed: $($PY $S/speed_result.py --results $R $label)"
}

restore() {  # §6 = PLAN_OVERNIGHT_MASK_GRID.md §5
  log "restore: stop every test"
  (cd $S && MASK_K=${MASK_K:-160} docker compose $TEST_COMPOSE down) >> "$LOG" 2>&1
  if docker ps -a --format '{{.Names}}' | grep -q "$TEST_PROJECT"; then log "RESTORE FAIL: a test container is left; the guard restores at 04:15"; exit 3; fi
  if ! main_running; then
    wait_mem 600 || { log "RESTORE FAIL: less than 100 GiB available ($(avail_gib)); the guard restores at 04:15"; exit 3; }
    log "start the main server ($(avail_gib) GiB available)"
    (cd $SERVER && docker compose -f "$MAIN_COMPOSE" start) >> "$LOG" 2>&1
  fi
  local t0 try
  for try in 1 2; do
    t0=$(now)
    until curl -fsS --max-time 5 http://127.0.0.1:8888/health > /dev/null 2>&1; do
      [ $(( $(now) - t0 )) -gt 1200 ] && break
      sleep 15
    done
    curl -fsS --max-time 5 http://127.0.0.1:8888/health > /dev/null 2>&1 && break
    log "main server not healthy after 1200 s (try $try)"
    (cd $SERVER && docker compose -f "$MAIN_COMPOSE" logs --tail 80) > $R/main-restore-$RESTORE_TAG-$try.log 2>&1
    [ $try = 1 ] && (cd $SERVER && docker compose -f "$MAIN_COMPOSE" restart) >> "$LOG" 2>&1
  done
  if ! curl -fsS --max-time 5 http://127.0.0.1:8888/health > /dev/null 2>&1; then
    log "RESTORE FAIL: main server not healthy; logs in $R/main-restore-$RESTORE_TAG-*.log; the guard tries at 04:15"; exit 3
  fi
  local reply
  reply=$(curl -s --max-time 120 http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash-0731-k160t5","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"thinking":false}}' \
    | $PY -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>&1)
  log "RESTORE DONE: main server healthy at $(date +%H:%M); reply: ${reply:0:40}; hermes-gateway $(systemctl --user is-active hermes-gateway)"
}

# One point: stop the main server if it runs, boot, replay, speed tests (full) and stop.
# Returns 0 done, 1 boot failed, 2 skipped (deadline), 9 safety failure.
point() {  # $1 MASK_NAME, $2 MASK_K, $3 full|replay
  local label=$1-k$2 rc
  export MASK_NAME=$1 MASK_K=$2
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  [ -f $S/tp1-$label/rank-sliced-tp1-manifest.json ] || { log "ERROR: no folder $S/tp1-$label; point skipped"; return 2; }
  stop_main || return 9
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  replay $label
  [ $3 = full ] && speed $label
  down || return 9
}

FAILS=0
run_point() {  # point, with the rule of two failed boots in a row
  point "$@"; local rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
  return $rc
}

#!/usr/bin/env bash
# The top-k test boots of PLAN_OVERNIGHT_TOPK.md §5, then the restore of the main server (§6).
# Start it in its own tmux session after the folders of §4 pass the self-check:
#   tmux new -d -s topk-run ~/Projects/dsv4-specialize/topk_run.sh
# Safety (PLAN_OVERNIGHT_MASK_GRID.md §2):
# - never two models in memory: before each boot, no server container and at least 100 GiB available;
# - a boot that is not healthy in 15 min: down, record, go on; two failed boots in a row: restore;
# - free memory every 30 s during each client run; below 1 GiB, stop the client;
# - deadlines: no boot with a replay after BOOT_BY, no speed test after SPEED_BY, and every client stops
#   at RESTORE_AT, when the restore starts. Any other failure: the safe action (restore).
set -u
START_AFTER=${START_AFTER:-2026-11-04 23:45}
BOOT_BY=${BOOT_BY:-2026-11-05 03:00}
SPEED_BY=${SPEED_BY:-2026-11-05 03:05}
RESTORE_AT=${RESTORE_AT:-2026-11-05 03:45}
SERVER=/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark
S=/home/colin-spark/Projects/dsv4-specialize
W=/home/colin-spark/Projects/stock-research-model-eval
PY=/home/colin-spark/.hermes/stock-research/.venv/bin/python
R=$W/model_eval/results
URL=http://127.0.0.1:8890
MAIN=deepseek-v4-flash-k160t5-deepseek-v4-flash-1   # the live server since 2026-11-06 (compose-k160t5.yml)
MASK=dsv4-mask-deepseek-v4-flash-mask-1
LOG=${LOG:-/home/colin-spark/.hermes-e2e/topk-run.log}
SPEEDF=/home/colin-spark/.hermes-e2e/stepf-speed.log
BOOT_BY_S=$(date -d "$BOOT_BY" +%s); SPEED_BY_S=$(date -d "$SPEED_BY" +%s); RESTORE_S=$(date -d "$RESTORE_AT" +%s)

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
now() { date +%s; }
avail_gib() { free -g | awk '/^Mem:/ {print $7}'; }
avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
main_running() { [ "$(docker inspect -f '{{.State.Running}}' $MAIN 2>/dev/null)" = true ]; }
servers() { docker ps --format '{{.Names}}' | grep -E 'deepseek|dsv4'; }
busy() { curl -s --max-time 5 http://127.0.0.1:8888/metrics | awk '/^vllm:num_requests_(running|waiting)/ {s+=$2} END {print (s>0)}'; }
wait_mem() {  # $1: seconds; true when at least 100 GiB are available
  local end=$(( $(now) + ${1:-180} ))
  while [ "$(now)" -lt "$end" ]; do [ "$(avail_gib)" -ge 100 ] && return 0; sleep 5; done
  [ "$(avail_gib)" -ge 100 ]
}
memwatch_start() {  # $1: label
  ( while true; do m=$(avail_mb); echo "$(date +%T) available_mb=$m" >> /home/colin-spark/.hermes-e2e/memory-$1.log
      if [ "$m" -lt 1024 ]; then log "MEMORY below 1 GiB ($m MB): stop the client"
        pkill -f "model_eval.replay|model_eval.speed|acceptance_c1"; fi
      sleep 30; done ) &
  MW=$!
}
memwatch_stop() { kill "$MW" 2>/dev/null; wait "$MW" 2>/dev/null; }

down() {  # stop the test server; true when no test container is left
  log "stop the test server ($MASK_NAME-k$MASK_K)"
  (cd $S && docker compose -f compose-mask.yml down) >> "$LOG" 2>&1
  if docker ps -a --format '{{.Names}}' | grep -q dsv4-mask; then log "ERROR: a test container is left"; return 1; fi
}

stop_main() {  # stop the main server when the bot is idle (up to 30 min); 9 on a safety failure
  if main_running; then
    local end=$(( $(now) + 1800 ))
    while [ "$(busy)" != 0 ] && [ "$(now)" -lt "$end" ]; do sleep 15; done
    [ "$(busy)" = 0 ] || { log "the bot stayed busy for 30 min"; return 9; }
    log "stop the main server"
    (cd $SERVER && docker compose -f compose-k160t5.yml stop) >> "$LOG" 2>&1
    main_running && { log "ERROR: the main server still runs"; return 9; }
  fi
  if servers > /dev/null; then log "ERROR: a server container runs: $(servers | tr '\n' ' ')"; return 9; fi
  wait_mem 300 || { log "ERROR: less than 100 GiB available ($(avail_gib))"; return 9; }
}

boot() {  # $1: label; 0 healthy, 1 boot failed (down done), 9 safety failure
  local label=$1 ok=0 t0 boot_log=$R/boot-$1.log
  if servers > /dev/null; then log "ERROR: a server container runs: $(servers | tr '\n' ' ')"; return 9; fi
  wait_mem 300 || { log "ERROR: less than 100 GiB available ($(avail_gib))"; return 9; }
  log "boot $label ($(avail_gib) GiB available)"
  (cd $S && docker compose -f compose-mask.yml up -d) >> "$LOG" 2>&1
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

speed() {  # $1: label
  local label=$1 model=deepseek-v4-flash-0731-$1 left n
  [ "$(now)" -ge $SPEED_BY_S ] && { log "$label: after $SPEED_BY, no speed test"; return; }
  memwatch_start "$label-speed"
  left=$(( RESTORE_S - $(now) ))
  log "$label bot decode test ($left s left)"
  (cd $W && timeout $left $PY -m model_eval.speed --base-url $URL --out $R/speed-bot-$label.jsonl --runs 3 > $R/speed-bot-$label.log 2>&1)
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
  (cd $S && MASK_K=${MASK_K:-160} docker compose -f compose-mask.yml down) >> "$LOG" 2>&1
  if docker ps -a --format '{{.Names}}' | grep -q dsv4-mask; then log "RESTORE FAIL: a test container is left; the guard restores at 04:15"; exit 3; fi
  if ! main_running; then
    wait_mem 600 || { log "RESTORE FAIL: less than 100 GiB available ($(avail_gib)); the guard restores at 04:15"; exit 3; }
    log "start the main server ($(avail_gib) GiB available)"
    (cd $SERVER && docker compose -f compose-k160t5.yml start) >> "$LOG" 2>&1
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
    (cd $SERVER && docker compose -f compose-k160t5.yml logs --tail 80) > $R/main-restore-topk-$try.log 2>&1
    [ $try = 1 ] && (cd $SERVER && docker compose -f compose-k160t5.yml restart) >> "$LOG" 2>&1
  done
  if ! curl -fsS --max-time 5 http://127.0.0.1:8888/health > /dev/null 2>&1; then
    log "RESTORE FAIL: main server not healthy; logs in $R/main-restore-topk-*.log; the guard tries at 04:15"; exit 3
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

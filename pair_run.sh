#!/usr/bin/env bash
# Phase D (plans/PLAN_QWEN_PRUNE.md §8, PLAN_SECOND_MODEL_FIT.md §7): Qwen3.6-35B-A3B K 208 with the NVFP4 draft
# (§7.9) beside the live DeepSeek, with no stop of DeepSeek. TAG names the run (nocap or cap).
#   1. Checks: DeepSeek healthy, no Qwen server, the memory watch on (pair_watch.sh), at least NEED_MB available,
#      not in the cron window (05:00 to 10:00 on weekdays).
#   2. Qwen boots on port 8889 as in point d2 of qwen_fit.sh, with --gpu-memory-utilization 0.005: with a pinned KV
#      cache, vLLM uses the value only for its free-memory check at the boot, and that check does not count the
#      page cache (§7.2 of the fit plan). The pid goes to the pid file of the watch.
#   3. The tests of §7.3 of the fit plan, each with the lowest available memory during it:
#      idle    both servers idle, 60 s
#      qlong   Qwen: one prompt of 120,000 tokens, while DeepSeek decodes 500 tokens
#      dlong   DeepSeek: one prompt of 128,000 tokens, while Qwen runs the bot decode test
#      qspeed  the bot decode test on Qwen alone;  dspeed  on DeepSeek alone;  both  on both at the same time
#   4. KEEP=1 (default) leaves Qwen up for the 24-hour gate; KEEP=0 stops it.
# CAUTION: out of memory, the Spark can do a hard reset. The watch stops Qwen below 2 GiB. Start this only with the
# user's go:   tmux new -d -s pair "TAG=nocap ~/Projects/dsv4-specialize/pair_run.sh"
set -u
S=${PAIR_S:-/home/colin-spark/Projects/dsv4-specialize}   # PAIR_S: a test folder (test_pair_scripts.py)
R=$S/results W=/home/colin-spark/Projects/stock-research-model-eval
PY=/home/colin-spark/.hermes/stock-research/.venv/bin/python QV=${QV:-$HOME/venvs/vllm033/bin}
TAG=${TAG:?set TAG, e.g. nocap or cap}
LOG=${LOG:-$S/logs/pair-$TAG.log} MEM=${MEM:-$R/pair-mem.log} PIDFILE=${PIDFILE:-$R/pair-qwen.pid}
NEED_MB=${NEED_MB:-28600}   # the peak of d2 (25.42 GiB) + the floor of the watch (2 GiB) + 0.5 GiB
KEEP=${KEEP:-1}
DS=http://127.0.0.1:8888 DSNAME=deepseek-v4-flash-0731-k160t5 QURL=http://127.0.0.1:8889 QNAME=second-model-fit
QSNAP=$(ls -d $HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/ 2>/dev/null | head -1)
DRAFT="{\"method\":\"mtp\",\"model\":\"$S/qwen-draft-nvfp4\",\"num_speculative_tokens\":3,\"moe_backend\":\"marlin\"}"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
now() { date +%s; }
avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
healthy() { curl -fsS --max-time 5 "$1/health" > /dev/null 2>&1; }
qwen_up() { [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
lowest() {  # $1 $2: epochs; the lowest available MB of the watch between them
  awk -v a="$1" -v b="$2" '$1 >= a && $1 <= b && (m == "" || $2 < m) {m = $2} END {print m}' "$MEM"
}
rate() { $PY $S/qwen_fit.py --rate "$1" 2>&1 | tail -1; }
stop_qwen() {  # the process group of the server (setsid), or the pid alone
  local pid
  qwen_up || { rm -f "$PIDFILE"; return 0; }
  pid=$(cat "$PIDFILE")
  kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid"
  sleep 10
  kill -0 "$pid" 2>/dev/null && { kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid"; }
  rm -f "$PIDFILE"
}

checks() {
  local h=$(date +%H) d=$(date +%u)
  healthy $DS || { log "ERROR: DeepSeek is not healthy"; return 1; }
  qwen_up && { log "ERROR: a Qwen server runs"; return 1; }
  healthy $QURL && { log "ERROR: port 8889 answers"; return 1; }
  tmux has-session -t pair-watch 2>/dev/null || { log "ERROR: the memory watch (tmux pair-watch) is not running"; return 1; }
  [ "$d" -le 5 ] && [ "$h" -ge 5 ] && [ "$h" -lt 10 ] && { log "ERROR: the cron window (05:00 to 10:00 on weekdays)"; return 1; }
  [ "$(avail_mb)" -ge "$NEED_MB" ] || { log "ERROR: $(avail_mb) MB available, below $NEED_MB"; return 1; }
  [ -f $S/qwen-k208/config.json ] && [ -f $S/qwen-draft-nvfp4/config.json ] || { log "ERROR: the K 208 or draft folder is missing"; return 1; }
}

boot() {
  local t0
  env PYTHONPATH=$S/qwen_mtp_patch HF_HUB_OFFLINE=1 VLLM_USE_FASTOKENS=0 CUTE_DSL_ARCH=sm_121a PATH=$QV:$PATH \
    setsid "$QV/vllm" serve $S/qwen-k208 --host 127.0.0.1 --port 8889 --served-model-name $QNAME \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.005 --kv-cache-memory-bytes 1965377034 --max-model-len 128000 \
    --max-num-seqs 1 --max-num-batched-tokens 2048 --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --async-scheduling \
    --enable-auto-tool-choice --generation-config vllm --attention-backend flashinfer --moe-backend flashinfer_cutlass \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --language-model-only --speculative-config "$DRAFT" \
    > $R/pair-$TAG-server.log 2>&1 &
  echo $! > "$PIDFILE"
  t0=$(now)
  while [ $(( $(now) - t0 )) -lt 900 ]; do
    healthy $QURL && return 0
    qwen_up || return 1
    sleep 5
  done
  return 1
}

run_test() {  # $1 name, then the command; logs the exit and the lowest available memory
  local name=$1 t0 rc
  shift
  t0=$(now)
  "$@"; rc=$?
  log "pair-$TAG $name: exit $rc; lowest available $(lowest $t0 $(now)) MB; Qwen $(qwen_up && echo up || echo DOWN)"
  qwen_up || { log "ERROR: Qwen is down (the watch, or a crash): no more tests"; return 1; }
}
t_idle() { sleep 60; }
t_qlong() {
  $PY $S/qwen_fit.py --base-url $QURL --model $QNAME --long 120000 > $R/pair-$TAG-qlong.json 2>&1 &
  local q=$!
  $PY $S/qwen_fit.py --base-url $DS --model $DSNAME --long 2000 > $R/pair-$TAG-ddecode.json 2>&1
  wait $q
}
t_dlong() {
  (cd $W && $PY -m model_eval.speed --base-url $QURL --model $QNAME --sessions 5 --runs 1 --out $R/pair-$TAG-qdecode.jsonl > $R/pair-$TAG-qdecode.log 2>&1) &
  local q=$!
  timeout 1200 $PY $S/qwen_fit.py --base-url $DS --model $DSNAME --long 128000 > $R/pair-$TAG-dlong.json 2>&1
  wait $q
}
speed_of() {  # $1 url, $2 model, $3 out
  (cd $W && timeout 1800 $PY -m model_eval.speed --base-url $1 --model $2 --sessions 5 --runs 1 --out $3.jsonl > $3.log 2>&1)
}
t_qspeed() { speed_of $QURL $QNAME $R/pair-$TAG-qspeed; }
t_dspeed() { speed_of $DS $DSNAME $R/pair-$TAG-dspeed; }
t_both() {   # wait for the Qwen client by its pid: a bare wait also waits for the Qwen server, a child of this shell
  speed_of $QURL $QNAME $R/pair-$TAG-bothq &
  local q=$!
  speed_of $DS $DSNAME $R/pair-$TAG-bothd
  wait $q
}

[ -n "${PAIR_STUB:-}" ] && source "$PAIR_STUB"   # the dry run of test_pair_scripts.py
log "phase D ($TAG) start: $(avail_mb) MB available beside DeepSeek"
checks || exit 1
log "boot Qwen K 208 with the NVFP4 draft on port 8889"
if ! boot; then
  log "ERROR: Qwen did not boot: $(grep -iE 'error|exception' $R/pair-$TAG-server.log | tail -2 | cut -c1-300)"
  stop_qwen; exit 1
fi
log "Qwen healthy: $(avail_mb) MB available; $(grep -oE 'Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|torch.compile took [0-9.]+ s' $R/pair-$TAG-server.log | sort -u | tr '\n' ';')"
for t in idle qlong dlong qspeed dspeed both; do run_test $t t_$t || break; done
log "rates: Qwen alone $(rate $R/pair-$TAG-qspeed.jsonl); DeepSeek alone $(rate $R/pair-$TAG-dspeed.jsonl); together Qwen $(rate $R/pair-$TAG-bothq.jsonl), DeepSeek $(rate $R/pair-$TAG-bothd.jsonl)"
log "long prompts: Qwen $(tail -1 $R/pair-$TAG-qlong.json 2>/dev/null | cut -c1-120); DeepSeek $(tail -1 $R/pair-$TAG-dlong.json 2>/dev/null | cut -c1-120)"
if [ "$KEEP" = 1 ] && qwen_up; then
  log "phase D ($TAG) done: Qwen stays up on port 8889 for the 24-hour gate, with the watch in tmux pair-watch"
else
  stop_qwen; log "phase D ($TAG) done: Qwen stopped"
fi

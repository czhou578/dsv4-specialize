#!/usr/bin/env bash
# Phase A of plans/PLAN_SECOND_MODEL_FIT.md §4: measure the memory of Qwen 27B NVFP4 alone, at the settings of a second
# model beside DeepSeek (256k context, one sequence), then restore the live server. Qwen runs as a plain process of
# the vLLM 0.33 venv on port 8889, with no restart. The points (POINTS, default "f1 f2 f3"):
#   f1  prefill budget 2,048 tokens, no draft: the smallest setup
#   f2  prefill budget 8,192 tokens, no draft: what the prefill budget costs
#   f3  prefill budget 2,048 tokens, the MTP draft (K 3): what the draft costs and gains
# All points: --max-model-len 256000, a KV pin of 1.04x of the need, FP8 KV, --max-num-seqs 1, no images.
# Each point: the free GPU memory that torch sees, boot, smoke (a short request and a tool call), one prompt of
# about 250,000 tokens, the bot decode test (5 sessions, 1 run), stop. The available memory is sampled every 5 s.
# The report of a point (results/qwen-fit-<point>.json) gives the overhead X of §2.3: the available memory before the
# boot, minus its lowest value after the health check, minus the loaded weights and the KV pin. The load itself has
# a short, larger peak (--load-format fastsafetensors); the report gives it as load_peak.
# CAUTION: this stops the bot's model server for about 45 min. Start it only with the user's go, not 05:00-10:00 on
# weekdays while the cron jobs are active, and not during another test:
#   tmux new -d -s qwen-fit "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/qwen_fit.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/qwen-fit.log}
RESTORE_TAG=qwen-fit
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-f1 f2 f3}
QV=${QV:-$HOME/venvs/vllm033/bin}          # the venv of vLLM 0.33.0 (model-benchmarks docs/dgx-spark.md)
QMODEL=nvidia/Qwen3.8-27B-NVFP4 QNAME=qwen3.8-27b-fit QURL=http://127.0.0.1:8889
PIN0=${PIN0:-8933531976}                   # 8.32 GiB: 256,000 tokens x 33.6 KB x 1.04 (the benchmark: 16 GiB hold 512,023)
declare -A BUDGET=([f1]=2048 [f2]=8192 [f3]=2048) DRAFT=([f1]=0 [f2]=0 [f3]=1)
QPID= SAMPLER= PIN=$PIN0
QPAT='^[^ ]*python[^ ]* [^ ]*vllm serve '   # a real server process only: a shell that names the command does not match
qrunning() { pgrep -f "$QPAT" > /dev/null; }
sample_start() {  # $1 file: the available memory every 5 s
  ( while true; do echo "$(date +%s) $(avail_mb)" >> "$1"; sleep 5; done ) & SAMPLER=$!
}
sample_stop() { [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=; }
qstop() {  # stop Qwen and wait until it is gone (the sampler goes on: a boot retry must not end it)
  [ -n "$QPID" ] && kill -TERM -- -"$QPID" 2>/dev/null
  local t0=$(now)
  while qrunning && [ $(( $(now) - t0 )) -lt 60 ]; do sleep 2; done
  qrunning && pkill -KILL -f "$QPAT"
  QPID=
}
launch() {  # $1 point, $2 server log: start vllm and wait for /health; 0 healthy, 1 failed
  local p=$1 extra=() t0
  [ ${DRAFT[$p]} = 1 ] && extra=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
  HF_HUB_OFFLINE=1 VLLM_USE_FASTOKENS=0 PATH=$QV:$PATH setsid "$QV/vllm" serve $QMODEL --host 127.0.0.1 --port 8889 \
    --served-model-name $QNAME --tensor-parallel-size 1 --gpu-memory-utilization 0.30 --kv-cache-memory-bytes $PIN \
    --max-model-len 256000 --max-num-seqs 1 --max-num-batched-tokens ${BUDGET[$p]} --kv-cache-dtype fp8_e4m3 \
    --attention-backend flashinfer --enable-chunked-prefill --async-scheduling --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder --enable-auto-tool-choice --generation-config vllm --load-format fastsafetensors \
    --limit-mm-per-prompt '{"image":0,"video":0}' "${extra[@]}" > "$2" 2>&1 &
  QPID=$!
  t0=$(now)
  while [ $(( $(now) - t0 )) -lt 900 ]; do
    curl -fsS --max-time 5 $QURL/health > /dev/null 2>&1 && return 0
    kill -0 "$QPID" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
qboot() {  # $1 point: launch; if the KV pin is too small, one more launch with the need of the error plus 4%
  local p=$1 f=$R/qwen-fit-$1-server.log need
  launch $p $f && return 0
  qstop
  need=$(grep -oE '[0-9.]+ GiB KV cache is needed' $f | head -1 | cut -d' ' -f1)
  [ -n "$need" ] || { log "ERROR: qwen-$p did not boot: $(grep -iE 'error|exception' $f | tail -2 | cut -c1-200)"; return 1; }
  PIN=$(awk -v n=$need 'BEGIN {printf "%d", n * 1.04 * 2^30}')
  log "qwen-$p: the KV pin was too small, the need is $need GiB: one more boot with $PIN bytes"
  wait_mem 120 100
  launch $p $f && return 0
  qstop; log "ERROR: qwen-$p did not boot with the larger pin"; return 1
}
tests() {  # $1 point: smoke, the long prompt, the bot decode test
  local p=$1 left
  $PY $S/qwen_fit.py --base-url $QURL --model $QNAME --smoke > $R/qwen-fit-$p-smoke.json 2>&1
  log "qwen-$p smoke: exit $?; $(tail -1 $R/qwen-fit-$p-smoke.json | cut -c1-200)"
  left=$(( RESTORE_S - $(now) - 300 ))
  timeout $left $PY $S/qwen_fit.py --base-url $QURL --model $QNAME --long 250000 > $R/qwen-fit-$p-long.json 2>&1
  log "qwen-$p long prompt: exit $?; $(tail -1 $R/qwen-fit-$p-long.json | cut -c1-200)"
  left=$(( RESTORE_S - $(now) - 300 ))
  (cd $W && timeout $left $PY -m model_eval.speed --base-url $QURL --model $QNAME --sessions 5 --runs 1 \
    --out $R/qwen-fit-$p-speed.jsonl > $R/qwen-fit-$p-speed.log 2>&1)
  log "qwen-$p bot decode test: exit $?; $(tail -1 $R/qwen-fit-$p-speed.log | cut -c1-200)"
  log "qwen-$p decode rate from the rows: $($PY $S/qwen_fit.py --rate $R/qwen-fit-$p-speed.jsonl 2>&1 | tail -1)"
}
gpu_free() { PATH=$QV:$PATH python -c "import torch; f, t = torch.cuda.mem_get_info(); print(f'{f / 2**30:.1f} of {t / 2**30:.1f} GiB')" 2>&1 | tail -1; }
finish() { sample_stop; qstop; restore; }
[ -n "${QWEN_STUB:-}" ] && source "$QWEN_STUB"   # the dry run of test_qwen_fit.py: stub boots and clients

fpoint() {  # $1 point; 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 before healthy healthy_at m=$R/qwen-fit-$1-mem.log
  [ $(( $(now) + 20 * 60 )) -gt $RESTORE_S ] && { log "qwen-$p: needs 20 min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "qwen-$p: after $BOOT_BY, not booted"; return 2; }
  stop_main || return 9
  qrunning && { log "ERROR: a vllm serve process runs"; return 9; }
  wait_mem 300 100 || { log "ERROR: less than 100 GiB available ($(avail_gib))"; return 9; }
  before=$(avail_mb)
  log "qwen-$p: prefill budget ${BUDGET[$p]}, draft ${DRAFT[$p]}; $before MB available; torch sees $(gpu_free) free"
  rm -f $m; sample_start $m
  qboot $p || { sample_stop; return 1; }
  healthy=$(avail_mb) healthy_at=$(now)
  log "qwen-$p healthy: $healthy MB available; $(grep -oE 'Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens' $R/qwen-fit-$p-server.log | sort -u | tr '\n' ';')"
  tests $p
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > $R/qwen-fit-$p-gpu.txt 2>&1
  sample_stop; qstop
  $PY $S/qwen_fit.py --report --before $before --healthy $healthy --healthy-at $healthy_at --pin $PIN --mem-log $m \
    --server-log $R/qwen-fit-$p-server.log --gpu $R/qwen-fit-$p-gpu.txt --out $R/qwen-fit-$p.json 2>&1 | while read -r line; do log "qwen-$p: $line"; done
}

log "qwen fit window start (points $POINTS; boot by $BOOT_BY; restore at $RESTORE_AT)"
for need in $QV/vllm $S/qwen_fit.py $W/model_eval/speed.py; do [ -e $need ] || { log "ERROR: no $need"; exit 1; }; done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
[ "$(now)" -ge $BOOT_BY_S ] && { log "after $BOOT_BY: no boot"; exit 1; }

FAILS=0
for p in $POINTS; do
  fpoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; finish; exit 1; };;
    9) log "safety failure: the safe action, restore"; finish; exit 1;;
  esac
done
finish
log "qwen fit window done"

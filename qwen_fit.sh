#!/usr/bin/env bash
# Phase A of plans/PLAN_SECOND_MODEL_FIT.md §4 and §6: measure the memory and the speed of a second model alone, at the
# settings of a model beside DeepSeek (256k context, one sequence), then restore the live server. The model runs as a
# plain process of the vLLM 0.33 venv on port 8889, with no restart. The points (POINTS, default "f1 f2 f3"):
#   f1 f2 f3  Qwen3.8-27B NVFP4: prefill budget 2,048 with no draft, 8,192 with no draft, 2,048 with the MTP draft
#   q1 q2     Qwen3.6-35B-A3B NVFP4 (unsloth "Fast", a MoE with 3B active parameters): no draft, the MTP draft (K 3)
#   n1 n2     Nemotron-3.5-Lightning-30B-A3B NVFP4 (the recipe of its model card): no draft, the DSpark draft (K 3)
# The q and n points use a prefill budget of 2,048 tokens and the default loader (the fastsafetensors loader of the
# f points has a large peak at the load). Download a model first: the boots run with HF_HUB_OFFLINE=1.
# The windows of plans/PLAN_QWEN_PRUNE.md, all with Qwen3.6-35B-A3B at a context of 128,000 tokens:
#   c1          §6 the capture: the full model with --enable-return-routed-experts. The capture of qwen-data with 256
#               generated tokens per segment (capture.py), the recorded set (qwen_gate.py --record), the tool-call
#               suite of model-benchmarks, the bot decode test
#   c2          the full model with the MTP draft and the capture flag: a short capture (does it cover the draft
#               layer?), the bot decode test
#   c3          the full model with the ngram draft, which needs no weights: the bot decode test
#   g256        §7 the reference of the gate window: the full model with the MTP draft. The score of the recorded set
#               (qwen_gate.py --score), the tool-call suite, the bot decode test
#   g192 g176 g160   the pruned folders qwen-k192, qwen-k176, qwen-k160 (qwen_prune.py) with the MTP draft: the same
#               tests, and the comparison with g256 (qwen_gate.py --compare)
# All points: a KV pin of 1.04x of the need, FP8 KV, --max-num-seqs 1, no images. The f, q and n points use
# --max-model-len 256000 and these tests: smoke (a short request and a tool call), one prompt of about 250,000
# tokens, and the bot decode test (5 sessions, 1 run). The available memory is sampled every 5 s.
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
QNAME=second-model-fit QURL=http://127.0.0.1:8889
MTP='{"method":"mtp","num_speculative_tokens":3}' NEMO=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
Q27="--attention-backend flashinfer --reasoning-parser qwen3 --tool-call-parser qwen3_coder --load-format fastsafetensors --limit-mm-per-prompt {\"image\":0,\"video\":0}"
Q35="--attention-backend flashinfer --moe-backend flashinfer_cutlass --reasoning-parser qwen3 --tool-call-parser qwen3_xml --language-model-only"
NEM="--moe-backend marlin --mamba-backend flashinfer --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder"
# The first KV pin of a point is 1.04x of the estimated need for 256,000 tokens: 33.6 KB per token for the 27B
# (measured), 10.5 KB for the 35B-A3B (10 attention layers), 3.1 KB for Nemotron (6 attention layers). qboot corrects it.
declare -A BUDGET=([f1]=2048 [f2]=8192 [f3]=2048 [q1]=2048 [q2]=2048 [n1]=2048 [n2]=2048) \
  DRAFT=([f3]="$MTP" [q2]="$MTP" [n2]="{\"model\":\"$NEMO-DSpark\",\"num_speculative_tokens\":3}") \
  MODEL=([f1]=nvidia/Qwen3.8-27B-NVFP4 [f2]=nvidia/Qwen3.8-27B-NVFP4 [f3]=nvidia/Qwen3.8-27B-NVFP4 \
         [q1]=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast [q2]=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast [n1]=$NEMO [n2]=$NEMO) \
  FLAGS=([f1]="$Q27" [f2]="$Q27" [f3]="$Q27" [q1]="$Q35" [q2]="$Q35" [n1]="$NEM" [n2]="$NEM") \
  PIN0=([f1]=8933531976 [f2]=8933531976 [f3]=8933531976 [q1]=2800000000 [q2]=2800000000 [n1]=820000000 [n2]=820000000)
QM=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast MB=${MB:-$HOME/Projects/model-benchmarks}
NGRAM='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
declare -A CTX TESTS NEED   # the context (default 256000), the tests (default "smoke long speed"), the minutes (default 20)
for p in c1 c2 c3 g256 g192 g176 g160; do
  BUDGET[$p]=2048 CTX[$p]=128000 MODEL[$p]=$QM FLAGS[$p]=$Q35 DRAFT[$p]=$MTP PIN0[$p]=1730000000 TESTS[$p]="smoke score tools speed" NEED[$p]=25
done
for p in g192 g176 g160; do MODEL[$p]=$S/qwen-k${p#g}; done
FLAGS[c1]="$Q35 --enable-return-routed-experts" FLAGS[c2]=${FLAGS[c1]} DRAFT[c3]=$NGRAM
unset "DRAFT[c1]"; PIN0[c1]=1480000000 PIN0[c3]=1480000000   # no draft weights: the KV need is smaller
TESTS[c1]="smoke capture record tools speed" NEED[c1]=60 TESTS[c2]="smoke draftcapture speed" TESTS[c3]="smoke speed"
QPID= SAMPLER= PIN=
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
  [ -n "${DRAFT[$p]:-}" ] && extra=(--speculative-config "${DRAFT[$p]}")
  HF_HUB_OFFLINE=1 VLLM_USE_FASTOKENS=0 CUTE_DSL_ARCH=sm_121a PATH=$QV:$PATH setsid "$QV/vllm" serve ${MODEL[$p]} \
    --host 127.0.0.1 --port 8889 --served-model-name $QNAME --tensor-parallel-size 1 --gpu-memory-utilization 0.30 \
    --kv-cache-memory-bytes $PIN --max-model-len ${CTX[$p]:-256000} --max-num-seqs 1 --max-num-batched-tokens ${BUDGET[$p]} \
    --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --async-scheduling --enable-auto-tool-choice \
    --generation-config vllm ${FLAGS[$p]} "${extra[@]}" > "$2" 2>&1 &
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
  local p=$1 f=$R/qwen-fit-$1-server.log need o
  PIN=${PIN0[$p]}
  launch $p $f && return 0
  qstop
  need=$(grep -oE '[0-9.]+ GiB KV cache is needed' $f | head -1 | cut -d' ' -f1)
  [ -n "$need" ] || { log "ERROR: qwen-$p did not boot: $(grep -iE 'error|exception' $f | tail -2 | cut -c1-200)"; return 1; }
  PIN=$(awk -v n=$need 'BEGIN {printf "%d", n * 1.04 * 2^30}')
  for o in "${!PIN0[@]}"; do   # the points with the same flags, context and kind of draft start with the corrected pin
    [ "${FLAGS[$o]/ --enable-return-routed-experts/}" = "${FLAGS[$p]/ --enable-return-routed-experts/}" ] &&
      [ "${CTX[$o]:-256000}" = "${CTX[$p]:-256000}" ] && [ "${DRAFT[$o]:-}" = "${DRAFT[$p]:-}" ] && PIN0[$o]=$PIN
  done
  log "qwen-$p: the KV pin was too small, the need is $need GiB: one more boot with $PIN bytes"
  wait_mem 120 100
  launch $p $f && return 0
  qstop; log "ERROR: qwen-$p did not boot with the larger pin"; return 1
}
tests() {  # $1 point: the tests of TESTS, in order
  local p=$1 t left o=$R/qwen-fit-$1 gate="$PY $S/qwen_gate.py --base-url $QURL --model $QNAME --data $S/qwen-data --recorded $R/qwen-recorded.jsonl"
  local cap="$PY $S/capture.py --base-url $QURL --model $QNAME --data $S/qwen-data --sources claude,discord,repo --experts 256"
  for t in ${TESTS[$p]:-smoke long speed}; do
    left=$(( RESTORE_S - $(now) - 300 ))
    case $t in
      smoke) $PY $S/qwen_fit.py --base-url $QURL --model $QNAME --smoke > $o-smoke.json 2>&1
        log "qwen-$p smoke: exit $?; $(tail -1 $o-smoke.json | cut -c1-200)";;
      long) timeout $left $PY $S/qwen_fit.py --base-url $QURL --model $QNAME --long 250000 > $o-long.json 2>&1
        log "qwen-$p long prompt: exit $?; $(tail -1 $o-long.json | cut -c1-200)";;
      speed) (cd $W && timeout $left $PY -m model_eval.speed --base-url $QURL --model $QNAME --sessions 5 --runs 1 \
          --out $o-speed.jsonl > $o-speed.log 2>&1)
        log "qwen-$p bot decode test: exit $?; $(tail -1 $o-speed.log | cut -c1-200)"
        log "qwen-$p decode rate from the rows: $($PY $S/qwen_fit.py --rate $o-speed.jsonl 2>&1 | tail -1)";;
      capture) timeout $left $cap --generate 256 --out $R/qwen-capture.jsonl > $o-capture.log 2>&1
        log "qwen-$p capture: exit $?; $(wc -l < $R/qwen-capture.jsonl 2>/dev/null) rows, $(grep -c '"error"' $R/qwen-capture.jsonl 2>/dev/null) errors";;
      draftcapture) timeout $left $cap --limit 24 --out $R/qwen-capture-draft.jsonl > $o-capture.log 2>&1
        log "qwen-$p capture with the draft: exit $?; layers $(grep -oE '"layers": [0-9]+' $R/qwen-capture-draft.jsonl 2>/dev/null | sort -u | tr '\n' ' ')";;
      record) timeout $left $gate --record > $o-record.log 2>&1
        log "qwen-$p record: exit $?; $(tail -1 $o-record.log | cut -c1-200)";;
      score) timeout $left $gate --score --out $o-score.jsonl > $o-score.log 2>&1
        log "qwen-$p score: exit $?; $(tail -1 $o-score.log | cut -c1-200)"
        [ $p != g256 ] && [ -s $R/qwen-fit-g256-score.jsonl ] && [ -s $o-score.jsonl ] &&
          log "qwen-$p against g256: $($gate --compare $R/qwen-fit-g256-score.jsonl $o-score.jsonl 2>&1 | grep -E 'nll change|PASS|FAIL' | tr '\n' ';')";;
      tools) (cd $MB && timeout $left $QV/python -m benchmarks.tool_calling --base-url $QURL/v1 --model $QNAME \
          --output $o-tools.json > $o-tools.log 2>&1)
        log "qwen-$p tool-call suite: exit $?; $(grep -oE '"total_(tasks|correct)": [0-9]+|"composite_score": [0-9.]+' $o-tools.json 2>/dev/null | tr '\n' ' ')";;
    esac
  done
}
gpu_free() { PATH=$QV:$PATH python -c "import torch; f, t = torch.cuda.mem_get_info(); print(f'{f / 2**30:.1f} of {t / 2**30:.1f} GiB')" 2>&1 | tail -1; }
finish() { sample_stop; qstop; restore; }
[ -n "${QWEN_STUB:-}" ] && source "$QWEN_STUB"   # the dry run of test_qwen_fit.py: stub boots and clients

fpoint() {  # $1 point; 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 before healthy healthy_at m=$R/qwen-fit-$1-mem.log
  [ -n "${MODEL[$p]:-}" ] || { log "ERROR: no point $p"; return 2; }
  [ $(( $(now) + ${NEED[$p]:-20} * 60 )) -gt $RESTORE_S ] && { log "qwen-$p: needs ${NEED[$p]:-20} min, no time before $RESTORE_AT"; return 2; }
  case ${MODEL[$p]} in /*) [ -f ${MODEL[$p]}/config.json ] || { log "ERROR: no folder ${MODEL[$p]}"; return 2; };; esac
  [ "$(now)" -ge $BOOT_BY_S ] && { log "qwen-$p: after $BOOT_BY, not booted"; return 2; }
  stop_main || return 9
  qrunning && { log "ERROR: a vllm serve process runs"; return 9; }
  wait_mem 300 100 || { log "ERROR: less than 100 GiB available ($(avail_gib))"; return 9; }
  before=$(avail_mb)
  log "qwen-$p: ${MODEL[$p]}, context ${CTX[$p]:-256000}, prefill budget ${BUDGET[$p]}, draft ${DRAFT[$p]:-none}; $before MB available; torch sees $(gpu_free) free"
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

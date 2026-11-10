#!/usr/bin/env bash
# The test night of plans/PLAN_FEWER_EXPERTS.md: Phase A (§5, the routing profile) and Phase C (§7, the replay
# points), then the restore of the live server.
# All boots use tp1-botmask-k160 (K 160 at top-k 6). The safety rules and the shared functions are in run_lib.sh.
# CAUTION: this stops the bot's model server. Start it only with the user's go:
#   tmux new -d -s fewer "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/fewer_night.sh"
# - PHASES (default "A C") and POINTS (default "c1 c2 c3 c4 c5 c0") select the work.
# - C1 boots first among the replay points: if its smoke request fails (the -1 ids in the FusedMoE glue), restore
#   and stop.
# - C5 takes its threshold from Phase A (results/profile-k160.json) and the better anchor of C3 and C4; no
#   threshold: C5 is skipped (the gate of §5).
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/fewer-night.log}
RESTORE_TAG=fewer
TEST_COMPOSE="-f compose-mask.yml -f compose-drop.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

PHASES=${PHASES:-A C}
POINTS=${POINTS:-c1 c2 c3 c4 c5 c0}
declare -A DROP=([c0]=none [c1]=lowest:1 [c2]=lowest:1 [c3]=lowest:2 [c4]=lowest:2)
declare -A NORM=([c0]=kept [c1]=kept [c2]=anchor:6 [c3]=anchor:6 [c4]=anchor:8)
export MASK_NAME=botmask MASK_K=160 MOE_LABEL=fewer-none   # compose-drop.yml needs MOE_LABEL, also for a down
BOOT_MIN_GIB=110   # vLLM's boot check at util 0.90: 109.5 GiB free (a restart failed at 107.6 on 2026-11-07)
on() { [[ " $PHASES " == *" $1 "* ]]; }

[ -f $S/image-patch/fused_topk_bias_router.py ] || $S/image-patch/build_router.sh >> "$LOG" 2>&1 \
  || { log "ERROR: image-patch/build_router.sh failed"; exit 1; }
[ -f $S/tp1-botmask-k160/rank-sliced-tp1-manifest.json ] || { log "ERROR: no tp1-botmask-k160"; exit 1; }
log "fewer night start (phases $PHASES; points $POINTS; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

phase_a() {  # the capture boot with the routing profile, on the capture compose; a safety failure restores here
  local keep="$TEST_COMPOSE" keep_p="$TEST_PROJECT" keep_m="$MASK" rc
  TEST_COMPOSE="-f compose-capture.yml"; TEST_PROJECT=dsv4-capture; MASK=dsv4-capture-deepseek-v4-flash-capture-1
  export CAPTURE_DIR=$S/tp1-botmask-k160 ROUTING_PROFILE=1 CAPTURE_UTIL=0.90
  export CAPTURE_EXTRA_ARGS="--kv-cache-memory-bytes 8589934592"
  capture_profile; rc=$?
  [ $rc = 9 ] && { log "safety failure in Phase A: restore"; restore; exit 1; }
  TEST_COMPOSE="$keep"; TEST_PROJECT="$keep_p"; MASK="$keep_m"
  return $rc
}

capture_profile() {  # 0 done, 1 boot failed, 9 safety failure
  local label=capture-profile-k160 sums=$S/capture-out/routing_sums.npz rc left
  rm -f "$sums" "$sums.tmp.npz"
  stop_main || return 9
  boot $label; rc=$?
  if [ $rc = 0 ]; then
    curl -s --max-time 300 $URL/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash-0731-capture","messages":[{"role":"user","content":"Say OK."}],"max_tokens":1}' > /dev/null
    sleep 3
    if $PY -c "import numpy as np,sys; z=np.load('$sums'); sys.exit(0 if z['prof_kept'].sum()>0 else 1)" 2>/dev/null; then
      log "smoke: the profile moves"
      memwatch_start $label
      left=$(( RESTORE_S - $(now) )); [ $left -gt 3000 ] && left=3000   # at most 50 min: C needs the night
      log "capture of the rank data ($left s)"
      (cd $W && timeout $left $PY $S/capture.py --base-url $URL --out $R/$label.jsonl --data $W/model_eval/data/rank \
        --sources discord,subagent,e2e,cron --sums "$sums" > $R/$label.log 2>&1)
      log "capture exit $?: $(wc -l < $R/$label.jsonl) rows, $(grep -c '"error"' $R/$label.jsonl) errors"
      memwatch_stop
      $PY $S/profile_report.py $R/$label.jsonl $R/profile-k160 2>&1 | tail -1 | tee -a "$LOG"
    else
      log "ERROR: the profile did not move after the smoke request; no capture"
      docker logs $MASK 2>&1 | grep -iE "moe_drop|profile|Traceback" | tail -5 | tee -a "$LOG"
    fi
    down || return 9
  fi
  return $rc
}

patch_ok() {  # one [moe_drop] line per target layer 0-42 in the boot log, hash=True for layers 0-2 only
  local f=$R/boot-$1.log n h
  [ "$MOE_DROP" = none ] && [ "$MOE_NORM" = kept ] && return 0
  n=$(grep -oE '\[moe_drop\] layer [0-9]+ ' "$f" | sort -u | wc -l)
  h=$(grep -E '\[moe_drop\] layer [0-9]+ .*hash=True' "$f" | grep -oE 'layer [0-9]+' | sort -u | tr '\n' ' ')
  log "$1 patch: $n target layers, hash: ${h:-none}; $(grep -m1 -oE 'drop=[^ ]+ norm=[^ ]+ layers=[^ ]+' "$f")"
  [ "$n" = 43 ] && [ "$h" = "layer 0 layer 1 layer 2 " ]
}

smoke() {  # one short answer from the test server (the first request through the -1 ids); false: no completion
  local reply rc
  reply=$(curl -s --max-time 300 $URL/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash-0731-$1\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 3? Answer with the number only.\"}],\"max_tokens\":16,\"chat_template_kwargs\":{\"thinking\":false}}" \
    | $PY -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>&1); rc=$?
  log "$1 smoke reply ($([[ "$reply" == *51* ]] && echo right || echo WRONG)): ${reply:0:60}"
  [ $rc = 0 ] && [ -n "$reply" ]
}

cpoint() {  # $1: c0 … c5; 0 done, 1 boot failed, 2 skipped, 8 smoke failed, 9 safety failure
  local label=fewer-$1 rc
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  export MOE_LABEL=$label MOE_DROP=${DROP[$1]} MOE_NORM=${NORM[$1]}
  log "$label: MOE_DROP=$MOE_DROP MOE_NORM=$MOE_NORM"
  stop_main || return 9
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  patch_ok $label || { log "ERROR: $label: the patch did not reach the 43 target layers as planned"; down || return 9; return 1; }
  smoke $label || { log "ERROR: $label: the smoke request failed"; docker logs $MASK 2>&1 | grep -iE "error|Traceback" | tail -5 | tee -a "$LOG"; down; return 8; }
  replay $label
  down || return 9
}

if on A; then
  phase_a; [ $? = 1 ] && FAILS=1
fi

if on C; then
  for p in $POINTS; do
    if [ $p = c5 ]; then
      thr=$($PY -c "import json; print(json.load(open('$R/profile-k160.json'))['c5'] or '')" 2>/dev/null)
      [ -z "$thr" ] && { log "fewer-c5: no threshold from Phase A (the gate of §5): skipped"; continue; }
      DROP[c5]=$thr; NORM[c5]=$($PY $S/fewer_report.py --c5-norm)
      echo "${DROP[c5]} ${NORM[c5]}" > $R/fewer-c5.rule
    fi
    cpoint $p; rc=$?
    case $rc in
      0) FAILS=0;;
      1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
      8) log "the -1 ids or the patch fail at run time: restore, the night stops"; restore; exit 1;;
      9) log "safety failure: the safe action, restore"; restore; exit 1;;
    esac
  done
  $PY $S/fewer_report.py > $R/fewer-phase-c.md 2>&1
  log "Phase C table: results/fewer-phase-c.md; decision: $(tail -1 $R/fewer-phase-c.md)"
fi

restore
log "fewer night done"

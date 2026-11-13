#!/usr/bin/env bash
# The replay window of plans/PLAN_COLD_EXPERTS.md §7 and plans/PLAN_COLD_CUT2.md §6 on port 8890, then the restore
# of the live server. Masks of data/tp1 (cold_sets.py, mask_checkpoint.py --drop-file, topk_checkpoint.py --top-k 5).
# Each drops the live drops plus a set of cold experts:
#   R0   nothing more: the live set as a mask (tp1-coldr05-k160), the reference
#   M1   the mandatory experts below 0.1% of the weight (tp1-coldm15-k154)
#   D1   the coldest ordinary experts, up to 1% of the weight per layer (tp1-coldd15-k155)
#   MD   M1 and D1 (tp1-coldmd5-k149)
#   S15  MD plus every expert below 0.15% of the weight (tp1-colds15t5-k145)
#   S20  MD plus every expert below 0.2% of the weight (tp1-colds20t5-k139)
# A point with the suffix -2 (r0-2, s20-2) is a repeat: a new boot of the same folder, read against the R0 of its own
# half (cold-r0-2). The end of the window prints the mean of the two reports of each repeated point: the gate of a
# small cut (PLAN_COLD_CUT2.md §3). The two R0 points each get the R0 check.
# Each point: boot, the smoke requests (prefill_probe.py --smoke), the replay, the comparisons against baseline-k216
# (run_lib.sh) and against R0 (cold_sets.py --report), stop. The R0 check: R0 must match botmask5-k160 within 0.5% of
# the NLL, or the window restores and stops. A point that cannot end before RESTORE_AT is skipped.
# The test boots pin the KV cache at 7.25 GiB, as the live server since 2026-11-10: a mask of 216 experts leaves little
# free memory, and the smaller pin adds 0.75 GiB.
# CAUTION: this stops the bot's model server for about 44 min per point, plus 10 min for the restore. Start it only
# with the user's go, not 05:00-10:00 on weekdays unless the cron jobs are paused, and not during another test:
#   tmux new -d -s cold "POINTS='r0 s20 s15 r0-2 s20-2 s15-2' START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/cold_replay.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/cold-replay.log}
RESTORE_TAG=cold
TEST_COMPOSE="-f compose-mask.yml -f serve-tune/compose-tune.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-r0 m1 d1 md}
declare -A FOLDER=([r0]=coldr05:160 [m1]=coldm15:154 [d1]=coldd15:155 [md]=coldmd5:149 [s15]=colds15t5:145 [s20]=colds20t5:139)
export TUNE_KV_BYTES=7784628224 TUNE_LABEL=cold-r0 MASK_NAME=coldr05 MASK_K=160   # compose-tune.yml needs TUNE_LABEL
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
FAILS=0
pct() { awk '$1 == "nll" {printf "%+.2f", 100 * ($3 / $2 - 1)}' "$1" 2>/dev/null; }
mean2() {  # $1 point: the mean of the first lines of cold-$1-report.txt and cold-$1-2-report.txt, and the gate on it
  cat <(head -1 $R/cold-$1-report.txt) <(head -1 $R/cold-$1-2-report.txt) | awk -F'[ %;]+' '
    {for (i = 1; i <= NF; i++) {if ($i == "NLL") n += $(i+1); if ($i == "decisions") d += $(i+1); if ($i == "names") t += $(i+1)}; r++}
    END {printf "NLL %+.2f%%; decisions %+.1f; tool names %+.1f; no measurable cost (without the smoke test): %s", n/r, d/r, t/r,
         (n/r <= 0.5 && d/r >= -1 && t/r >= -1) ? "yes" : "no"}'
}

cpoint() {  # $1 point; returns 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 label=cold-$1 base=${1%-*} rep= rc
  [ "$base" = "$1" ] || rep=-${1##*-}        # r0-2 is the repeat of r0: base r0, rep -2, reference cold-r0-2
  local name=${FOLDER[$base]%:*} k=${FOLDER[$base]#*:} r0=cold-r0$rep
  export TUNE_LABEL=$label MASK_NAME=$name MASK_K=$k
  [ $(( $(now) + 44 * 60 )) -gt $RESTORE_S ] && { log "$label: needs 44 min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  [ -f $S/tp1-$name-k$k/rank-sliced-tp1-manifest.json ] || { log "ERROR: no folder $S/tp1-$name-k$k"; return 2; }
  stop_main || return 9
  log "$label: folder tp1-$name-k$k"
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  log "$label: $(avail_gib) GiB available after the boot"
  timeout 300 $PY $S/serve-tune/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$label --smoke \
    --out $R/smoke-$label.json > /dev/null 2> $R/smoke-$label.err; rc=$?
  log "$label smoke: exit $rc, $(docker logs $MASK 2>&1 | grep -c AssertionError) assertion lines; $(tail -1 $R/smoke-$label.err | cut -c1-200)"
  replay $label
  if [ $base != r0 ] && [ -s $R/$r0.jsonl ]; then
    (cd $W && $PY -m model_eval.replay --compare $R/$r0.jsonl $R/$label.jsonl > $R/$label-vs-$r0.txt 2>&1)
    $PY $S/cold_sets.py --report $R/$r0.jsonl $R/$label.jsonl > $R/$label-report.txt 2>&1
    log "$label against $r0: $(head -1 $R/$label-report.txt)"
  fi
  down || return 9
}

log "cold window start (points $POINTS; boot by $BOOT_BY; restore at $RESTORE_AT)"
for need in $S/serve-tune/compose-tune.yml $S/serve-tune/prefill_probe.py $S/cold_sets.py $R/baseline-k216.jsonl \
  $R/botmask5-k160.jsonl; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

for p in $POINTS; do
  cpoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
  if [ ${p%-*} = r0 ] && [ $rc = 0 ]; then   # the R0 check of §5.2, for r0 and for its repeat r0-2
    (cd $W && $PY -m model_eval.replay --compare $R/botmask5-k160.jsonl $R/cold-$p.jsonl > $R/cold-$p-vs-botmask5-k160.txt 2>&1)
    d=$(pct $R/cold-$p-vs-botmask5-k160.txt)
    log "$p check: NLL ${d:-none}% against botmask5-k160 (the limit is 0.5%)"
    awk -v d="${d:-9}" 'BEGIN {exit !(d >= -0.5 && d <= 0.5)}' || { log "$p check fails: restore and stop"; restore; exit 1; }
  fi
done
restore
for p in $POINTS; do
  log "cold-$p: NLL $(pct $R/cold-$p-compare.txt)% against baseline-k216; $(grep -E '^(PASS|FAIL)' $R/cold-$p-compare.txt 2>/dev/null | cut -c1-6 | tr '\n' ' ')"
done
for p in $POINTS; do   # the mean of the two replays of each repeated point (PLAN_COLD_CUT2.md §3)
  case $p in *-2|r0) continue;; esac
  [ -s $R/cold-$p-report.txt ] && [ -s $R/cold-$p-2-report.txt ] && log "cold-$p, mean of two against R0: $(mean2 $p)"
done
log "cold window done"

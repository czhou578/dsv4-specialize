#!/usr/bin/env bash
# The replay window of plans/PLAN_FEWER_LAYERS.md §8 on port 8890, then the restore of the live server.
# Every point boots the live folder (the MD prune, compose-perlayer.yml) with the bot draft and the live settings, and
# compose-skip.yml with a fixed skip set (LAYER_SKIP) and, for a heal point, a heal file (LAYER_HEAL). The points
# (POINTS, default "s0 s1 s1h"; the larger sets first, §8.1):
#   s0   no skip (the patch on). The S0 check: the mean NLL of its two replays within 1% of the live replays (0.4562);
#        else the window restores and stops (§8.3 and the refresh §0).
#   s3, s2, s1   the sets of 3, 2 and 1 block (S3_SET, S2_SET, S1_SET from the probe report).
#   s1h  S1_SET with the heal maps of HEAL_FILE (layer_screen.py --fit), copied to skip-out/heal.npz.
#   sa   the 6 A blocks (SA_SET).
# Each point: boot, the boot check (the skip line of the set, the heal line, no graph from the compile cache), the
# smoke requests (prefill_probe.py --smoke), two replays, the gate on their mean against baseline-k216 (NLL at most +3%,
# tool names at least 98% in both, decisions down by at most 3), and the speed rule of §8.2: the bot decode x 5 for s0
# and sa, and for the first point that passes. A point that cannot end before RESTORE_AT is skipped.
# CAUTION: this stops the bot's model server for about 90 min per point, plus 10 min for the restore. Start it only
# with the user's go, pause the Hermes cron jobs first, and do not start it during another test:
#   tmux new -d -s replay "S1_SET=F24 HEAL_FILE=... START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... \
#     ~/Projects/dsv4-specialize/skip_replay.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/skip-replay.log}
RESTORE_TAG=skip-replay
TEST_COMPOSE="-f compose-mask.yml -f serve-tune/compose-tune.yml -f compose-perlayer.yml -f compose-skip.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-s0 s1 s1h}
LIVE_NLL=0.4562 PASSED=
declare -A NEED=([s0]=95 [s3]=90 [s2]=90 [s1]=90 [s1h]=90 [sa]=95)
export MASK_NAME=prunemd5 MASK_K=149 TUNE_LABEL=skip-s0 TUNE_KV_BYTES=7784628224
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
canon() { tr ',' '\n' <<< "$1" | sed -nE 's/^([ALF])([0-9]+)$/\2 \1\2/p' | sort -n | cut -d' ' -f2 | paste -sd, -; }
set_of() { case $1 in s0) echo;; s1|s1h) canon "${S1_SET:?}";; s2) canon "${S2_SET:?}";; s3) canon "${S3_SET:?}";;
  sa) canon "${SA_SET:?}";; esac; }
smoke() {
  timeout 300 $PY $S/serve-tune/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$1 --smoke \
    --out $R/smoke-$1.json > /dev/null 2> $R/smoke-$1.err
  log "$1 smoke: exit $?; $(tail -1 $R/smoke-$1.err | cut -c1-200)"
}
boot_check() {  # $1 label, $2 set, $3 heal (0|1): the skip line of §6.1 and no graph from the compile cache
  local f=$R/boot-$1.log want="layer skip: set ${2:-none} (fixed);"
  grep -qF "$want" $f || { log "ERROR: $1: no line '$want'; $(grep -o 'layer skip: set [^;]*' $f | head -1)"; return 1; }
  if [ $3 = 1 ]; then grep -qE "layer skip: .*; heal [ALF][0-9].* sha256 " $f || { log "ERROR: $1: no heal line"; return 1; }
  else grep -qF "; heal none;" $f || { log "ERROR: $1: a heal map is on"; return 1; }; fi
  if grep -qE 'Directly load (AOT compilation|the compiled graph)' $f; then log "ERROR: $1: a graph from the compile cache"; return 1; fi
}
gate() {  # $1 label: the mean of the two replays against baseline-k216, one line, PASS or FAIL at the end
  $PY - "$R" "$1" <<'EOF'
import json, sys
from pathlib import Path
r, label = Path(sys.argv[1]), sys.argv[2]
load = lambda p: {(x["session"], x["segment"], x["index"]): x for x in map(json.loads, p.read_text().splitlines())
                  if "error" not in x}
files = [r / "baseline-k216.jsonl", r / f"{label}-a.jsonl", r / f"{label}-b.jsonl"]
if not all(p.exists() for p in files):
    sys.exit(print(f"{label}: a replay file is missing: FAIL"))
base, *runs = map(load, files)
keys = set(base).intersection(*runs)
nll = lambda d: sum(d[k]["reasoning_nll"] + d[k]["answer_nll"] for k in keys) / sum(
    d[k]["reasoning_tokens"] + d[k]["answer_tokens"] for k in keys)
names = lambda d: sum(d[k]["tool_names_ok"] for k in keys) / max(1, sum(d[k]["tool_calls"] for k in keys))
dec = lambda d: sum(int(d[k]["decision_ok"]) for k in keys)
mean = sum(map(nll, runs)) / 2
ok = (mean <= 1.03 * nll(base) and all(names(d) >= 0.98 * names(base) for d in runs)
      and sum(map(dec, runs)) / 2 >= dec(base) - 3 and len(keys) >= 60)
print(f"{label}: mean NLL {mean:.4f} ({100 * (mean / nll(base) - 1):+.2f}% against K 216; the gate +3%), tool names "
      f"{names(runs[0]):.4f} and {names(runs[1]):.4f} (98% of {names(base):.4f}), decisions "
      f"{sum(map(dec, runs)) / 2 - dec(base):+.1f}, {len(keys)} targets: {'PASS' if ok else 'FAIL'}")
EOF
}
mean_nll() { grep -oE 'mean NLL [0-9.]+' $R/$1-gate.txt 2>/dev/null | awk '{print $3}'; }
[ -n "${SKIP_STUB:-}" ] && source "$SKIP_STUB"   # the dry run of test_skip_scripts.py: stub boots and replays

spoint() {  # $1 point; 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 label=skip-$1 blocks heal=0 rc
  blocks=$(set_of $p) || return 2
  [ $p = s1h ] && heal=1
  export TUNE_LABEL=$label LAYER_SKIP=$blocks LAYER_HEAL=
  if [ $heal = 1 ]; then
    [ -s "${HEAL_FILE:-}" ] || { log "ERROR: $label needs HEAL_FILE"; return 2; }
    mkdir -p $S/skip-out && cp "$HEAL_FILE" $S/skip-out/heal.npz && export LAYER_HEAL=/skip/heal.npz
  fi
  [ $(( $(now) + ${NEED[$p]} * 60 )) -gt $RESTORE_S ] && { log "$label: needs ${NEED[$p]} min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  stop_main || return 9
  log "$label: set ${blocks:-none}, heal $heal"
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  boot_check $label "$blocks" $heal || { down || return 9; return 1; }
  smoke $label
  replay $label-a
  replay $label-b      # the replay skips the targets of its output file, so the repeat needs its own label
  gate $label > $R/$label-gate.txt 2>&1
  log "$(cat $R/$label-gate.txt)"
  if [ $p = s0 ] || [ $p = sa ] || { grep -q PASS$ $R/$label-gate.txt && [ -z "$PASSED" ]; }; then
    BOT_RUNS=5 speed $label
  fi
  grep -q PASS$ $R/$label-gate.txt && [ $p != s0 ] && [ $p != sa ] && PASSED=${PASSED:-$p}
  down || return 9
}

log "replay window start (points $POINTS; boot by $BOOT_BY; restore at $RESTORE_AT)"
for need in $S/image-patch/skip/deepseek_v4_model.py $S/compose-skip.yml $S/serve-tune/prefill_probe.py \
  $R/baseline-k216.jsonl $S/tp1-$MASK_NAME-k$MASK_K/rank-sliced-tp1-manifest.json; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
for p in $POINTS; do   # the replay skips the targets of its out file: an old file would join this window's result
  ls $R/skip-$p-[ab].jsonl > /dev/null 2>&1 && { log "ERROR: old replay files of $p in $R; move them away"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

FAILS=0
for p in $POINTS; do
  spoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
  if [ $p = s0 ] && [ $rc = 0 ]; then   # the S0 check of §8.3: the patch must not change the model with no skip
    d=$(awk -v m="$(mean_nll skip-s0)" -v l=$LIVE_NLL 'BEGIN {if (m == "") print "none"; else printf "%+.2f", 100 * (m / l - 1)}')
    log "S0 check: mean NLL $(mean_nll skip-s0) is ${d}% against the live replays ($LIVE_NLL; the limit is 1%)"
    awk -v d="$d" 'BEGIN {exit !(d != "none" && d >= -1 && d <= 1)}' || { log "S0 check fails: restore and stop"; restore; exit 1; }
  fi
done
restore
S0=$(mean_nll skip-s0)
for p in $POINTS; do
  m=$(mean_nll skip-$p)
  [ -n "$m" ] && log "skip-$p: $(grep -oE '(PASS|FAIL)$' $R/skip-$p-gate.txt); mean NLL $m$( [ -n "$S0" ] && [ $p != s0 ] &&
    awk -v m=$m -v s=$S0 'BEGIN {printf ", %+.2f%% against S0", 100 * (m / s - 1)}'); speed: $($PY $S/speed_result.py --results $R skip-$p 2>/dev/null | tail -1 | cut -c1-200)"
done
log "replay window done"

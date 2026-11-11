#!/usr/bin/env bash
# The test window of plans/PLAN_LOADER_PATCH.md §6 on port 8890, then the restore of the live server.
# Both points boot the patched image (compose-perlayer.yml) at the live settings, with the KV pin of 7.25 GiB:
#   P0  tp1-prune5-k160: equal to the live folder, file by file. The patch must not change a uniform folder.
#       Boot, smoke, the speed test (5 runs), the prefill probe.
#   P1  tp1-prunemd5-k149: the prune of MD (PLAN_COLD_EXPERTS.md §5.5), K 140 to 160 per layer.
#       Boot, smoke, two replays, the speed test (5 runs), the prefill probe.
# After the restore, the script logs the gates of §6.1: the loaded GiB and the pool line of each boot, the free
# memory after each boot, the speed of P1 against P0, and the mean NLL of the two P1 replays against the mean of the
# two MD mask replays (results/cold-w1/cold-md.jsonl, results/cold-md.jsonl).
# Build image-patch/perlayer/ first (image-patch/build_perlayer.sh), or the boots fail.
# CAUTION: this stops the bot's model server for about 2 h 30 min. Start it only with the user's go, pause the Hermes
# cron jobs first, and do not start it during another test:
#   tmux new -d -s loader "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... \
#     ~/Projects/dsv4-specialize/loader_window.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/loader-window.log}
RESTORE_TAG=loader
TEST_COMPOSE="-f compose-mask.yml -f serve-tune/compose-tune.yml -f compose-perlayer.yml"
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-p0 p1}
declare -A FOLDER=([p0]=prune5:160 [p1]=prunemd5:149) NEED=([p0]=35 [p1]=110) FREE_MB
export TUNE_KV_BYTES=7784628224 TUNE_LABEL=lp-p0 MASK_NAME=prune5 MASK_K=160   # compose-tune.yml needs TUNE_LABEL
export DRAFT_DIR=$SERVER/data/dspark-draft-k64-bot
BOOT_MIN_GIB=110                   # vLLM's boot check at util 0.90 needs 109.5 GiB free
FAILS=0
pct() { awk '$1 == "nll" {printf "%+.2f", 100 * ($3 / $2 - 1)}' "$1" 2>/dev/null; }

probe() {  # $1 label, $2 kind (probe|smoke), the rest: prefill_probe.py arguments
  local label=$1 kind=$2 left=$(( RESTORE_S - $(now) )) rc; shift 2
  [ $left -le $([ $kind = probe ] && echo 900 || echo 120) ] && { log "$label: no time for the $kind"; return; }
  memwatch_start "$label-$kind"
  timeout $left $PY $S/serve-tune/prefill_probe.py --base-url $URL --model deepseek-v4-flash-0731-$label \
    --out $R/$kind-$label.json "$@" > /dev/null 2> $R/$kind-$label.err; rc=$?
  memwatch_stop
  log "$label $kind: exit $rc, $(docker logs $MASK 2>&1 | grep -c AssertionError) assertion lines; $(tail -1 $R/$kind-$label.err | cut -c1-300)"
}

lpoint() {  # $1 point; returns 0 done, 1 boot failed, 2 skipped, 9 safety failure
  local p=$1 label=lp-$1 name=${FOLDER[$1]%:*} k=${FOLDER[$1]#*:} rc
  export TUNE_LABEL=$label MASK_NAME=$name MASK_K=$k
  [ $(( $(now) + ${NEED[$p]} * 60 )) -gt $RESTORE_S ] && { log "$label: needs ${NEED[$p]} min, no time before $RESTORE_AT"; return 2; }
  [ "$(now)" -ge $BOOT_BY_S ] && { log "$label: after $BOOT_BY, not booted"; return 2; }
  [ -f $S/tp1-$name-k$k/rank-sliced-tp1-manifest.json ] || { log "ERROR: no folder $S/tp1-$name-k$k"; return 2; }
  stop_main || return 9
  log "$label: folder tp1-$name-k$k"
  boot $label; rc=$?
  [ $rc != 0 ] && return $rc
  FREE_MB[$p]=$(avail_mb)
  log "$label: ${FREE_MB[$p]} MB available after the boot; $(grep -oE 'per-layer experts: .*' $R/boot-$label.log | head -1)"
  probe $label smoke --smoke
  if [ $p = p1 ]; then
    replay $label-a
    replay $label-b    # the replay skips the targets of its output file, so the repeat needs its own label
  fi
  BOT_RUNS=5 speed $label
  probe $label probe --cold 8000 32000 128000 --warm 200000 8000
  down || return 9
}

log "loader window start (points $POINTS; boot by $BOOT_BY; speed by $SPEED_BY; restore at $RESTORE_AT)"
for need in $S/compose-perlayer.yml $S/image-patch/perlayer/exl3.py $S/image-patch/perlayer/deepseek_v4_model.py \
  $S/serve-tune/prefill_probe.py $R/baseline-k216.jsonl $R/cold-md.jsonl $R/cold-w1/cold-md.jsonl; do
  [ -e $need ] || { log "ERROR: no $need"; exit 1; }
done
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done

for p in $POINTS; do
  lpoint $p; rc=$?
  case $rc in
    0) FAILS=0;;
    1) FAILS=$((FAILS + 1)); [ $FAILS -ge 2 ] && { log "two failed boots in a row: restore"; restore; exit 1; };;
    9) log "safety failure: the safe action, restore"; restore; exit 1;;
  esac
done
restore

for p in $POINTS; do
  log "lp-$p boot: $(grep -oE 'Model loading took [0-9.]+ GiB' $R/boot-lp-$p.log | head -1); ${FREE_MB[$p]:-?} MB available"
  log "lp-$p speed: $($PY $S/speed_result.py --results $R lp-$p 2>&1 | tail -1 | cut -c1-300)"
done
[ -n "${FREE_MB[p0]:-}" ] && [ -n "${FREE_MB[p1]:-}" ] && log "P1 frees $(( (FREE_MB[p1] - FREE_MB[p0]) / 1024 )) GiB more than P0 (the gate: 4)"
for l in lp-p1-a lp-p1-b; do
  log "$l: NLL $(pct $R/$l-compare.txt)% against baseline-k216; $(grep -E '^(PASS|FAIL)' $R/$l-compare.txt 2>/dev/null | cut -c1-6 | tr '\n' ' ')"
done
$PY - "$R" <<'EOF' 2>&1 | while read -r line; do log "$line"; done
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
load = lambda p: {(x["session"], x["segment"], x["index"]): x for x in map(json.loads, p.read_text().splitlines())}
files = {"mask": [r / "cold-w1/cold-md.jsonl", r / "cold-md.jsonl"], "prune": [r / "lp-p1-a.jsonl", r / "lp-p1-b.jsonl"]}
if not all(p.exists() for ps in files.values() for p in ps):
    sys.exit("P1 against the MD mask: a replay file is missing")
rows = {k: [load(p) for p in ps] for k, ps in files.items()}
keys = set.intersection(*(set(d) for ds in rows.values() for d in ds))
nll = {k: sum(d[t]["reasoning_nll"] + d[t]["answer_nll"] for d in ds for t in keys) for k, ds in rows.items()}
dec = {k: sum(int(d[t]["decision_ok"]) for d in ds for t in keys) for k, ds in rows.items()}
change = 100 * (nll["prune"] / nll["mask"] - 1)
print(f"P1 against the MD mask, mean of two replays each: NLL {change:+.2f}% (the gate: within 0.6%), "
      f"decisions {dec['prune'] - dec['mask']:+d} over two replays, {len(keys)} targets: "
      f"{'PASS' if abs(change) <= 0.6 else 'FAIL'}")
EOF
log "loader window done"

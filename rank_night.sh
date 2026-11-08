#!/usr/bin/env bash
# The test nights of PLAN_BETTER_RANK.md §6: masks on the replay screen, then the restore of the live server.
# POINTS lists the masks as NAME:K (folder tp1-NAME-kK, top-k as built). The default is the three masks of §6.1:
#   wmask5:160   the floor rank, K 160, top-k 5   (does the new rank beat +2.27% at the same size?)
#   wmask5:144   the floor rank, K 144, top-k 5   (does K 144 pass, and at what cost?)
#   botmask5:144 the count rank, K 144, top-k 5   (the control)
# §6.2 ran POINTS=wmask5q:144 LOG=logs/rank-night2.log (the 40 k floor). The folders must exist (rank_masks.sh).
# The safety rules and the shared functions are in run_lib.sh.
# CAUTION: this stops the bot's model server. Start it only with the user's go:
#   tmux new -d -s rank-night "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/rank_night.sh"
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/logs/rank-night.log}
RESTORE_TAG=rank
source "$(cd "$(dirname "$0")" && pwd)/run_lib.sh"

POINTS=${POINTS:-wmask5:160 wmask5:144 botmask5:144}
log "rank night start (points $POINTS; start after $START_AFTER; boot by $BOOT_BY; restore at $RESTORE_AT)"
until [ "$(now)" -ge "$(date -d "$START_AFTER" +%s)" ]; do sleep 30; done
for p in $POINTS; do
  [ -f $S/tp1-${p%:*}-k${p#*:}/rank-sliced-tp1-manifest.json ] || { log "ERROR: no folder tp1-${p%:*}-k${p#*:}"; restore; exit 1; }
done
for p in $POINTS; do run_point ${p%:*} ${p#*:} replay; done
for p in $POINTS; do
  l=${p%:*}-k${p#*:}
  [ -f $R/$l.jsonl ] && (cd $W && $PY -m model_eval.replay --compare $R/botmask5-k160.jsonl $R/$l.jsonl > $R/$l-vs-botmask5-k160.txt 2>&1) \
    && log "$l against botmask5-k160: $(grep -E '^(nll|tool_names|PASS|FAIL)' $R/$l-vs-botmask5-k160.txt | tr -s ' ' | tr '\n' ';')"
done
restore
log "rank night done"

#!/usr/bin/env bash
# The deploy of a prune on the live server. DEPLOY picks the profile (default md149t5):
#   md149t5    the MD prune in place of K 160 (plans/PLAN_MD_DEPLOY.md, done on 2026-11-12)
#   code226t5  code226 in place of MD (plans/PLAN_CODE226_DEPLOY.md), with the context cap of 160k (lever B). After
#              the checks pass, Hermes gets context_length 160000 for DeepSeek and the router, and the compression
#              threshold 0.8, with a backup of config.yaml (a rollback puts the backup back by hand).
# The env can set EXPECT_GIB (the weights of the boot check) after the review window measures them.
# The modes:
#   ./deploy_md.sh prepare        §6: the files in the server folder, with the live server up (no downtime)
#   ./deploy_md.sh readings TAG   §5 and §8: the /metrics counters and the bot decode test (5 runs) on port 8888
#   ./deploy_md.sh check [LOG]    the checks of §7 step 4 on a boot log (default: the log of the new container)
#   ./deploy_md.sh switch         §7: about 4 min of downtime. Only the user starts it, with a `!` command:
#     tmux new -d -s deploy "DEPLOY=code226t5 ~/Projects/dsv4-specialize/deploy_md.sh switch"
#   ./deploy_md.sh vars           print the values of the profile
#   ./deploy_md.sh hermes-cap     the Hermes change of the cap alone (NO_RESTART=1: no restart of hermes-gateway)
# DRY_RUN=1 with switch runs every check of the files and of the live server, then the checks of §7 step 4 on the
# boot log of the review window, and ends before any server action.
# The switch waits until the bot is idle (up to 30 min) and refuses to start in the cron window. If the new server
# is not healthy in 10 min, or a check fails, it stops the new server and starts MAIN_COMPOSE again (restore() of
# run_lib.sh). The tmux session deploy-guard restores MAIN_COMPOSE if port 8888 is down 90 min after the start.
set -u
S=/home/colin-spark/Projects/dsv4-specialize
case ${DEPLOY:-md149t5} in
  md149t5) NEW_NAME=md149t5 FOLDER_SRC=$S/tp1-prunemd5-k149 OLD_NAME=k160t5 MIN_KV=256000 CAP= REVIEW=boot-md-review
           : "${EXPECT_GIB:=69.79}";;
  code226t5) NEW_NAME=code226t5 FOLDER_SRC=$S/tp1-prunecode226t5-k154 OLD_NAME=md149t5 MIN_KV=160000 CAP=160000
             REVIEW=boot-rv-code226; : "${EXPECT_GIB:=71.78}";;   # 68.53 GiB at mtp0 plus the 3.25 GiB of the draft
  *) echo "unknown DEPLOY ${DEPLOY}: md149t5 or code226t5"; exit 2;;
esac
EXPECT_COUNTS=15
NEW_SRC=$S/deploy/compose-$NEW_NAME.yml
HERMES=${HERMES:-$HOME/.hermes/config.yaml}
# The old live server: the defaults of run_lib.sh name the live server until the switch, so the switch names it
export MAIN_COMPOSE=${MAIN_COMPOSE:-/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark/compose-$OLD_NAME.yml}
export MAIN_CONTAINER=${MAIN_CONTAINER:-deepseek-v4-flash-$OLD_NAME-deepseek-v4-flash-1}
# run_lib.sh needs the deadlines of a test window; the functions that this script calls do not use them
export LOG=${LOG:-$S/logs/deploy-$NEW_NAME.log} RESTORE_TAG=deploy START_AFTER=now BOOT_BY=now SPEED_BY=now RESTORE_AT=now
source "$S/run_lib.sh"   # SERVER, W, PY, R, MAIN, MAIN_COMPOSE, log, busy, stop_main, wait_mem, restore, ...

NEW=$SERVER/compose-$NEW_NAME.yml
NEW_PROJECT=deepseek-v4-flash-$NEW_NAME
NEW_CONTAINER=$NEW_PROJECT-deepseek-v4-flash-1
FOLDER=$SERVER/data/tp1-$NEW_NAME
CACHE=$SERVER/cache-$NEW_NAME
PATCHES="exl3.py:vllm/model_executor/layers/quantization/exl3.py deepseek_v4_model.py:vllm/models/deepseek_v4/nvidia/model.py"
IMAGE=$(awk '/^ *image:/ {print $2; exit}' "$NEW_SRC")
LIVE=http://127.0.0.1:8888
NAMES="deepseek-v4-flash-0731-k160t5 deepseek-v4-flash-0731"
GUARD_AT="(no guard yet)"

fail() { log "ERROR: $*"; exit 1; }
healthy() { curl -fsS --max-time 5 $LIVE/health > /dev/null 2>&1; }
new_running() { [ "$(docker inspect -f '{{.State.Running}}' $NEW_CONTAINER 2>/dev/null)" = true ]; }
in_cron_window() {  # the cron jobs run from 05:00 to 09:40 on weekdays; the idle wait can take 30 min
  local t=$((10#$(date +%H%M)))
  [ "$(date +%u)" -le 5 ] && [ $t -ge 415 ] && [ $t -le 945 ]
}

check_folder() {  # $1: a prune folder; true when it has its config, its manifest and no broken link
  [ -f "$1/config.json" ] && [ -f "$1/EXL3_MANIFEST.json" ] && grep -q n_routed_experts_per_layer "$1/config.json" &&
    [ -z "$(find "$1" -xtype l)" ]
}

check_files() {  # the files of prepare; true when all are in place
  local ok=0 p
  check_folder "$FOLDER" || { log "missing or incomplete: $FOLDER"; ok=1; }
  for p in $PATCHES; do
    cmp -s "$S/image-patch/perlayer/${p%%:*}" "$SERVER/image-patch/${p#*:}" ||
      { log "missing or different: $SERVER/image-patch/${p#*:}"; ok=1; }
  done
  [ -d "$CACHE/sparkinfer" ] || { log "missing: $CACHE"; ok=1; }
  cmp -s "$NEW_SRC" "$NEW" || { log "missing or different: $NEW"; ok=1; }
  [ $ok = 0 ] && { docker compose -f "$NEW" config -q 2>> "$LOG" || { log "the compose check failed: $NEW"; ok=1; }; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "the image is not local: $IMAGE"; ok=1; }
  return $ok
}

prepare() {
  log "prepare: the files of $NEW_NAME in $SERVER (the live server does not change)"
  if [ -d "$FOLDER" ]; then
    log "the folder is in place: $FOLDER"
  else
    [ -d "$FOLDER_SRC" ] && [ ! -L "$FOLDER_SRC" ] || fail "no folder at $FOLDER_SRC"
    check_folder "$FOLDER_SRC" || fail "$FOLDER_SRC is incomplete or has a broken link"
    mv "$FOLDER_SRC" "$FOLDER" && ln -s "$FOLDER" "$FOLDER_SRC" || fail "the move of the folder failed"
    log "moved the folder to $FOLDER; a link stays at $FOLDER_SRC for the test boots"
  fi
  for p in $PATCHES; do
    mkdir -p "$(dirname "$SERVER/image-patch/${p#*:}")" &&
      cp -p "$S/image-patch/perlayer/${p%%:*}" "$SERVER/image-patch/${p#*:}" || fail "the copy of ${p%%:*} failed"
  done
  if [ -d "$CACHE" ]; then
    log "the kernel cache is in place: $CACHE"
  else
    # root owns some cache files (mode 600), so a container copies them
    mkdir "$CACHE" && docker run --rm --network none --entrypoint cp -v "$S/cache:/src:ro" -v "$CACHE:/dst" "$IMAGE" \
      -a /src/. /dst/ >> "$LOG" 2>&1 || fail "the copy of the kernel cache failed"
    log "copied the kernel cache to $CACHE ($(du -sh "$CACHE" 2>/dev/null | cut -f1))"
  fi
  if [ -f "$NEW" ] && ! cmp -s "$NEW_SRC" "$NEW"; then cp -p "$NEW" "$NEW.bak.$(date +%Y%m%d_%H%M%S)"; fi
  cp "$NEW_SRC" "$NEW" || fail "the copy of the compose file failed"
  check_files || fail "a check of the files failed"
  log "prepare done: every file is in place"
}

readings() {  # $1: the tag of the files, e.g. before or after
  local tag=${1:?readings needs a tag, e.g. before or after} out
  out=$R/speed-bot-deploy-$tag
  [ -e "$out.jsonl" ] && fail "$out.jsonl exists: move it first"
  healthy || fail "port 8888 is not healthy"
  curl -s --max-time 30 $LIVE/metrics > "$R/metrics-deploy-$tag.txt" || fail "no /metrics"
  log "readings $tag: /metrics in $R/metrics-deploy-$tag.txt; the bot decode test (5 runs)"
  (cd $W && $PY -m model_eval.speed --base-url $LIVE --out "$out.jsonl" --runs 5 > "$out.log" 2>&1)
  log "readings $tag: the bot decode test exit $?: $(tail -1 "$out.log" | cut -c1-160)"
}

checks() {  # $1: a boot log; the API checks go to port 8888; true when every check passes
  local f=$1 ok=0 gib kv models reply n
  gib=$(grep -oE 'Model loading took [0-9.]+ GiB' "$f" | grep -oE '[0-9.]+' | tail -1)
  log "check: weights ${gib:-?} GiB ($EXPECT_GIB expected)"
  awk -v g="${gib:-0}" -v e="$EXPECT_GIB" 'BEGIN {exit !(g > e - 0.1 && g < e + 0.1)}' || ok=1
  log "check: $(grep -oE 'per-layer experts: [0-9]+ counts' "$f" | head -1) ($EXPECT_COUNTS expected)"
  grep -q "per-layer experts: $EXPECT_COUNTS counts" "$f" || ok=1
  kv=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$f" | tail -1 | tr -dc 0-9)
  log "check: KV cache ${kv:-?} tokens (at least $MIN_KV)"
  [ "${kv:-0}" -ge "$MIN_KV" ] || ok=1
  models=$(curl -s --max-time 10 $LIVE/v1/models | $PY -c "import json,sys; print(*(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null)
  log "check: served names: ${models:-none}"
  for n in $NAMES; do case " $models " in *" $n "*) ;; *) ok=1;; esac; done
  for n in 1 2 3; do   # a bot request can come first after the boot: up to 3 tries of 5 min
    reply=$(curl -s --max-time 300 $LIVE/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash-0731-k160t5","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"thinking":false}}' \
      | $PY -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>&1)
    log "check: reply ${reply:0:40} (try $n)"
    case "$reply" in *OK*) break ;; esac
  done
  case "$reply" in *OK*) ;; *) ok=1;; esac
  log "check: hermes-gateway $(systemctl --user is-active hermes-gateway); $(avail_gib) GiB available"
  systemctl --user is-active -q hermes-gateway || ok=1
  [ $ok = 0 ] && log "every check passes" || log "ERROR: a check failed"
  return $ok
}

hermes_cap() {  # the cap: context_length $CAP for DeepSeek and the router, compression threshold 0.8, with a backup
  [ -n "$CAP" ] || return 0
  local bak=$HERMES.bak.$(date +%Y%m%d_%H%M%S)_$NEW_NAME
  cp -p "$HERMES" "$bak" || { log "ERROR: no backup of $HERMES"; return 1; }
  if ! CAP=$CAP python3 -c '
import os, sys
p, cap = sys.argv[1], os.environ["CAP"]
t = open(p).read()
for a, b, n in (("\n        context_length: 256000\n", f"\n        context_length: {cap}\n", 2),   # DeepSeek and auto
                ("\ncompression:\n  enabled: true\n  progress_notices: false\n  threshold: 0.5\n",
                 "\ncompression:\n  enabled: true\n  progress_notices: false\n  threshold: 0.8\n", 1)):
    assert t.count(a) == n, f"{a.strip()!r}: {t.count(a)} times, {n} expected"
    t = t.replace(a, b)
open(p, "w").write(t)
' "$HERMES" 2>> "$LOG"; then
    cp -p "$bak" "$HERMES"; log "ERROR: the Hermes change failed; $HERMES is restored"; return 1
  fi
  log "Hermes: context_length $CAP for DeepSeek and the router, compression threshold 0.8 (backup $bak)"
  [ "${NO_RESTART:-0}" = 1 ] && return 0
  systemctl --user restart hermes-gateway && sleep 5
  log "hermes-gateway restarted: $(systemctl --user is-active hermes-gateway)"
}

rollback() {
  log "ROLLBACK: stop $NEW_NAME, then start $MAIN_COMPOSE"
  docker logs $NEW_CONTAINER > "$R/boot-deploy-$NEW_NAME-failed.log" 2>&1
  (cd $SERVER && docker compose -f "$NEW" down) >> "$LOG" 2>&1
  docker ps -a --format '{{.Names}}' | grep -qx "$NEW_CONTAINER" && { log "ROLLBACK FAIL: $NEW_CONTAINER is left; the deploy-guard acts at $GUARD_AT"; exit 3; }
  restore   # run_lib.sh: 100 GiB free, start, health, reply; exit 3 on a failure
  log "ROLLBACK DONE: the live server is $MAIN_COMPOSE again"
}

switch() {
  local t0 ok=0 boot_log=$R/boot-deploy-$NEW_NAME.log
  log "deploy $NEW_NAME: stop $MAIN_COMPOSE, then start $NEW (DRY_RUN=${DRY_RUN:-0})"
  check_files || fail "the files are not in place: run ./deploy_md.sh prepare"
  new_running && fail "$NEW_CONTAINER runs already"
  main_running && healthy || fail "the live server is not up and healthy: no deploy"
  [ "$(servers | wc -l)" = 1 ] || fail "another server container runs: $(servers | tr '\n' ' ')"
  in_cron_window && fail "the cron jobs run from 05:00 to 09:40 on weekdays: start after 09:45, or before 04:15"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    log "dry run: the checks of step 4 on the boot log of the review window"
    checks "$R/$REVIEW.log"
    log "dry run done: no server action"; exit 0
  fi
  GUARD_AT=$(date -d 'now + 90 minutes' '+%F %T')
  tmux kill-session -t deploy-guard 2> /dev/null
  tmux new -d -s deploy-guard "GUARD_AT='$GUARD_AT' RECHECK_AT='$GUARD_AT' STOP_PROJECTS='dsv4-mask $NEW_PROJECT' \
MAIN_COMPOSE='$MAIN_COMPOSE' MAIN_CONTAINER='$MAIN' LOG=$S/logs/deploy-guard.log $S/overnight_guard.sh"
  log "the deploy-guard restores $MAIN_COMPOSE if port 8888 is down at $GUARD_AT"
  curl -s --max-time 30 $LIVE/metrics > "$R/metrics-deploy-old-final.txt"
  if ! stop_main; then
    main_running && fail "no deploy: the live server still runs"
    rollback; exit 1
  fi
  wait_mem 300 110 || { log "ERROR: less than 110 GiB available ($(avail_gib))"; rollback; exit 1; }
  log "start $NEW ($(avail_gib) GiB available)"
  (cd $SERVER && docker compose -f "$NEW" up -d) >> "$LOG" 2>&1
  t0=$(now)
  while [ $(( $(now) - t0 )) -lt 600 ]; do
    healthy && { ok=1; break; }
    case "$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' $NEW_CONTAINER 2>/dev/null || echo gone)" in
      exited\ [1-9]*|dead*|gone*) break;;
    esac
    sleep 10
  done
  docker logs $NEW_CONTAINER > "$boot_log" 2>&1
  [ $ok = 1 ] || { log "ERROR: $NEW_NAME is not healthy after $(( $(now) - t0 )) s"; rollback; exit 1; }
  log "$NEW_NAME healthy after $(( $(now) - t0 )) s"
  checks "$boot_log" || { rollback; exit 1; }
  hermes_cap || log "ERROR: Hermes keeps its old values; set them by hand (./deploy_md.sh hermes-cap)"
  log "DEPLOY DONE: the live server is $NEW. Next: DEPLOY=${DEPLOY:-md149t5} ./deploy_md.sh readings after"
}

case "${1:-}" in
  prepare) prepare ;;
  readings) readings "${2:-}" ;;
  check) if [ -n "${2:-}" ]; then checks "$2"; else docker logs $NEW_CONTAINER > "$R/boot-deploy-$NEW_NAME-check.log" 2>&1
           checks "$R/boot-deploy-$NEW_NAME-check.log"; fi ;;
  switch) switch ;;
  vars) echo "DEPLOY=${DEPLOY:-md149t5} NEW=$NEW FOLDER_SRC=$FOLDER_SRC FOLDER=$FOLDER MAIN_COMPOSE=$MAIN_COMPOSE" \
          "MAIN_CONTAINER=$MAIN_CONTAINER EXPECT_GIB=$EXPECT_GIB EXPECT_COUNTS=$EXPECT_COUNTS MIN_KV=$MIN_KV CAP=${CAP:-none}" ;;
  hermes-cap) hermes_cap ;;
  *) echo "usage: $0 prepare | readings TAG | check [BOOT_LOG] | switch | vars | hermes-cap   (see the header)"; exit 2 ;;
esac

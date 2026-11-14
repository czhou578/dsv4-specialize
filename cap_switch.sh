#!/usr/bin/env bash
# The DeepSeek context cap (D1 of PLAN_SECOND_MODEL_FIT.md §3.1; plans/PLAN_QWEN_PRUNE.md §8): 160,000 tokens
# instead of 256,000, and a KV pin of about 6.92 GiB (1.04x of the need at 160k; the need is about 6.54 GiB at 128k
# and 6.97 GiB at 256k) instead of 7.25 GiB. About 0.35 GiB more for Qwen.
#   1. Checks: DeepSeek healthy, not in the cron window. Stops the Qwen server of Phase D (pid file), if any.
#   2. Waits until the bot is idle (up to 30 min).
#   3. Writes compose-md149t5-c160.yml in the server folder: a copy of compose-md149t5.yml with exactly two changes
#      (MAX_MODEL_LEN and --kv-cache-memory-bytes). The project name stays, so the container keeps its name, and the
#      guard and run_lib.sh (docker compose start) go on working. The old file does not change.
#   4. docker compose up -d with the new file (about 4 min of downtime). If the KV pin is too small, one more boot
#      with the need of the error plus 4%. If DeepSeek is not healthy then, the rollback: up -d with the old file.
#   5. After a healthy boot at 160k: Hermes gets context_length 160000 for DeepSeek and the compression threshold 0.8
#      (the compression still starts at 128k), with a backup of config.yaml, and the gateway restarts.
# DRY_RUN=1 runs the checks and writes the new compose file, with no change to the servers or to Hermes.
# The rollback by hand:  cd ~/Projects/DeepSeek-v4-Flash-One-DGX-Spark && docker compose -f compose-md149t5.yml up -d
#   then copy ~/.hermes/config.yaml.bak.<date>_cap160 back to ~/.hermes/config.yaml and restart hermes-gateway.
# CAUTION: this restarts the bot's model server. Start it only with the user's go:
#   tmux new -d -s cap ~/Projects/dsv4-specialize/cap_switch.sh
set -u
S=/home/colin-spark/Projects/dsv4-specialize SERVER=${SERVER:-/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark}
LOG=${LOG:-$S/logs/cap-switch.log} PIDFILE=${PIDFILE:-$S/results/pair-qwen.pid}
OLD=$SERVER/compose-md149t5.yml NEW=$SERVER/compose-md149t5-c160.yml
HERMES=${HERMES:-$HOME/.hermes/config.yaml}
PIN=${PIN:-7431600000} OLD_PIN=7784628224 CTX=160000
MAIN=deepseek-v4-flash-md149t5-deepseek-v4-flash-1
DRY=${DRY_RUN:-0}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
now() { date +%s; }
healthy() { curl -fsS --max-time 5 http://127.0.0.1:8888/health > /dev/null 2>&1; }
busy() { curl -s --max-time 5 http://127.0.0.1:8888/metrics | awk '/^vllm:num_requests_(running|waiting)/ {s+=$2} END {print (s>0)}'; }
max_len() { curl -s --max-time 5 http://127.0.0.1:8888/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0].get('max_model_len'))"; }
compose() { (cd $SERVER && docker compose -f "$1" "${@:2}") >> "$LOG" 2>&1; }

write_new() {  # $1 pin: the new compose file, and a check that it differs from the old one in two lines only
  sed -e "s/MAX_MODEL_LEN: \"256000\"/MAX_MODEL_LEN: \"$CTX\"/" -e "s/--kv-cache-memory-bytes $OLD_PIN/--kv-cache-memory-bytes $1/" \
    -e "1i # The DeepSeek context cap (dsv4-specialize cap_switch.sh): compose-md149t5.yml with MAX_MODEL_LEN $CTX and a KV pin of $1 bytes." \
    "$OLD" > "$NEW"
  [ "$(diff "$OLD" "$NEW" | grep -c '^>')" = 3 ] || { log "ERROR: the new compose file does not differ in exactly two lines"; return 1; }
  docker compose -f "$NEW" config -q 2>> "$LOG" || { log "ERROR: the compose check failed: $NEW"; return 1; }
}
wait_healthy() {  # up to 20 min; 0 healthy, 1 not
  local t0=$(now)
  sleep 30
  while [ $(( $(now) - t0 )) -lt 1200 ]; do
    healthy && return 0
    [ "$(docker inspect -f '{{.State.Running}}' $MAIN 2>/dev/null)" = true ] || { sleep 20; [ "$(docker inspect -f '{{.State.Running}}' $MAIN 2>/dev/null)" = true ] || return 1; }
    sleep 10
  done
  return 1
}
hermes() {  # the two values, with a backup
  local bak=$HERMES.bak.$(date +%Y%m%d_%H%M%S)_cap160
  cp "$HERMES" "$bak" || return 1
  python3 - "$HERMES" <<'EOF' || { cp "$bak" "$HERMES"; return 1; }
import sys
p = sys.argv[1]; t = open(p).read()
for a, b in (("\n        context_length: 256000\n", "\n        context_length: 160000\n"),
             ("\ncompression:\n  enabled: true\n  progress_notices: false\n  threshold: 0.5\n",
              "\ncompression:\n  enabled: true\n  progress_notices: false\n  threshold: 0.8\n")):
    assert t.count(a) == 1, a
    t = t.replace(a, b)
open(p, "w").write(t)
EOF
  log "Hermes: context_length 160000 for DeepSeek, compression threshold 0.8 (backup $bak)"
  systemctl --user restart hermes-gateway && sleep 5 && systemctl --user is-active hermes-gateway >> "$LOG" 2>&1 &&
    log "hermes-gateway restarted: $(systemctl --user is-active hermes-gateway)"
}

[ -n "${CAP_STUB:-}" ] && source "$CAP_STUB"   # the dry run of test_cap_switch.py
log "cap switch start (dry run $DRY): DeepSeek at $(max_len) tokens"
healthy || { log "ERROR: DeepSeek is not healthy"; exit 1; }
[ "$(date +%u)" -le 5 ] && [ "$(date +%H)" -ge 5 ] && [ "$(date +%H)" -lt 10 ] && { log "ERROR: the cron window"; exit 1; }
[ "$(max_len)" = "$CTX" ] && { log "DeepSeek already runs at $CTX: nothing to do"; exit 0; }
write_new $PIN || exit 1
log "new compose file: $NEW ($(diff "$OLD" "$NEW" | grep '^>' | tr -s ' ' | cut -c1-90 | tr '\n' ';'))"
[ "$DRY" = 1 ] && { log "dry run: no change to the servers or to Hermes"; exit 0; }
if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  log "stop the Qwen server of Phase D (pid $(cat "$PIDFILE"))"
  kill -TERM -- -"$(cat "$PIDFILE")" 2>/dev/null; sleep 15; kill -KILL -- -"$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"
fi
t0=$(now)
while [ "$(busy)" = 1 ] && [ $(( $(now) - t0 )) -lt 1800 ]; do sleep 20; done
[ "$(busy)" = 1 ] && { log "ERROR: the bot is still busy after 30 min: no switch"; exit 1; }
log "switch DeepSeek to $CTX tokens (KV pin $PIN bytes)"
compose "$NEW" up -d
if ! wait_healthy; then
  need=$(docker logs --tail 300 $MAIN 2>&1 | grep -oE '[0-9.]+ GiB KV cache is needed' | head -1 | cut -d' ' -f1)
  if [ -n "$need" ]; then
    PIN=$(awk -v n=$need 'BEGIN {printf "%d", n * 1.04 * 2^30}')
    log "the KV pin was too small, the need is $need GiB: one more boot with $PIN bytes"
    write_new $PIN && compose "$NEW" up -d && wait_healthy || need=
  fi
  if ! healthy; then
    log "ERROR: DeepSeek is not healthy at $CTX: the rollback to $OLD"
    docker logs --tail 120 $MAIN > $S/results/cap-switch-failed.log 2>&1
    compose "$OLD" up -d
    wait_healthy && log "ROLLBACK DONE: DeepSeek healthy at $(max_len) tokens" || log "ERROR: DeepSeek is not healthy after the rollback"
    exit 1
  fi
fi
log "DeepSeek healthy at $(max_len) tokens; $(docker logs --tail 400 $MAIN 2>&1 | grep -oE 'reserved [0-9.]+ GiB memory for KV Cache|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' | sort -u | tr '\n' ';')"
log "reply: $(curl -s --max-time 120 http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731-k160t5","max_tokens":300,"messages":[{"role":"user","content":"Reply with OK."}]}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>&1 | head -c 60)"
hermes || log "ERROR: the Hermes change failed: config.yaml is restored"
log "cap switch done"

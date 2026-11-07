#!/usr/bin/env bash
# Overnight guard: make sure the main server serves on port 8888 before the 05:00 cron jobs.
# It does not depend on Claude Code. Start it in its own tmux session before the overnight run:
#   tmux new -d -s guard ~/Projects/dsv4-specialize/overnight_guard.sh
# At GUARD_AT (default 04:15) and again at RECHECK_AT (default 04:35):
#   - main server healthy: do nothing;
#   - main container running but not healthy (it boots): wait up to 20 min for /health;
#   - main container stopped: stop every dsv4-mask container, wait until the memory is free,
#     start the main container again (docker compose start: no pull, no new compose file).
# The main server is MAIN_COMPOSE (default: compose-k160t5.yml, the pruned model, live since 2026-11-06).
# DRY_RUN=1 logs the actions instead of running them. GUARD_NOW=1 skips the wait for GUARD_AT.
set -u
GUARD_AT=${GUARD_AT:-04:15}
RECHECK_AT=${RECHECK_AT:-04:35}
MAIN_URL=${MAIN_URL:-http://127.0.0.1:8888}
SERVER=/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark
MAIN_COMPOSE=${MAIN_COMPOSE:-$SERVER/compose-k160t5.yml}
MAIN_CONTAINER=${MAIN_CONTAINER:-deepseek-v4-flash-k160t5-deepseek-v4-flash-1}
LOG=${LOG:-/home/colin-spark/Projects/dsv4-specialize/overnight_guard.log}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() {
  if [ "${DRY_RUN:-0}" = 1 ]; then log "dry run: $*"; return 0; fi
  log "run: $*"
  "$@" >> "$LOG" 2>&1
}
healthy() { curl -fsS --max-time 5 "$MAIN_URL/health" > /dev/null 2>&1; }
main_running() { [ "$(docker inspect -f '{{.State.Running}}' "$MAIN_CONTAINER" 2>/dev/null)" = true ]; }
available_gib() { free -g | awk '/^Mem:/ {print $7}'; }

wait_until() {  # sleep until the next HH:MM; $2=today: if HH:MM is past today, do not wait
  local target now
  target=$(date -d "$1" +%s); now=$(date +%s)
  if [ "$target" -le "$now" ]; then
    # The recheck must not slip a day when the first restore waits past RECHECK_AT (2026-11-03).
    [ "${2:-}" = today ] && { log "$1 already passed; no wait"; return; }
    target=$((target + 86400))
  fi
  log "waiting until $(date -d "@$target" '+%F %T')"
  sleep $((target - now))
}

wait_healthy() {  # $1: seconds
  if [ "${DRY_RUN:-0}" = 1 ]; then log "dry run: wait up to $1 s for /health"; return 0; fi
  local end=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    healthy && { log "main server healthy"; return 0; }
    sleep 15
  done
  log "ERROR: main server not healthy after $1 s"; return 1
}

restore() {
  if healthy; then log "main server healthy on $MAIN_URL; nothing to do"; return 0; fi
  if main_running; then
    log "main container runs but is not healthy yet; waiting"
    wait_healthy 1200; return
  fi
  log "main server down; restoring"
  for c in $(docker ps -aq --filter label=com.docker.compose.project=dsv4-mask); do
    run docker stop -t 120 "$c"
  done
  if [ "${DRY_RUN:-0}" != 1 ]; then
    for _ in $(seq 1 36); do  # up to 3 min for the memory of a test model to free
      [ "$(available_gib)" -ge 100 ] && break
      sleep 5
    done
    log "available memory: $(available_gib) GiB"
    if [ "$(available_gib)" -lt 100 ]; then
      log "ERROR: less than 100 GiB free; not starting the main server (two models do not fit)"
      return 1
    fi
  fi
  # start reuses the stopped container; up -d only if it is gone (that one pulls the pinned image)
  run docker compose -f "$MAIN_COMPOSE" start || run docker compose -f "$MAIN_COMPOSE" up -d
  wait_healthy 1200
}

log "guard started (GUARD_AT=$GUARD_AT, RECHECK_AT=$RECHECK_AT, MAIN_COMPOSE=$MAIN_COMPOSE, DRY_RUN=${DRY_RUN:-0})"
[ "${GUARD_NOW:-0}" = 1 ] || wait_until "$GUARD_AT"
restore
[ "${GUARD_NOW:-0}" = 1 ] && exit 0
wait_until "$RECHECK_AT" today
restore
log "guard done"

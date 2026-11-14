#!/usr/bin/env bash
# The memory watch of Phase D (plans/PLAN_QWEN_PRUNE.md §8, PLAN_SECOND_MODEL_FIT.md §7.1): Qwen beside the live
# DeepSeek. Every second it records the available memory. Below FLOOR_MB (default 2048) it stops the Qwen server, and
# never DeepSeek: it kills only the process group in PIDFILE, and only when the command line of that process is the
# vLLM 0.33 venv with --port 8889 (DeepSeek also shows on the host as a vLLM process, on port 8888). It keeps watching
# after a stop. Start it in its own tmux session before Qwen, and leave it on while Qwen runs:
#   tmux new -d -s pair-watch ~/Projects/dsv4-specialize/pair_watch.sh
# WATCH_ONCE=1 runs one check and exits (the tests). AVAIL_CMD replaces the memory reading (the tests).
set -u
S=/home/colin-spark/Projects/dsv4-specialize
LOG=${LOG:-$S/logs/pair-watch.log}
MEM=${MEM:-$S/results/pair-mem.log}
PIDFILE=${PIDFILE:-$S/results/pair-qwen.pid}
FLOOR_MB=${FLOOR_MB:-2048}
QV=${QV:-$HOME/venvs/vllm033/bin}
AVAIL_CMD=${AVAIL_CMD:-"free -m | awk '/^Mem:/ {print \$7}'"}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
qwen_pid() {  # the recorded Qwen pid, if that process is the Qwen server on port 8889
  local pid cmd
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  [ -n "$pid" ] && cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || return 1
  case "$cmd" in "$QV/"*" serve "*"--port 8889 "*) echo "$pid";; *) return 1;; esac
}
check() {
  local m pid
  m=$(eval "$AVAIL_CMD")
  echo "$(date +%s) $m" >> "$MEM"
  if [ "$m" -lt "$FLOOR_MB" ]; then
    if pid=$(qwen_pid); then
      kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid"
      log "WATCH: $m MB available, below $FLOOR_MB MB: the Qwen server (pid $pid) is stopped"
    else
      log "WATCH: $m MB available, below $FLOOR_MB MB, and no Qwen server to stop"
    fi
  fi
}

[ "${WATCH_ONCE:-0}" = 1 ] && { check; exit 0; }
log "watch start: floor $FLOOR_MB MB, pid file $PIDFILE"
while true; do check; sleep 1; done

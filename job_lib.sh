#!/usr/bin/env bash
# Report a script to the machine-dash page (http://127.0.0.1:8470; ~/Projects/machine-dash, README "Jobs").
# run_lib.sh and overnight_guard.sh source this file. Then:
#   job_wrap [job run options] -- "$@"   in a tmux pane, run this script again under `job run` (exec), so the page
#                                        shows it with its log, its phase and its exit code
#   job_say step|wait|set ARGS...       report to the page; does nothing outside `job run` and never fails
# JOB_NAME names the job (default: the script name without .sh). JOB_WRAP=0 turns the wrap off. Outside tmux there
# is no wrap, so the tests and the runs in a terminal do not show on the page. Under `job run` the script runs again
# from the top, so a script must do no work before the wrap.
job_wrap() {
  local opts=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do opts+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  [ -z "${MACHINE_DASH_JOB:-}" ] && [ -n "${TMUX:-}" ] && [ "${JOB_WRAP:-1}" != 0 ] && [ -f "$0" ] \
    && command -v job > /dev/null || return 0
  # `bash "$0"` gives the child the command line of the shebang start, so `pgrep -f "^bash $WINDOW"`
  # (review_finish.sh) still finds it. If the exec fails, the script goes on without the page.
  shopt -s execfail
  exec job run "${opts[@]}" "${JOB_NAME:-$(basename "$0" .sh)}" -- bash "$0" "$@"
  shopt -u execfail
  return 0
}
job_say() {
  [ -n "${MACHINE_DASH_JOB:-}" ] && command -v job > /dev/null && job "$@" > /dev/null 2>&1
  return 0
}

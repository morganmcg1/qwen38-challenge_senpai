#!/bin/bash
# Wait for one Yukon submission row to leave `validating`, then exit.
#
# Run this only under `run_job`, which owns the cadence and wakes the
# conversation on the terminal state. Do not run it in the interactive
# terminal.
set -u

export PATH="${HOME}/.local/bin:${PATH}"

prefix="${1:?usage: e135_receipt_watch.sh SUBMISSION_ID_PREFIX [MAX_SECONDS]}"
max_seconds="${2:-5100}"
interval=60
start="$(date +%s)"

# Yukon colours the status column even when stdout is a pipe or a file, so the
# raw third field is an ANSI-wrapped string that no plain pattern can match.
strip_ansi() { sed $'s/\033\\[[0-9;]*[A-Za-z]//g'; }

while :; do
  row="$(yukon submissions --all 2>/dev/null | strip_ansi | grep -E "^${prefix}" | tail -1)"
  status="$(printf '%s' "${row}" | awk '{print $3}')"
  printf '%s %s %s\n' "$(date -u +%H:%M:%SZ)" "${prefix}" "${status:-absent}"

  case "${status}" in
    validating|absent|"")
      ;;
    *)
      echo "TERMINAL ${status}"
      echo "${row}"
      exit 0
      ;;
  esac

  if (( $(date +%s) - start > max_seconds )); then
    echo "WATCH TIMEOUT after ${max_seconds}s, still ${status:-absent}"
    exit 3
  fi
  sleep "${interval}"
done

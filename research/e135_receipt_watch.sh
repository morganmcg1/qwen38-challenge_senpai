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

# Yukon truncates the submission column to 7 characters, so a longer prefix
# taken from an API id never matches and the watch spins to timeout.
prefix="${prefix:0:7}"

# A row is briefly absent right after submission. A row that stays absent means
# the prefix is wrong, so fail loudly instead of waiting out the whole budget.
absent_limit=5
absent_count=0

# Yukon colours the status column even when stdout is a pipe or a file, so the
# raw third field is an ANSI-wrapped string that no plain pattern can match.
strip_ansi() { sed $'s/\033\\[[0-9;]*[A-Za-z]//g'; }

while :; do
  row="$(yukon submissions --all 2>/dev/null | strip_ansi | grep -E "^${prefix}" | tail -1)"
  status="$(printf '%s' "${row}" | awk '{print $3}')"
  printf '%s %s %s\n' "$(date -u +%H:%M:%SZ)" "${prefix}" "${status:-absent}"

  case "${status}" in
    validating)
      absent_count=0
      ;;
    "")
      absent_count=$((absent_count + 1))
      if (( absent_count >= absent_limit )); then
        echo "WATCH ABORT: row ${prefix} absent for ${absent_count} polls, check the prefix"
        exit 4
      fi
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

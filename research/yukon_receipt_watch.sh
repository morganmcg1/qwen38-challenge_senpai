#!/usr/bin/env bash
# Watch one Yukon submission until it leaves `validating`, then print its row
# and exit. Research-only; Yukon does not submit anything under research/.
#
#   usage: research/yukon_receipt_watch.sh SUBMISSION_ID_PREFIX [INTERVAL_SECONDS]
#
# Run this through the Senpai job supervisor, never in an interactive shell:
# the supervisor owns the cadence and wakes the conversation on the terminal
# state. The job timeout is the only deadline; this script does not impose one.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: yukon_receipt_watch.sh SUBMISSION_ID_PREFIX [INTERVAL_SECONDS]}"
interval="${2:-60}"

export PATH="${HOME}/.local/bin:${PATH}"
strip_ansi='s/\x1b\[[0-9;]*m//g'

while true; do
  # Keep stderr and the exit status. A missing credential used to look
  # identical to "the row is not on the board yet", which hid two failures.
  board="$(yukon submissions --all 2>&1)" || {
    rc=$?
    echo "receipt-watch: FAILED to query yukon, exit ${rc}: ${board}" >&2
    exit 3
  }

  # Yukon truncates the printed id to 7 characters, so a longer requested
  # prefix never matches with a plain "^p" test. Accept the row when either
  # string is a prefix of the other.
  row="$(sed -e "${strip_ansi}" <<<"${board}" \
    | awk -v p="${prefix}" \
        'length($1) >= 6 && (index(p, $1) == 1 || index($1, p) == 1) { print; exit }')"

  if [[ -n "${row}" ]]; then
    status="$(awk '{print $3}' <<<"${row}")"
    # `promotion failed` is two words; the third field is then `promotion`.
    if [[ "${status}" != "validating" && "${status}" != "pending" \
       && "${status}" != "queued" && "${status}" != "running" ]]; then
      echo "receipt-watch: ${prefix} resolved"
      echo "${row}"
      exit 0
    fi
    echo "receipt-watch: ${prefix} status=${status} $(date -u +%H:%M:%SZ)"
  else
    echo "receipt-watch: ${prefix} not listed yet $(date -u +%H:%M:%SZ)"
  fi

  sleep "${interval}"
done

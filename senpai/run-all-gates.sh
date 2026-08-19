#!/bin/bash
# run-all-gates.sh -- run EVERY campaign gate, with its arguments, and report.
#
# WHY THIS EXISTS
# ---------------
# I was about to write "all gates are green" into a student note from memory. I
# ran them instead. Five were green. Two exited **2**, which is *usage* -- they
# need the crown SHA as an argument and I had invoked them bare. Given the
# argument, one was green and one was RED, and had been red since `7f89dd5`.
#
# That is the whole defect: a gate that refuses to run and a gate that fails
# both look like "not the word PASS", and a gate invoked without its arguments
# is a gate never run. `verify-campaign-overlay.sh` hid a real `.gitignore`
# violation for most of a campaign behind an exit code that looked like a broken
# script rather than a failing check.
#
# So: one command, every gate, arguments supplied, exit codes captured
# individually, and usage-exits reported as LOUDLY as failures -- because a gate
# that could not run is not evidence of anything.
#
# The crown SHA is resolved LIVE from `upstream` via ls-remote rather than
# hardcoded, for the same reason `verify-base-drift.sh` does it: this checkout's
# fetch refspecs cannot be trusted to have updated any local mirror.
#
# Exit 0 only if every gate returned 0.

set -u

cd "$(git rev-parse --show-toplevel)" || exit 99

OUTDIR="${TMPDIR:-/tmp}/senpai-gates-$$"
mkdir -p "$OUTDIR"

echo "resolving crown from upstream (live, not a local mirror)..."
CROWN=$(git ls-remote upstream refs/heads/main 2>/dev/null | awk '{print $1}' | head -1)
if [ -z "$CROWN" ]; then
  echo "FAIL: could not resolve upstream/main via ls-remote. Refusing to run" >&2
  echo "      parity/overlay gates against a guessed SHA." >&2
  exit 1
fi
case "$CROWN" in
  ????????????????????????????????????????) : ;;
  *) echo "FAIL: upstream/main SHA is not 40 chars: '$CROWN'" >&2; exit 1 ;;
esac
echo "crown = $CROWN"
echo

fail=0
usage=0
ran=0

run() {
  name="$1"; shift
  out="$OUTDIR/$(echo "$name" | tr -c 'a-zA-Z0-9' '_').txt"
  "$@" > "$out" 2>&1
  rc=$?
  ran=$((ran + 1))
  if [ "$rc" -eq 0 ]; then
    printf 'PASS      %s\n' "$name"
  elif [ "$rc" -eq 2 ]; then
    # Reported as loudly as a failure, on purpose. This is the exit code that
    # let a red gate hide.
    printf '🔴 USAGE  %s  rc=2 -- THE GATE DID NOT RUN. Not evidence.  (%s)\n' \
      "$name" "$out"
    usage=$((usage + 1))
  else
    printf '🔴 FAIL   %s  rc=%s  (%s)\n' "$name" "$rc" "$out"
    fail=$((fail + 1))
  fi
}

run scored-surface            bash research/scored-surface-gate.sh
run shipped-surface           bash research/shipped-surface-gate.sh
run inherited-surface         bash research/inherited-surface-gate.sh
run frontier-revert           bash research/frontier-revert-gate.sh
run frontier-revert-selftest  bash research/selftest-frontier-revert-gate.sh
run base-drift                bash senpai/verify-base-drift.sh
run trusted-parity            bash senpai/verify-trusted-parity.sh "$CROWN"
run campaign-overlay          bash senpai/verify-campaign-overlay.sh "$CROWN"
run kernel-table-selftest     bash senpai/verify-kernel-table.sh selftest
run student-instruments-selftest bash senpai/verify-student-instruments.sh selftest
run census-selftest           python3 research/stream_dispatch_census.py selftest
run stream-optimality-selftest python3 research/stream_optimality.py selftest
run crown-leg-selftest        python3 research/crown_leg_decomposition.py --selftest

echo
echo "gates run: $ran   failures: $fail   did-not-run: $usage"
if [ "$fail" -gt 0 ] || [ "$usage" -gt 0 ]; then
  echo "🔴 NOT ALL GATES GREEN -- do not quote a clean bill of health."
  exit 1
fi
echo "all $ran gates PASS against crown $CROWN"
echo "outputs in $OUTDIR"
exit 0

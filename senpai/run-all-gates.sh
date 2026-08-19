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

# The growth base for the byte-budget gate. Read from senpai/campaign-base.json
# rather than hardcoded, for the same reason the crown is: a stale constant here
# would silently re-baseline the budget and hide growth. Fail closed.
BASE_SHA=$(python3 -c 'import json,sys;print(json.load(open("senpai/campaign-base.json"))["baseSha"])' 2>/dev/null)
case "${BASE_SHA:-}" in
  ????????????????????????????????????????) : ;;
  *) echo "FAIL: could not read baseSha from senpai/campaign-base.json" >&2; exit 1 ;;
esac
echo "base  = $BASE_SHA"
echo

# PREFLIGHT: is the packaged surface dirty?
#
# Several gates below fail-closed on uncommitted changes to packaged paths, and
# the CONTROL suites degrade worse than that -- they report "right_reason=no",
# because the gate they are probing failed for the dirty-tree reason instead of
# the reason the control injected. That reads like a broken control rather than
# a dirty tree, and it cost me a diagnosis. Say it once, up front, plainly.
DIRTY=$(python3 - "$PWD" <<'PY' 2>/dev/null
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
paths = set(json.loads((root / "benchmark.json").read_text())["editablePaths"])
out = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                     cwd=root, capture_output=True, text=True).stdout.split()
print(" ".join(sorted(p for p in out if p in paths)))
PY
)
if [ -n "${DIRTY:-}" ]; then
  echo "🔴 PREFLIGHT: packaged paths have UNCOMMITTED changes:"
  for p in $DIRTY; do echo "     $p"; done
  echo "   Surface gates will refuse to certify, and the CONTROL suites will"
  echo "   report 'right_reason=no' because they are probing a gate that is"
  echo "   already failing for this reason. Commit or stash, then re-run."
  echo
fi

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
# ADDED 2026-08-19, after a STUDENT found this red and I did not.
#
# `twin_audit.py` had been exiting 1 -- correctly, on a real divergence between
# quantized.h and its mlx-generated twin -- for an unknown number of turns, and
# nobody saw it, because it was never in this file. It was a working gate that
# was simply never invoked. I had been reporting "15/15 gates GREEN"; that
# sentence was true and useless.
#
# A red gate outside the suite is indistinguishable from no gate. The
# distinction that matters is not "does a check exist" but "does anything run
# it".
run twin-audit                python3 research/twin_audit.py
run twin-waiver-control       python3 research/twin_waiver_negative_control.py
# MUTATES the checked-in twin blob and restores it. Safe here only because this
# suite runs strictly sequentially. Do not parallelise the suite without giving
# this one exclusive access -- I raced it against twin-audit by hand and got a
# confident, wrong RED. "Independent" means no shared mutable state, not just
# "different commands".
run twin-waiver-digests-selftest python3 research/twin_waiver_digests_selftest.py
run frontier-revert           bash research/frontier-revert-gate.sh
run frontier-revert-selftest  bash research/selftest-frontier-revert-gate.sh
# Line-granular companion to frontier-revert, which is FILE-granular: a sync
# that reintroduces `reachedStopToken` while the file still differs for other
# reasons satisfies the ack and leaves frontier-revert GREEN on a broken tree.
run campaign-invariants       bash senpai/verify-campaign-invariants.sh
run campaign-invariants-selftest bash senpai/selftest-campaign-invariants.sh
run base-drift                bash senpai/verify-base-drift.sh
run trusted-parity            bash senpai/verify-trusted-parity.sh "$CROWN"
run campaign-overlay          bash senpai/verify-campaign-overlay.sh "$CROWN"
run kernel-table-selftest     bash senpai/verify-kernel-table.sh selftest
run student-instruments-selftest bash senpai/verify-student-instruments.sh selftest
run census-selftest           python3 research/stream_dispatch_census.py selftest
run stream-optimality-selftest python3 research/stream_optimality.py selftest
run crown-leg-selftest        python3 research/crown_leg_decomposition.py --selftest
# ALSO ADDED 2026-08-19, from a sweep for "checks that exist but nothing runs".
# twin_audit was not a one-off; it was a CATEGORY. 34 of 47 candidate checks
# were named nowhere in this file. Most are frozen per-experiment analyses and
# belong nowhere near a suite, but these were standing invariants, and
# noise-floors-selftest is one I wrote LAST TURN and never wired in -- the same
# mistake, one turn later, by me.
#
# The two CONTROL suites are the ones that matter most: they are what makes the
# surface gates evidence rather than decoration, by injecting a defect and
# requiring the gate to fail FOR THE RIGHT REASON. Both were 12/12 green and
# nothing had run them.
run noise-floors-selftest     python3 research/noise_floors.py selftest
# Companion to noise-floors: that file is the authority on DENOMINATORS, this
# one on the two leverage NUMERATORS. Its first assertion is that a uniform QMV
# speedup has a NEGATIVE score derivative. If that ever flips green-to-red the
# campaign's whole direction changes, so it is a gate and not a note.
run qmv-leverage-selftest     python3 research/qmv_score_leverage.py selftest
run scored-surface-controls   bash research/scored-surface-gate-controls.sh
run frontier-revert-controls  bash research/frontier-revert-gate-controls.sh
run metallib-guard-controls   bash research/metallib-guard-controls.sh
run forward-dispatch-inventory python3 research/verify_forward_dispatch_inventory.py
run trusted-parity-selftest   bash senpai/selftest-trusted-parity.sh "$CROWN"
run editable-budget           bash senpai/check-editable-budget.sh "$BASE_SHA"

echo
echo "gates run: $ran   failures: $fail   did-not-run: $usage"
if [ "$fail" -gt 0 ] || [ "$usage" -gt 0 ]; then
  echo "🔴 NOT ALL GATES GREEN -- do not quote a clean bill of health."
  exit 1
fi
echo "all $ran gates PASS against crown $CROWN"
echo "outputs in $OUTDIR"
exit 0

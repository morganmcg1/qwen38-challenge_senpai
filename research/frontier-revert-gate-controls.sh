#!/usr/bin/env bash
# Mutation negative controls for research/frontier-revert-gate.sh.
#
# Three gates in research/ have now shipped with defects my own careful reading
# missed, and in all three cases the mutation controls found them, not me. So the
# rule is: no gate is trusted on inspection, and a control that only checks
# "exit 1" is too weak, because a syntax error also exits 1. Each case must also
# emit the gate's own diagnostic naming the thing that was broken.
#
# This gate has an additional trap the others do not. Its correct answer TODAY is
# BLOCKED, so "it failed" is not evidence that it works -- a gate hardcoded to
# fail would look identical. Control 2 therefore drives it to a clean PASS, and
# control 12 proves it is not simply reporting every changed file in the repo.
#
# Usage: research/frontier-revert-gate-controls.sh

set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GATE="research/frontier-revert-gate.sh"
ACKS="research/frontier-revert-acks.txt"
[ -f "${GATE}" ] || { echo "controls: missing ${GATE}" >&2; exit 1; }
[ -f "${ACKS}" ] || { echo "controls: missing ${ACKS}" >&2; exit 1; }

work="$(mktemp -d "${TMPDIR:-/tmp}/frontier-gate-controls.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

pass=0
nfail=0

# expect_fail <label> <needle> <env assignments...> -- run the pristine gate with
# a mutated ACK table and/or env and require its own FAIL line to mention needle.
expect_fail() {
  local label="$1" needle="$2"; shift 2
  env "$@" bash "${GATE}" > "${work}/out.txt" 2>&1
  local rc=$?
  local saw_fail="no" right="no"
  grep -q "^FAIL: " "${work}/out.txt" && saw_fail="yes"
  grep -q "${needle}" "${work}/out.txt" && right="yes"
  if [ "${rc}" -eq 1 ] && [ "${saw_fail}" = "yes" ] && [ "${right}" = "yes" ]; then
    printf '  PASS  %-58s (exit 1, own FAIL line, right reason)\n' "${label}"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-58s rc=%s own_fail=%s right_reason=%s\n' "${label}" "${rc}" "${saw_fail}" "${right}"
    head -6 "${work}/out.txt" | sed 's/^/          /'
    nfail=$((nfail + 1))
  fi
}

echo "frontier-revert gate: mutation negative controls"
echo "  subject: ${GATE}"
echo

# --- the target file every mutation control needs ----------------------------
# LEDGER 162: controls 3 and 5 used to hardcode
# Sources/MLXFastModel/Qwen35RuntimeWeights.swift as "a packaged file that
# differs from the frontier". The rebase made that file byte-identical to the
# frontier, so removing or shortening its entry changed nothing and both controls
# silently became vacuous -- they reported rc=0 and looked like gate defects when
# in fact the CONTROLS had rotted. The rule that follows: a control must
# CONSTRUCT its own input and never depend on the live tree's state. So the
# target is derived from the acknowledgement table itself, whose every entry the
# gate independently asserts describes a real difference (that is control 4).
TARGET="$(awk -F'|' '!/^[[:space:]]*#/ && NF >= 3 { print $1; exit }' "${ACKS}")"
if [ -z "${TARGET}" ]; then
  echo "  controls: the acknowledgement table has no entries, so controls 3 and 5"
  echo "            cannot be constructed. That is a legitimate state only if the"
  echo "            tree is fully reconciled; refusing to report them as passing."
  nfail=$((nfail + 2))
fi
echo "  derived mutation target: ${TARGET:-<none>}"
echo

# --- control 1: a MUST-REBASE entry BLOCKS ----------------------------------
# LEDGER 162: this control used to run the gate unmutated and assert BLOCKED,
# because BLOCKED was the correct answer on the day it was written. Then the
# rebase landed, the gate correctly went green, and the control failed while the
# gate was working perfectly -- it was asserting a fact about the TREE, not a
# property of the GATE. It is now the exact mirror of control 2: control 2
# promotes every entry to INTENTIONAL-REPLACEMENT and demands PASS, control 1
# downgrades one entry to MUST-REBASE and demands BLOCKED. Together they prove
# the verdict tracks the table in both directions, on any tree.
if [ -n "${TARGET}" ]; then
  awk -F'|' -v t="${TARGET}" 'BEGIN{OFS="|"} !/^[[:space:]]*#/ && NF >= 3 && $1 == t { $2 = "MUST-REBASE" } { print }' \
    "${ACKS}" > "${work}/one_must_rebase.txt"
else
  cp "${ACKS}" "${work}/one_must_rebase.txt"
fi
FRONTIER_ACKS="${work}/one_must_rebase.txt" bash "${GATE}" > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && grep -q "MUST-REBASE" "${work}/out.txt" \
   && grep -q "frontier-revert gate: BLOCKED" "${work}/out.txt"; then
  printf '  PASS  %-58s (exit 1, blocked on named MUST-REBASE)\n' "control 1  a MUST-REBASE entry blocks"
  pass=$((pass + 1))
else
  printf '  FAIL  %-58s rc=%s\n' "control 1  a MUST-REBASE entry blocks" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

# --- control 2: it CAN pass ---------------------------------------------------
# The load-bearing control. Today's correct answer is BLOCKED, so a gate that
# always failed would be indistinguishable from a working one. Promote every
# entry to INTENTIONAL-REPLACEMENT and the gate must go green -- proving the
# blocker is data-driven, not baked in.
sed 's/|MUST-REBASE|/|INTENTIONAL-REPLACEMENT|/' "${ACKS}" > "${work}/all_intentional.txt"
FRONTIER_ACKS="${work}/all_intentional.txt" bash "${GATE}" > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 0 ] && grep -q "frontier-revert gate: PASS" "${work}/out.txt"; then
  printf '  PASS  %-58s (exit 0, gate can go green)\n' "control 2  all-intentional table passes"
  pass=$((pass + 1))
else
  printf '  FAIL  %-58s rc=%s (gate may be hardcoded to fail)\n' "control 2  all-intentional table passes" "${rc}"
  head -8 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

# --- control 3: a packaged file with no entry at all -------------------------
# Target derived above, not hardcoded. See the LEDGER 162 note at its definition.
awk -F'|' -v t="${TARGET}" '!($1 == t && !/^[[:space:]]*#/ && NF >= 3)' \
  "${work}/all_intentional.txt" > "${work}/missing_entry.txt"
expect_fail "control 3  unlisted packaged file refused" "is not listed in" \
  "FRONTIER_ACKS=${work}/missing_entry.txt"

# --- control 4: a stale entry must be refused --------------------------------
# benchmark.json is inside editablePaths and is byte-identical to the frontier,
# so an entry for it describes a revert that does not exist.
cp "${work}/all_intentional.txt" "${work}/stale.txt"
printf '%s\n' 'benchmark.json|INTENTIONAL-REPLACEMENT|a long enough reason to clear the forty character minimum check' \
  >> "${work}/stale.txt"
expect_fail "control 4  stale acknowledgement refused" "no longer differs from the frontier" \
  "FRONTIER_ACKS=${work}/stale.txt"

# --- control 5: a reason too short to be a reason ----------------------------
# Target derived above, not hardcoded. See the LEDGER 162 note at its definition.
awk -F'|' -v t="${TARGET}" 'BEGIN{OFS="|"} !/^[[:space:]]*#/ && NF >= 3 && $1 == t { print $1, $2, "fine"; next } { print }' \
  "${work}/all_intentional.txt" > "${work}/shortreason.txt"
expect_fail "control 5  acknowledgement with no reason refused" "no substantive reason" \
  "FRONTIER_ACKS=${work}/shortreason.txt"

# --- control 6: an invented status word must not be honoured ------------------
sed 's/|INTENTIONAL-REPLACEMENT|/|PROBABLY-FINE|/' "${work}/all_intentional.txt" \
  > "${work}/badstatus.txt"
expect_fail "control 6  unknown status word refused" "unknown status" \
  "FRONTIER_ACKS=${work}/badstatus.txt"

# --- control 7: a missing acknowledgement table ------------------------------
expect_fail "control 7  missing ack table refused" "is missing" \
  "FRONTIER_ACKS=${work}/does-not-exist.txt"

# --- control 8: fetch staleness is a hard failure ---------------------------
expect_fail "control 8  stale fetch refused" "over the" \
  "FRONTIER_ACKS=${work}/all_intentional.txt" "FRONTIER_MAX_FETCH_AGE=0"

# --- control 9: an unresolvable frontier ref --------------------------------
FRONTIER_ACKS="${work}/all_intentional.txt" bash "${GATE}" refs/heads/no-such-frontier \
  > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && grep -q "does not resolve" "${work}/out.txt" \
   && grep -q "git fetch upstream" "${work}/out.txt"; then
  printf '  PASS  %-58s (exit 1, says how to fix it)\n' "control 9  unresolvable frontier ref refused"
  pass=$((pass + 1))
else
  printf '  FAIL  %-58s rc=%s\n' "control 9  unresolvable frontier ref refused" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

# --- control 10: pointing the gate at OUR OWN branch must be refused --------
# The single most dangerous false pass: aim the gate at a tree we authored and it
# reports zero reverts, which is true and useless. The organizer-shape assertion
# has to catch it.
FRONTIER_ACKS="${work}/all_intentional.txt" bash "${GATE}" HEAD > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && grep -q "not yukon-autoresearch\[bot\]" "${work}/out.txt"; then
  printf '  PASS  %-58s (exit 1, refuses our own branch as a frontier)\n' "control 10 self-referential frontier refused"
  pass=$((pass + 1))
else
  printf '  FAIL  %-58s rc=%s\n' "control 10 self-referential frontier refused" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

# --- control 11: the editablePaths filter must be real ----------------------
# The gate must report files a submission PACKAGES, not every file that differs.
# senpai/ and research/ are ours and differ from the frontier enormously, but a
# submission does not carry them. If they appear, the gate is measuring the wrong
# set and its line counts are fiction.
FRONTIER_ACKS="${work}/all_intentional.txt" bash "${GATE}" > "${work}/out.txt" 2>&1
leaked=0
grep -q "senpai/campaign-ledger.md" "${work}/out.txt" && leaked=1
grep -q "research/frontier-revert-gate.sh" "${work}/out.txt" && leaked=1
differs_outside=0
# LEDGER 162: this line used to be
#     git diff --numstat upstream/main HEAD -- senpai/ research/ | grep -q . && differs_outside=1
# and it silently stopped working. `grep -q` exits on the FIRST match, git then
# dies of SIGPIPE with status 141, and `set -o pipefail` makes the whole pipeline
# return 141 -- so the `&&` never fired and the control reported differs_outside=0,
# i.e. "there is nothing outside the packaged set to leak", which made the control
# vacuous. It passed for weeks because the diff was small enough to fit in the
# pipe buffer before grep exited; it broke when research/ grew. A control whose
# own MEASUREMENT can fail silently is worse than no control. No pipe now.
_outside="$(git diff --numstat upstream/main HEAD -- senpai/ research/ 2>/dev/null || true)"
[ -n "${_outside}" ] && differs_outside=1
if [ "${leaked}" -eq 0 ] && [ "${differs_outside}" -eq 1 ]; then
  printf '  PASS  %-58s (unpackaged files differ but are excluded)\n' "control 11 editablePaths filter is load-bearing"
  pass=$((pass + 1))
else
  printf '  FAIL  %-58s leaked=%s differs_outside=%s\n' "control 11 editablePaths filter is load-bearing" "${leaked}" "${differs_outside}"
  nfail=$((nfail + 1))
fi

# --- control 12: an empty editablePaths list must not mean "nothing to check" -
# A submission with no known packaged set is unreviewable, and silently reporting
# zero reverts would be the worst possible failure mode.
mkdir -p "${work}/emptyep/research"
git -C "${work}/emptyep" init -q 2>/dev/null
printf '%s\n' '{"editablePaths": []}' > "${work}/emptyep/benchmark.json"
cp "${GATE}" "${work}/emptyep/research/g.sh"
cp "${ACKS}" "${work}/emptyep/research/acks.txt"
( cd "${work}/emptyep" && FRONTIER_ACKS=research/acks.txt bash research/g.sh ) \
  > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && ! grep -q "frontier-revert gate: PASS" "${work}/out.txt"; then
  printf '  PASS  %-58s (exit 1, no silent zero)\n' "control 12 empty editablePaths refused"
  pass=$((pass + 1))
else
  printf '  FAIL  %-58s rc=%s\n' "control 12 empty editablePaths refused" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

echo
echo "  ${pass} passed, ${nfail} failed"
if [ "${nfail}" -ne 0 ]; then
  echo "frontier-revert gate controls: FAIL"
  exit 1
fi
echo "frontier-revert gate controls: PASS"

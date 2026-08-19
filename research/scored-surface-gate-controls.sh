#!/usr/bin/env bash
# Mutation negative controls for research/scored-surface-gate.sh.
#
# A gate written by the person with the blind spot inherits the blind spot. The
# first version of research/inherited-surface-gate.sh credited this campaign with
# +4708 lines because it summed our files against the wrong baseline -- the exact
# confusion that gate exists to expose -- and the mutation controls caught it,
# not a careful reading. So no gate in research/ is trusted on inspection.
#
# Each control copies the gate, mutates ONE thing, and requires the copy to fail.
# A control that merely checks "exit 1" is too weak, because a syntax error in the
# mutation would also exit 1; so each case must also emit the gate's own FAIL line
# mentioning the thing that was broken.
#
# Usage: research/scored-surface-gate-controls.sh

set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GATE="research/scored-surface-gate.sh"
[ -f "${GATE}" ] || { echo "controls: missing ${GATE}" >&2; exit 1; }

work="$(mktemp -d "${TMPDIR:-/tmp}/scored-gate-controls.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

pass=0
nfail=0

# run <mutant-file> -> exit code; output in ${work}/out.txt
run() {
  bash "$1" > "${work}/out.txt" 2>&1
  return $?
}

# expect_fail <label> <mutant> <substring the FAIL line must contain>
expect_fail() {
  local label="$1" mutant="$2" needle="$3"
  run "${mutant}"
  local rc=$?
  local saw_fail="no"
  if grep -q "^FAIL: " "${work}/out.txt"; then saw_fail="yes"; fi
  local right_reason="no"
  if grep -q "${needle}" "${work}/out.txt"; then right_reason="yes"; fi
  if [ "${rc}" -eq 1 ] && [ "${saw_fail}" = "yes" ] && [ "${right_reason}" = "yes" ]; then
    printf '  PASS  %-56s (exit 1, own FAIL line, right reason)\n' "${label}"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-56s rc=%s own_fail=%s right_reason=%s\n' "${label}" "${rc}" "${saw_fail}" "${right_reason}"
    printf '        first lines of output:\n'
    head -4 "${work}/out.txt" | sed 's/^/          /'
    nfail=$((nfail + 1))
  fi
}

expect_pass() {
  local label="$1" mutant="$2"
  run "${mutant}"
  local rc=$?
  if [ "${rc}" -eq 0 ] && grep -q "scored-surface gate: PASS" "${work}/out.txt"; then
    printf '  PASS  %-56s (exit 0, gate reports PASS)\n' "${label}"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-56s rc=%s (expected a clean pass)\n' "${label}" "${rc}"
    head -4 "${work}/out.txt" | sed 's/^/          /'
    nfail=$((nfail + 1))
  fi
}

echo "scored-surface gate: mutation negative controls"
echo "  subject: ${GATE}"
echo

# --- control 1: unmodified gate must pass -----------------------------------
cp "${GATE}" "${work}/m0.sh"
expect_pass "control 1  unmodified gate passes" "${work}/m0.sh"

# --- control 2: an unscored file with no acknowledgement must be refused ----
# This is the gate's primary job. Drop the Qwen35.swift acknowledgement and the
# gate must notice that a shipped file differs from the scored tree unexplained.
cp "${GATE}" "${work}/m1.sh"
sed -i '' '/^  "Vendor\/mlx-swift-lm\/Libraries\/MLXLLM\/Models\/Qwen35.swift|HOT-PATH-REFACTOR/d' "${work}/m1.sh"
expect_fail "control 2  unacknowledged unscored file refused" "${work}/m1.sh" "is not acknowledged in ACK_UNSCORED"

# --- control 3: the same for the other file, so one ACK cannot mask another --
cp "${GATE}" "${work}/m2.sh"
sed -i '' '/^  "Sources\/MLXFastModel\/Qwen36MTPBlockSession.swift|PROBE-OFF-BY-DEFAULT/d' "${work}/m2.sh"
expect_fail "control 3  the other file, unacknowledged, refused" "${work}/m2.sh" "Qwen36MTPBlockSession.swift"

# --- control 4: a stale acknowledgement must be refused ---------------------
# The table has to rot loudly. Add an entry for a path that does NOT differ from
# the scored tree; leaving such entries is how an ACK list becomes fiction.
cp "${GATE}" "${work}/m3.sh"
sed -i '' 's|^ACK_UNSCORED=(|ACK_UNSCORED=(\
  "Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift\|PROBE-OFF-BY-DEFAULT\|a plausible sounding reason that is long enough to pass the length check"|' "${work}/m3.sh"
expect_fail "control 4  stale acknowledgement refused" "${work}/m3.sh" "no longer differs from the scored tree"

# --- control 5: an acknowledgement with no substantive reason must be refused
# Done in python, not sed: the ACK entries are pipe-delimited and BSD sed has no
# way to write this substitution without the delimiter colliding with the data.
cp "${GATE}" "${work}/m4.sh"
python3 - "${work}/m4.sh" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(
    r'"Sources/MLXFastModel/Qwen36MTPBlockSession\.swift\|PROBE-OFF-BY-DEFAULT\|[^"]*"',
    '"Sources/MLXFastModel/Qwen36MTPBlockSession.swift|PROBE-OFF-BY-DEFAULT|fine"',
    s,
)
open(p, 'w').write(s)
PY
expect_fail "control 5  acknowledgement with no reason refused" "${work}/m4.sh" "has no substantive reason"

# --- control 6: a missing scored commit must fail with the fetch instruction -
# The pre-existing situation: the sha the API reports is not in the local object
# store. The gate must refuse to report an unscored delta it cannot compute, and
# must say how to fix it, rather than silently comparing against nothing.
cp "${GATE}" "${work}/m5.sh"
sed -i '' 's|^SCORED_COMMIT=.*|SCORED_COMMIT="0000000000000000000000000000000000000000"|' "${work}/m5.sh"
expect_fail "control 6  unresolvable scored commit refused" "${work}/m5.sh" "git fetch origin"

# --- controls 7-9: every provenance pin must be load-bearing ----------------
cp "${GATE}" "${work}/m6.sh"
sed -i '' 's|^SCORED_SUBJECT=.*|SCORED_SUBJECT="Validate submission not-the-real-one"|' "${work}/m6.sh"
expect_fail "control 7  wrong subject pin refused" "${work}/m6.sh" "subject drifted"

cp "${GATE}" "${work}/m7.sh"
sed -i '' 's|^SCORED_AUTHOR=.*|SCORED_AUTHOR="somebody-else"|' "${work}/m7.sh"
expect_fail "control 8  wrong author pin refused" "${work}/m7.sh" "author drifted"

cp "${GATE}" "${work}/m8.sh"
sed -i '' 's|^SCORED_PARENT=.*|SCORED_PARENT="5068eb8d0bae032faca6e901de398fc7325311ff"|' "${work}/m8.sh"
expect_fail "control 9  wrong lineage-parent pin refused" "${work}/m8.sh" "parent drifted"

# --- control 10: pointing the gate at the scored tree itself must be clean ---
# A true positive control on the measurement: with rev == the scored commit there
# is by construction no unscored delta, so every acknowledgement is stale and the
# gate must say so rather than reporting a comfortable pass.
cp "${GATE}" "${work}/m9.sh"
bash "${work}/m9.sh" 2b0c36a078b7660c9215adee933336ff46da25af > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && grep -q "no longer differs from the scored tree" "${work}/out.txt" \
   && grep -q "every shipped line has been measured" "${work}/out.txt"; then
  printf '  PASS  %-56s (empty delta reported, ACKs correctly stale)\n' "control 10 gate run against the scored tree itself"
  pass=$((pass + 1))
else
  printf '  FAIL  %-56s rc=%s\n' "control 10 gate run against the scored tree itself" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

# --- control 11: the root-resolution fallback must itself fail closed --------
# Controls 2-10 are only runnable because the gate falls back to the caller's
# repository when its own path is not in one. That fallback is a new hazard: a
# gate that guesses a root could guess wrong and then report a comfortable
# answer about a tree nobody asked about. So prove the guess has a floor -- with
# no repository reachable from either the script or the caller, the gate must
# refuse rather than produce output.
norepo="${work}/norepo"
mkdir -p "${norepo}"
cp "${GATE}" "${norepo}/m10.sh"
( cd "${norepo}" && GIT_CEILING_DIRECTORIES="${norepo}" bash ./m10.sh ) \
  > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && grep -q "not inside a git repository" "${work}/out.txt" \
   && ! grep -q "scored-surface gate: PASS" "${work}/out.txt"; then
  printf '  PASS  %-56s (exit 1, refuses to guess a root)\n' "control 11 no repository reachable at all"
  pass=$((pass + 1))
else
  printf '  FAIL  %-56s rc=%s\n' "control 11 no repository reachable at all" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

# --- control 12: script-relative root must WIN over the caller's repo --------
# The fallback must be a fallback, not a preference. Copy the gate into a real
# but unrelated git repo that contains none of the objects it pins, and run it
# with the cwd inside that decoy. The gate must resolve to the decoy (its own
# location) and fail on the missing scored commit -- NOT quietly reach back into
# the campaign repo and report a pass. Ordering here is the whole safety
# argument for control 11's fallback existing at all.
decoy="${work}/decoy"
mkdir -p "${decoy}/research"
git -C "${decoy}" init -q 2>/dev/null
git -C "${decoy}" commit -q --allow-empty -m "decoy" \
  --author="decoy <decoy@example.invalid>" 2>/dev/null
cp "${GATE}" "${decoy}/research/m11.sh"
( cd "${decoy}" && bash research/m11.sh ) > "${work}/out.txt" 2>&1
rc=$?
if [ "${rc}" -eq 1 ] && grep -q "not in the local object store" "${work}/out.txt" \
   && ! grep -q "scored-surface gate: PASS" "${work}/out.txt"; then
  printf '  PASS  %-56s (exit 1, used its own repo not the campaign one)\n' "control 12 script-relative root beats caller repo"
  pass=$((pass + 1))
else
  printf '  FAIL  %-56s rc=%s\n' "control 12 script-relative root beats caller repo" "${rc}"
  head -6 "${work}/out.txt" | sed 's/^/          /'
  nfail=$((nfail + 1))
fi

echo
echo "  ${pass} passed, ${nfail} failed"
if [ "${nfail}" -ne 0 ]; then
  echo "scored-surface gate controls: FAIL"
  exit 1
fi
echo "scored-surface gate controls: PASS"

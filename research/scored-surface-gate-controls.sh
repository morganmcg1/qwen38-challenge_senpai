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

# --- derive the mutation targets from the subject, never from memory ----------
# Controls 3, 4 and 5 used to name a path AND its STATUS word literally. One
# rebase renamed a status (PROBE-OFF-BY-DEFAULT -> WARM-PATH-ONLY) and added a
# FRONTIER-TAKEN class, and all three mutations silently matched nothing: they
# ran the UNMODIFIED gate, it passed, and the harness recorded "refusal absent"
# without ever saying the mutation had not been applied. A control that cannot
# fire is worse than no control, so the targets are now read out of the gate's
# own table and every mutation asserts that it changed the file.
ack_paths() {
  awk '/^ACK_UNSCORED=\(/ { f = 1; next } f && /^\)/ { exit } f' "${GATE}" \
    | awk -F'|' '{ sub(/^[[:space:]]*"/, "", $1); print $1 }'
}

# drop_ack <mutant-file> <path> -- remove that path's ACK entry, by path ONLY.
# Matching on "path| is enough to identify an entry and is immune to renaming
# the status word, which is what broke the literal versions.
drop_ack() {
  local file="$1" path="$2" before after
  before="$(wc -l < "${file}")"
  grep -vF "\"${path}|" "${file}" > "${file}.tmp" && mv "${file}.tmp" "${file}"
  after="$(wc -l < "${file}")"
  if [ "${before}" -eq "${after}" ]; then
    printf '  FAIL  %-56s (vacuous: no ACK entry matched)\n' "setup for ${path}"
    nfail=$((nfail + 1))
    return 1
  fi
  return 0
}

# No mapfile: the system bash here is 3.2.57, where mapfile does not exist and
# an unset array under `set -u` would take the targets out silently.
ACK_PATHS=()
while IFS= read -r line; do
  [ -n "${line}" ] && ACK_PATHS+=("${line}")
done < <(ack_paths)
ACK_COUNT="${#ACK_PATHS[@]}"
ACK_FIRST=""
ACK_LAST=""
if [ "${ACK_COUNT}" -ge 1 ]; then
  ACK_FIRST="${ACK_PATHS[0]}"
  ACK_LAST="${ACK_PATHS[$((ACK_COUNT - 1))]}"
fi

echo "scored-surface gate: mutation negative controls"
echo "  subject: ${GATE}"
echo "  derived ACK targets: ${ACK_COUNT} entries; first=${ACK_FIRST##*/} last=${ACK_LAST##*/}"
echo

# An empty or single-entry table must not let controls 2/3 pass by vacuum.
if [ "${ACK_COUNT}" -lt 2 ] || [ "${ACK_FIRST}" = "${ACK_LAST}" ]; then
  printf '  FAIL  %-56s (need >=2 distinct ACK entries to mutate)\n' "control setup  ACK table too small"
  nfail=$((nfail + 2))
fi

# --- control 1: unmodified gate must pass -----------------------------------
cp "${GATE}" "${work}/m0.sh"
expect_pass "control 1  unmodified gate passes" "${work}/m0.sh"

# --- control 2: an unscored file with no acknowledgement must be refused ----
# This is the gate's primary job. Drop the FIRST acknowledgement and the gate
# must notice that a shipped file differs from the scored tree unexplained. The
# needle names the derived path, so the control also proves the gate blames the
# right file rather than merely failing somewhere.
cp "${GATE}" "${work}/m1.sh"
if drop_ack "${work}/m1.sh" "${ACK_FIRST}"; then
  expect_fail "control 2  unacknowledged unscored file refused" "${work}/m1.sh" \
    "${ACK_FIRST}' differs from the scored tree and is not acknowledged"
fi

# --- control 3: the same for another entry, so one ACK cannot mask another ---
cp "${GATE}" "${work}/m2.sh"
if drop_ack "${work}/m2.sh" "${ACK_LAST}"; then
  expect_fail "control 3  another entry, unacknowledged, refused" "${work}/m2.sh" \
    "${ACK_LAST}' differs from the scored tree and is not acknowledged"
fi

# --- control 4: a stale acknowledgement must be refused ---------------------
# The table has to rot loudly. Add an entry for a path that CANNOT appear in the
# unscored delta: benchmark.json sits outside SURFACE_PATHS (Sources/, Vendor/,
# mtp-head.manifest.json), so the gate never diffs it and the entry is stale by
# construction rather than by today's tree. The previous version of this control
# named a Sources/ file that "does not differ" -- until a rebase made it differ,
# at which point the control tested nothing.
cp "${GATE}" "${work}/m3.sh"
python3 - "${work}/m3.sh" "benchmark.json|PROBE-OFF-BY-DEFAULT|a plausible sounding reason long enough to clear the length check" <<'PY'
import sys
path, entry = sys.argv[1], sys.argv[2]
s = open(path).read()
needle = "ACK_UNSCORED=(\n"
if needle not in s:
    sys.exit("control 4 vacuous: no ACK_UNSCORED=( opener in the gate")
open(path, "w").write(s.replace(needle, needle + '  "' + entry + '"\n', 1))
PY
if [ $? -eq 0 ]; then
  expect_fail "control 4  stale acknowledgement refused" "${work}/m3.sh" "no longer differs from the scored tree"
else
  printf '  FAIL  %-56s (mutation could not be applied)\n' "control 4  stale acknowledgement refused"
  nfail=$((nfail + 1))
fi

# --- control 5: an acknowledgement with no substantive reason must be refused
# Done in python, not sed: the ACK entries are pipe-delimited and BSD sed has no
# way to write this substitution without the delimiter colliding with the data.
# The STATUS word is matched as a class, not spelled out, and the substitution
# count is asserted -- spelling it out is what made this control vacuous.
cp "${GATE}" "${work}/m4.sh"
python3 - "${work}/m4.sh" "${ACK_FIRST}" <<'PY'
import re, sys
path, target = sys.argv[1], sys.argv[2]
s = open(path).read()
pat = re.compile(r'("' + re.escape(target) + r'\|[A-Z0-9-]+\|)[^"]*"')
s2, n = pat.subn(r'\1fine"', s)
if n != 1:
    sys.exit("control 5 vacuous: %d ACK entries matched %s" % (n, target))
open(path, "w").write(s2)
PY
if [ $? -eq 0 ]; then
  expect_fail "control 5  acknowledgement with no reason refused" "${work}/m4.sh" "has no substantive reason"
else
  printf '  FAIL  %-56s (mutation could not be applied)\n' "control 5  acknowledgement with no reason refused"
  nfail=$((nfail + 1))
fi

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

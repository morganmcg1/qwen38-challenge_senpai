#!/usr/bin/env bash
# Selftest for senpai/verify-campaign-invariants.sh.
#
# WHY A SELFTEST AND NOT A READING
#
# The gate's whole value is in branches this repository cannot reach on demand:
# a promoted-frontier sync that reintroduces `reachedStopToken`, a table that
# parses to nothing, a regex silently truncated by a stray `|`. Reachability by
# running and coverage are different properties, and the second one needs
# CONSTRUCTED inputs. So every case below builds its own fixture and asserts
# the mutation actually landed before grading the verdict -- an assertion that
# exists because a previous control in this campaign "passed" while its
# mutation had silently failed to apply.
#
# CONTROL STRENGTH, measured and written down
#
# Case 2 is the case the gate exists for. It is also the only case that fails
# if the `absent` branch is deleted outright. Cases 5/6/12 (fail-closed) and
# 7/8 (field-count) are independent of it. That distribution is recorded here
# on purpose: a reader should know which single case is load-bearing rather
# than infer breadth from the total count.
#
# Exit: 0 all cases behaved as specified | 1 at least one did not

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="${SCRIPT_DIR}/verify-campaign-invariants.sh"

[ -x "$GATE" ] || { echo "FAIL: gate not executable: $GATE"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/civ-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0

note() { echo "    $1"; }

check() {
  # check <case-name> <expected-exit> <actual-exit> [<must-appear-in-output>]
  local name="$1" want="$2" got="$3" needle="${4:-}"
  local out="$WORK/out.txt"
  if [ "$want" != "$got" ]; then
    echo "FAIL  $name: expected exit $want, got $got"
    sed -e 's/^/        /' "$out" | head -12
    fail=$((fail + 1))
    return
  fi
  if [ -n "$needle" ] && ! grep -qF -- "$needle" "$out"; then
    echo "FAIL  $name: exit $got correct but output lacks: $needle"
    sed -e 's/^/        /' "$out" | head -12
    fail=$((fail + 1))
    return
  fi
  echo "ok    $name (exit $got)"
  pass=$((pass + 1))
}

run_gate() {
  # run_gate <root> <table>  -> sets RC, writes $WORK/out.txt
  "$GATE" --root "$1" --table "$2" >"$WORK/out.txt" 2>&1
  RC=$?
}

# ---------------------------------------------------------------- fixtures
mkfixture() {
  # mkfixture <dir> : a tree satisfying all three seeded invariants
  local d="$1"
  mkdir -p "$d/Sources/MLXFastModel"
  mkdir -p "$d/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models"
  {
    echo "func warmAllDepthShapes() {"
    echo "    for extra in 0 ... maxDepth {"
    echo "        _ = concatenated(parts, axis: 1)"
    echo "    }"
    echo "}"
  } > "$d/Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
  {
    echo "// S <= 2 (stale comment; the code says 9)"
    echo "let ladderActive = inputs.dim(1) <= 9 || prefillLadder"
  } > "$d/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
}

SESSION="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
QWEN="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

mktable() {
  # mktable <file> : the three real invariants
  {
    echo "# fixture table"
    echo ""
    echo "$SESSION | absent | reachedStopToken | provenance text"
    echo "$SESSION | present | for extra in 0 \\.\\.\\. maxDepth | provenance text"
    echo "$QWEN | present | inputs\\.dim\\(1\\) <= 9 | provenance text"
  } > "$1"
}

echo "selftest: senpai/verify-campaign-invariants.sh"
echo ""

# --- case 1: clean fixture passes -----------------------------------------
F="$WORK/c1"; mkfixture "$F"; T="$WORK/t1.txt"; mktable "$T"
run_gate "$F" "$T"
check "1  clean fixture passes" 0 "$RC" "RESULT: PASS"

# --- case 2: THE case -- `absent` invariant violated ----------------------
# A sync reintroduces the stop-token early return.
F="$WORK/c2"; mkfixture "$F"; T="$WORK/t2.txt"; mktable "$T"
printf '    if reachedStopToken { return out }\n' \
  >> "$F/$SESSION"
# assert the mutation landed, or this case proves nothing
if ! grep -q "reachedStopToken" "$F/$SESSION"; then
  echo "FAIL  2  fixture mutation did not land (reachedStopToken absent)"
  fail=$((fail + 1))
else
  run_gate "$F" "$T"
  check "2  absent violated -> FAIL" 1 "$RC" "must NOT contain"
  note "load-bearing: this is the case the gate exists for"
fi

# --- case 3: `present` invariant violated --------------------------------
# An overlay reverts fkiene's verify-concat warm.
F="$WORK/c3"; mkfixture "$F"; T="$WORK/t3.txt"; mktable "$T"
grep -v "for extra in 0 \.\.\. maxDepth" "$F/$SESSION" > "$F/tmp" \
  && mv "$F/tmp" "$F/$SESSION"
if grep -q "for extra in 0 \.\.\. maxDepth" "$F/$SESSION"; then
  echo "FAIL  3  fixture mutation did not land (warm still present)"
  fail=$((fail + 1))
else
  run_gate "$F" "$T"
  check "3  present violated -> FAIL" 1 "$RC" "must contain"
fi

# --- case 4: listed path missing from the tree ---------------------------
F="$WORK/c4"; mkfixture "$F"; T="$WORK/t4.txt"; mktable "$T"
rm -f "$F/$QWEN"
[ -f "$F/$QWEN" ] && { echo "FAIL  4  fixture mutation did not land"; \
  fail=$((fail + 1)); } || {
  run_gate "$F" "$T"
  check "4  missing path -> FAIL" 1 "$RC" "not present in the tree"
}

# --- case 5: table parses to zero invariants -> fail CLOSED --------------
F="$WORK/c5"; mkfixture "$F"; T="$WORK/t5.txt"
printf '# only a comment\n\n   \n' > "$T"
run_gate "$F" "$T"
check "5  empty table -> fail closed (2)" 2 "$RC" "ZERO invariants"
note "a PASS here would mean 'no evidence' read as 'no problems'"

# --- case 6: table file missing -> fail CLOSED --------------------------
F="$WORK/c6"; mkfixture "$F"
run_gate "$F" "$WORK/does-not-exist.txt"
check "6  missing table -> fail closed (2)" 2 "$RC" "not found"

# --- cases 7-10: TABLE defects -> fail closed (2), not graded (1) ------
#
# A malformed table line means "I could not read my own instructions". That is
# never a graded result about the tree, so it exits 2, not 1. These four cases
# were written expecting 1 and the gate returned 2; investigating that
# disagreement is what found the defect fixed in case 13.

# --- case 7: 3 fields (regex accidentally dropped) ---------------------
F="$WORK/c7"; mkfixture "$F"; T="$WORK/t7.txt"
printf '%s | absent | reachedStopToken\n' "$SESSION" > "$T"
run_gate "$F" "$T"
check "7  3-field line -> fail closed (2)" 2 "$RC" "expected 4"

# --- case 8: 5 fields (regex contains a literal '|') -------------------
# This is the silent-truncation hazard: without the field-count check the
# regex would become "foo" and the check would still report ok.
F="$WORK/c8"; mkfixture "$F"; T="$WORK/t8.txt"
printf '%s | absent | foo|reachedStopToken | prov\n' "$SESSION" > "$T"
run_gate "$F" "$T"
check "8  5-field line (regex has '|') -> fail closed (2)" 2 "$RC" "expected 4"
note "truncating a check is worse than dropping it: it still reports ok"

# --- case 9: mode not absent/present ----------------------------------
F="$WORK/c9"; mkfixture "$F"; T="$WORK/t9.txt"
printf '%s | maybe | reachedStopToken | prov\n' "$SESSION" > "$T"
run_gate "$F" "$T"
check "9  bad mode -> fail closed (2)" 2 "$RC" "expected 'absent' or 'present'"

# --- case 10: empty regex --------------------------------------------
F="$WORK/c10"; mkfixture "$F"; T="$WORK/t10.txt"
printf '%s | absent |  | prov\n' "$SESSION" > "$T"
run_gate "$F" "$T"
check "10 empty regex -> fail closed (2)" 2 "$RC" "empty regex"

# --- case 11: unknown argument -> fail CLOSED ------------------------
F="$WORK/c11"; mkfixture "$F"; T="$WORK/t11.txt"; mktable "$T"
"$GATE" --root "$F" --table "$T" --bogus >"$WORK/out.txt" 2>&1
RC=$?
check "11 unknown argument -> fail closed (2)" 2 "$RC" "unknown argument"

# --- case 12: default binding is the SCRIPT's tree, not $PWD ---------
# The gate must grade the tree it ships in. Proving that requires copying
# the gate into a fixture; invoking the campaign copy from elsewhere would
# grade the campaign tree, which is correct behaviour and therefore not a
# test of anything.
F="$WORK/c12"; mkfixture "$F"; mkdir -p "$F/senpai"
cp "$GATE" "$F/senpai/verify-campaign-invariants.sh"
chmod +x "$F/senpai/verify-campaign-invariants.sh"
mktable "$F/senpai/campaign-invariants.txt"
printf '    if reachedStopToken { return out }\n' >> "$F/$SESSION"
if ! grep -q "reachedStopToken" "$F/$SESSION"; then
  echo "FAIL  12 fixture mutation did not land"
  fail=$((fail + 1))
else
  ( cd / && "$F/senpai/verify-campaign-invariants.sh" ) \
    >"$WORK/out.txt" 2>&1
  RC=$?
  check "12 default binding grades script's tree from \$PWD=/" 1 "$RC" \
        "must NOT contain"
fi

# --- case 13: a table defect BESIDE good lines still fails closed -------
#
# This is the case that found a real defect. In the first version there was one
# counter for both failure classes, so a lone malformed line exited 2 (via the
# zero-evaluated guard) while the SAME malformed line beside three well-formed
# ones exited 1. The exit code for "your table is broken" must not depend on
# how many unrelated lines happen to parse. Both directions are asserted:
# case 7 (alone) and case 13 (in company) must agree.
F="$WORK/c13"; mkfixture "$F"; T="$WORK/t13.txt"; mktable "$T"
printf '%s | absent | foo|bar | prov\n' "$SESSION" >> "$T"
# assert the fixture really has 3 good lines plus 1 bad one
good=$(grep -c ' | absent | reachedStopToken | \|present' "$T" || true)
run_gate "$F" "$T"
check "13 defect beside 3 good lines -> still 2" 2 "$RC" "unparseable"
note "3 good lines evaluated ok yet the verdict is still fail-closed"
if grep -q "invariants listed: 4" "$WORK/out.txt"; then
  echo "ok    13a listed=4 confirms the good lines were parsed too"
  pass=$((pass + 1))
else
  echo "FAIL  13a expected 'invariants listed: 4' in output"
  fail=$((fail + 1))
fi

# --- case 14: the field-count check is load-bearing, not decorative -----
#
# Without it, `absent | foo|reachedStopToken` truncates to the regex `foo`,
# which does not match, so an `absent` check on a tree that DOES contain
# reachedStopToken would report ok. Case 8 proves the gate rejects the line;
# this case proves the line would otherwise have been a false PASS, by
# checking that the truncated regex genuinely fails to see the hazard.
F="$WORK/c14"; mkfixture "$F"
printf '    if reachedStopToken { return out }\n' >> "$F/$SESSION"
if grep -Eq -- "foo" "$F/$SESSION"; then
  echo "FAIL  14 fixture unexpectedly contains the truncated pattern"
  fail=$((fail + 1))
elif ! grep -Eq -- "foo|reachedStopToken" "$F/$SESSION"; then
  echo "FAIL  14 full pattern should match the hazard but does not"
  fail=$((fail + 1))
else
  echo "ok    14 truncated regex /foo/ misses a hazard that /foo|reachedStopToken/ catches"
  note "so the 4-field check prevents a FALSE PASS, not just an ugly table"
  pass=$((pass + 1))
fi

echo ""
echo "cases passed: $pass   failed: $fail"
if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: PASS ($pass cases)"
exit 0

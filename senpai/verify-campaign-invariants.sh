#!/usr/bin/env bash
# Line-granular guard on the handful of facts about our tree that a
# "Sync promoted organizer frontier" commit has historically broken.
#
# WHY THIS EXISTS, and why the frontier-revert gate is not enough
#
# research/frontier-revert-acks.txt is file-granular: one verdict per path,
# answering "is it intentional that our copy differs from the tip?". That is
# the right shape for that question. It cannot answer this one:
#
#   A sync reintroduces `reachedStopToken` into Qwen36MTPBlockSession.swift.
#   The file still differs from the tip for the other three reasons we carry,
#   so the ack is still satisfied, and the frontier-revert gate still PASSES --
#   on a tree whose fixed decode window now terminates early.
#
# `reachedStopToken` has been removed and reintroduced five times here. Every
# reintroduction arrived inside a commit whose subject said "sync". Every
# removal needed a human to notice. This gate is that human, written down.
#
# FAIL-CLOSED CONTRACT
#
# Every check in this gate is keyed off senpai/campaign-invariants.txt. So an
# empty, missing or unparseable table must NOT produce a PASS: a PASS would
# read as "no problems" when the truth is "no evidence". Specifically these are
# all hard failures:
#
#   * the table file is missing or unreadable
#   * the table parses to zero invariants
#   * a data line does not have exactly four `|`-separated fields
#   * the mode field is not exactly `absent` or `present`
#   * a listed path does not exist in the tree
#   * zero invariants were actually evaluated
#
# The four-field check matters more than it looks: a regex containing a literal
# `|` would otherwise be silently truncated into a weaker pattern that still
# passes. Truncating a check is worse than dropping it, because the truncated
# form still reports success.
#
# SCOPE, stated so a PASS is not over-read: this is line existence, not
# semantics. `present` cannot distinguish live code from a comment or a string
# literal, and cannot distinguish one occurrence from five. See the header of
# campaign-invariants.txt.
#
# Usage:  senpai/verify-campaign-invariants.sh [--table FILE] [--root DIR]
# Exit:   0 all invariants hold  |  1 an invariant is violated
#         2 the gate could not evaluate (fail-closed)

set -uo pipefail

# Bind to this script's location, not $PWD, so the gate grades the tree it
# ships in. The frontier-revert gate does the same and its selftest had to
# `cp` the gate into the fixture to test it -- that is correct behaviour for a
# gate and a required cost for its selftest.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TABLE="${SCRIPT_DIR}/campaign-invariants.txt"

while [ $# -gt 0 ]; do
  case "$1" in
    --table) TABLE="$2"; shift 2 ;;
    --root)  ROOT="$2";  shift 2 ;;
    -h|--help)
      echo "usage: $0 [--table FILE] [--root DIR]"; exit 0 ;;
    *)
      echo "FAIL(2): unknown argument: $1" >&2
      echo "  A gate that ignores an argument is worse than one that errors:" >&2
      echo "  the caller believes it checked something it did not." >&2
      exit 2 ;;
  esac
done

fail_closed() {
  echo "FAIL(2): $1" >&2
  shift
  while [ $# -gt 0 ]; do echo "         $1" >&2; shift; done
  exit 2
}

[ -f "$TABLE" ] || fail_closed \
  "invariant table not found: $TABLE" \
  "Every check here is keyed off that table, so a missing table is" \
  "'no evidence', not 'no problems'."
[ -r "$TABLE" ] || fail_closed "invariant table not readable: $TABLE"

echo "campaign invariant gate"
echo "  root:  $ROOT"
echo "  table: $TABLE"
echo ""

# Two failure classes, deliberately counted apart.
#
#   parse_errors -> "I could not evaluate this invariant" (a TABLE defect:
#                   wrong field count, bad mode, empty regex). Always exit 2.
#   violations   -> "I evaluated it and the tree is wrong". Exit 1.
#
# They were one counter in the first version, and the selftest caught what
# that costs: with a single malformed line the table produced exit 2 (via the
# zero-evaluated guard), but the SAME malformed line beside three good ones
# produced exit 1. The exit code for a table defect must not depend on how
# many unrelated lines happen to be well-formed.
listed=0
evaluated=0
violations=0
parse_errors=0

# Strip comments and blank lines first so the parse loop sees only data.
DATA="$(sed -e 's/[[:space:]]*$//' "$TABLE" \
        | grep -v '^[[:space:]]*#' \
        | grep -v '^[[:space:]]*$' || true)"

if [ -z "$DATA" ]; then
  fail_closed \
    "the invariant table declares ZERO invariants" \
    "Refusing to report PASS. Every check in this gate is keyed off the" \
    "table, so an empty table means the gate verified nothing at all."
fi

# Avoid `while read` over a pipe (subshell would discard the counters), and
# avoid mapfile (bash 3.2 on this host has none).
OLDIFS="$IFS"
IFS='
'
set -f
for line in $DATA; do
  set +f
  IFS="$OLDIFS"
  listed=$((listed + 1))

  nfields=$(printf '%s' "$line" | awk -F'|' '{print NF}')
  if [ "$nfields" -ne 4 ]; then
    echo "FAIL: line $listed has $nfields '|'-separated fields, expected 4"
    echo "      $line"
    echo "      A regex containing a literal '|' would be silently truncated"
    echo "      into a weaker check that still passes. Hard failure instead."
    parse_errors=$((parse_errors + 1))
    IFS='
'
    set -f
    continue
  fi

  path=$(printf '%s' "$line" | awk -F'|' '{print $1}' \
         | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  mode=$(printf '%s' "$line" | awk -F'|' '{print $2}' \
         | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  regex=$(printf '%s' "$line" | awk -F'|' '{print $3}' \
         | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

  if [ "$mode" != "absent" ] && [ "$mode" != "present" ]; then
    echo "FAIL: line $listed mode is '$mode', expected 'absent' or 'present'"
    parse_errors=$((parse_errors + 1))
    IFS='
'
    set -f
    continue
  fi
  if [ -z "$regex" ]; then
    echo "FAIL: line $listed has an empty regex"
    echo "      An empty pattern matches every line, which would make an"
    echo "      'absent' check always fail and a 'present' check always pass."
    parse_errors=$((parse_errors + 1))
    IFS='
'
    set -f
    continue
  fi

  target="$ROOT/$path"
  if [ ! -f "$target" ]; then
    echo "FAIL: $path"
    echo "      listed in the invariant table but not present in the tree."
    echo "      A vanished path is a stronger signal than a changed one:"
    echo "      resolve by hand, do not delete the entry to get green."
    violations=$((violations + 1))
    IFS='
'
    set -f
    continue
  fi

  # grep -c counts matching LINES. Exit status is 1 for no match, which under
  # `set -o pipefail` would abort; capture explicitly instead.
  hits=$(grep -Ec -- "$regex" "$target" 2>/dev/null || true)
  [ -n "$hits" ] || hits=0
  evaluated=$((evaluated + 1))

  if [ "$mode" = "absent" ]; then
    if [ "$hits" -ne 0 ]; then
      echo "FAIL: $path must NOT contain /$regex/ but has $hits line(s)"
      echo "      If this arrived in a 'Sync promoted organizer frontier'"
      echo "      commit, the fix is to remove the code, NOT to relax this"
      echo "      entry. See the provenance field in the table."
      violations=$((violations + 1))
    else
      echo "ok   absent   $path  /$regex/"
    fi
  else
    if [ "$hits" -eq 0 ]; then
      echo "FAIL: $path must contain /$regex/ but has no matching line"
      echo "      This is the reverted-by-overlay direction: our carried"
      echo "      change is gone. Restore it; do not delete this entry."
      violations=$((violations + 1))
    else
      echo "ok   present  $path  /$regex/  ($hits line(s))"
    fi
  fi

  IFS='
'
  set -f
done
set +f
IFS="$OLDIFS"

echo ""
echo "invariants listed: $listed   evaluated: $evaluated" \
     "  violations: $violations   table defects: $parse_errors"

# Verdict order is deliberate and is the whole point of splitting the counters.
#
# 1. A table defect wins outright. "I could not read my own instructions" is
#    never a graded result about the tree, and its exit code must not depend on
#    how many other lines happened to parse.
if [ "$parse_errors" -ne 0 ]; then
  fail_closed \
    "$parse_errors unparseable line(s) in $TABLE" \
    "Those invariants were NOT evaluated. Fix the table; do not read the" \
    "remaining $evaluated 'ok' line(s) as coverage of the whole table."
fi

# 2. Nothing evaluated is also "no evidence", not "no problems".
if [ "$evaluated" -eq 0 ]; then
  fail_closed \
    "ZERO invariants were evaluated although $listed line(s) were listed" \
    "Refusing to report PASS: nothing was actually checked."
fi

# 3. Only now can a violation be a statement about the tree.
if [ "$violations" -ne 0 ]; then
  echo "RESULT: FAIL ($violations violation(s))"
  exit 1
fi

echo "RESULT: PASS ($evaluated invariant(s) hold)"
exit 0

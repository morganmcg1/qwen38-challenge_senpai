#!/bin/bash
# verify-kernel-table.sh -- name the QMV dispatch table, and the weight-stream
# boundaries it implies, FOR A NAMED TREE.
#
# Why this exists
# ---------------
# On 2026-08-19 I told a student "the 1->2 weight-stream boundary is at 4->5,
# not 5->6" and sent him to probe a location where, ON HIS TREE, both sides were
# single-stream. The boundary I quoted was real -- at the advisor tip. His
# merge-base was 04ad6bf1, inside the window where E27 had set M5=5/M9=5 and
# raised the wide helper's bound to NA<=5. Two trees, two tables, and I quoted
# one while reasoning about the other.
#
# The same defect had already fired twice that day: origin/main looked frozen
# because the fetch refspec made it incapable of changing, and a merged result
# was read as describing our tree because the merge was inert on the scored
# surface. "The merge changes nothing" and "the measurement describes our tree"
# are different claims, and for an inert merge they can have OPPOSITE answers:
# an inert merge is exactly the case where the student's numbers came from the
# student's base, not from ours.
#
# So: never quote a boundary, an IPG, or a register ceiling without its tree.
# This script makes that mechanical.
#
# Modes
#   selftest              pinned trees with known tables
#   table [REV...]        report the table for each REV (default: HEAD)
#   students              merge-base table for every origin/qwen-* branch,
#                         compared against the advisor branch tip
#
# Exit status is nonzero if a requested comparison DIVERGES, so this can gate.

set -u

HEADER='Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h'
TWIN='Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp'
ADVISOR_REF='senpai/qwen38-mtp-r1'

FAILURES=0

die() { printf 'FATAL %s\n' "$*" >&2; exit 3; }

# ceil(a/b) in integer arithmetic
ceil_div() {
  if [ "$2" -le 0 ]; then echo 0; return; fi
  echo $(( ( $1 + $2 - 1 ) / $2 ))
}

# ipg_table REV -> "3 4 5 3 4 4 5" for M=3..9, or empty on failure.
# Reads the explicit template arguments of the >=4096 tier switch. A width whose
# case is absent prints "-" so a missing cell can never be silently read as a
# number.
ipg_table() {
  local rev="$1" body m ipg out=''
  body=$(git show "${rev}:${HEADER}" 2>/dev/null) || return 1
  [ -n "$body" ] || return 1
  for m in 3 4 5 6 7 8 9; do
    ipg=$(printf '%s\n' "$body" \
      | grep -o "qmv_fast_crossrow_affine4_g64_m<T, ${m}, [0-9]" \
      | head -1 | sed 's/.*, //')
    [ -n "$ipg" ] || ipg='-'
    out="${out}${ipg} "
  done
  printf '%s' "${out% }"
}

# na_bound REV -> the wide helper's compile-time NA ceiling, or "-"
na_bound() {
  local rev="$1" v
  v=$(git show "${rev}:${HEADER}" 2>/dev/null \
    | grep -o 'static_assert(NA >= 2 && NA <= [0-9]' | head -1 | sed 's/.*<= //')
  [ -n "$v" ] || v='-'
  printf '%s' "$v"
}

# narrow_ipg REV -> inputs_per_group hard-coded in the sub-4096 tier helper.
# This tier is table-INVARIANT across the E27 revert, which matters: it dilutes
# any stream effect but cannot relocate one.
narrow_ipg() {
  local rev="$1" v
  v=$(git show "${rev}:${HEADER}" 2>/dev/null \
    | grep -o 'constexpr int inputs_per_group = [0-9]' | head -1 | sed 's/.*= //')
  [ -n "$v" ] || v='-'
  printf '%s' "$v"
}

# streams_of "3 4 5 3 4 4 5" -> "1 1 1 2 2 2 2"
streams_of() {
  local m=3 ipg out=''
  for ipg in $1; do
    if [ "$ipg" = '-' ]; then
      out="${out}- "
    else
      out="${out}$(ceil_div "$m" "$ipg") "
    fi
    m=$(( m + 1 ))
  done
  printf '%s' "${out% }"
}

# boundaries_of "1 1 1 2 2 2 2" -> "5->6"
boundaries_of() {
  local m=3 s prev='' out=''
  for s in $1; do
    if [ -n "$prev" ] && [ "$s" != "$prev" ] && [ "$s" != '-' ] && [ "$prev" != '-' ]; then
      out="${out}$(( m - 1 ))->${m} "
    fi
    prev="$s"
    m=$(( m + 1 ))
  done
  [ -n "$out" ] || out='(none in M=3..9)'
  printf '%s' "${out% }"
}

# twin_agrees REV -> yes/no/unknown. The metallib is built from the generated
# twin; a twin that disagrees with the header means the source you read is not
# the source that ran.
twin_agrees() {
  local rev="$1" a b
  a=$(git show "${rev}:${HEADER}" 2>/dev/null \
    | grep -o 'qmv_fast_crossrow_affine4_g64_m<T, [0-9], [0-9]')
  b=$(git show "${rev}:${TWIN}" 2>/dev/null \
    | grep -o 'qmv_fast_crossrow_affine4_g64_m<T, [0-9], [0-9]')
  if [ -z "$a" ] || [ -z "$b" ]; then printf 'unknown'; return; fi
  if [ "$a" = "$b" ]; then printf 'yes'; else printf 'no'; fi
}

report_one() {
  local rev="$1" label="${2:-}" ipg streams bounds na nipg twin short
  short=$(git rev-parse --short "$rev" 2>/dev/null) || { printf 'unresolvable rev %s\n' "$rev"; return 1; }
  ipg=$(ipg_table "$rev") || { printf '%s  no %s in tree\n' "$short" "$HEADER"; return 1; }
  streams=$(streams_of "$ipg")
  bounds=$(boundaries_of "$streams")
  na=$(na_bound "$rev")
  nipg=$(narrow_ipg "$rev")
  twin=$(twin_agrees "$rev")
  printf '%s %s\n' "$short" "$label"
  printf '    M           3 4 5 6 7 8 9\n'
  printf '    IPG         %s\n' "$ipg"
  printf '    streams     %s\n' "$streams"
  printf '    boundaries  %s\n' "$bounds"
  printf '    NA ceiling  %s        sub-4096 tier IPG %s (table-invariant)\n' "$na" "$nipg"
  printf '    twin==hdr   %s\n' "$twin"
  if [ "$twin" = 'no' ]; then
    printf '    WARN  generated twin disagrees with the header at this rev;\n'
    printf '          the metallib is built from the twin, so the header is not what ran.\n'
  fi
  return 0
}

cmd_table() {
  local rev
  if [ $# -eq 0 ]; then set -- HEAD; fi
  for rev in "$@"; do
    report_one "$rev" '' || FAILURES=$(( FAILURES + 1 ))
    printf '\n'
  done
}

cmd_students() {
  local tip tip_ipg tip_na br mb mb_ipg mb_na n=0 diverged=0
  tip=$(git rev-parse "$ADVISOR_REF" 2>/dev/null) \
    || tip=$(git rev-parse "origin/$ADVISOR_REF" 2>/dev/null) \
    || die "cannot resolve advisor ref $ADVISOR_REF"
  tip_ipg=$(ipg_table "$tip") || die "cannot read table at advisor tip"
  tip_na=$(na_bound "$tip")
  printf 'advisor tip %s\n' "$(git rev-parse --short "$tip")"
  printf '    IPG %s   NA<=%s   streams %s   boundaries %s\n\n' \
    "$tip_ipg" "$tip_na" "$(streams_of "$tip_ipg")" "$(boundaries_of "$(streams_of "$tip_ipg")")"

  # NB: a for-each-ref pattern does not match across '/', and every student
  # branch has a slash in it, so 'refs/remotes/origin/qwen-*' silently matched
  # nothing and this gate reported PASS over an empty set. Enumerate and filter.
  for br in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin \
              | grep '^origin/qwen-'); do
    mb=$(git merge-base "$tip" "$br" 2>/dev/null) || continue
    mb_ipg=$(ipg_table "$mb") || continue
    mb_na=$(na_bound "$mb")
    n=$(( n + 1 ))
    if [ "$mb_ipg" = "$tip_ipg" ] && [ "$mb_na" = "$tip_na" ]; then
      printf 'ok        %s\n              merge-base %s table matches tip\n' \
        "$br" "$(git rev-parse --short "$mb")"
    else
      diverged=$(( diverged + 1 ))
      printf 'DIVERGED  %s\n' "$br"
      printf '              merge-base %s\n' "$(git rev-parse --short "$mb")"
      printf '              IPG %s   NA<=%s\n' "$mb_ipg" "$mb_na"
      printf '              streams %s   boundaries %s\n' \
        "$(streams_of "$mb_ipg")" "$(boundaries_of "$(streams_of "$mb_ipg")")"
      printf '              => per-width times from this branch are NOT comparable\n'
      printf '                 with the tip, at ANY width: one shared register\n'
      printf '                 allocation is taken as the max over all cells, so a\n'
      printf '                 different NA ceiling moves occupancy everywhere.\n'
    fi
  done
  printf '\n%d branches examined, %d diverged from the tip table\n' "$n" "$diverged"
  # Fail closed. A gate that examined nothing has not agreed with anything, and
  # the first version of this script reported PASS over an empty set because the
  # ref pattern never matched. "The gate did not refuse" is not "the gate ran".
  if [ "$n" -eq 0 ]; then
    printf 'FAIL  examined zero student branches; either none are fetched or the\n'
    printf '      enumeration is broken. Fetch them explicitly -- note that\n'
    printf '      remote.origin.fetch in this checkout is scoped to the advisor\n'
    printf '      branch alone, so student refs are absent until asked for:\n'
    printf "        git fetch origin '+refs/heads/qwen-*:refs/remotes/origin/qwen-*'\n"
    FAILURES=$(( FAILURES + 1 ))
    return
  fi
  [ "$diverged" -eq 0 ] || FAILURES=$(( FAILURES + 1 ))
}

# ---------------------------------------------------------------- selftest ----
# Pinned trees. These are commits on our own branch whose tables I have read by
# hand from the source, so a change in this script's parsing shows up as a
# mismatch rather than as a plausible-looking new number.
SELF_T1='04ad6bf11437c269df85a47e91faa769c74fe6da'   # inside the E27 window
SELF_T2='efff400c1b5554be2e8993b01856653d55de7664'   # after the E27 revert
SELF_T3='527306761f70e2c4024f347915328894db80c181'   # the old campaign base

expect() {
  local what="$1" got="$2" want="$3"
  if [ "$got" = "$want" ]; then
    printf 'ok    %s = %s\n' "$what" "$got"
  else
    printf 'FAIL  %s\n        got  %s\n        want %s\n' "$what" "$got" "$want"
    FAILURES=$(( FAILURES + 1 ))
  fi
}

cmd_selftest() {
  local t

  # --- pure arithmetic, independent of the repo -----------------------------
  expect 'ceil_div 5 3'  "$(ceil_div 5 3)"  '2'
  expect 'ceil_div 5 5'  "$(ceil_div 5 5)"  '1'
  expect 'ceil_div 9 3'  "$(ceil_div 9 3)"  '3'
  expect 'ceil_div 9 5'  "$(ceil_div 9 5)"  '2'
  expect 'ceil_div 4 4'  "$(ceil_div 4 4)"  '1'
  expect 'ceil_div 8 4'  "$(ceil_div 8 4)"  '2'

  # The two rules that have been confused all campaign. Both are "read from
  # source"; they disagree at M=5 and M=9, which is the entire story.
  expect 'streams under the E27 table'     "$(streams_of '3 4 5 3 4 4 5')" '1 1 1 2 2 2 2'
  expect 'streams under the shipped table' "$(streams_of '3 4 3 3 4 4 3')" '1 1 2 2 2 2 3'
  expect 'boundary under E27 table'        "$(boundaries_of '1 1 1 2 2 2 2')" '5->6'
  expect 'boundary under shipped table'    "$(boundaries_of '1 1 2 2 2 2 3')" '4->5 8->9'

  # A missing cell must stay "-" and never become a number.
  expect 'missing cell stays unknown'      "$(streams_of '3 4 - 3 4 4 3')" '1 1 - 2 2 2 3'
  expect 'unknown suppresses a boundary'   "$(boundaries_of '1 1 - 2 2 2 3')" '8->9'
  expect 'flat table has no boundary'      "$(boundaries_of '1 1 1 1 1 1 1')" '(none in M=3..9)'

  # --- pinned trees ---------------------------------------------------------
  for t in "$SELF_T1" "$SELF_T2" "$SELF_T3"; do
    if ! git cat-file -e "${t}^{commit}" 2>/dev/null; then
      printf 'FAIL  pinned commit %s is not in this repo\n' "$t"
      FAILURES=$(( FAILURES + 1 ))
      continue
    fi
  done

  expect 'E27-window table (04ad6bf1)'  "$(ipg_table "$SELF_T1")" '3 4 5 3 4 4 5'
  expect 'E27-window NA ceiling'        "$(na_bound  "$SELF_T1")" '5'
  expect 'post-revert table (efff400c)' "$(ipg_table "$SELF_T2")" '3 4 3 3 4 4 3'
  expect 'post-revert NA ceiling'       "$(na_bound  "$SELF_T2")" '4'
  expect 'old campaign base (5273067)'  "$(ipg_table "$SELF_T3")" '3 4 3 3 4 4 3'
  expect 'old campaign base NA ceiling' "$(na_bound  "$SELF_T3")" '4'

  # The pinned pair must actually DIFFER, otherwise this whole gate is vacuous
  # and would pass even if parsing returned a constant.
  if [ "$(ipg_table "$SELF_T1")" = "$(ipg_table "$SELF_T2")" ]; then
    printf 'FAIL  pinned trees are indistinguishable; the gate would be vacuous\n'
    FAILURES=$(( FAILURES + 1 ))
  else
    printf 'ok    pinned trees differ, so a constant-returning parser would fail\n'
  fi

  # The sub-4096 tier must be table-invariant across the revert.
  expect 'sub-4096 IPG at E27 window'  "$(narrow_ipg "$SELF_T1")" '2'
  expect 'sub-4096 IPG post-revert'    "$(narrow_ipg "$SELF_T2")" '2'

  # Twin/header agreement at the current tip.
  expect 'twin agrees with header at HEAD' "$(twin_agrees HEAD)" 'yes'
}

usage() {
  printf 'usage: %s {selftest | table [REV...] | students}\n' "$(basename "$0")"
}

case "${1:-table}" in
  selftest) shift; cmd_selftest "$@" ;;
  table)    shift; cmd_table "$@" ;;
  students) shift; cmd_students "$@" ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf 'PASS kernel-table-gate\n'
  exit 0
fi
printf 'FAIL kernel-table-gate (%d)\n' "$FAILURES"
exit 1

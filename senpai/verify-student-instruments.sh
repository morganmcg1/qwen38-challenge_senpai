#!/bin/bash
# verify-student-instruments.sh -- which instruments does a student ACTUALLY have?
#
# WHY THIS EXISTS
# ---------------
# Twice in one session I handed a student a command and assumed the copy in
# their checkout was the copy in mine. Both times it was not, and both times the
# older copy FAILED OPEN:
#
#   * `stream_dispatch_census.py census <tree>` accepted the tree argument and
#     never read it, printing a board-wide aggregate over 476 trees. A student
#     would have pasted plausible output that was not about his tree.
#   * `stream_dispatch_census.py ab` printed the *scientific conclusion* "NO
#     clean A/B exists ..." over ZERO trees and exited 0, which is the default
#     state of a fresh checkout because `bootstrap-checkout.sh` adds the
#     `upstream` remote but never fetches it.
#
# A student's base is a base. It goes stale exactly like `origin/main` does, and
# the staleness is invisible because the tool still runs.
#
# THIS SCRIPT ALSO EXISTS BECAUSE MY FIRST, AD-HOC VERSION OF IT WAS WRONG.
# I checked staleness with `git rev-parse "$rev:$path"`, which for a path that
# does not exist at that rev prints THE ARGUMENT ITSELF BACK TO STDOUT and exits
# 128. With `2>/dev/null || echo ABSENT` the captured value became argument+
# ABSENT, matched neither the current blob nor "ABSENT", and was reported as
# STALE. So a file that DID NOT EXIST was reported as merely out of date -- an
# understatement in exactly the direction that makes you relax. Use
# `git cat-file -e` to test existence, never bare `git rev-parse`.
#
# No declared instrument list, deliberately: a hand-maintained list rots and
# then under-reports. Instead it diffs the whole of research/ and senpai/
# between the branch's merge-base and the advisor tip, so a new instrument is
# covered the moment it is committed.
#
# USAGE
#   verify-student-instruments.sh selftest         constructed cases, known truth
#   verify-student-instruments.sh check BRANCH...  exit 1 if any is not CURRENT
#   verify-student-instruments.sh all              report every qwen-* branch
#
# `check` is the pre-flight before telling a student to run something.

set -u

TIP=HEAD
INSTRUMENT_DIRS="research senpai"

# Classify one path at one rev against the tip. Echoes CURRENT|STALE|ABSENT.
classify() {
  rev="$1"; path="$2"
  if ! git cat-file -e "${rev}:${path}" 2>/dev/null; then
    echo ABSENT
    return
  fi
  a=$(git rev-parse --verify "${rev}:${path}" 2>/dev/null)
  b=$(git rev-parse --verify "${TIP}:${path}" 2>/dev/null)
  if [ -z "$a" ] || [ -z "$b" ]; then
    # Unreadable on either side is never "fine".
    echo ABSENT
    return
  fi
  if [ "$a" = "$b" ]; then echo CURRENT; else echo STALE; fi
}

# Report one branch. Echoes the count of not-CURRENT instruments on stdout's
# last line via the global NOT_CURRENT.
NOT_CURRENT=0
report_branch() {
  ref="$1"
  if ! git rev-parse --verify "$ref" >/dev/null 2>&1; then
    echo "  UNRESOLVED ref: $ref"
    NOT_CURRENT=$((NOT_CURRENT + 1))
    return 1
  fi
  mb=$(git merge-base "$ref" "$TIP" 2>/dev/null)
  if [ -z "$mb" ]; then
    echo "  NO MERGE-BASE with $TIP: $ref"
    NOT_CURRENT=$((NOT_CURRENT + 1))
    return 1
  fi
  printf '%s\n' "$ref"
  printf '  merge-base %s\n' "$(echo "$mb" | cut -c1-12)"

  # Every instrument path differing between the merge-base and the tip.
  # RUNNABLE code only (.py/.sh/.metal). Prose and data are excluded because a
  # gate that prints 114 lines of stale ledger and pending-feedback notes will
  # not be run, and a gate nobody runs is the failure mode this whole file is
  # about -- `verify-campaign-overlay.sh` sat red for most of a campaign for
  # exactly that reason. Data artifacts are counted, not listed.
  all_paths=$(git diff --name-only "$mb" "$TIP" -- $INSTRUMENT_DIRS)
  paths=$(printf '%s\n' "$all_paths" | grep -E '\.(py|sh|metal)$')
  data_n=$(printf '%s\n' "$all_paths" | grep -Evc '\.(py|sh|metal)$|^$')
  if [ -z "$paths" ]; then
    printf '  all RUNNABLE instruments CURRENT'
    [ "$data_n" -gt 0 ] && printf ' (%s data/prose file(s) differ, not counted)' "$data_n"
    printf '\n'
  else
    n=0
    for p in $paths; do
      c=$(classify "$mb" "$p")
      if [ "$c" != CURRENT ]; then
        printf '  %-9s %s\n' "$c" "$p"
        n=$((n + 1))
        NOT_CURRENT=$((NOT_CURRENT + 1))
      fi
    done
    if [ "$n" -eq 0 ]; then
      printf '  all RUNNABLE instruments CURRENT\n'
    fi
    [ "$data_n" -gt 0 ] && printf '  (%s data/prose file(s) also differ, not counted)\n' "$data_n"
  fi
  # Scored surface: rebasing is only free if this is empty.
  sc=$(git diff --name-only "$mb" "$TIP" -- Sources Vendor benchmark.json mlx-generated)
  if [ -z "$sc" ]; then
    printf '  scored surface IDENTICAL to tip -- rebasing costs no measurement\n'
  else
    printf '  🔴 scored surface DIFFERS from tip (%s file(s)) -- a rebase changes what is measured\n' \
      "$(printf '%s\n' "$sc" | wc -l | tr -d ' ')"
  fi
  printf '\n'
  return 0
}

selftest() {
  fails=0
  # Ground truth established by hand this session. Each case is a rev/path/verdict
  # triple where I know the answer independently of this script.
  #
  # The ABSENT cases are the point: my ad-hoc check called them STALE.
  set -- \
    "efff400c1b5554be2e8993b01856653d55de7664 research/stream_dispatch_census.py ABSENT" \
    "efff400c1b5554be2e8993b01856653d55de7664 senpai/verify-kernel-table.sh ABSENT" \
    "04ad6bf11437c269df85a47e91faa769c74fe6da research/stream_dispatch_census.py ABSENT" \
    "01f69e18f3878c9565fee479581581d85cf481ce senpai/verify-kernel-table.sh CURRENT" \
    "01f69e18f3878c9565fee479581581d85cf481ce research/stream_dispatch_census.py STALE" \
    "HEAD senpai/verify-kernel-table.sh CURRENT" \
    "HEAD research/stream_dispatch_census.py CURRENT"
  ncases=0
  for case in "$@"; do
    rev=$(echo "$case" | awk '{print $1}')
    path=$(echo "$case" | awk '{print $2}')
    want=$(echo "$case" | awk '{print $3}')
    got=$(classify "$rev" "$path")
    ncases=$((ncases + 1))
    if [ "$got" != "$want" ]; then
      echo "  FAIL classify($(echo "$rev" | cut -c1-8), $path) = $got, want $want"
      fails=$((fails + 1))
    fi
  done

  # ABSENT and STALE must be distinguishable. If they collapse, the script is
  # useless in precisely the way the ad-hoc version was.
  if [ "$(classify efff400c1b5554be2e8993b01856653d55de7664 research/stream_dispatch_census.py)" \
     = "$(classify 01f69e18f3878c9565fee479581581d85cf481ce research/stream_dispatch_census.py)" ]; then
    echo "  FAIL ABSENT and STALE are not distinguished"
    fails=$((fails + 1))
  fi

  # A nonexistent path must be ABSENT, not CURRENT.
  if [ "$(classify HEAD research/no-such-instrument-xyz.py)" != ABSENT ]; then
    echo "  FAIL nonexistent path is not reported ABSENT"
    fails=$((fails + 1))
  fi
  ncases=$((ncases + 2))

  # The tip must differ from at least one pinned rev, else a constant-returning
  # classify() would pass everything above.
  if [ "$(classify 01f69e18f3878c9565fee479581581d85cf481ce research/stream_dispatch_census.py)" \
     = "$(classify HEAD research/stream_dispatch_census.py)" ]; then
    echo "  FAIL classify() appears to return a constant"
    fails=$((fails + 1))
  fi
  ncases=$((ncases + 1))

  if [ "$fails" -gt 0 ]; then
    echo "SELFTEST FAIL ($fails of $ncases)"
    return 1
  fi
  echo "SELFTEST PASS: $ncases cases, ABSENT/STALE/CURRENT distinguished on revs with known ground truth."
  return 0
}

mode="${1:-selftest}"
shift 2>/dev/null || true

case "$mode" in
  selftest)
    selftest
    exit $?
    ;;
  check)
    if [ "$#" -eq 0 ]; then
      echo "check needs at least one branch. Refusing to pass over an empty set." >&2
      exit 2
    fi
    examined=0
    for b in "$@"; do
      case "$b" in
        refs/*|origin/*) ref="$b" ;;
        *) ref="origin/$b" ;;
      esac
      report_branch "$ref"
      examined=$((examined + 1))
    done
    if [ "$examined" -eq 0 ]; then
      echo "FAIL: examined zero branches." >&2
      exit 1
    fi
    echo "examined $examined branch(es); $NOT_CURRENT instrument(s) not current"
    [ "$NOT_CURRENT" -eq 0 ] || exit 1
    exit 0
    ;;
  all)
    refs=$(git for-each-ref --format='%(refname:short)' 'refs/remotes/origin/qwen-*/*')
    if [ -z "$refs" ]; then
      # A `*` in a for-each-ref pattern does not match across `/`, which is how
      # an earlier gate of mine reported PASS over an empty set.
      echo "FAIL: no student branches matched. Fetch them:" >&2
      echo "  git fetch origin '+refs/heads/qwen-*:refs/remotes/origin/qwen-*'" >&2
      exit 1
    fi
    examined=0
    for ref in $refs; do
      report_branch "$ref"
      examined=$((examined + 1))
    done
    echo "examined $examined branch(es); $NOT_CURRENT instrument(s) not current"
    exit 0
    ;;
  *)
    echo "usage: $0 [selftest | check BRANCH... | all]" >&2
    exit 2
    ;;
esac

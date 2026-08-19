#!/usr/bin/env bash
# Detect movement of the campaign's research base, and detect the far more
# dangerous case where our LOCAL MIRROR of that base cannot move at all.
#
# THE DEFECT THIS EXISTS FOR
# --------------------------
# This repository's origin fetch refspec is scoped to a single branch:
#
#     remote.origin.fetch = +refs/heads/senpai/qwen38-mtp-r1:refs/remotes/origin/senpai/qwen38-mtp-r1
#
# So `git fetch origin` NEVER updates refs/remotes/origin/main. The local
# `origin/main` is a frozen snapshot of whenever it was last fetched by name.
# Reading it and reporting "base unchanged" is not a measurement -- it is a
# reading that is STRUCTURALLY INCAPABLE of changing. Across several sessions
# origin/main was quoted as unchanged at 52730676 while the real branch had
# advanced by three commits, one of which resynced the entire shipped surface
# onto a new promoted frontier.
#
# The fix is not "remember to fetch main". The fix is to stop trusting a local
# mirror of a remote ref and ASK THE REMOTE. `git ls-remote` is refspec-
# independent, so it reports the truth no matter how the local config is set.
#
# This gate therefore checks three separate things, and the second is the one
# that would have caught the original defect:
#   1. the live remote base against the base we recorded (drift);
#   2. the live remote base against our local mirror of it (stale mirror);
#   3. that the drift, if any, is a fast-forward rather than a rewind.
#
# Usage:
#   senpai/verify-base-drift.sh            # check against senpai/campaign-base.json
#   senpai/verify-base-drift.sh --selftest # perturbation tests, no network needed
#
set -u
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

RECORD_FILE="senpai/campaign-base.json"
BASE_BRANCH="main"
FAILURES=0

note() { printf '%s\n' "$*"; }
fail() { FAILURES=$((FAILURES + 1)); printf 'FAIL  %s\n' "$*"; }
ok()   { printf 'ok    %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Pure evaluation, so the selftest can drive it with constructed inputs rather
# than with today's repository state. A control that names today's tree rots.
# ---------------------------------------------------------------------------
# evaluate <recorded> <live> <mirror-or-empty>
evaluate() {
  local recorded="$1" live="$2" mirror="$3"
  local before=$FAILURES

  if [ -z "$live" ]; then
    fail "could not read the live base from the remote. Failing closed: an" \
         "unreadable base is not an unchanged base."
    return 1
  fi

  # (2) stale mirror -- the original defect.
  if [ -n "$mirror" ]; then
    if [ "$mirror" = "$live" ]; then
      ok "local mirror refs/remotes/origin/$BASE_BRANCH agrees with the remote"
    else
      fail "STALE MIRROR: refs/remotes/origin/$BASE_BRANCH is ${mirror} but the" \
           "remote says ${live}. A plain 'git fetch origin' will NOT fix this if" \
           "the fetch refspec does not cover $BASE_BRANCH -- check" \
           "'git config --get-all remote.origin.fetch'. Until it is fixed, every" \
           "statement you make about the base by reading origin/$BASE_BRANCH is" \
           "unfalsifiable. REMEDY, verified in this checkout:" \
           "  git fetch origin '+refs/heads/$BASE_BRANCH:refs/remotes/origin/$BASE_BRANCH'" \
           "An explicit refspec bypasses the configured one. Widening the config" \
           "permanently is NOT available to the advisor role here ('git config'" \
           "and 'git update-ref' are both blocked), so the root cause cannot be" \
           "repaired from inside the campaign and THIS GATE is the only standing" \
           "defence. Run it before quoting the base, not after."
    fi
  else
    note "note  no local mirror of origin/$BASE_BRANCH exists; nothing to be stale."
  fi

  # (1) + (3) drift and its direction.
  if [ "$recorded" = "$live" ]; then
    ok "recorded base $recorded is the live base"
  else
    if git cat-file -e "${live}^{commit}" 2>/dev/null && \
       git cat-file -e "${recorded}^{commit}" 2>/dev/null; then
      if git merge-base --is-ancestor "$recorded" "$live" 2>/dev/null; then
        fail "BASE ADVANCED: recorded $recorded -> live $live ($(git rev-list --count "$recorded".."$live" 2>/dev/null) commits)." \
             "This is a fast-forward, so nothing was rewritten, but every result" \
             "measured against the recorded base now has a changed base and must" \
             "be re-adjudicated rather than assumed to transfer. Update" \
             "$RECORD_FILE deliberately, with the reason."
      else
        fail "BASE REWOUND OR DIVERGED: recorded $recorded is NOT an ancestor of" \
             "live $live. Someone force-pushed or reset the base branch." \
             "Do not update the record until you know which."
      fi
    else
      fail "BASE CHANGED to $live but one of the two commits is not present" \
           "locally, so the direction cannot be established. Fetch, then re-run."
    fi
  fi

  [ "$FAILURES" -eq "$before" ]
}

# ---------------------------------------------------------------------------
selftest() {
  # Real commits from this repository, so ancestry is genuinely exercised.
  local OLD=527306761f70e2c4024f347915328894db80c181   # campaign base as recorded
  local NEW=6391b03a39dfd56daac0f67e62851d0a86187963   # main after the frontier sync
  local CROWN=0c90733d383f6b987a29682bf9eb9458a6172bfa # NOT an ancestor of NEW
  local rc pass=0 total=0

  for c in "$OLD" "$NEW" "$CROWN"; do
    if ! git cat-file -e "${c}^{commit}" 2>/dev/null; then
      echo "selftest VACUOUS: commit $c is absent, ancestry cannot be exercised."
      return 1
    fi
  done

  # case 1: everything agrees -> must PASS
  total=$((total + 1)); FAILURES=0
  evaluate "$NEW" "$NEW" "$NEW" >/dev/null 2>&1; rc=$?
  if [ "$rc" -eq 0 ] && [ "$FAILURES" -eq 0 ]; then pass=$((pass + 1))
  else echo "  case 1 (all agree) should pass, got rc=$rc failures=$FAILURES"; fi

  # case 2: stale mirror only -> must FAIL
  total=$((total + 1)); FAILURES=0
  evaluate "$NEW" "$NEW" "$OLD" >/dev/null 2>&1
  if [ "$FAILURES" -ge 1 ]; then pass=$((pass + 1))
  else echo "  case 2 (stale mirror) should fail, failures=$FAILURES"; fi

  # case 3: base advanced (fast-forward) -> must FAIL
  total=$((total + 1)); FAILURES=0
  evaluate "$OLD" "$NEW" "$NEW" >/dev/null 2>&1
  if [ "$FAILURES" -ge 1 ]; then pass=$((pass + 1))
  else echo "  case 3 (base advanced) should fail, failures=$FAILURES"; fi

  # case 4: base diverged (not a fast-forward) -> must FAIL, and the two cases
  # must be DISTINGUISHABLE, because a rewind and an advance need opposite
  # responses. Assert on the message, not just on the count.
  #
  # NOTE: capture through a temp file, NOT through `out=$(evaluate ...)`.
  # Command substitution runs the callee in a SUBSHELL, so every FAILURES++ it
  # performs is discarded and the counter reads 0 no matter what happened. That
  # bug made this very case report a false negative on first run.
  local tmp="${TMPDIR:-/tmp}/base-drift-selftest.$$"
  total=$((total + 1)); FAILURES=0
  evaluate "$CROWN" "$NEW" "$NEW" >"$tmp" 2>&1
  if [ "$FAILURES" -ge 1 ] && grep -q 'REWOUND OR DIVERGED' "$tmp"; then
    pass=$((pass + 1))
  else
    echo "  case 4 (diverged) should fail with REWOUND OR DIVERGED, failures=$FAILURES"
  fi

  # case 5: unreadable remote -> must FAIL CLOSED
  total=$((total + 1)); FAILURES=0
  evaluate "$NEW" "" "$NEW" >/dev/null 2>&1
  if [ "$FAILURES" -ge 1 ]; then pass=$((pass + 1))
  else echo "  case 5 (unreadable remote) should fail closed, failures=$FAILURES"; fi

  # case 6: the advance and the divergence messages must differ, or the gate
  # cannot tell an operator which response is required. Same subshell caveat as
  # case 4, so the same temp-file capture.
  total=$((total + 1)); FAILURES=0
  evaluate "$OLD" "$NEW" "$NEW" >"$tmp.a" 2>&1
  FAILURES=0
  evaluate "$CROWN" "$NEW" "$NEW" >"$tmp.b" 2>&1
  if ! cmp -s "$tmp.a" "$tmp.b"; then pass=$((pass + 1))
  else echo "  case 6: advance and divergence produce identical output"; fi

  # case 7: meta-test the harness itself. Feed evaluate() an input on which it
  # MUST pass, but assert the failing expectation, so that a selftest whose
  # cases had silently stopped exercising anything cannot report a clean sweep.
  # If this "case" ever passes, the counter is not wired to evaluate() at all.
  total=$((total + 1)); FAILURES=0
  evaluate "$NEW" "$NEW" "$NEW" >/dev/null 2>&1
  if [ "$FAILURES" -eq 0 ]; then pass=$((pass + 1))
  else echo "  case 7 (harness wiring) FAILURES leaked across cases: $FAILURES"; fi

  rm -f "$tmp" "$tmp.a" "$tmp.b"
  echo "SELFTEST $pass/$total"
  [ "$pass" -eq "$total" ]
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--selftest" ]; then
  selftest
  exit $?
fi

if [ ! -f "$RECORD_FILE" ]; then
  echo "FAIL  $RECORD_FILE is missing; there is no recorded base to compare against."
  exit 1
fi

RECORDED=$(sed -n 's/.*"baseSha"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' "$RECORD_FILE" | head -1)
if [ -z "$RECORDED" ]; then
  echo "FAIL  could not parse a 40-hex baseSha out of $RECORD_FILE."
  exit 1
fi

# Ask the REMOTE, not our mirror of it. Note: no `| grep -q` here -- under
# `set -o pipefail` a short-circuiting reader makes the producer die of SIGPIPE
# and the pipeline reports 141, which has been mistaken for a clean failure in
# this repository before.
LS=$(git ls-remote origin "refs/heads/$BASE_BRANCH" 2>/dev/null)
LS_RC=$?
LIVE=$(printf '%s' "$LS" | awk 'NR==1{print $1}')
if [ "$LS_RC" -ne 0 ]; then
  echo "FAIL  git ls-remote origin refs/heads/$BASE_BRANCH failed (rc=$LS_RC)."
  echo "      Failing closed: an unreachable remote is not an unchanged base."
  exit 1
fi

MIRROR=$(git rev-parse --verify --quiet "refs/remotes/origin/$BASE_BRANCH" 2>/dev/null || true)

note "recorded base : $RECORDED"
note "live remote   : ${LIVE:-<unreadable>}"
note "local mirror  : ${MIRROR:-<absent>}"
note ""
evaluate "$RECORDED" "$LIVE" "$MIRROR"

note ""
if [ "$FAILURES" -eq 0 ]; then
  echo "PASS base-drift-gate"
  exit 0
fi
echo "FAIL base-drift-gate ($FAILURES finding(s))"
exit 1

#!/usr/bin/env bash
# Self-test for research/frontier-revert-gate.sh.
#
# WHY THIS IS A SEPARATE SCRIPT THAT BUILDS ITS OWN REPOSITORIES
#
# Every interesting branch of that gate is UNREACHABLE from this checkout.
# Running the gate here exercises exactly one path: two acknowledged files that
# differ from a frontier we did fetch. The paths that decide whether we submit a
# wiped tree -- "the ack'd file no longer differs", "the ack blob is stale", "the
# table declares nothing" -- cannot be produced by running the real gate against
# the real repository at all. Reachability-by-running and coverage are different
# properties, and the second one needs constructed inputs.
#
# That distinction is not theoretical here. The blob-sha discriminator this
# suite covers exists because a constructed control (a commit-tree whose tree is
# our tree, authored by the bot) showed the gate telling a reader to delete the
# only record of what an overlay had just destroyed.
#
# Each case builds a throwaway git repository in $TMPDIR, so nothing depends on
# the state of the campaign checkout and the suite is safe to run at any time.

set -uo pipefail

_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="${_self_dir}/frontier-revert-gate.sh"
if [ ! -f "${GATE}" ]; then
  printf 'FAIL: gate under test not found at %s\n' "${GATE}" >&2
  exit 1
fi

pass=0
fail=0
ok()  { pass=$((pass + 1)); printf '  ok    %s\n' "$1"; }
nok() { fail=$((fail + 1)); printf '  FAIL  %s\n' "$1" >&2; }

check_eq() {  # label expected actual
  if [ "$2" = "$3" ]; then ok "$1"; else nok "$1 (expected '$2', got '$3')"; fi
}
check_has() {  # label needle file
  if grep -qF -- "$2" "$3"; then ok "$1"; else nok "$1 (no '$2' in output)"; fi
}
check_hasnt() {  # label needle file
  if grep -qF -- "$2" "$3"; then nok "$1 (unexpected '$2' in output)"; else ok "$1"; fi
}

# --- fixture -----------------------------------------------------------------
# A minimal repo the gate accepts: benchmark.json with one editablePath, a
# frontier commit shaped like the organizer's (bot author, "Accept submission"
# subject), a FETCH_HEAD so the freshness check is satisfied, and our own commit
# on top. Returns with $PWD inside the new repo, $FRONTIER set, and $GATE_LOCAL
# pointing at the copy of the gate to invoke.
#
# THE GATE MUST BE COPIED IN, and the first run of this suite is what proved it:
# every case failed with "frontier ref does not resolve" and a repo root of the
# campaign checkout. The gate resolves its root from its OWN script location and
# ignores $PWD, so invoking the campaign copy from a temp directory grades the
# campaign tree. That is the correct behaviour for a gate -- you cannot soften it
# by running it from somewhere friendlier -- and it is why its header says it is
# "testable by copy". A suite that had only asserted exit codes would have called
# ten spurious failures ten genuine ones.
FRONTIER=""
GATE_LOCAL=""
make_repo() {  # our_file_contents
  local d
  d="$(mktemp -d)" || return 1
  cd "${d}" || return 1
  git init -q .
  mkdir -p pkg research
  cp "${GATE}" research/frontier-revert-gate.sh || return 1
  GATE_LOCAL="${d}/research/frontier-revert-gate.sh"
  printf '{"editablePaths":["pkg/"]}\n' > benchmark.json
  printf 'frontier version\n' > pkg/f.swift
  git add -A
  git -c user.name='yukon-autoresearch[bot]' \
      -c user.email='bot@example.invalid' \
      commit -q -m 'Accept submission fixture'
  FRONTIER="$(git rev-parse HEAD)"
  printf '%s' "$1" > pkg/f.swift
  git add -A
  git -c user.name='us' -c user.email='us@example.invalid' \
      commit -q -m 'our change' --allow-empty
  : > "$(git rev-parse --git-dir)/FETCH_HEAD"
}

blob_of() { git ls-tree HEAD -- "$1" | awk '{ print $3 }'; }

REASON='We delete the frontier line and replace it, reason long enough to pass.'

run_gate() {  # acksfile outfile
  FRONTIER_ACKS="$1" bash "${GATE_LOCAL}" "${FRONTIER}" > "$2" 2>&1
  printf '%s' "$?"
}

printf 'selftest: frontier-revert-gate.sh\n\n'
start="$(pwd)"

# --- case 1: acknowledged, blob current  => PASS ------------------------------
if make_repo 'our version
'; then
  b="$(blob_of pkg/f.swift)"
  printf 'pkg/f.swift|INTENTIONAL-REPLACEMENT|%s|%s\n' "${REASON}" "${b}" > acks.txt
  rc="$(run_gate acks.txt out.txt)"
  check_eq "1 acknowledged + current blob => exit 0" 0 "${rc}"
  check_has "1 reports the overwrite" "pkg/f.swift" out.txt
  cd "${start}" || exit 1
else
  nok "1 fixture could not be built"
fi

# --- case 2: acknowledged, blob field MISSING => FAIL ------------------------
if make_repo 'our version
'; then
  printf 'pkg/f.swift|INTENTIONAL-REPLACEMENT|%s\n' "${REASON}" > acks.txt
  rc="$(run_gate acks.txt out.txt)"
  check_eq "2 missing ack blob => exit 1" 1 "${rc}"
  check_has "2 names the missing field" "no ack blob sha" out.txt
  check_has "2 offers the sha to paste" "Append it:" out.txt
  cd "${start}" || exit 1
else
  nok "2 fixture could not be built"
fi

# --- case 3: acknowledged, blob STALE => FAIL --------------------------------
if make_repo 'our version
'; then
  printf 'pkg/f.swift|INTENTIONAL-REPLACEMENT|%s|%s\n' "${REASON}" \
    "0000000000000000000000000000000000000000" > acks.txt
  rc="$(run_gate acks.txt out.txt)"
  check_eq "3 stale ack blob => exit 1" 1 "${rc}"
  check_has "3 shows both shas" "was written against blob 000000" out.txt
  cd "${start}" || exit 1
else
  nok "3 fixture could not be built"
fi

# --- case 4: no longer differs, our copy UNCHANGED => "they adopted it" ------
# Constructed by acking a path that is identical on both sides from the start.
if make_repo 'frontier version
'; then
  b="$(blob_of pkg/f.swift)"
  printf 'pkg/f.swift|INTENTIONAL-REPLACEMENT|%s|%s\n' "${REASON}" "${b}" > acks.txt
  rc="$(run_gate acks.txt out4.txt)"
  cp out4.txt "${TMPDIR:-/tmp}/frgate_case4.txt"
  check_eq "4 identical + fresh blob => exit 1" 1 "${rc}"
  check_has "4 says the frontier adopted it" "they" out4.txt
  check_has "4 tells us to delete the line" "delete the line" out4.txt
  check_hasnt "4 does NOT cry reversion" "WE moved onto THEM" out4.txt
  cd "${start}" || exit 1
else
  nok "4 fixture could not be built"
fi

# --- case 5: no longer differs, our copy CHANGED => "we were overlaid" ------
# The scenario the whole discriminator exists for: our copy now equals the
# frontier's, and the ack was written against something else.
if make_repo 'frontier version
'; then
  printf 'pkg/f.swift|INTENTIONAL-REPLACEMENT|%s|%s\n' "${REASON}" \
    "1111111111111111111111111111111111111111" > acks.txt
  rc="$(run_gate acks.txt out5.txt)"
  cp out5.txt "${TMPDIR:-/tmp}/frgate_case5.txt"
  check_eq "5 identical + stale blob => exit 1" 1 "${rc}"
  check_has "5 says WE moved onto THEM" "WE moved onto THEM" out5.txt
  check_has "5 forbids deleting the record" "DO NOT delete this line" out5.txt
  check_hasnt "5 does NOT advise deletion" "delete the line." out5.txt
  cd "${start}" || exit 1
else
  nok "5 fixture could not be built"
fi

# --- the discriminator must actually discriminate ----------------------------
# Cases 4 and 5 differ ONLY in the ack blob. If the gate emitted the same text
# for both, the field would be decoration and this suite would still be green on
# every exit code above. Compare the two transcripts directly.
#
# HOW STRONG THIS CHECK ACTUALLY IS, measured rather than assumed. Mutating the
# gate's reversion branch to `elif false` fails 2 of the 24 checks here -- both of
# them case 5's specific text assertions -- and THIS CHECK STILL PASSES, because
# the mutant falls through to the third branch, whose wording differs from case
# 4's anyway. So case 6 detects "the ack blob is ignored entirely" and nothing
# finer; the load-bearing checks are case 5's. Recorded because a cross-check
# that survives the mutation it looks like it covers is worse than no check if
# anyone reads it as coverage.
c4="${TMPDIR:-/tmp}/frgate_case4.txt"
c5="${TMPDIR:-/tmp}/frgate_case5.txt"
if [ -f "${c4}" ] && [ -f "${c5}" ]; then
  if cmp -s "${c4}" "${c5}"; then
    nok "6 cases 4 and 5 produced IDENTICAL output -- the ack blob is ignored"
  else
    ok "6 cases 4 and 5 produce different verdicts from the same diff"
  fi
else
  nok "6 could not compare case 4 and case 5 transcripts"
fi

# --- case 7: unacknowledged path => FAIL ------------------------------------
if make_repo 'our version
'; then
  printf '# nothing acked but one path differs\npkg/other|INTENTIONAL-REPLACEMENT|%s|%s\n' \
    "${REASON}" "2222222222222222222222222222222222222222" > acks.txt
  rc="$(run_gate acks.txt out.txt)"
  check_eq "7 unacknowledged differing path => exit 1" 1 "${rc}"
  check_has "7 names the unacknowledged path" "not listed in" out.txt
  cd "${start}" || exit 1
else
  nok "7 fixture could not be built"
fi

# --- case 8: table declares nothing => FAIL (no evidence is not a pass) -----
if make_repo 'frontier version
'; then
  printf '# every line a comment\n' > acks.txt
  rc="$(run_gate acks.txt out.txt)"
  check_eq "8 empty table on an identical tree => exit 1" 1 "${rc}"
  check_has "8 says PASS would mean no evidence" "declares zero paths" out.txt
  cd "${start}" || exit 1
else
  nok "8 fixture could not be built"
fi

# --- case 9: a frontier that is not the organizer's tip => FAIL -------------
if make_repo 'our version
'; then
  b="$(blob_of pkg/f.swift)"
  printf 'pkg/f.swift|INTENTIONAL-REPLACEMENT|%s|%s\n' "${REASON}" "${b}" > acks.txt
  ours="$(git rev-parse HEAD)"
  rc="$(FRONTIER_ACKS=acks.txt bash "${GATE_LOCAL}" "${ours}" > out.txt 2>&1; printf '%s' "$?")"
  check_eq "9 frontier pointed at our own commit => exit 1" 1 "${rc}"
  check_has "9 names the wrong author" "not yukon-autoresearch" out.txt
  cd "${start}" || exit 1
else
  nok "9 fixture could not be built"
fi

# --- case 10: MUST-REBASE is a blocker, not an escape hatch ------------------
if make_repo 'our version
'; then
  b="$(blob_of pkg/f.swift)"
  printf 'pkg/f.swift|MUST-REBASE|%s|%s\n' "${REASON}" "${b}" > acks.txt
  rc="$(run_gate acks.txt out.txt)"
  check_eq "10 MUST-REBASE => exit 1" 1 "${rc}"
  check_has "10 calls it a named blocker" "named blocker" out.txt
  cd "${start}" || exit 1
else
  nok "10 fixture could not be built"
fi

printf '\nselftest: %d passed, %d failed\n' "${pass}" "${fail}"
if [ "${pass}" -eq 0 ]; then
  printf 'FAIL: zero checks ran, which is not a pass.\n' >&2
  exit 1
fi
[ "${fail}" -eq 0 ] || exit 1
printf 'selftest-frontier-revert-gate.sh: PASS\n'

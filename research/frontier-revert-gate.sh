#!/usr/bin/env bash
# The fourth baseline, and the only one that can lose us points: what does our
# next submission DELETE from the live leaderboard tip?
#
# WHY THIS EXISTS
#
# `yukon submit` is a whole-file REPLACE overlay, not a merge. That is not an
# assumption; it is refuted-alternative:
#
#   * fkiene's tree 1cb1f43a added a 19-line verify-concat JIT warm to
#     Sources/MLXFastModel/Qwen36MTPBlockSession.swift and scored 3.24418.
#   * ofou branched from 5068eb8d, which PREDATES fkiene, and never opened that
#     file: `git diff 5068eb8d ef42e043` is 2 files, +11/-5, memory policy only.
#   * Yet `git diff 1cb1f43a 0c90733` -- the overlay the organizer actually
#     applied for ofou -- DELETES all 19 of fkiene's lines.
#
# A three-way merge preserves a hunk the author never touched. This did not. So
# every file we package overwrites the tip's copy wholesale, in regions we have
# never read, and a stale checkout silently reverts other people's accepted work.
#
# We have already paid for this twice:
#
#   1. Our scored overlay deleted 17 lines of quantized.h we never wrote,
#      including a 12-line frontier comment. Filed as a curiosity.
#   2. As of this writing HEAD would revert all three hunks of the current crown
#      (MLX_MAX_MB_PER_BUFFER 512 -> 128, setenv overwrite 1 -> 0, full-profile
#      512/50 -> 320/128), whose measured value over our submit base is +0.186 %.
#
# The three existing gates cannot see this. shipped-surface diffs against the
# campaign baseline, inherited-surface against pristine upstream, scored-surface
# against our own last scored tree. All three look BACKWARD at trees we authored
# or inherited. None looks at the tip we are about to overwrite.
#
# WHY EACH ACK CARRIES A BLOB SHA
#
# The first version of this gate had a hole that only shows up in the scenario it
# was written to prevent. Constructed control (a commit-tree whose tree IS our
# tree, authored by the bot with an Accept subject, i.e. "an overlay landed and
# we are now byte-identical to the tip"): the gate correctly exits 1, because the
# anti-rot loop at the bottom notices that the ack'd paths no longer differ. But
# its REMEDIATION TEXT said "Remove the stale entry." Following that advice on a
# tree that had just been wiped turns the gate green and deletes the only written
# record of what we were carrying.
#
# That is the campaign's standing lesson -- ADOPTED-BY-FRONTIER AND REVERTED-BY-
# OVERLAY ARE IDENTICAL IN A DIFF AND NEED OPPOSITE RESPONSES -- landing on the
# one instrument where it decides whether we submit. A diff alone cannot tell
# them apart, because both end at "these two blobs are equal". What tells them
# apart is WHICH SIDE MOVED, and that needs a third point: the blob our reason
# was written against.
#
#   ack blob == HEAD blob   the frontier came to us. Our content is intact; the
#                           entry is genuinely stale and should be deleted.
#   ack blob != HEAD blob   OUR content moved onto theirs. Whatever the reason
#                           says we delete, we no longer delete. Do not delete
#                           the entry: re-read, then rewrite it.
#
# The same field also catches a quieter rot: an ack whose reason names specific
# lines it deletes, written against a version of the file that no longer exists.
# Those two files change only by deliberate scored-surface edit, and a scored-
# surface edit is exactly when you want to be forced to re-read what you delete,
# so this is treated as a hard failure rather than a warning.
#
# Usage:
#   research/frontier-revert-gate.sh [frontier-ref]        (default upstream/main)
#   research/frontier-revert-gate.sh --selftest            (delegates, see below)
# Env:
#   FRONTIER_ACKS          path to the acknowledgement table
#                          (default research/frontier-revert-acks.txt)
#   FRONTIER_MAX_FETCH_AGE seconds of fetch staleness tolerated (default 43200)
#
# Exit 0 only if every packaged file that differs from the frontier is
# acknowledged as an intentional replacement, with a reason of at least 40
# characters and the blob sha that reason was written against. Any unreviewed
# revert is a BLOCK.
#
# Self-test: research/selftest-frontier-revert-gate.sh (constructs its own
# throwaway repositories; the interesting branches are unreachable from this
# checkout, so running the gate here can never exercise them).

set -uo pipefail

# Script-relative first, caller's repo as a fallback, so the gate is testable by
# copy. See the same reasoning in research/scored-surface-gate.sh.
_gate_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_gate_root="$(git -C "${_gate_script_dir}/.." rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${_gate_root}" ]; then
  _gate_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "${_gate_root}" ]; then
  printf 'FAIL: frontier-revert-gate.sh is not inside a git repository and was\n' >&2
  printf '      not invoked from one, so there is no tree to compare.\n' >&2
  exit 1
fi
cd "${_gate_root}" || exit 1

if [ "${1:-}" = "--selftest" ] || [ "${1:-}" = "selftest" ]; then
  exec bash "${_gate_script_dir}/selftest-frontier-revert-gate.sh"
fi

FRONTIER_REF="${1:-upstream/main}"
ACKS="${FRONTIER_ACKS:-research/frontier-revert-acks.txt}"
MAX_FETCH_AGE="${FRONTIER_MAX_FETCH_AGE:-43200}"

fail=0
bad() { printf 'FAIL: %s\n' "$1" >&2; fail=1; }
note() { printf '%s\n' "$1"; }

note "frontier-revert gate"
note "  question: what would our next submission DELETE from the live tip?"
note "  repo root  : ${_gate_root}"
note ""

# --- the frontier must exist -------------------------------------------------
frontier_sha="$(git rev-parse --verify "${FRONTIER_REF}^{commit}" 2>/dev/null || true)"
if [ -z "${frontier_sha}" ]; then
  bad "frontier ref '${FRONTIER_REF}' does not resolve."
  printf '      The leaderboard tip is a real git ref. Get it with:\n\n' >&2
  printf '        git fetch upstream --prune\n\n' >&2
  printf '      Refusing to guess what we would overwrite.\n' >&2
  exit 1
fi

f_subject="$(git log -1 --format=%s "${frontier_sha}")"
f_author="$(git log -1 --format=%an "${frontier_sha}")"
f_date="$(git log -1 --format=%ci "${frontier_sha}")"

# --- it must actually BE the organizer's tip ---------------------------------
# Pinning the sha is wrong (it moves hourly), but pointing this gate at our own
# branch would make it report a comfortable zero. Assert the shape instead.
if [ "${f_author}" != "yukon-autoresearch[bot]" ]; then
  bad "frontier ref '${FRONTIER_REF}' is authored by '${f_author}', not yukon-autoresearch[bot]."
  printf '      That is not the leaderboard tip. A gate pointed at our own branch\n' >&2
  printf '      reports zero reverts and means nothing.\n' >&2
fi
case "${f_subject}" in
  "Accept submission "*|"Validate submission "*) : ;;
  *)
    bad "frontier subject '${f_subject}' is not an Accept/Validate commit."
    ;;
esac

note "  frontier   : ${FRONTIER_REF} = ${frontier_sha}"
note "               ${f_subject}"
note "               ${f_author}, ${f_date}"

# --- and it must be freshly fetched -----------------------------------------
# A frontier we fetched yesterday gives false comfort: the tip has moved and we
# would be diffing against a tree that is no longer anybody's baseline. This is
# a hard failure, not a warning, because the whole output is meaningless without
# it. Measure FETCH time, not commit time: main can legitimately sit still.
git_dir="$(git rev-parse --git-dir)"
fetch_stamp="${git_dir}/FETCH_HEAD"
if [ -f "${fetch_stamp}" ]; then
  now="$(date +%s)"
  mtime="$(stat -f %m "${fetch_stamp}" 2>/dev/null || stat -c %Y "${fetch_stamp}" 2>/dev/null || echo 0)"
  age=$((now - mtime))
  note "  last fetch : ${age}s ago (budget ${MAX_FETCH_AGE}s)"
  if [ "${age}" -gt "${MAX_FETCH_AGE}" ]; then
    bad "the last fetch was ${age}s ago, over the ${MAX_FETCH_AGE}s budget."
    printf '      Run: git fetch upstream --prune\n' >&2
  fi
else
  bad "no ${fetch_stamp}; cannot establish that the frontier is current."
  printf '      Run: git fetch upstream --prune\n' >&2
fi
note ""

# --- which files does a submission actually package? ------------------------
if [ ! -f benchmark.json ]; then
  bad "benchmark.json is missing, so editablePaths is unknown and the set of"
  bad "files a submission packages cannot be determined."
  note "frontier-revert gate: FAIL"
  exit 1
fi

# --- refuse to certify a tree we have not committed -------------------------
# Found the hard way at ledger 162. Every comparison below is `frontier..HEAD`,
# so an uncommitted edit to a packaged file is INVISIBLE to this gate. That is
# precisely the state you are in while reconciling with the tip: I reverted E27
# in the worktree, re-ran the gates, and they certified the OLD tree while
# reporting the new one's repo root. A gate that grades a commit while you are
# about to submit a worktree is not a gate, it is a decoration. So: if any
# packaged path is dirty, fail closed and say which. You cannot certify what you
# have not committed.
dirty="$(git status --porcelain -- $(python3 -c '
import json, shlex
paths = json.load(open("benchmark.json")).get("editablePaths") or []
print(" ".join(shlex.quote(p) for p in paths))
' 2>/dev/null) 2>/dev/null)"
if [ -n "${dirty}" ]; then
  bad "packaged paths have UNCOMMITTED changes, and every comparison this gate"
  bad "makes is against HEAD, so those changes would not be examined at all:"
  printf '%s\n' "${dirty}" | sed 's/^/        /' >&2
  printf '      Commit (or stash) them, then re-run. Do not read the report\n' >&2
  printf '      below as covering the tree you are holding.\n' >&2
  note "frontier-revert gate: FAIL -- dirty packaged surface, nothing certified"
  exit 1
fi

changed="$(git diff --numstat "${frontier_sha}" HEAD 2>/dev/null)"
packaged="$(printf '%s\n' "${changed}" | python3 -c '
import json, sys

try:
    paths = json.load(open("benchmark.json")).get("editablePaths") or []
except Exception as exc:                       # noqa: BLE001
    sys.stderr.write("FAIL: cannot parse benchmark.json editablePaths: %s\n" % exc)
    sys.exit(2)
if not paths:
    sys.stderr.write("FAIL: benchmark.json has an empty editablePaths list.\n")
    sys.exit(2)

dirs = [p.rstrip("/") + "/" for p in paths]
exact = set(paths)

for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    add, dele, path = parts[0], parts[1], parts[2]
    if path in exact or any(path.startswith(d) for d in dirs):
        print("%s\t%s\t%s" % (add, dele, path))
' 2>&1)"
pkg_rc=$?
if [ "${pkg_rc}" -ne 0 ]; then
  printf '%s\n' "${packaged}" >&2
  bad "could not determine the packaged file set."
  note "frontier-revert gate: FAIL"
  exit 1
fi

# --- load the acknowledgement table -----------------------------------------
if [ ! -f "${ACKS}" ]; then
  bad "acknowledgement table '${ACKS}' is missing."
  note "frontier-revert gate: FAIL"
  exit 1
fi

ack_status_for() {
  awk -F'|' -v p="$1" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1) }
    $1 == p { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit }
  ' "${ACKS}"
}
ack_reason_for() {
  awk -F'|' -v p="$1" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1) }
    $1 == p { print $3; exit }
  ' "${ACKS}"
}
ack_blob_for() {
  awk -F'|' -v p="$1" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1) }
    $1 == p { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4; exit }
  ' "${ACKS}"
}

# HEAD's blob sha for a path, or the empty string if the path is absent there.
#
# Deliberately NOT `git rev-parse HEAD:path`: on an unresolvable argument that
# command ECHOES THE ARGUMENT BACK TO STDOUT and exits 128, so
# `$(git rev-parse HEAD:p 2>/dev/null)` yields the literal string "HEAD:p"
# rather than nothing, and every downstream comparison then compares two
# non-shas. That inverted a staleness check earlier in this campaign. `ls-tree`
# prints nothing for a missing path, which is the behaviour the callers assume.
head_blob_for() {
  git ls-tree HEAD -- "$1" 2>/dev/null | awk '{ print $3 }'
}

note "  FILES OUR SUBMISSION WOULD OVERWRITE ON THE LIVE TIP:"
if [ -z "${packaged}" ]; then
  note "    (none -- our packaged files are byte-identical to the frontier)"
else
  blocked=0
  total_del=0
  while IFS=$'\t' read -r a d p; do
    [ -n "${p}" ] || continue
    st="$(ack_status_for "${p}")"
    [ -n "${st}" ] || st="UNACKNOWLEDGED"
    printf '    %6s %6s  %-62s %s\n' "+${a}" "-${d}" "${p}" "${st}"
    case "${d}" in ''|*[!0-9]*) : ;; *) total_del=$((total_del + d)) ;; esac
    case "${st}" in
      INTENTIONAL-REPLACEMENT)
        r="$(ack_reason_for "${p}")"
        if [ "${#r}" -lt 40 ]; then
          bad "'${p}' is marked INTENTIONAL-REPLACEMENT with no substantive reason."
        fi
        # The reason names specific lines of OUR copy that the overlay drops.
        # Pin it to the copy it was written against, so (a) the bottom loop can
        # tell adoption from reversion, and (b) editing the file forces a
        # re-read instead of inheriting a reason about text that is gone.
        ab="$(ack_blob_for "${p}")"
        hb="$(head_blob_for "${p}")"
        if [ -z "${hb}" ]; then
          bad "'${p}' differs from the frontier but is absent from HEAD. A packaged"
          bad "      path that does not exist in the tree we would submit is a gate bug"
          bad "      or a deleted file; either way, do not submit."
        elif [ -z "${ab}" ]; then
          bad "'${p}' has no ack blob sha (4th |-field). Without it, a future overlay"
          bad "      that reverts us onto the frontier is indistinguishable from the"
          bad "      frontier adopting our change, and the two need opposite fixes."
          printf "      Append it:  ...|%s\n" "${hb}" >&2
        elif [ "${ab}" != "${hb}" ]; then
          bad "'${p}' ack was written against blob ${ab} but HEAD holds ${hb}."
          bad "      The reason below describes lines that may no longer be the lines"
          bad "      we delete. Re-read the frontier's copy, rewrite the reason, and"
          bad "      update the sha. Diff what changed under you with:"
          printf '        git diff %s %s\n' "${ab}" "${hb}" >&2
        fi
        ;;
      MUST-REBASE)
        blocked=$((blocked + 1))
        ;;
      UNACKNOWLEDGED)
        bad "'${p}' differs from the frontier and is not listed in ${ACKS}."
        ;;
      *)
        bad "'${p}' has unknown status '${st}'; use INTENTIONAL-REPLACEMENT or MUST-REBASE."
        ;;
    esac
  done <<EOF
${packaged}
EOF
  note ""
  note "    ${total_del} lines of the live tip would be deleted by our overlay."
  if [ "${blocked}" -gt 0 ]; then
    bad "${blocked} packaged file(s) are marked MUST-REBASE: we would revert frontier"
    bad "work we have not read. MUST-REBASE is a named blocker, not an escape hatch."
  fi
fi
note ""

# --- the table must not rot, and rot has two opposite causes -----------------
# An ack'd path that no longer differs from the frontier is EITHER good news
# (they took our change) or the worst news this gate can carry (an overlay took
# our change away). The diff is identical in both cases; the ack blob decides.
listed=0
while IFS='|' read -r p st r; do
  case "${p}" in ''|'#'*) continue ;; esac
  p="$(printf '%s' "${p}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -n "${p}" ] || continue
  listed=$((listed + 1))
  if ! printf '%s\n' "${packaged}" | awk -F'\t' -v q="${p}" '$3 == q { f = 1 } END { exit !f }'; then
    ab="$(ack_blob_for "${p}")"
    hb="$(head_blob_for "${p}")"
    if [ -n "${ab}" ] && [ "${ab}" = "${hb}" ]; then
      bad "${ACKS} lists '${p}' but it no longer differs from the frontier, and our"
      bad "      copy is UNCHANGED (blob ${hb}). So the frontier moved to us: they"
      bad "      adopted it. This entry is genuinely stale -- delete the line."
    elif [ -n "${ab}" ]; then
      bad "${ACKS} lists '${p}' and it no longer differs from the frontier, but OUR"
      bad "      copy changed under the ack: ${ab} -> ${hb:-<absent from HEAD>}."
      bad "      🔴 WE moved onto THEM. Whatever the reason says we carry, we no"
      bad "      longer carry. DO NOT delete this line -- it is the only written"
      bad "      record of what was lost. Restore it or retract it deliberately:"
      printf '        git diff %s %s\n' "${ab}" "${hb:-HEAD}" >&2
      printf '        git log --oneline -S<token> -- %s\n' "${p}" >&2
    else
      bad "${ACKS} lists '${p}', it no longer differs from the frontier, and the"
      bad "      entry has no ack blob sha, so adoption and reversion CANNOT be"
      bad "      distinguished. Resolve by hand; do not delete the line blind."
    fi
  fi
done < "${ACKS}"

# Fail closed on a table that evaluated nothing. An empty (or all-comment) acks
# file makes every check above vacuous: the packaged loop finds no status and
# reports UNACKNOWLEDGED only for paths that differ, so a tree that differs in
# nothing and acks nothing would sail through with no evidence examined at all.
if [ "${listed}" -eq 0 ]; then
  bad "${ACKS} declares zero paths. This gate then examines nothing and its PASS"
  bad "      would mean 'no evidence', not 'no reverts'. If our packaged surface is"
  bad "      genuinely identical to the frontier, say so in the table with a"
  bad "      comment AND record why we are submitting a tree with no changes."
fi

if [ "${fail}" -ne 0 ]; then
  note "frontier-revert gate: BLOCKED -- do not submit from this tree"
  exit 1
fi
note "frontier-revert gate: PASS (every overwrite of the live tip is intentional)"

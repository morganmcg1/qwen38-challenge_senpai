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
# Usage:
#   research/frontier-revert-gate.sh [frontier-ref]        (default upstream/main)
# Env:
#   FRONTIER_ACKS          path to the acknowledgement table
#                          (default research/frontier-revert-acks.txt)
#   FRONTIER_MAX_FETCH_AGE seconds of fetch staleness tolerated (default 43200)
#
# Exit 0 only if every packaged file that differs from the frontier is
# acknowledged as an intentional replacement. Any unreviewed revert is a BLOCK.

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

# --- the table must not rot --------------------------------------------------
while IFS='|' read -r p st r; do
  case "${p}" in ''|'#'*) continue ;; esac
  p="$(printf '%s' "${p}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -n "${p}" ] || continue
  if ! printf '%s\n' "${packaged}" | awk -F'\t' -v q="${p}" '$3 == q { f = 1 } END { exit !f }'; then
    bad "${ACKS} lists '${p}' but it no longer differs from the frontier. Remove the stale entry."
  fi
done < "${ACKS}"

if [ "${fail}" -ne 0 ]; then
  note "frontier-revert gate: BLOCKED -- do not submit from this tree"
  exit 1
fi
note "frontier-revert gate: PASS (every overwrite of the live tip is intentional)"

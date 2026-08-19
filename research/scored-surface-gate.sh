#!/usr/bin/env bash
# The third baseline: what has actually been SCORED?
#
# WHY THIS EXISTS
#
# research/shipped-surface-gate.sh answers "what did this campaign change?" by
# diffing HEAD against the campaign baseline. research/inherited-surface-gate.sh
# answers "how much of what we ship did we write?" by diffing against pristine
# upstream. Neither answers the question a leaderboard actually cares about:
#
#     which of the lines we would submit have ever been measured by the board?
#
# That question was unanswerable for most of this campaign because our own
# submission commit sha, as reported by the benchmark API, was not a resolvable
# git object locally -- so the shipped-surface gate was verifying HEAD and
# reporting it as "the shipped surface", which is a different tree. It turns out
# the object is fetchable and simply was not in the local store:
#
#     git fetch origin 2b0c36a078b7660c9215adee933336ff46da25af
#
# Once fetched, three facts appear that no existing gate could see:
#
#   1. HEAD and the scored tree are on DIVERGENT lineages. Neither is an
#      ancestor of the other. The scored tree was built by the organizer on top
#      of its own accept/validate lineage, not on top of our main.
#   2. HEAD carries lines in the shipped surface that have NEVER been scored.
#   3. Our submitted overlay is not the same diff as our campaign diff, because
#      the organizer lineage had already moved. Reading "+229/-74" as "what was
#      scored" is wrong in both directions.
#
# This gate makes (2) impossible to lose track of. Every unscored file in the
# shipped surface must be acknowledged here with a status word and a reason, and
# the table cannot rot: an entry that is no longer unscored is also a failure.
#
# Usage: research/scored-surface-gate.sh [rev]     (rev defaults to HEAD)
# Exit 0 iff every unscored shipped file is acknowledged and every
# acknowledgement still describes something unscored.

set -uo pipefail

# --- repo root ---------------------------------------------------------------
# Resolve script-relative FIRST, because that is the intended meaning: this gate
# describes the checkout it is committed into. Fall back to the caller's
# repository only if the script is not sitting in one.
#
# The fallback is not cosmetic. The original form was an unconditional
#
#     cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#
# which means a copy of this file placed anywhere else cd's somewhere useless.
# That made the gate impossible to mutation-test: research/scored-surface-gate-
# controls.sh copies the gate to a temp dir, mutates one line, and requires the
# copy to fail for the right reason -- and 9 of 10 controls "failed" for the
# uninteresting reason that a /tmp copy cd'd to /tmp. A gate that can only run
# from its own path cannot be tested by copy, and an untested gate is an opinion.
# Falling back to the caller's repo is safe here because every provenance pin
# below is asserted against real git objects, so a wrong root fails loudly.
_gate_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_gate_root="$(git -C "${_gate_script_dir}/.." rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${_gate_root}" ]; then
  _gate_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "${_gate_root}" ]; then
  printf 'FAIL: scored-surface-gate.sh is not inside a git repository and was not\n' >&2
  printf '      invoked from one, so there is no tree to compare.\n' >&2
  exit 1
fi
cd "${_gate_root}" || exit 1

# --- pins --------------------------------------------------------------------
# Our best official submission. Every field is asserted below, so a wrong pin
# fails loudly rather than silently comparing against some other tree.
SCORED_COMMIT="2b0c36a078b7660c9215adee933336ff46da25af"
SCORED_SCORE="3.23250848263467"
SCORED_CREATED="2026-08-18T22:44:18.032Z"
SCORED_SUBJECT="Validate submission ca9251b8-58cd-4d90-9a52-fa05f5657216"
SCORED_AUTHOR="yukon-autoresearch[bot]"
SCORED_PARENT="5068eb8d0bae032faca6e901de398fc732531160"

SURFACE_PATHS=(
  "Sources/"
  "Vendor/"
  "mtp-head.manifest.json"
)

# --- acknowledgements --------------------------------------------------------
# One line per shipped file that differs from the scored tree.
#   path | STATUS | why this unscored delta is acceptable to carry
# STATUS words:
#   PROBE-OFF-BY-DEFAULT  gated instrumentation; the added path cannot execute
#                         unless an env var is set, so the timed configuration
#                         is the scored one plus one boolean test.
#   HOT-PATH-REFACTOR     changes code that runs on every decode step. Behaviour
#                         is intended to be identical but the emitted work is
#                         not textually identical. Must carry a modelled cost.
ACK_UNSCORED=(
  "Sources/MLXFastModel/Qwen36MTPBlockSession.swift|PROBE-OFF-BY-DEFAULT|E29 head-chain drain probe behind MLX_QWEN_MTP_TRACE_SYNC_HEAD; body is one eval() inside 'if Self.traceSyncHeadChain', so a timed run pays one static-let bool test per round. Comment already warns it destroys head/verify overlap if enabled."
  "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift|HOT-PATH-REFACTOR|E29(c) makes the decode asyncEval rung schedule overridable via MLX_QWEN_MTP_LADDER. Default rung set [0,1,9,19,29,39,49,57] is identical to the scored switch, but a Set<Int>.contains hash lookup replaced a jump table, 64x per forward pass. Modelled cost ~1 us/step: ~0.002 % of a 38 ms serial step and ~0.002 % of a 59 ms MTP round, and it appears on BOTH legs of raw_p so it largely cancels in the ratio. That is ~50x below sigma_score = 0.0978 %."
)

fail=0
bad() { printf 'FAIL: %s\n' "$1" >&2; fail=1; }
note() { printf '%s\n' "$1"; }

note "scored-surface gate"
note "  question: which shipped lines have ever been measured by the board?"
note "  repo root     : ${_gate_root}"
note ""

# --- the scored commit must be present and must be the right one -------------
if ! git cat-file -e "${SCORED_COMMIT}^{commit}" 2>/dev/null; then
  bad "the scored submission commit ${SCORED_COMMIT} is not in the local object store."
  printf '      It IS fetchable. Run:\n\n        git fetch origin %s\n\n' "${SCORED_COMMIT}" >&2
  printf '      Refusing to report an unscored delta I cannot compute. Without this\n' >&2
  printf '      commit the only honest statement is that we do not know which of the\n' >&2
  printf '      lines we ship have been measured.\n' >&2
  exit 1
fi

actual_subject="$(git log -1 --format=%s "${SCORED_COMMIT}")"
actual_author="$(git log -1 --format=%an "${SCORED_COMMIT}")"
actual_parent="$(git log -1 --format=%P "${SCORED_COMMIT}")"
[ "${actual_subject}" = "${SCORED_SUBJECT}" ] || bad "scored commit subject drifted: expected '${SCORED_SUBJECT}', got '${actual_subject}'"
[ "${actual_author}" = "${SCORED_AUTHOR}" ] || bad "scored commit author drifted: expected '${SCORED_AUTHOR}', got '${actual_author}'"
[ "${actual_parent}" = "${SCORED_PARENT}" ] || bad "scored commit parent drifted: expected '${SCORED_PARENT}', got '${actual_parent}'"

note "  scored commit : ${SCORED_COMMIT}"
note "  official score: ${SCORED_SCORE}   (created ${SCORED_CREATED})"
note "  built by      : ${SCORED_AUTHOR} on organizer lineage ${SCORED_PARENT:0:12}"

REV="${1:-HEAD}"
rev_full="$(git rev-parse "${REV}" 2>/dev/null || true)"
if [ -z "${rev_full}" ]; then
  bad "cannot resolve rev '${REV}'"
  exit 1
fi
note "  subject rev   : ${rev_full}  (${REV})"

# --- lineage relation, stated rather than assumed ----------------------------
if git merge-base --is-ancestor "${SCORED_COMMIT}" "${rev_full}" 2>/dev/null; then
  note "  lineage       : the scored commit IS an ancestor of ${REV}"
else
  note "  lineage       : the scored commit is NOT an ancestor of ${REV} -- divergent"
  note "                  histories, so only the TREE comparison below is meaningful."
fi
note ""

# --- informational: what the organizer actually applied for us ---------------
# Our submitted overlay is the diff from the organizer lineage tip to the scored
# tree. It is NOT the same as our campaign diff, and reading one as the other is
# the specific mistake this gate exists to prevent.
if git cat-file -e "${SCORED_PARENT}^{commit}" 2>/dev/null; then
  note "  THE OVERLAY THE ORGANIZER APPLIED (lineage tip -> scored tree):"
  overlay="$(git diff --numstat "${SCORED_PARENT}" "${SCORED_COMMIT}" -- "${SURFACE_PATHS[@]}")"
  if [ -z "${overlay}" ]; then
    note "    (empty)"
  else
    printf '%s\n' "${overlay}" | while IFS=$'\t' read -r a d p; do
      printf '    %6s %6s  %s\n' "+${a}" "-${d}" "${p}"
    done
  fi
  note "    Note the deletions: our REPLACE overlay removes lineage lines as well as"
  note "    adding ours. A submission can silently revert organizer-accepted work."
  note ""
fi

# --- the load-bearing output: what we ship that was never scored -------------
unscored="$(git diff --numstat "${SCORED_COMMIT}" "${rev_full}" -- "${SURFACE_PATHS[@]}")"
note "  UNSCORED SHIPPED DELTA (scored tree -> ${REV}):"
if [ -z "${unscored}" ]; then
  note "    (none -- every shipped line has been measured by the board)"
else
  total_add=0
  total_del=0
  while IFS=$'\t' read -r a d p; do
    [ -n "${p}" ] || continue
    status="UNACKNOWLEDGED"
    for entry in "${ACK_UNSCORED[@]}"; do
      if [ "${entry%%|*}" = "${p}" ]; then
        rest="${entry#*|}"
        status="${rest%%|*}"
      fi
    done
    printf '    %6s %6s  %-64s %s\n' "+${a}" "-${d}" "${p}" "${status}"
    total_add=$((total_add + a))
    total_del=$((total_del + d))
    if [ "${status}" = "UNACKNOWLEDGED" ]; then
      bad "shipped file '${p}' differs from the scored tree and is not acknowledged in ACK_UNSCORED."
    fi
  done <<EOF
${unscored}
EOF
  note ""
  note "    ${total_add} inserted and ${total_del} deleted lines in the surface we would"
  note "    submit next have never been measured by the board."
fi
note ""

# --- the table must not rot --------------------------------------------------
for entry in "${ACK_UNSCORED[@]}"; do
  path="${entry%%|*}"
  if ! printf '%s\n' "${unscored}" | awk -F'\t' -v p="${path}" '$3 == p { found = 1 } END { exit !found }'; then
    bad "ACK_UNSCORED lists '${path}' but it no longer differs from the scored tree. Remove the entry rather than leaving a stale acknowledgement."
  fi
done

# --- reasons must be present ------------------------------------------------
for entry in "${ACK_UNSCORED[@]}"; do
  reason="${entry##*|}"
  path="${entry%%|*}"
  if [ ${#reason} -lt 40 ]; then
    bad "ACK_UNSCORED entry for '${path}' has no substantive reason."
  fi
done

note "  WHY EACH UNSCORED DELTA IS CARRIED:"
for entry in "${ACK_UNSCORED[@]}"; do
  path="${entry%%|*}"
  rest="${entry#*|}"
  status="${rest%%|*}"
  reason="${entry##*|}"
  note "    ${path}"
  note "      [${status}] ${reason}"
done
note ""

if [ "${fail}" -ne 0 ]; then
  note "scored-surface gate: FAIL"
  exit 1
fi
note "scored-surface gate: PASS (every unscored shipped delta is acknowledged)"

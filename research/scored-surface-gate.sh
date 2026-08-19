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

# The live research frontier. FRONTIER-TAKEN acknowledgements are asserted
# against this ref, so it must resolve; a gate that silently skips its own
# strongest assertion because a ref is missing is worse than no gate.
FRONTIER_REF="${SCORED_GATE_FRONTIER_REF:-upstream/main}"

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
#   FRONTIER-TAKEN        the file is BYTE-IDENTICAL to the live research
#                         frontier (${FRONTIER_REF}). The delta versus the scored
#                         tree is therefore the ORGANIZER'S OWN promoted work,
#                         not ours: it is unscored only because our last scored
#                         row predates that promotion. This is the strongest
#                         justification available and it is the only status word
#                         this gate VERIFIES rather than believes -- see the
#                         FRONTIER-TAKEN assertion loop below. An entry that
#                         stops being byte-identical fails, so this class of
#                         acknowledgement cannot rot into a lie the way a
#                         free-text reason can. Ledger 162 added it after four
#                         hand-written acks went stale in a single rebase.
#   WARM-PATH-ONLY        added code executes only in the warm-up path, outside
#                         the timed window, so it cannot appear in a scored
#                         measurement except by moving cost OUT of it.
#   FRONTIER-PLUS-PINNED-DIFF:<sha256>
#                         the file is the live frontier's copy PLUS exactly one
#                         declared diff, whose content digest is pinned in the
#                         status word. Use this, never FRONTIER-TAKEN, when a
#                         candidate edits a file we otherwise only adopt.
#
#                         WHY THIS CLASS EXISTS. 2026-08-19, askeladd's E55.
#                         FRONTIER-TAKEN asserts byte-identity, so the first
#                         legitimate edit to an adopted file has no honest label
#                         available: the choices were to weaken FRONTIER-TAKEN,
#                         to demote the file to a prose-only status and lose the
#                         one verified class, or to add this. The safety property
#                         FRONTIER-TAKEN really protects is not identity, it is
#                         "our whole-file overlay does not SILENTLY REVERT
#                         organizer-accepted work". A pinned diff protects that
#                         property exactly, because anything the overlay changes
#                         beyond the declared hunks moves the digest and fails.
#
#                         The digest covers only the '+'/'-' CONTENT lines of
#                         `git diff -U0` against the frontier, so it is stable
#                         under line-number drift elsewhere in the file and it
#                         cannot be satisfied by a diff that touches anything
#                         else. Same principle as the twin-audit waiver: pin the
#                         DIVERGENCE, never the BODY.
#
#                         An entry whose diff has become empty also fails: at
#                         that point the file has returned to the frontier and
#                         the honest label is FRONTIER-TAKEN again.
ACK_UNSCORED=(
  "Sources/MLXFastModel/Qwen35RuntimeWeights.swift|FRONTIER-TAKEN|Buffer cap MLX_MAX_MB_PER_BUFFER raised 128 to 512. Adopted verbatim from the frontier so our whole-file overlay stops REVERTING an organizer-accepted promotion. Byte-identity is asserted by this gate, so no cost model is owed: the frontier is already the measured configuration."
  "Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift|FRONTIER-TAKEN|Frontier memory policy taken verbatim: setenv overwrite flag 0 to 1 and the 320/128 MiB pair to 512/50. These were the crown's actual mechanism, not dead constants; a previous turn nearly deleted them as unused. Byte-identity to the frontier is asserted below."
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp|FRONTIER-PLUS-PINNED-DIFF:08c42cf7891adc9104329bee5bc4c648d049887b57adbdc954d476cd7b7e69f6|E27 per-width register widening REVERTED to the frontier. E27 cost 0.3321 percent of score: mean MTP leg plus 0.1995 percent, slower on every wide prompt (beagle 0.2353, essays 0.4803, republic 0.2375, botany 0.5225 against an MTP replicate sd of 0.0995 percent). See research/crown_leg_decomposition.py. JIT twin of the kernel header below. DECLARED DIFF (E55): the same 4 content lines as the header twin, relaxing the wide-helper assert to NA in [2, 5] and moving case 9 from T,9,3 to T,9,5. It IS the E27 register mechanism minus case 5: research/e55_reg_census.py measures the candidate at the identical kernel-wide max of 129 and at 181 of E27's 183 entry registers, so the shared-allocation channel E27 was reverted for is open here too."
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h|FRONTIER-PLUS-PINNED-DIFF:08c42cf7891adc9104329bee5bc4c648d049887b57adbdc954d476cd7b7e69f6|E27 reverted with its twin. Cells return to T,5,3 and T,9,3, dropping the kernel-wide register max from 129 to 108 and the production entry affine_qmv_fast bfloat16_t,64,4,false from 183 to 163. Because there is exactly one [[kernel]] and every helper is METAL_FUNC inline (alphonse E40), that allocation is shared by every width, which is why two local per-width wins lost the score. DECLARED DIFF (E55): 4 content lines, the assert relaxed to NA in [2, 5] and case 9 moved from T,9,3 to T,9,5, with case 5 and case 8 untouched. It IS that register mechanism minus case 5, and the census shows case 5 was never the cause: E27's case-5 cell measures 125, below the 129 max that T,9,5 sets on its own."
  "Sources/MLXFastModel/Qwen36MTPBlockSession.swift|WARM-PATH-ONLY|Two additions. (1) fkiene's verify-concat JIT warm, promoted at 1cb1f43a7246d57af8b96dad468583364779aa73 scoring 3.24417896624589 against base 3.24326223889754 (plus 0.0283 percent), deleted from the frontier by a later whole-file overlay whose author never opened the file; restored inside warmAllDepthShapes, i.e. OUTSIDE the timed window, so it can only move JIT cost out of the measured region. (2) E29 head-chain drain probe behind MLX_QWEN_MTP_TRACE_SYNC_HEAD: body is one eval() inside 'if Self.traceSyncHeadChain', so a timed run pays one static-let bool test per round."
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

# --- refuse to certify a worktree we have not committed ----------------------
# See the header of research/lib/dirty-packaged-surface.sh. This gate reports
# what we would SUBMIT, so certifying an uncommitted tree here is the worst case.
#
# ORDERING IS DELIBERATE and it is not cosmetic. This block sits AFTER the pin
# assertions above, because a gate must establish that its own pins are sound
# before it makes any statement at all -- and because putting it first broke
# control 12, which copies this gate into a decoy repo containing none of the
# objects it pins and requires it to fail on the MISSING SCORED COMMIT rather
# than on anything else. A control that demands a specific diagnostic is a
# control that pins the gate's reasoning ORDER, which is worth preserving.
if [ -r "${_gate_root}/research/lib/dirty-packaged-surface.sh" ]; then
  # shellcheck source=research/lib/dirty-packaged-surface.sh
  . "${_gate_root}/research/lib/dirty-packaged-surface.sh"
  if ! refuse_if_packaged_surface_dirty "${REV}" "scored-surface gate"; then
    note "scored-surface gate: FAIL -- dirty packaged surface, nothing certified"
    exit 1
  fi
else
  bad "research/lib/dirty-packaged-surface.sh is missing, so this gate cannot establish that the packaged surface is committed. Failing closed rather than certifying a tree I have not read."
  note "scored-surface gate: FAIL"
  exit 1
fi

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

# --- FRONTIER-TAKEN entries must EARN the label -------------------------------
# This is the one acknowledgement class the gate verifies instead of believing.
# The claim "this delta is the organizer's own promoted work, not ours" is
# exactly the claim `git diff --quiet <frontier> HEAD -- <path>` decides, so
# decide it. Four hand-written acknowledgements went stale in a single rebase
# (ledger 162); the failure mode of a prose reason is that it keeps passing
# after it stops being true.
frontier_sha="$(git rev-parse --verify "${FRONTIER_REF}" 2>/dev/null || true)"
have_frontier_ack=0
for entry in "${ACK_UNSCORED[@]}"; do
  rest="${entry#*|}"
  [ "${rest%%|*}" = "FRONTIER-TAKEN" ] && have_frontier_ack=1
done
if [ "${have_frontier_ack}" -eq 1 ] && [ -z "${frontier_sha}" ]; then
  bad "ACK_UNSCORED contains FRONTIER-TAKEN entries but frontier ref '${FRONTIER_REF}' does not resolve, so their central claim cannot be checked. Run 'git fetch upstream' or set SCORED_GATE_FRONTIER_REF."
elif [ "${have_frontier_ack}" -eq 1 ]; then
  note "  FRONTIER-TAKEN VERIFICATION (against ${FRONTIER_REF} = ${frontier_sha:0:12}):"
  for entry in "${ACK_UNSCORED[@]}"; do
    path="${entry%%|*}"
    rest="${entry#*|}"
    [ "${rest%%|*}" = "FRONTIER-TAKEN" ] || continue
    if git diff --quiet "${frontier_sha}" "${rev_full}" -- "${path}" 2>/dev/null; then
      note "    byte-identical to frontier   ${path}"
    else
      note "    NOT identical to frontier    ${path}"
      bad "ACK_UNSCORED marks '${path}' FRONTIER-TAKEN, but it is NOT byte-identical to ${FRONTIER_REF}. Either the frontier moved under us or we edited a file we claimed only to adopt. Re-adjudicate with 'git diff ${FRONTIER_REF} ${REV} -- ${path}' and 'git blame' on the deleted lines before changing the label. If the edit is a DELIBERATE candidate change, relabel to FRONTIER-PLUS-PINNED-DIFF:<sha256> instead of dropping the verified class."
    fi
  done
  note ""
fi

# --- FRONTIER-PLUS-PINNED-DIFF entries must match their pinned divergence -----
# The digest is taken over the '+'/'-' content lines of a -U0 diff with an
# explicit algorithm, so it is reproducible and it ignores line-number drift
# elsewhere in the file. Anything the overlay changes beyond the declared hunks
# moves the digest, which is the whole point.
pinned_diff_digest() {
  git diff --no-color --no-ext-diff --diff-algorithm=myers -U0 \
    "$1" "$2" -- "$3" \
    | grep -E '^[+-]' \
    | grep -Ev '^(\+\+\+|---)' \
    | shasum -a 256 \
    | awk '{ print $1 }'
}

have_pinned_ack=0
for entry in "${ACK_UNSCORED[@]}"; do
  rest="${entry#*|}"
  case "${rest%%|*}" in FRONTIER-PLUS-PINNED-DIFF:*) have_pinned_ack=1 ;; esac
done
if [ "${have_pinned_ack}" -eq 1 ] && [ -z "${frontier_sha}" ]; then
  bad "ACK_UNSCORED contains FRONTIER-PLUS-PINNED-DIFF entries but frontier ref '${FRONTIER_REF}' does not resolve, so their pinned divergence cannot be checked. Run 'git fetch upstream' or set SCORED_GATE_FRONTIER_REF."
elif [ "${have_pinned_ack}" -eq 1 ]; then
  note "  PINNED-DIFF VERIFICATION (against ${FRONTIER_REF} = ${frontier_sha:0:12}):"
  for entry in "${ACK_UNSCORED[@]}"; do
    path="${entry%%|*}"
    rest="${entry#*|}"
    status="${rest%%|*}"
    case "${status}" in FRONTIER-PLUS-PINNED-DIFF:*) ;; *) continue ;; esac
    pinned="${status#FRONTIER-PLUS-PINNED-DIFF:}"
    if git diff --quiet "${frontier_sha}" "${rev_full}" -- "${path}" 2>/dev/null; then
      bad "ACK_UNSCORED marks '${path}' FRONTIER-PLUS-PINNED-DIFF but it is now byte-identical to ${FRONTIER_REF}. The declared diff is gone, so this label is a lie about a change that no longer exists. Relabel to FRONTIER-TAKEN."
      continue
    fi
    observed="$(pinned_diff_digest "${frontier_sha}" "${rev_full}" "${path}")"
    if [ "${observed}" = "${pinned}" ]; then
      note "    declared diff matches pin    ${path}  ${observed:0:12}"
    else
      note "    PINNED DIFF MOVED            ${path}"
      bad "ACK_UNSCORED pins '${path}' to diff digest ${pinned} but the observed digest is ${observed}. Our overlay changes this frontier file somewhere the acknowledgement does not declare. Read 'git diff ${FRONTIER_REF} ${REV} -- ${path}', confirm every hunk is intended, then re-pin to the observed digest in the SAME commit that introduces the hunks."
    fi
  done
  note ""
fi

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

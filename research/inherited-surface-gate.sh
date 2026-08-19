#!/usr/bin/env bash
# Inherited-surface gate for the Qwen 3.8 native-MTP campaign.
#
# WHY THIS EXISTS (ledger 156, then 159). `shipped-surface-gate.sh` answers
# "what did THIS CAMPAIGN change?" -- 5 files, +229/-74 against the campaign
# baseline. For fifteen turns I quoted that number as the search space for
# defects. It is not. It is the search space for *our* defects.
#
# Against PRISTINE UPSTREAM we ship 18 files and +5027 inserted lines, so we
# authored 4.6 % of the code we submit. The other 95.4 % is inherited from the
# validated-submission lineage: never written by us, never reviewed by us, and
# on the scored path.
#
# That blind spot cost us a concrete fifteen turns. Ledger 156: I derived the
# SDPA width wall from source three times, wrote a brief calling it "the primary
# target", and was about to assign a student to build a chunked-SDPA fix -- which
# has been sitting in AttentionUtils.swift, inherited, reachable and completely
# untested, since before our first submission. I found it by accident.
#
# So this gate answers the other question, and it FAILS CLOSED: if a shipped
# file appears that is neither ours nor on the acknowledged list below, that is
# new unaudited code on the scored path and you must look at it and record what
# it does before adding it here.
#
# Usage:
#   research/inherited-surface-gate.sh [<rev>]     # default HEAD

set -uo pipefail

# --- pristine upstream, named and pinned ------------------------------------
PRISTINE="5d029178765cf727e7ee530b0b4c731d566f908a"
PRISTINE_SUBJECT="Qwen 3.8 27B native-MTP challenge"

SURFACE_PATHS=("Sources/" "Vendor/" "mtp-head.manifest.json")

# The campaign baseline, so "what WE wrote" is measured against the point this
# campaign started rather than against pristine upstream. Getting this wrong is
# the exact confusion this gate exists to expose -- the first version of this
# script credited the campaign with 4708 lines by diffing our five files against
# pristine, which counts the inherited code living inside our own files.
CAMPAIGN_BASE="527306761f70e2c4024f347915328894db80c181"

# --- the five files THIS campaign changed (ledger 140) ----------------------
# LEDGER 162: this list used to be hardcoded, and it went stale the moment the
# shipped surface was rebased onto the frontier -- Qwen35RuntimeWeights.swift
# entered our delta and mlx-generated/quantized.cpp left it, so the gate called
# a file we changed "inherited" and demanded an audit acknowledgement for it.
# It is now DERIVED from CAMPAIGN_BASE below, because git already knows which
# files this campaign touched and a second hand-maintained copy of that fact can
# only drift. research/shipped-surface-gate.sh owns the question "did the set
# change?" and fails loudly on drift; this gate owns "how much did we write?"
# and should simply ask. Single source of truth, one place to update.
OURS=""

# --- inherited files we have SEEN, with audit status ------------------------
# A file here is acknowledged as present, NOT certified as correct. The status
# word is load-bearing: only AUDITED means someone traced what it does.
#   AUDITED   -- mechanism traced and written up in the ledger
#   PARTIAL   -- extent established, mechanism not traced
#   UNAUDITED -- known to be shipped, nothing more
ACK="Sources/MLXFastCLI/main.swift|UNAUDITED|CLI entry; +10 lines
Sources/MLXFastHarness/QwenRuntimeLocalIterate.swift|UNAUDITED|local-iterate fixture; +2/-2
Sources/MLXFastModel/Qwen35Block.swift|UNAUDITED|+2 lines in the block
Sources/MLXFastModel/Qwen35GatedDelta.swift|UNAUDITED|+2 lines in GDN; GDN is 28 % of step time
Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift|UNAUDITED|+76/-10 head wiring, scored path
Sources/MLXFastModel/Qwen36MTPTarget.swift|UNAUDITED|+62/-6 MTP target, scored path
Sources/MLXFastTrustedHarness/QwenRuntimeLocalIterate.swift|UNAUDITED|twin of the harness fixture
Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift|UNAUDITED|+15 lines
Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift|UNAUDITED|+6 lines; owns effectiveDraftLengths
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift|UNAUDITED|+40/-1 MTP head forward
Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp|AUDITED|ledger 162: the JIT twin of kernels/quantized.h. Was OURS until E27 was reverted; it is now byte-identical to the frontier, asserted by scored-surface-gate.sh FRONTIER-TAKEN, so it left our authored set and re-entered the inherited one. It differs from pristine only by the frontier's own promoted cross-row work. Its code must stay in lockstep with kernels/quantized.h or the AOT metallib and the runtime JIT compile different kernels; shipped-surface-gate.sh checks that over CODE lines only, because generated source does not carry the header's comments
Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift|AUDITED|ledger 156: chunked SDPA at qL 6..9, fires on scored verify path, ZERO test coverage
Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift|PARTIAL|ledger 159: one 58-line append at :1243 adding rollbackCheckpoints + prefixReplayTape to MambaCache; does NOT touch makeMask/KVCacheSimple/createAttentionMask, so ledger 156 reachability rests on pristine code
mtp-head.manifest.json|UNAUDITED|+5/-13; selects the head artifact"

REV="${1:-HEAD}"
fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

# LEDGER 162: membership below was tested with `printf ... | grep -qxF`. Under
# `set -o pipefail` that construct is unsafe: grep -q exits on the first match,
# the producer can then die of SIGPIPE (141), and pipefail propagates 141 as the
# condition's status -- silently inverting the test. A control in
# research/frontier-revert-gate-controls.sh broke in exactly that way and reported
# a vacuous pass for weeks before I noticed. The three sites that use this helper
# decide whether a shipped file counts as OURS or INHERITED, i.e. they produce the
# authorship percentage this gate exists to report, so they must not be able to
# fail quietly. Pure builtin, no pipe, no external process.
contains_line() { # $1 = newline-separated haystack, $2 = exact needle
  # NOT $(printf ...): command substitution strips trailing newlines, which would
  # make the LAST entry of the haystack unmatchable. Literal $'\n' concatenation
  # keeps both sentinels.
  local hay=$'\n'"$1"$'\n'
  local needle=$'\n'"$2"$'\n'
  case "${hay}" in
    *"${needle}"*) return 0 ;;
    *) return 1 ;;
  esac
}

note "inherited-surface gate"
note "  pristine $PRISTINE"
note "  rev      $(git rev-parse --short "$REV") $(git log -1 --format=%s "$REV")"

# Refuse to certify a worktree we have not committed; see the header of
# research/lib/dirty-packaged-surface.sh.
_gate_root_ih="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "${_gate_root_ih}" ] && [ -r "${_gate_root_ih}/research/lib/dirty-packaged-surface.sh" ]; then
  # shellcheck source=research/lib/dirty-packaged-surface.sh
  . "${_gate_root_ih}/research/lib/dirty-packaged-surface.sh"
  ( cd "${_gate_root_ih}" && refuse_if_packaged_surface_dirty "$REV" "inherited-surface gate" ) || {
    note "inherited-surface gate: FAIL -- dirty packaged surface, nothing certified"
    exit 1
  }
else
  bad "research/lib/dirty-packaged-surface.sh is missing, so this gate cannot establish that the packaged surface is committed. Failing closed."
  exit 1
fi

if ! git cat-file -e "${PRISTINE}^{commit}" 2>/dev/null; then
  bad "pristine commit $PRISTINE does not exist in this repository"
  exit 1
fi
actual_subject="$(git log -1 --format=%s "$PRISTINE")"
if [ "$actual_subject" != "$PRISTINE_SUBJECT" ]; then
  bad "pristine subject drifted: expected '$PRISTINE_SUBJECT', got '$actual_subject'"
fi
if ! git merge-base --is-ancestor "$PRISTINE" "$REV"; then
  bad "pristine is NOT an ancestor of $REV -- the diff below is not a shipped delta"
fi

numstat="$(git diff --numstat "$PRISTINE" "$REV" -- "${SURFACE_PATHS[@]}")"

# Derive "our" files rather than trusting a second copy of the fact. See the
# comment at the OURS declaration: the hardcoded list went stale on the rebase.
if ! git cat-file -e "${CAMPAIGN_BASE}^{commit}" 2>/dev/null; then
  bad "campaign base $CAMPAIGN_BASE is not in this repository, so the set of files this campaign changed cannot be derived and every authorship number below would be a guess."
  exit 1
fi
OURS="$(git diff --name-only "$CAMPAIGN_BASE" "$REV" -- "${SURFACE_PATHS[@]}")"
if [ -z "$OURS" ]; then
  bad "this campaign changed NO packaged file relative to $CAMPAIGN_BASE. That is either a wrong base or an empty submission; refusing to report a 0 % authorship figure as if it were meaningful."
  exit 1
fi
note ""
note "  files THIS campaign changed, derived from ${CAMPAIGN_BASE:0:12}:"
printf '%s\n' "$OURS" | sed 's/^/    /'
ours_sorted="$(printf '%s\n' "$OURS" | sort)"
ack_paths="$(printf '%s\n' "$ACK" | awk -F'|' 'NF{print $1}' | sort)"

tot_files=$(printf '%s\n' "$numstat" | grep -c . || true)
tot_ins=$(printf '%s\n' "$numstat" | awk -F'\t' '{s+=$1} END {print s+0}')
tot_del=$(printf '%s\n' "$numstat" | awk -F'\t' '{s+=$2} END {print s+0}')

note ""
note "  SHIPPED vs pristine upstream: $tot_files files, +$tot_ins/-$tot_del"

# What THIS CAMPAIGN wrote: measured against the campaign baseline, NOT against
# pristine. Diffing our five files against pristine would also count the
# inherited code that lives inside those same files.
own_ins=$(git diff --numstat "$CAMPAIGN_BASE" "$REV" -- "${SURFACE_PATHS[@]}" \
          | awk -F'\t' '{s+=$1} END {print s+0}')

# Inherited insertions that sit INSIDE our own five files -- the subtlest part of
# the blind spot: opening one of "our" files does not mean reading our own code.
own_files_vs_pristine=0
while IFS=$'\t' read -r add del path; do
  [ -n "${path:-}" ] || continue
  if contains_line "$ours_sorted" "$path"; then
    own_files_vs_pristine=$((own_files_vs_pristine + add))
  fi
done <<< "$numstat"
inherited_inside_ours=$((own_files_vs_pristine - own_ins))

note ""
note "  INHERITED files (never authored by this campaign):"
inherited_count=0
while IFS=$'\t' read -r add del path; do
  [ -n "${path:-}" ] || continue
  contains_line "$ours_sorted" "$path" && continue
  inherited_count=$((inherited_count + 1))
  status="$(printf '%s\n' "$ACK" | awk -F'|' -v p="$path" '$1==p{print $2}')"
  if [ -z "$status" ]; then
    printf '    +%-6s -%-5s %-70s %s\n' "$add" "$del" "$path" "*** UNACKNOWLEDGED ***"
    bad "unacknowledged inherited shipped file: $path -- read it, then add it to ACK"
  else
    printf '    +%-6s -%-5s %-70s %s\n' "$add" "$del" "$path" "$status"
  fi
done <<< "$numstat"

# an ACK entry for a file that is no longer shipped is also drift
while IFS= read -r p; do
  [ -n "$p" ] || continue
  _shipped_paths="$(awk -F'\t' 'NF{print $3}' <<< "$numstat")"
  if ! contains_line "$_shipped_paths" "$p"; then
    bad "ACK lists '$p' but it is no longer in the shipped delta -- remove it"
  fi
done <<< "$ack_paths"

share="$(awk -v a="$own_ins" -v b="$tot_ins" 'BEGIN{printf "%.1f", (b?a/b*100:0)}')"
note ""
note "  AUTHORSHIP BREAKDOWN of the $tot_ins inserted lines we ship:"
note "    this campaign wrote                          +$own_ins"
note "    inherited, inside our own five files         +$inherited_inside_ours"
note "    inherited, in the $inherited_count files we never touched   +$((tot_ins - own_ins - inherited_inside_ours))"
note ""
note "  => we authored $share % of the code we ship"
note "  => $inherited_count of $tot_files shipped files are inherited"
note ""
note "  Read this as: a defect hunt confined to our own diff searches $share % of"
note "  the code we submit. And +$inherited_inside_ours inherited lines live INSIDE the five files"
note "  we call 'the shipped surface', so opening one of our files is not the"
note "  same as reading our own code."

note ""
if [ "$fail" = 0 ]; then
  note "inherited-surface gate: PASS (no unacknowledged inherited shipped file)"
else
  note "inherited-surface gate: FAIL"
fi
exit "$fail"

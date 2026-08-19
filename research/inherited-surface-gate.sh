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
OURS="Sources/MLXFastModel/Qwen36MTPBlockSession.swift
Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"

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
Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift|AUDITED|ledger 156: chunked SDPA at qL 6..9, fires on scored verify path, ZERO test coverage
Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift|PARTIAL|ledger 159: one 58-line append at :1243 adding rollbackCheckpoints + prefixReplayTape to MambaCache; does NOT touch makeMask/KVCacheSimple/createAttentionMask, so ledger 156 reachability rests on pristine code
mtp-head.manifest.json|UNAUDITED|+5/-13; selects the head artifact"

REV="${1:-HEAD}"
fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

note "inherited-surface gate"
note "  pristine $PRISTINE"
note "  rev      $(git rev-parse --short "$REV") $(git log -1 --format=%s "$REV")"

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
  if printf '%s\n' "$ours_sorted" | grep -qxF "$path"; then
    own_files_vs_pristine=$((own_files_vs_pristine + add))
  fi
done <<< "$numstat"
inherited_inside_ours=$((own_files_vs_pristine - own_ins))

note ""
note "  INHERITED files (never authored by this campaign):"
inherited_count=0
while IFS=$'\t' read -r add del path; do
  [ -n "${path:-}" ] || continue
  printf '%s\n' "$ours_sorted" | grep -qxF "$path" && continue
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
  if ! printf '%s\n' "$numstat" | awk -F'\t' 'NF{print $3}' | grep -qxF "$p"; then
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

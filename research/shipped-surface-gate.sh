#!/usr/bin/env bash
# Shipped-surface gate for the Qwen 3.8 native-MTP campaign.
#
# WHY THIS EXISTS (ledger 140). For the length of the campaign I quoted the
# shipped surface as "E27, 4 files, +117/-87" and re-verified it before every
# submission. The check ran, passed, and was believed -- against the WRONG
# BASELINE. The real surface is 5 files, +229/-74, and the file I had never
# counted, RuntimeStartupMemoryPolicy.swift, turned out to control the ranked
# box's command-buffer geometry and wired residency: the exact mechanism family
# that E35 found to be the only ranked positive on the leaderboard and that we
# had twice placed on our own established-negatives list.
#
# A check you never audit is not a check. So this script:
#   1. names its baseline explicitly instead of inferring one,
#   2. asserts the baseline commit still resolves to the expected subject,
#   3. prints the per-file numstat, never just a total,
#   4. fails on ANY drift, including a NEW file, and tells you what changed.
#
# Usage:
#   research/shipped-surface-gate.sh [<rev>]        # default HEAD
#   EXPECT_FILES=6 research/shipped-surface-gate.sh # after an intentional change
#
# When a submission intentionally changes the surface, update the EXPECT_*
# block below IN THE SAME COMMIT and say so in the ledger. Do not pass
# overrides to silence it.

set -uo pipefail

# --- the baseline, named and pinned -----------------------------------------
BASELINE="527306761f70e2c4024f347915328894db80c181"
BASELINE_SUBJECT="Record organizer and promoted frontiers"

# --- what the surface is expected to be -------------------------------------
# Ledger 140 pinned 5 files, +229/-74. Ledger 162 REBASED the shipped surface
# onto the live research frontier, which changed the set as well as the counts:
# Qwen35RuntimeWeights.swift ENTERED (we adopted the frontier's 512 MiB buffer
# cap instead of silently reverting it) and mlx-generated/quantized.cpp LEFT
# (E27 reverted, so the JIT twin is byte-identical to the frontier again).
# Both movements are verified byte-for-byte against the frontier by
# research/scored-surface-gate.sh's FRONTIER-TAKEN assertion; this gate pins the
# shape, that gate pins the provenance.
EXPECT_FILES="${EXPECT_FILES:-5}"
EXPECT_INSERTIONS="${EXPECT_INSERTIONS:-281}"
EXPECT_DELETIONS="${EXPECT_DELETIONS:-72}"

# Every path that `yukon submit` packages. Directory entries are prefixes.
SURFACE_PATHS=(
  "Sources/"
  "Vendor/"
  "mtp-head.manifest.json"
)

REV="${1:-HEAD}"
fail=0

note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

note "shipped-surface gate"
note "  baseline $BASELINE"
note "  rev      $(git rev-parse --short "$REV") $(git log -1 --format=%s "$REV")"

# 0. refuse to certify a worktree we have not committed. See the header of
#    research/lib/dirty-packaged-surface.sh: a mutation control for the twin
#    check below could not fire because this gate read the commit, not the tree.
_gate_root_sh="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "${_gate_root_sh}" ] && [ -r "${_gate_root_sh}/research/lib/dirty-packaged-surface.sh" ]; then
  # shellcheck source=research/lib/dirty-packaged-surface.sh
  . "${_gate_root_sh}/research/lib/dirty-packaged-surface.sh"
  ( cd "${_gate_root_sh}" && refuse_if_packaged_surface_dirty "$REV" "shipped-surface gate" ) || {
    note "shipped-surface gate: FAIL -- dirty packaged surface, nothing certified"
    exit 1
  }
else
  bad "research/lib/dirty-packaged-surface.sh is missing, so this gate cannot establish that the packaged surface is committed. Failing closed rather than certifying a tree I have not read."
  note "shipped-surface gate: FAIL"
  exit 1
fi

# 1. the baseline must still be the commit we think it is.
if ! git cat-file -e "${BASELINE}^{commit}" 2>/dev/null; then
  bad "baseline commit $BASELINE does not exist in this repository"
  exit 1
fi
actual_subject="$(git log -1 --format=%s "$BASELINE")"
if [ "$actual_subject" != "$BASELINE_SUBJECT" ]; then
  bad "baseline subject drifted: expected '$BASELINE_SUBJECT', got '$actual_subject'"
fi

# 2. the baseline must be an ancestor of the rev under test, or the diff is
#    meaningless rather than merely surprising.
if ! git merge-base --is-ancestor "$BASELINE" "$REV"; then
  bad "baseline is NOT an ancestor of $REV -- the diff below is not a campaign delta"
fi

# 3. per-file numstat, always printed.
note ""
note "  per-file numstat (+ / - / path):"
numstat="$(git diff --numstat "$BASELINE" "$REV" -- "${SURFACE_PATHS[@]}")"
if [ -z "$numstat" ]; then
  note "    (empty)"
else
  printf '%s\n' "$numstat" | while IFS=$'\t' read -r add del path; do
    printf '    %6s %6s  %s\n' "$add" "$del" "$path"
  done
fi

files=$(printf '%s\n' "$numstat" | grep -c . || true)
ins=$(printf '%s\n' "$numstat" | awk -F'\t' '{s+=$1} END {print s+0}')
dels=$(printf '%s\n' "$numstat" | awk -F'\t' '{s+=$2} END {print s+0}')

note ""
note "  totals: $files files, +$ins/-$dels"
note "  expect: $EXPECT_FILES files, +$EXPECT_INSERTIONS/-$EXPECT_DELETIONS"

[ "$files" = "$EXPECT_FILES" ] || bad "file count $files != expected $EXPECT_FILES"
[ "$ins" = "$EXPECT_INSERTIONS" ] || bad "insertions $ins != expected $EXPECT_INSERTIONS"
[ "$dels" = "$EXPECT_DELETIONS" ] || bad "deletions $dels != expected $EXPECT_DELETIONS"

# 4. name any file that is new relative to the expected set, since a NEW shipped
#    file is the specific failure this gate was written to catch.
EXPECTED_SET="Sources/MLXFastModel/Qwen35RuntimeWeights.swift
Sources/MLXFastModel/Qwen36MTPBlockSession.swift
Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
actual_set="$(printf '%s\n' "$numstat" | awk -F'\t' 'NF{print $3}' | sort)"
expected_sorted="$(printf '%s\n' "$EXPECTED_SET" | sort)"
if [ "$actual_set" != "$expected_sorted" ]; then
  note ""
  note "  set difference vs the ledger-140 surface:"
  diff <(printf '%s\n' "$expected_sorted") <(printf '%s\n' "$actual_set") \
    | sed 's/^/    /' || true
  bad "the set of shipped files changed"
fi

# 5. the twin invariant: quantized.h and quantized.cpp must move together.
#
# LEDGER 162: this check compared raw numstat and was therefore a FALSE POSITIVE
# GENERATOR. mlx-generated/quantized.cpp is a GENERATED source string and it does
# not carry the kernel header's comments, so a comment-only edit to quantized.h
# can never appear in the twin. The rebase produced exactly that case: the whole
# +13/-3 delta on quantized.h is comment text, zero code, and the gate reported
# "the JIT twin is out of sync". A gate that cries wolf on a zero-risk edit
# trains you to ignore it, which is strictly worse than having no gate -- so the
# invariant is now stated over CODE lines, which is what it always meant.
#
# The comment in question is worth recording: the frontier carries a 12-line
# comment asserting M = 8 uses "3+3+2, not 4+4" and "IPG 3" directly above code
# that reads <T, 8, 4, true>, i.e. 4+4. The frontier's comment contradicts the
# frontier's own code. We keep it byte-identical anyway, because byte-identity
# with the frontier is what stops the next whole-file overlay reverting our work
# silently; the defect is recorded rather than patched.
code_delta() { # $1 = path -> "adds/dels" over non-comment, non-blank lines
  local p="$1" a d
  a=$(git diff "$BASELINE" "$REV" -- "$p" | grep -E '^\+' | grep -vE '^\+\+\+' \
      | sed -E 's/^\+[[:space:]]*//' | grep -vE '^(//|/\*|\*|$)' | grep -c . || true)
  d=$(git diff "$BASELINE" "$REV" -- "$p" | grep -E '^-' | grep -vE '^---' \
      | sed -E 's/^-[[:space:]]*//' | grep -vE '^(//|/\*|\*|$)' | grep -c . || true)
  printf '%s/%s' "$a" "$d"
}
h_raw=$(printf '%s\n' "$numstat" | awk -F'\t' '$3 ~ /kernels\/quantized\.h$/ {print $1"/"$2}')
c_raw=$(printf '%s\n' "$numstat" | awk -F'\t' '$3 ~ /mlx-generated\/quantized\.cpp$/ {print $1"/"$2}')
h_code=$(code_delta "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h")
c_code=$(code_delta "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp")
note ""
note "  twin check (raw, comments included):  quantized.h ${h_raw:-0/0}  vs  quantized.cpp ${c_raw:-0/0}"
note "  twin check (CODE only, load-bearing): quantized.h ${h_code}  vs  quantized.cpp ${c_code}"
if [ "$h_code" != "$c_code" ]; then
  bad "quantized.h and quantized.cpp diverged in CODE -- the JIT twin is out of sync. The AOT metallib and the runtime JIT would compile different kernels."
elif [ "${h_raw:-0/0}" != "${c_raw:-0/0}" ]; then
  note "    (raw counts differ but code is in sync: comment-only divergence, no runtime effect)"
fi

note ""
if [ "$fail" = 0 ]; then
  note "shipped-surface gate: PASS"
else
  note "shipped-surface gate: FAIL"
fi
exit "$fail"

#!/usr/bin/env bash
# E121 rung 3 -- the 512-token exactness gate for the shipped gated
# chunk-sum share.
#
#   usage: research/e121_exact512.sh
#
# WHY 512 AND NOT 64. The seed is 512 tokens, so only a 512-token decode window
# walks the key length past 1024 and exercises the boundary the row ledger has
# to close over. A 64-token leg never reaches it.
#
# WHY THIS GATE AND NOT AN ARGMAX MATCH. The arm changes WHERE a per-lane
# activation chunk sum is computed, not the order in which its terms are added:
# lane L of simdgroup 0 and lane L of simdgroup 1 walk the same k range in the
# same order, so the exchanged float is bit-identical to the one the receiving
# simdgroup would have produced. The claim is bit exactness, so the check is
# over `mtp-row:` lines whose top-two values are hex float literals.
#
# THE CONTROL. `research/e116_row_digest_check.py` runs a value control and an
# order control on its own. The runtime control has to change the decode
# window, because no compliant candidate knob can move this digest
# (`research/e116-results.md:191-204`); `MLX_E80_FORCE_DRAFTS` changes `d=` in
# the trace and leaves the rows alone. So the second leg here is 128 tokens and
# its digest MUST NOT match the pin.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PIN=719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e121_exact512: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  exit 1
fi

mkdir -p research/e121-artifacts
failures=0

echo "=== e121x512cand: candidate, tokens=512 ==="
research/e79_trace_leg.sh e121x512cand 512 \
  || { echo "e121_exact512: 512-token leg failed" >&2; failures=$((failures + 1)); }

echo
echo "=== e121x128neg: candidate, tokens=128 (runtime negative control) ==="
research/e79_trace_leg.sh e121x128neg 128 \
  || { echo "e121_exact512: 128-token control leg failed" >&2
       failures=$((failures + 1)); }

echo
python3 research/e116_row_digest_check.py e121x512cand \
  --pin "${PIN}" \
  --expect-rows 1025 \
  --negative-control e121x128neg \
  --json research/e121-artifacts/row-digest-512.json \
  || failures=$((failures + 1))

echo
echo "--- wrapper verdicts ---"
for tag in e121x512cand e121x128neg; do
  echo "${tag}: $(grep -oE 'all_tokens_matched[": ]*[a-z]+' \
    "research/out/${tag}/wrapper.out" 2>/dev/null | tail -1)"
  echo "${tag}: $(grep -oE '"passed"[: ]*[a-z]+' \
    "research/out/${tag}/score.json" 2>/dev/null | tail -1)"
done

echo
echo "e121_exact512: ${failures} failures"
exit "${failures}"

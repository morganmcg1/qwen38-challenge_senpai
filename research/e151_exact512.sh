#!/usr/bin/env bash
# E151 R1 -- the 512-token exactness gate for the arm-on NAX seed-prefill
# retile, plus the 512-token `--local-submit` leg that carries row-ledger
# closure and post-EOS continuation.
#
#   usage: research/e151_exact512.sh
#
# WHAT THIS CAN AND CANNOT PROVE. This host reports `is_nax_available()`
# false, so `qmm_t_nax` never runs here and the retiled kernel never executes.
# Nothing below is an effect estimate for the arm. What it does prove is that
# the arm-on tree still builds, still loads, and still reproduces the campaign
# row evidence bit for bit on the non-NAX path, which is the only failure mode
# a local run can catch.
#
# WHY 512 AND NOT 64. The seed is 512 tokens, so only a 512-token decode window
# walks the key length past 1024 and exercises the boundary the row ledger has
# to close over. The public golden also puts its first EOS at generated index
# 301, so only a window longer than 301 carries post-EOS continuation evidence.
#
# WHY A DIGEST AND NOT AN ARGMAX MATCH. `program.md` requires actual
# floating-point values at the touched cells. A `mtp-row:` line carries the
# top-two target scores as hex float literals, so the digest moves on a single
# changed low bit. An argmax match would not.
#
# THE CONTROL. `research/e116_row_digest_check.py` runs a value control and an
# order control on the rows themselves. The runtime control has to change the
# decode window, because no compliant candidate knob may move this digest: a
# `mtp-row:` line is emitted once per emitted token position for each of the
# wrapper's two passes, so a 512-token leg carries 1025 rows and a 128-token
# leg carries 257. The 128-token leg's digest MUST NOT match the pin.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Pinned by E121 rung 3, several campaign bases before this experiment existed.
# STALE PIN. This 1025-row digest was recorded on E89 base f18400c4 with 78
# trace rounds. Base de8ce44c now produces 1024 rows over 82 trace rounds and
# digest d070b3978940ee13f5a3ce224f1ac56e8c105c87d8d12220c7c6256ec683e158.
# research/e151_arm_attribution.sh proved the drift belongs to the intervening
# scheduler commits, not to the retile arm: the arm-off base and two
# independent arm-on rebuilds all emit d070b397 (research/e151-arm-attribution.json).
# Left unchanged so the recorded E151 failure stays auditable. Re-pinning is a
# campaign decision, and this constant is shared with E101, E110, E116, E121
# and E129.
PIN=719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e151_exact512: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

# E151_LABEL keeps one rung's evidence from overwriting another's. Unset
# reproduces the R1 tags and filenames exactly.
LABEL="${E151_LABEL:-}"
CAND_TAG="e151x512cand${LABEL}"
NEG_TAG="e151x128neg${LABEL}"
SUF="${LABEL:+-${LABEL}}"

out=research/e151-artifacts
mkdir -p "${out}"
failures=0

echo "=== ${CAND_TAG}: arm-on candidate, tokens=512, row evidence ==="
research/e79_trace_leg.sh "${CAND_TAG}" 512 \
  || { echo "e151_exact512: 512-token leg failed" >&2
       failures=$((failures + 1)); }

echo
echo "=== ${NEG_TAG}: arm-on candidate, tokens=128, runtime control ==="
research/e79_trace_leg.sh "${NEG_TAG}" 128 \
  || { echo "e151_exact512: 128-token control leg failed" >&2
       failures=$((failures + 1)); }

echo
python3 research/e116_row_digest_check.py "${CAND_TAG}" \
  --pin "${PIN}" \
  --expect-rows 1025 \
  --negative-control "${NEG_TAG}" \
  --json "${out}/row-digest-512${SUF}.json" \
  || failures=$((failures + 1))

echo
echo "--- wrapper verdicts ---"
for tag in "${CAND_TAG}" "${NEG_TAG}"; do
  echo "${tag}: $(grep -oE 'all_tokens_matched[": ]*[a-z]+' \
    "research/out/${tag}/wrapper.out" 2>/dev/null | tail -1)"
  echo "${tag}: $(grep -oE '"passed"[: ]*[a-z]+' \
    "research/out/${tag}/score.json" 2>/dev/null | tail -1)"
done

echo
echo "=== e151 --local-submit at 512 tokens: the gate-qualified leg ==="
# The real 40 C gate stays on here. This leg is the submission-readiness
# verdict and it carries `reference_checked_rows=<done>/<total>`, which is the
# row-ledger closure evidence.
(
  export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512
  export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
  export MLXFAST_SCORE_PATH="${PWD}/${out}/local-submit-512${SUF}.json"
  ./benchmark-qwen-mtp.sh --local-submit
) > "${out}/local-submit-512${SUF}.log" 2>&1 \
  || { echo "e151_exact512: --local-submit leg failed" >&2
       failures=$((failures + 1)); }

echo "--- local-submit verdict ---"
grep -oE '"passed"[: ]*[a-z]+' \
  "${out}/local-submit-512${SUF}.json" 2>/dev/null | tail -1
grep -oE 'reference_checked_rows=[0-9]+/[0-9]+' \
  "${out}/local-submit-512${SUF}.log" 2>/dev/null | sort -u
grep -oE 'cool-down gate passed \(current [0-9.]+C[^)]*\)' \
  "${out}/local-submit-512${SUF}.log" 2>/dev/null

echo
echo "e151_exact512: failures=${failures}"
exit "${failures}"

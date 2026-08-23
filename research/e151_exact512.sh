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
PIN=719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e151_exact512: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

out=research/e151-artifacts
mkdir -p "${out}"
failures=0

echo "=== e151x512cand: arm-on candidate, tokens=512, row evidence ==="
research/e79_trace_leg.sh e151x512cand 512 \
  || { echo "e151_exact512: 512-token leg failed" >&2
       failures=$((failures + 1)); }

echo
echo "=== e151x128neg: arm-on candidate, tokens=128, runtime control ==="
research/e79_trace_leg.sh e151x128neg 128 \
  || { echo "e151_exact512: 128-token control leg failed" >&2
       failures=$((failures + 1)); }

echo
python3 research/e116_row_digest_check.py e151x512cand \
  --pin "${PIN}" \
  --expect-rows 1025 \
  --negative-control e151x128neg \
  --json "${out}/row-digest-512.json" \
  || failures=$((failures + 1))

echo
echo "--- wrapper verdicts ---"
for tag in e151x512cand e151x128neg; do
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
  export MLXFAST_SCORE_PATH="${PWD}/${out}/local-submit-512.json"
  ./benchmark-qwen-mtp.sh --local-submit
) > "${out}/local-submit-512.log" 2>&1 \
  || { echo "e151_exact512: --local-submit leg failed" >&2
       failures=$((failures + 1)); }

echo "--- local-submit verdict ---"
grep -oE '"passed"[: ]*[a-z]+' "${out}/local-submit-512.json" 2>/dev/null | tail -1
grep -oE 'reference_checked_rows=[0-9]+/[0-9]+' \
  "${out}/local-submit-512.log" 2>/dev/null | sort -u
grep -oE 'cool-down gate passed \(current [0-9.]+C[^)]*\)' \
  "${out}/local-submit-512.log" 2>/dev/null

echo
echo "e151_exact512: failures=${failures}"
exit "${failures}"

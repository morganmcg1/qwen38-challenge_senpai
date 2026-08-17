#!/usr/bin/env bash
# Research-only driver: E15 Phase 2 exactness pair, control then candidate,
# back to back in one process so both arms share one build and one hot state.
#
#   research/run-draft-bits-phase2.sh TAG_PREFIX [TOKENS] [BASE_SHA]
#
# Two jobs cannot be launched in parallel for this: await-lock-then-run.sh waits
# for the benchmark lock to clear but does not hold it, so concurrent arms would
# both see it free and the loser would die on benchmark.sh's fail-fast take.
#
# TOKENS defaults to 256. The current base re-stops the block session on
# `stopTokens` (config.json eos_token_id = [248046, 248044]), and the
# longcopy-gate golden emits 248044 at decode index 301, so any window past 302
# ends in `notBegun` on both arms regardless of draft precision. 256 is the
# public --local-iterate golden's own length and contains no stop token, so both
# arms can assert a full-length exact match over the whole fixture.
set -euo pipefail

prefix="${1:?usage: run-draft-bits-phase2.sh TAG_PREFIX [TOKENS] [BASE_SHA]}"
tokens="${2:-256}"
base_sha="${3:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLX_QWEN_MTP_TRACE="${MLX_QWEN_MTP_TRACE:-1}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen38-r1-e15-draft-readout-3bit}"

for bits in 4 3; do
  echo "=== run-draft-bits-phase2: arm bits=${bits} tokens=${tokens} ==="
  research/await-lock-then-run.sh 900 \
    research/run-draft-bits-arm.sh "${bits}" "${prefix}-b${bits}" \
    "${tokens}" "${base_sha}"
done

echo "=== run-draft-bits-phase2: exactness summary ==="
for bits in 4 3; do
  python3 - "${repo_root}/.mlxfast-private/draft-bits/${prefix}-b${bits}/amdahl.json" "${bits}" <<'PY'
import json, sys
path, bits = sys.argv[1], sys.argv[2]
r = json.load(open(path))
for leg in ("serial_leg", "mtp_leg"):
    d = r.get(leg, {})
    print("phase2 bits=%s %s matched=%s emitted=%s spt=%s" % (
        bits, leg, d.get("all_tokens_matched"),
        d.get("emitted_token_total"),
        d.get("parent_measured_seconds_per_token")))
PY
done

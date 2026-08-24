#!/usr/bin/env bash
# E171 pre-flight: prove the staged S1 worker runs end to end before the gated
# session spends four thermal legs on it.
#
#   usage: research/e171_preflight.sh [TOKENS]
#
# The staged arm workers live outside the checkout, so this also proves the
# worker sandbox accepts an executable and a metallib on that path. It is
# ungated and short: it is a smoke test, not a measurement, and it reports no
# timing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-8}"
workers_root="${MLXFAST_E171_WORKERS:-$(cd ../.. && pwd)/e171-workers}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden_path="correctness_prompts/public_longcopy_gate_english_512_1024.json"
out="research/out/e171/preflight"

rm -rf "${out}"; mkdir -p "${out}"
eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"
export MLXFAST_USE_RUNTIME_WORKER=1
export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${workers_root}/s1/mlxfast-runtime-worker"
export MLXFAST_MLX_METALLIB="${workers_root}/s1/mlx.metallib"

jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden_path}" > "${out}/seed-plan.json"

"${swift_bin}" mtp-verify \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --emitted "${out}/seed-plan.json" \
  --generate "$(( tokens + 1 ))" \
  --mtp-depth 8 \
  --output "${out}/golden-rows.json" \
  --plan-output "${out}/generated-plan.json" \
  > "${out}/verify.out" 2> "${out}/verify.err"
status=$?

echo "worker      ${MLXFAST_RUNTIME_WORKER_EXECUTABLE}"
echo "exit        ${status}"
if [[ "${status}" != "0" ]]; then
  tail -20 "${out}/verify.err" >&2
  echo "e171_preflight: FAIL"
  exit "${status}"
fi
jq -r '"rows        \(.rows | length)\nself_consistent \(.reference_self_consistent)"' \
  "${out}/golden-rows.json"
echo "e171_preflight: PASS"

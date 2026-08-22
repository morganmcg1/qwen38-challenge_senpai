#!/usr/bin/env bash
# Bisect the rung 5e failure: run the wrapper's own MTP reference pass directly
# so the worker's stderr is visible, at several offered draft depths.
#
#   usage: research/e120_refprobe.sh [depths...]   (default "2 4 8")
#
# No timing is taken here, so no cool gate is required.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

depths=("$@")
[[ ${#depths[@]} -gt 0 ]] || depths=(2 4 8)

out_dir="research/out/e120-refprobe"
mkdir -p "${out_dir}"

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden="correctness_prompts/public_longcopy_gate_english_512_256.json"

plan_path="${out_dir}/seed-plan.json"
jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' "${public_golden}" > "${plan_path}"

echo "swift_bin=${swift_bin}"
echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
echo "arm=${MLXFAST_QWEN_E120_QMV:-<unset>}"

overall=0
for depth in "${depths[@]}"; do
  echo
  echo "=== mtp-verify --generate 4 --mtp-depth ${depth} ==="
  "${swift_bin}" mtp-verify \
    --weights "${weights_path}" \
    --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
    --emitted "${plan_path}" \
    --generate 4 \
    --mtp-depth "${depth}" \
    --output "${out_dir}/golden-d${depth}.json" \
    --plan-output "${out_dir}/plan-d${depth}.json"
  rc=$?
  echo "depth=${depth} rc=${rc}"
  [[ ${rc} -eq 0 ]] || overall=${rc}
done

echo
echo "overall=${overall}"
exit "${overall}"

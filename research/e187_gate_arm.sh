#!/usr/bin/env bash
# E187 Stage 1: run ONE offline arm of the GDN recurrent-state storage gate.
#
#   usage: research/e187_gate_arm.sh ARM [TOKENS] [LABEL]
#
# ARM is the value of MLX_E187_STATE_STORE: fp32 (pinned), bf16, fp16, ulp1,
# ulpbf16. The run generates the serial reference chain on the public fixture
# with `mtp-verify --generate`, which records per-token argmax and top-2 logits.
# Nothing here is timed, so no cool gate and no score are involved.
#
# The ulp arms perturb ONE state cell at the FIRST store only, so their R5
# self-consistency replay (which rebuilds the frame from scratch, after the
# probe has fired) is EXPECTED to disagree. The golden file is written before
# that check runs, so a non-zero exit still leaves usable rows.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e187_gate_arm.sh ARM [TOKENS] [LABEL]}"
tokens="${2:-512}"
label="${3:-${arm}}"

out_dir="research/out/e187-gate"
mkdir -p "${out_dir}"

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden="correctness_prompts/public_longcopy_gate_english_512_256.json"

plan_path="${out_dir}/seed-plan.json"
jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden}" > "${plan_path}"

echo "arm=${arm} tokens=${tokens} label=${label}"
echo "swift_bin=${swift_bin}"
echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "commit=$(git rev-parse HEAD)"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

MLX_E187_STATE_STORE="${arm}" "${swift_bin}" mtp-verify \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --emitted "${plan_path}" \
  --generate "${tokens}" \
  --output "${out_dir}/golden-${label}.json" \
  --plan-output "${out_dir}/plan-${label}.json"
rc=$?

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "rc=${rc}"
exit "${rc}"

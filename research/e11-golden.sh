#!/usr/bin/env bash
# Research-only (qwen38-r1-e11-depth-lever-showdown, T3): generate ONE extra
# public golden from a prose prompt, so the depth levers can be re-measured on
# a prompt whose acceptance profile is not the copy fixture's.
#
# WHY: --local-iterate's only fixture is public_longcopy_gate_english_512_256,
# a copy task. It realises accepted-draft rates of 0.89..0.95, near the best
# case for deep drafting, while the hidden pool is eight prose prompts whose
# calibration ratios span 0.8467..1.0726. Every E11 depth conclusion is
# therefore fitted to one unusually draft-friendly prompt until something else
# is measured. This produces that something else.
#
# NOT A FIXTURE CHANGE: nothing under fixtures/ or correctness_prompts/ is
# touched, the output lands outside Git in .mlxfast-private/, and the benchmark
# selects it through the override MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE that
# benchmark-qwen-mtp.sh already publishes. Yukon submits none of it.
#
# The prompt text IS committed (research/e11_prose_gate_english_512.txt) so the
# golden is regenerable; the generated JSON is not, per the campaign rule on
# large measurement artifacts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prompt="research/e11_prose_gate_english_512.txt"
out_dir=".mlxfast-private/e11/goldens"
out="${out_dir}/e11_prose_512_256.json"
name="e11_prose_gate_english_512_256"
steps="${E11_GOLDEN_STEPS:-256}"

mkdir -p "${out_dir}"

# Generate with whichever arm binary is resident: generate-golden runs the
# SERIAL reference path (QwenRuntime.generateGreedyTokens), which no E11 arm
# touches -- every arm edits only Qwen36MTPBlockSession constants. Recorded
# anyway, so the claim is checkable rather than asserted.
echo "e11-golden: resident worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e11-golden: resident cli    $(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
echo "e11-golden: prompt sha256   $(shasum -a 256 "${prompt}" | cut -d' ' -f1)"

.build/release/mlxfast-swift generate-golden \
  --prompt-file "${prompt}" \
  --output "${out}" \
  --name "${name}" \
  --steps "${steps}"
rc=$?
if ((rc == 0)); then
  echo "e11-golden: wrote ${out} ($(wc -c < "${out}" | tr -d ' ') bytes)"
  shasum -a 256 "${out}"
fi
exit "${rc}"

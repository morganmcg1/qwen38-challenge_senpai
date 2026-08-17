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

# usage: research/e11-golden.sh [PROMPT_FILE ...]
#
# With no argument this keeps E11's single prose prompt and filename. With
# arguments it generates one golden per prompt file, named after the prompt's
# basename, so a prompt SET can be measured without a second copy of this
# script. E11_GOLDEN_DIR relocates the output for a later experiment.
out_dir="${E11_GOLDEN_DIR:-.mlxfast-private/e11/goldens}"
steps="${E11_GOLDEN_STEPS:-512}"
prompts=("$@")
if ((${#prompts[@]} == 0)); then
  prompts=("research/e11_prose_gate_english_512.txt")
fi

mkdir -p "${out_dir}"

golden_paths() {
  local prompt="$1" stem
  stem="$(basename "${prompt}" .txt)"
  if [[ "${prompt}" == "research/e11_prose_gate_english_512.txt" \
        && "${out_dir}" == ".mlxfast-private/e11/goldens" ]]; then
    # E11's committed filename, kept so its runs stay reproducible.
    echo ".mlxfast-private/e11/goldens/e11_prose_512_${steps}.json" \
         "e11_prose_gate_english_512_${steps}"
  else
    echo "${out_dir}/${stem}_${steps}.json" "${stem}_${steps}"
  fi
}

# Generate with whichever arm binary is resident: generate-golden runs the
# SERIAL reference path (QwenRuntime.generateGreedyTokens), which no E11 arm
# touches -- every arm edits only Qwen36MTPBlockSession constants. Recorded
# anyway, so the claim is checkable rather than asserted.
echo "e11-golden: resident worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e11-golden: resident cli    $(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"

status=0
for prompt in "${prompts[@]}"; do
  read -r out name <<<"$(golden_paths "${prompt}")"
  echo "=== e11-golden: ${prompt} -> ${out} (${steps} steps) ==="
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
  else
    echo "e11-golden: ${prompt}: generate-golden exited ${rc}" >&2
    status=1
    break
  fi
done
exit "${status}"

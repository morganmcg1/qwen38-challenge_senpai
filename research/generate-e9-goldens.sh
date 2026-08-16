#!/usr/bin/env bash
# Research-only: author the two new public-shape correctness fixtures that the
# E9 multi-prompt arms run against.
#
#   research/generate-e9-goldens.sh [STEPS]
#
# `generate-golden` tokenizes the seed with the weights-dir tokenizer, keeps
# exactly correctnessPromptTokens (512) tokens, greedy-generates STEPS
# reference tokens with the reference model, and writes a strict-loader-valid
# version-1 fixture. STEPS must be >= correctnessSteps (64); the drift tripwire
# in benchmark-qwen-mtp.sh calls `correctness` with no --steps, so 64 is what it
# actually checks. The timed legs never read expected_tokens: they seed the MTP
# reference from cases[0].prompt_tokens and regenerate their own rows.
#
# These goldens are generated on THIS host, so unlike the checked-in M5-authored
# public fixtures they cannot produce a cross-machine near-tie mismatch here.
set -euo pipefail

steps="${1:-64}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker

for name in technical narrative; do
  seed="correctness_prompts/research_e9_${name}_512.txt"
  out="correctness_prompts/research_e9_${name}_512_${steps}.json"
  echo "generate-e9-goldens: ${seed} -> ${out} (${steps} steps)" >&2
  .build/release/mlxfast-swift generate-golden \
    --prompt-file "${seed}" \
    --weights "${weights_path}" \
    --output "${out}" \
    --name "research_e9_${name}_512" \
    --steps "${steps}"

  # The fixture is only useful if the tripwire it will gate actually passes on
  # this host and this build; run that exact verb now rather than discovering it
  # inside a timed arm.
  .build/release/mlxfast-swift correctness \
    --weights "${weights_path}" \
    --golden "${out}" \
    | jq -e '.passed == true' >/dev/null
  echo "generate-e9-goldens: ${out} passes the drift tripwire verb" >&2
done

for name in technical narrative; do
  out="correctness_prompts/research_e9_${name}_512_${steps}.json"
  jq -r --arg p "${out}" '
    "\($p): prompt_tokens=\(.cases[0].prompt_tokens|length) " +
    "expected_tokens=\(.cases[0].expected_tokens|length) " +
    "name=\(.cases[0].name)"' "${out}"
  shasum -a 256 "${out}"
done

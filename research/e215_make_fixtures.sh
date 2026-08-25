#!/usr/bin/env bash
# Research-only (qwen38-r1-e215): turn the E215 prompt texts into local golden
# fixtures the Qwen MTP wrapper accepts.
#
#   usage: research/e215_make_fixtures.sh [STEPS]     # default 64
#
# `benchmark-qwen-mtp.sh` uses the fixture for exactly two things: the public
# drift tripwire (`mlxfast-swift correctness`) and `cases[0].prompt_tokens`,
# which seeds the MTP reference pass. The expected tokens are generated HERE on
# this host by the same greedy target, so the tripwire compares this build with
# itself and cannot fail for a hardware reason. 64 steps is the smallest window
# the strict fixture loader accepts, and the tripwire is a drift check, not the
# experiment.
#
# These fixtures live under research/ and never enter fixtures/ or
# correctness_prompts/. They are not a reference for the hidden pool and are
# not a submitted path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

steps="${1:-64}"
prompts=(e215-eos-short e215-code e215-story)
prompt_dir="research/e215-artifacts/prompts"
out_dir="research/e215-artifacts/fixtures"
cli=".build/release/mlxfast-swift"

mkdir -p "${out_dir}"

for name in "${prompts[@]}"; do
  prompt="${prompt_dir}/${name}.txt"
  [[ -s "${prompt}" ]] || {
    echo "e215_make_fixtures.sh: missing ${prompt}; run research/e215_make_prompts.py" >&2
    exit 2; }
  out="${out_dir}/${name}_512_${steps}.json"
  echo "=== ${name}: generating ${steps} reference tokens"
  "${cli}" generate-golden \
    --prompt-file "${prompt}" \
    --weights weights \
    --tokenizer weights \
    --output "${out}" \
    --name "${name}" \
    --steps "${steps}" || exit $?
  python3 - "${out}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
case = doc["cases"][0]
stops = {248044, 248046}
hits = [i for i, t in enumerate(case["expected_tokens"]) if t in stops]
print(f"    prompt_tokens={len(case['prompt_tokens'])} "
      f"expected_tokens={len(case['expected_tokens'])} "
      f"stop_token_positions={hits[:5]}")
PY
done

echo "=== done"

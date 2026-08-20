#!/usr/bin/env bash
# E82: build every head arm of the rung-0 screen, outside Git.
#
# The screen is a 2x2 plus one. Trunk weights (stock master vs xkm fine-tune)
# cross with trunk coding (4-bit g64 vs BF16), and every arm carries the SAME
# byte-identical affine-2 draft readout, so the fine-tune effect and the
# quantization effect can be read separately instead of as one confounded
# difference:
#
#            stock trunk                 xkm trunk
#   4-bit    A  declared (shipped)       B  e82-soup-q4
#   BF16     C  e82-master-bf16          D  Kamciosz (published)
#   plus     E  e82-qat-q4  -- xkm's quantization-aware parent, the only
#                             published head trained to survive step B
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
py=python3

echo "=== island selection rule replay against the shipped declared head ==="
${py} research/e82_build_head.py verify

for spec in "soup q4 e82-soup-q4" "qat q4 e82-qat-q4" "master bf16 e82-master-bf16"; do
  read -r source trunk tag <<<"${spec}"
  echo "=== build ${tag} (source=${source} trunk=${trunk}) ==="
  ${py} research/e82_build_head.py build --source "${source}" --trunk "${trunk}" --tag "${tag}"
done

# The published Kamciosz graft needs the same run-tree shape as the built arms:
# one model.safetensors on the single-file declared-head branch, plus the head
# config.json that benchmark-qwen-mtp.sh insists on and the loader ignores.
kam_run="${cache}/e82/built/e82-kamciosz-run"
mkdir -p "${kam_run}"
rm -f "${kam_run}/model.safetensors"
ln "${cache}/e82/kamciosz-graft/model.safetensors" "${kam_run}/model.safetensors"
cp "${cache}/mtp-head/config.json" "${kam_run}/config.json"
echo "run tree staged at ${kam_run}"

echo
echo "=== arm run trees ==="
for d in "${cache}/mtp-head-declared-run" "${cache}/e82/built"/*-run "${kam_run}"; do
  [[ -d "${d}" ]] || continue
  printf '%-70s %s\n' "${d}" "$(wc -c < "${d}/model.safetensors" | tr -d ' ') B"
done

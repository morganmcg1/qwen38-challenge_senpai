#!/usr/bin/env bash
# E82 rung 0: the untimed acceptance screen.
#
# Two stages, because the reference trajectory must not depend on the head:
#
#   goldens  generate-golden walks the SERIAL greedy chain once per seed. No
#            head participates, so every arm is later scored against the same
#            token stream and any acceptance difference is the head's alone.
#   verify   mtp-verify replays each golden under each head at --mtp-depth 8.
#            It is untimed, so no thermal gate applies and the numbers are
#            acceptance evidence only -- never a speedup.
#
# Usage:
#   research/e82_screen.sh goldens [--steps N] [--only SEED[,SEED...]]
#   research/e82_screen.sh verify  [--depth D] [--arms A[,A...]] [--only SEED]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
seeds_dir="${cache}/e82/corpus/seeds"
out="${cache}/e82/screen"
cli=.build/release/mlxfast-swift

# Arm -> head run tree. Every arm carries the byte-identical affine-2 draft
# readout except `pinned`, which is the organizer default and is included so
# this screen ties back to the E79 timed numbers.
declare -a ARM_ORDER=(declared soup-q4 qat-q4 master-bf16 kamciosz pinned)
arm_head() {
  case "$1" in
    declared)    echo "${cache}/mtp-head-declared-run" ;;
    soup-q4)     echo "${cache}/e82/built/e82-soup-q4-run" ;;
    qat-q4)      echo "${cache}/e82/built/e82-qat-q4-run" ;;
    master-bf16) echo "${cache}/e82/built/e82-master-bf16-run" ;;
    kamciosz)    echo "${cache}/e82/built/e82-kamciosz-run" ;;
    pinned)      echo "${cache}/mtp-head" ;;
    *) echo "e82-screen: unknown arm '$1'" >&2; return 1 ;;
  esac
}

stage="${1:-}"; shift || true
steps=512
depth=8
only=""
arms="$(IFS=,; echo "${ARM_ORDER[*]}")"
while (($#)); do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --only)  only="$2"; shift 2 ;;
    --arms)  arms="$2"; shift 2 ;;
    *) echo "e82-screen: unknown flag '$1'" >&2; exit 2 ;;
  esac
done

seed_names() {
  python3 - "$only" <<'PY'
import json, sys
only = sys.argv[1]
keep = set(filter(None, only.split(","))) if only else None
for s in json.load(open("research/e82-corpus-manifest.json"))["seeds"]:
    if keep is None or s["name"] in keep:
        print(s["name"])
PY
}

echo "e82-screen: cli    $(shasum -a 256 ${cli} | cut -d' ' -f1)"
echo "e82-screen: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e82-screen: head   $(git rev-parse HEAD)"

case "${stage}" in
goldens)
  mkdir -p "${out}/goldens"
  for name in $(seed_names); do
    dest="${out}/goldens/${name}_${steps}.json"
    if [[ -s "${dest}" ]]; then echo "=== skip ${name}: ${dest} exists ==="; continue; fi
    echo "=== golden ${name} (${steps} steps) ==="
    start=$(date +%s)
    ${cli} generate-golden \
      --prompt-file "${seeds_dir}/e82-${name}.txt" \
      --output "${dest}" \
      --name "e82_${name}_${steps}" \
      --steps "${steps}" || { echo "e82-screen: golden ${name} FAILED" >&2; exit 1; }
    echo "e82-screen: ${name} golden in $(( $(date +%s) - start ))s"
  done
  ;;
verify)
  mkdir -p "${out}/verify"
  IFS=, read -r -a arm_list <<<"${arms}"
  # Arm-major order: one head load serves every seed, and the arms stay in a
  # fixed order so a partial run is still a complete prefix of the design.
  for arm in "${arm_list[@]}"; do
    head_dir="$(arm_head "${arm}")" || exit 2
    mkdir -p "${out}/verify/${arm}"
    for name in $(seed_names); do
      golden="${out}/goldens/${name}_${steps}.json"
      [[ -s "${golden}" ]] || { echo "e82-screen: missing ${golden}" >&2; exit 1; }
      dest="${out}/verify/${arm}/${name}.json"
      if [[ -s "${dest}" ]]; then echo "=== skip ${arm}/${name} ==="; continue; fi
      echo "=== verify ${arm} / ${name} (depth ${depth}, ${steps} tokens) ==="
      start=$(date +%s)
      ${cli} mtp-verify \
        --golden "${golden}" \
        --mtp-head "${head_dir}" \
        --mtp-depth "${depth}" \
        --tokens "${steps}" \
        --output "${dest}" || { echo "e82-screen: ${arm}/${name} FAILED" >&2; exit 1; }
      echo "e82-screen: ${arm}/${name} in $(( $(date +%s) - start ))s"
    done
  done
  ;;
*)
  echo "usage: research/e82_screen.sh {goldens|verify} [flags]" >&2
  exit 2
  ;;
esac
echo "e82-screen: ${stage} done"

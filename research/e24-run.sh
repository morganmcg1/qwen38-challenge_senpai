#!/usr/bin/env bash
# Research-only (qwen38-r1-e24-constant-scalar-dispatch-tax): drive the paired
# BASE/MEMO measurement over the pre-registered prose prompt set.
#
#   research/e24-run.sh --goldens [ID ...]      generate the 512-step goldens
#   research/e24-run.sh --arms BASE,MEMO ID ... time that arm pair per prompt
#
# This is e17-run.sh's driver retargeted at E24's arms and roots. Its design
# commitments are inherited deliberately and restated here because they are the
# reason the output is believable, not boilerplate:
#
#  * The prompt set and its ORDER are `prompt_ids` below, fixed before any arm
#    was timed. Prompts are never added, dropped or reordered afterwards, and
#    every completed prompt is reported. A headline recovered by dropping a
#    prompt is worthless.
#  * Arms are interleaved WITHIN prompt, never blocked by arm, so thermal drift
#    across the session cannot correlate with arm.
#  * The within-prompt ORDER ALTERNATES with the prompt index, so a systematic
#    first-arm/second-arm effect (cold cache, cool-gate history) cannot
#    correlate with arm either. For two arms this is exactly ABBA.
#  * Each --local-iterate run measures its own byte-identical serial control at
#    depth 0 alongside the arm. E24 touches only the GDN QK-norm constants,
#    which the depth-0 serial leg executes just as the MTP leg does, so the
#    serial leg is NOT an inert control here: it is a second, independent
#    witness of the same effect, and its spread is the prompt's noise floor.
#
# WHY NOT THE PUBLIC GOLDEN: --local-iterate's shipped fixture is a copy task
# capped near 300 decode tokens by a stop-token defect, so it cannot carry a
# 512-token window. The prose set is generated here at the full 512 steps.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Eight registers mirroring the shape of the hidden pool.
prompt_ids=(english narrative technical dramatic travel philosophy
  natural_history medicine)

goldens_dir=".mlxfast-private/e24/goldens"
export E11_GOLDEN_DIR="${goldens_dir}"
export E11_GOLDEN_STEPS=512
export E11_BINS_ROOT="${PWD}/.mlxfast-private/e24/bins"
export E11_RUNS_ROOT="${PWD}/.mlxfast-private/e24/runs"
export E11_TOKENS=512

prompt_file_for() {
  case "$1" in
    english) echo "research/e11_prose_gate_english_512.txt" ;;
    *) echo "research/e17_prose_$1_512.txt" ;;
  esac
}

golden_for() {
  local stem
  stem="$(basename "$(prompt_file_for "$1")" .txt)"
  echo "${goldens_dir}/${stem}_512.json"
}

index_of() {
  local id="$1" i
  for i in "${!prompt_ids[@]}"; do
    [[ "${prompt_ids[i]}" == "${id}" ]] && { echo "${i}"; return 0; }
  done
  echo "e24-run: ${id} is not in the pre-registered prompt set" >&2
  return 1
}

if [[ "${1:-}" == "--goldens" ]]; then
  shift
  ids=("$@")
  ((${#ids[@]})) || ids=("${prompt_ids[@]}")
  files=()
  for id in "${ids[@]}"; do
    index_of "${id}" >/dev/null || exit 2
    files+=("$(prompt_file_for "${id}")")
  done
  # generate-golden runs the SERIAL reference path at depth 0. That path does
  # execute the GDN constants E24 changes, so the two arms could in principle
  # disagree here -- which is exactly why the golden is pinned to ONE named
  # build (BASE by default) rather than whatever binary happens to be resident.
  # Both arms are then checked against the same fixed reference rows.
  pin="${E11_BINS_ROOT}/${E24_GOLDEN_PIN:-BASE}"
  [[ -f "${pin}/sha256.txt" ]] || {
    echo "e24-run: no built arm at ${pin}; run research/e24-build.sh first" >&2
    exit 2
  }
  install -m 755 "${pin}/mlxfast-swift" .build/release/mlxfast-swift
  install -m 755 "${pin}/mlxfast-runtime-worker" \
    .build-worker/release/mlxfast-runtime-worker
  exec research/e11-golden.sh "${files[@]}"
fi

pair=()
if [[ "${1:-}" == "--arms" ]]; then
  IFS=, read -r -a pair <<< "${2:?--arms needs A,B}"
  shift 2
fi
((${#pair[@]} >= 1)) || {
  echo "usage: research/e24-run.sh --arms A[,B,...] ID [ID ...]" >&2; exit 2; }
for arm in "${pair[@]}"; do
  [[ -s "${E11_BINS_ROOT}/${arm}/sha256.txt" ]] || {
    echo "e24-run: arm ${arm} is not built" >&2; exit 2; }
done

((${#@})) || { echo "usage: research/e24-run.sh --arms A,B ID [ID ...]" >&2; exit 2; }

status=0
for id in "$@"; do
  idx="$(index_of "${id}")" || { status=2; break; }
  golden="$(golden_for "${id}")"
  if [[ ! -s "${golden}" ]]; then
    echo "e24-run: missing golden ${golden}; run --goldens ${id} first" >&2
    status=2; break
  fi
  rot=$((idx % ${#pair[@]}))
  arms=("${pair[@]:rot}" "${pair[@]:0:rot}")
  echo "=== e24-run: prompt ${id} (index ${idx}) arms ${arms[*]} ==="
  export E11_GOLDEN="${golden}"
  for arm in "${arms[@]}"; do
    research/e11-run.sh "${id}-${arm}=${arm}" || { status=1; break 2; }
  done
done
exit "${status}"

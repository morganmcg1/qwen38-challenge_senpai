#!/usr/bin/env bash
# Research-only (qwen38-r1-e17-curve-transfer-and-refit): drive the per-prompt
# paired measurement that asks whether the merged h curve survives the ranked
# aggregation (median over eight prose prompts), rather than winning on the one
# prompt E11 happened to hold out.
#
#   research/e17-run.sh --goldens [ID ...]   generate the 512-step goldens
#   research/e17-run.sh --arms A,B ID [ID ...]  time that arm pair per prompt
#
# DESIGN COMMITMENTS, fixed here before any arm was timed:
#
#  * The prompt set and its ORDER are the `prompt_ids` list below. Prompts are
#    never added, dropped or reordered after timing starts, and every completed
#    prompt is reported. That is the whole point of the experiment: a headline
#    recovered by prompt selection is worthless.
#  * Arms are interleaved WITHIN prompt, never blocked by arm, so thermal drift
#    across the session cannot correlate with arm.
#  * The within-prompt ORDER ALTERNATES with the prompt's index in the list, so
#    a systematic first-arm/second-arm effect (cold cache, cool-gate history)
#    cannot correlate with arm either.
#  * Each `--local-iterate` run measures its own byte-identical serial control
#    at depth 0 alongside the arm, so every prompt carries two independent
#    serial legs whose spread is that prompt's noise floor.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Eight registers mirroring the shape of the hidden pool (expository history,
# narrative, technical, dramatic dialogue, travel, philosophical argument,
# natural history, clinical medicine). `english` is E11's held-out prompt.
prompt_ids=(english narrative technical dramatic travel philosophy
  natural_history medicine)

goldens_dir=".mlxfast-private/e17/goldens"
export E11_GOLDEN_DIR="${goldens_dir}"
export E11_GOLDEN_STEPS=512
export E11_BINS_ROOT="${PWD}/.mlxfast-private/e17/bins"
export E11_RUNS_ROOT="${PWD}/.mlxfast-private/e17/runs"
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
  echo "e17-run: ${id} is not in the pre-registered prompt set" >&2
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
  # generate-golden runs the SERIAL reference path, which neither arm touches,
  # so either build yields the same rows; pin CURVE anyway so the goldens name
  # a hash-verified binary instead of whatever happened to be resident.
  pin="${E11_BINS_ROOT}/${E17_GOLDEN_PIN:-S18}"
  [[ -f "${pin}/sha256.txt" ]] || {
    echo "e17-run: no built arm at ${pin}; run research/e17-build.sh first" >&2
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
  echo "usage: research/e17-run.sh --arms A[,B,...] ID [ID ...]" >&2; exit 2; }
for arm in "${pair[@]}"; do
  [[ -s "${E11_BINS_ROOT}/${arm}/sha256.txt" ]] || {
    echo "e17-run: arm ${arm} is not built" >&2; exit 2; }
done

((${#@})) || { echo "usage: research/e17-run.sh --arms A,B ID [ID ...]" >&2; exit 2; }

status=0
for id in "$@"; do
  idx="$(index_of "${id}")" || { status=2; break; }
  golden="$(golden_for "${id}")"
  if [[ ! -s "${golden}" ]]; then
    echo "e17-run: missing golden ${golden}; run --goldens ${id} first" >&2
    status=2; break
  fi
  # Rotate the within-prompt order by prompt index. For the two-arm case this is
  # exactly the ABBA alternation the header commits to; it generalises to N arms
  # without giving any arm a fixed slot in the session.
  rot=$((idx % ${#pair[@]}))
  arms=("${pair[@]:rot}" "${pair[@]:0:rot}")
  echo "=== e17-run: prompt ${id} (index ${idx}) arms ${arms[*]} ==="
  export E11_GOLDEN="${golden}"
  for arm in "${arms[@]}"; do
    research/e11-run.sh "${id}-${arm}=${arm}" || { status=1; break 2; }
  done
done
exit "${status}"

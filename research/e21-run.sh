#!/usr/bin/env bash
# Research-only (qwen38-r1-e21-depth-preserving-row-declination): drive the
# per-prompt paired measurement for the row-declination arms.
#
#   research/e21-run.sh --probe ID ARM        one TRACED, non-timed pass
#   research/e21-run.sh --arm ID ARM          one TIMED arm on one prompt
#
# WHY ONE ARM PER INVOCATION. E17's `english` pair took 23.1 min wall
# (S18 10.1 + CURVE 13.0) because `benchmark-qwen-mtp.sh` runs its 40C cool
# gate three times INSIDE each arm. The launch limit here is 30 min per job, so
# a pair does not reliably fit and a job killed mid-arm would silently drop one
# half of a counterbalanced pair. One arm per job always fits, and the ABBA
# order is then owned by the caller's job sequence.
#
# DESIGN COMMITMENTS, inherited verbatim from E17 so the two experiments remain
# comparable, and fixed here before any arm was timed:
#
#  * The prompt set and its ORDER are the `prompt_ids` list below. Prompts are
#    never added, dropped or reordered after timing starts, and every completed
#    prompt is reported.
#  * Arms are interleaved WITHIN prompt, never blocked by arm, so thermal drift
#    across the session cannot correlate with arm.
#  * The within-prompt ORDER ALTERNATES with the prompt's index in the list, so
#    a systematic first-arm/second-arm effect (cold cache, cool-gate history)
#    cannot correlate with arm either. `--order` prints that schedule; it is
#    derived from the prompt index alone, never chosen per run.
#  * Each `--local-iterate` run measures its own byte-identical serial control
#    at depth 0 alongside the arm, so every prompt carries two independent
#    serial legs whose spread is that prompt's noise floor.
#
# The goldens are E17's: they are 512-step SERIAL reference rows, which neither
# arm's schedule can touch, and regenerating eight of them costs about a GPU
# hour. The first probe pass re-checks `all_tokens_matched` against them on this
# base, so a golden invalidated by the merges under this base fails loudly
# before any timed arm is spent.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prompt_ids=(english narrative technical dramatic travel philosophy
  natural_history medicine)

goldens_dir=".mlxfast-private/e17/goldens"
export E11_GOLDEN_DIR="${goldens_dir}"
export E11_GOLDEN_STEPS=512
export E11_BINS_ROOT="${PWD}/.mlxfast-private/e21/bins"
export E11_RUNS_ROOT="${PWD}/.mlxfast-private/e21/runs"
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
  echo "e21-run: ${id} is not in the pre-registered prompt set" >&2
  return 1
}

# The ABBA schedule, derived from the prompt index so it cannot be chosen per
# run. Printed rather than executed, because the caller owns the job sequence.
if [[ "${1:-}" == "--order" ]]; then
  IFS=, read -r -a pair <<< "${2:?--order needs A,B}"
  for i in "${!prompt_ids[@]}"; do
    rot=$((i % ${#pair[@]}))
    arms=("${pair[@]:rot}" "${pair[@]:0:rot}")
    echo "${prompt_ids[i]} ${arms[*]}"
  done
  exit 0
fi

mode="${1:?usage: research/e21-run.sh --probe|--arm ID ARM}"
id="${2:?usage: research/e21-run.sh --probe|--arm ID ARM}"
arm="${3:?usage: research/e21-run.sh --probe|--arm ID ARM}"

index_of "${id}" >/dev/null || exit 2
[[ -s "${E11_BINS_ROOT}/${arm}/sha256.txt" ]] || {
  echo "e21-run: arm ${arm} is not built" >&2; exit 2; }

golden="$(golden_for "${id}")"
[[ -s "${golden}" ]] || {
  echo "e21-run: missing golden ${golden}" >&2; exit 2; }
export E11_GOLDEN="${golden}"

case "${mode}" in
  --probe)
    # Never a source of headline timing: the trace gate buys per-round file I/O
    # inside the timed round. This pass exists to recover the depth histogram,
    # the per-round reach walk and the schedule's input scalars.
    export E11_TRACE=1
    exec research/e11-run.sh "probe-${id}-${arm}=${arm}"
    ;;
  --arm)
    unset E11_TRACE
    exec research/e11-run.sh "${id}-${arm}=${arm}"
    ;;
  *)
    echo "e21-run: unknown mode ${mode}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Research-only (qwen38-r1-e21-depth-preserving-row-declination): drive the
# per-prompt paired measurement for the row-declination arms.
#
#   research/e21-run.sh --probe ID ARM        one TRACED, non-timed pass
#   research/e21-run.sh --probe-sweep ARM ID... several TRACED passes
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

# THE REAL 40C GATE IS UNSATISFIABLE ON THIS HOST. E17 entered `english` at
# 39.14C and passed; this host now idles at ~42.6C, and the first E21 probe
# died after watching the GPU asymptote to 42.6C over 190s with nothing else
# loading it (`min seen 42.8C`). `COOL_GATE_TEMP_C = 40` is a compile-time
# constant in benchmark.sh:28, so no amount of waiting will clear it.
#
# The advisor authorised the bypass for exactly this case, conditional on:
# ABBA-in-one-session, entry/exit temps per arm, the two flags below carried
# VERBATIM, and the spread reported next to the effect. All four hold here.
# The flags are appended to every run's meta.txt below so that no downstream
# reader can mistake one of these numbers for a gate-qualified measurement.
export MLXFAST_LOCAL_COOL_GATE=0
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

run_one() {
  local mode="$1" id="$2" arm="$3" golden label rc

  index_of "${id}" >/dev/null || return 2
  [[ -s "${E11_BINS_ROOT}/${arm}/sha256.txt" ]] || {
    echo "e21-run: arm ${arm} is not built" >&2; return 2; }

  golden="$(golden_for "${id}")"
  [[ -s "${golden}" ]] || {
    echo "e21-run: missing golden ${golden}" >&2; return 2; }
  export E11_GOLDEN="${golden}"

  case "${mode}" in
    --probe)
      # Never a source of headline timing: the trace gate buys per-round file
      # I/O inside the timed round. This pass exists to recover the depth
      # histogram, the per-round reach walk and the schedule's input scalars.
      export E11_TRACE=1
      label="probe-${id}-${arm}"
      ;;
    --arm)
      unset E11_TRACE
      label="${id}-${arm}"
      ;;
    *)
      echo "e21-run: unknown mode ${mode}" >&2; return 2 ;;
  esac

  research/e11-run.sh "${label}=${arm}"
  rc=$?

  # Appended after the run because e11-run.sh recreates the directory. These
  # two names are the advisor's, spelled exactly as required, so that a grep
  # for either one finds every affected measurement.
  local meta="${E11_RUNS_ROOT}/${label}/meta.txt"
  if [[ -f "${meta}" ]]; then
    {
      echo "cool_gate_passed_real_gate=false"
      echo "gate_qualified_for_timing=false"
      echo "cool_gate_temp_c=40"
      echo "cool_gate_bypass_reason=host idles above the compile-time 40C gate"
    } >> "${meta}"
  fi
  return "${rc}"
}

# Sweep mode exists only for the untimed probe pass, where the runs are
# independent histogram samples rather than halves of a counterbalanced pair.
# Timed arms stay one-per-invocation so the caller keeps owning the ABBA order.
if [[ "${1:-}" == "--probe-sweep" ]]; then
  sweep_arm="${2:?--probe-sweep needs ARM ID...}"
  shift 2
  worst=0
  for sweep_id in "$@"; do
    echo "e21-run: probe ${sweep_id} ${sweep_arm}"
    run_one --probe "${sweep_id}" "${sweep_arm}" || worst=$?
  done
  exit "${worst}"
fi

mode="${1:?usage: research/e21-run.sh --probe|--arm ID ARM}"
id="${2:?usage: research/e21-run.sh --probe|--arm ID ARM}"
arm="${3:?usage: research/e21-run.sh --probe|--arm ID ARM}"

run_one "${mode}" "${id}" "${arm}"
exit $?

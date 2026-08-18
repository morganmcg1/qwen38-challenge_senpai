#!/usr/bin/env bash
# Research-only (qwen38-r1-e25-per-row-draft-price): drive the per-prompt paired
# measurement for the per-row draft-price arms.
#
#   research/e25-run.sh --order A,B            print the ABBA schedule
#   research/e25-run.sh --probe ID ARM         one TRACED, non-timed pass
#   research/e25-run.sh --arm ID ARM           one TIMED arm on one prompt
#   research/e25-run.sh --pair ID A B          both TIMED arms, ABBA order
#   research/e25-run.sh --pairs A,B ID...      several TIMED pairs, one job
#
# DESIGN COMMITMENTS, inherited verbatim from E17/E21 so all three remain
# comparable, and fixed here before any arm was timed:
#
#  * The prompt set and its ORDER are the `prompt_ids` list below. Prompts are
#    never added, dropped or reordered after timing starts, and every completed
#    prompt is reported.
#  * Arms are interleaved WITHIN prompt, never blocked by arm, so thermal drift
#    across the session cannot correlate with arm.
#  * The within-prompt ORDER ALTERNATES with the prompt's index in the list, so
#    a systematic first-arm/second-arm effect (cold cache, cool-gate history)
#    cannot correlate with arm either. `--pair` derives that order from the
#    prompt index alone; it is never chosen per run.
#  * Each `--local-iterate` run measures its own byte-identical serial control
#    at depth 0 alongside the arm, so every prompt carries two independent
#    serial legs whose spread is that prompt's noise floor.
#
# `--pair` exists because E21's one-arm-per-job rule was forced by the 40C cool
# gate running three times inside each arm (E17's `english` pair took 23.1 min).
# With the gate bypassed an arm is far shorter, so a pair fits the 30-min launch
# limit and keeps both halves of a counterbalanced pair inside one job -- which
# is what the counterbalancing is for. `--arm` remains for repairing one half.
#
# The goldens are E17's 512-step SERIAL reference rows, which neither arm's
# schedule can touch. The first probe pass re-checks `all_tokens_matched`
# against them on this base, so an invalidated golden fails loudly before any
# timed arm is spent.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prompt_ids=(english narrative technical dramatic travel philosophy
  natural_history medicine)

goldens_dir=".mlxfast-private/e17/goldens"
export E11_GOLDEN_DIR="${goldens_dir}"
export E11_GOLDEN_STEPS=512

# THE REAL 40C GATE IS UNSATISFIABLE ON THIS HOST. E17 entered `english` at
# 39.14C and passed; this host now idles at ~42.6C, and E21's first probe died
# after watching the GPU asymptote to 42.6C over 190s with nothing else loading
# it. `COOL_GATE_TEMP_C = 40` is a compile-time constant in benchmark.sh:28, so
# no amount of waiting will clear it.
#
# The advisor authorised the bypass for exactly this case (PR #29 section 8),
# conditional on: ABBA-in-one-session, entry/exit temps per arm, the four flags
# below carried VERBATIM, and the spread reported next to the effect. All hold
# here. The flags are appended to every run's meta.txt so that no downstream
# reader can mistake one of these numbers for a gate-qualified measurement.
export MLXFAST_LOCAL_COOL_GATE=0
export E11_BINS_ROOT="${PWD}/.mlxfast-private/e25/bins"
export E11_RUNS_ROOT="${PWD}/.mlxfast-private/e25/runs"
export E11_TOKENS=512

# THE HEAD THE RANKED CANDIDATE LEG ACTUALLY EXECUTES. `e11-run.sh` defaults to
# `mtp-head-declared`, which on this host still holds the PREVIOUS declaration
# (238934129 bytes, q4-qkv-islands-v1). Base d7619a7's mtp-head.manifest.json
# declares q2-q4-rerank-v1 (427742600 bytes, sha256 559b24eb...), and that head
# is what the new draft path was built for -- so a cost constant fitted against
# the old head is fitted against a head no ranked leg runs. Provision it with
# `research/fetch-declared-head.sh <this-path-without--run>`, which
# digest-verifies the declared tree and stages this wrapper-readable sibling.
export E11_HEAD_DIR="${E11_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-q2q4-run}"
[[ -s "${E11_HEAD_DIR}/model.safetensors" ]] || {
  echo "e25-run: declared head missing at ${E11_HEAD_DIR}; run research/fetch-declared-head.sh" >&2
  exit 2; }

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
  echo "e25-run: ${id} is not in the pre-registered prompt set" >&2
  return 1
}

order_for() {
  local id="$1" i rot
  shift
  local -a pair=("$@")
  i="$(index_of "${id}")" || return 1
  rot=$((i % ${#pair[@]}))
  echo "${pair[@]:rot}" "${pair[@]:0:rot}"
}

if [[ "${1:-}" == "--order" ]]; then
  IFS=, read -r -a pair <<< "${2:?--order needs A,B}"
  for id in "${prompt_ids[@]}"; do
    echo "${id} $(order_for "${id}" "${pair[@]}")"
  done
  exit 0
fi

run_one() {
  local mode="$1" id="$2" arm="$3" golden label rc

  index_of "${id}" >/dev/null || return 2
  [[ -s "${E11_BINS_ROOT}/${arm}/sha256.txt" ]] || {
    echo "e25-run: arm ${arm} is not built" >&2; return 2; }

  golden="$(golden_for "${id}")"
  [[ -s "${golden}" ]] || {
    echo "e25-run: missing golden ${golden}" >&2; return 2; }
  export E11_GOLDEN="${golden}"

  case "${mode}" in
    --probe)
      # Never a source of headline timing: the trace gate buys per-round file
      # I/O inside the timed round. This pass exists to recover the REALISED
      # depth histogram, which is what Phase 1 is actually testing.
      export E11_TRACE=1
      label="probe-${id}-${arm}"
      ;;
    --arm)
      unset E11_TRACE
      label="${id}-${arm}"
      ;;
    *)
      echo "e25-run: unknown mode ${mode}" >&2; return 2 ;;
  esac

  research/e11-run.sh "${label}=${arm}"
  rc=$?

  # Appended after the run because e11-run.sh recreates the directory. These
  # four names are the advisor's, spelled exactly as required, so that a grep
  # for any one of them finds every affected measurement.
  local meta="${E11_RUNS_ROOT}/${label}/meta.txt"
  if [[ -f "${meta}" ]]; then
    {
      echo "cool_gate_passed_real_gate=false"
      echo "gate_qualified_for_timing=false"
      echo "cool_gate_temp_c=40"
      echo "cool_gate_bypass_reason=host idles above the compile-time 40C gate"
      echo "head_dir=${E11_HEAD_DIR}"
      echo "head_safetensors_sha256=$(shasum -a 256 "${E11_HEAD_DIR}/model.safetensors" | awk '{print $1}')"
    } >> "${meta}"
  fi
  return "${rc}"
}

run_pair() {
  local pair_id="$1" rc=0 pair_arm
  shift
  local -a ordered
  read -r -a ordered <<< "$(order_for "${pair_id}" "$@")" || return 2
  echo "e25-run: pair ${pair_id} order: ${ordered[*]}"
  for pair_arm in "${ordered[@]}"; do
    run_one --arm "${pair_id}" "${pair_arm}" || { rc=$?; break; }
  done
  return "${rc}"
}

if [[ "${1:-}" == "--pair" ]]; then
  pair_id="${2:?--pair needs ID A B}"
  shift 2
  run_pair "${pair_id}" "$@"
  exit $?
fi

# Several pairs per job. Each pair is still counterbalanced within its own
# prompt and still adjacent in time, so batching cannot move a thermal trend
# onto one arm. A failing pair does not abandon the pairs after it: every
# completed prompt is reported, and the worst status is returned so the failure
# is still visible.
if [[ "${1:-}" == "--pairs" ]]; then
  IFS=, read -r -a pairs_arms <<< "${2:?--pairs needs A,B ID...}"
  shift 2
  ((${#@})) || { echo "e25-run: --pairs needs at least one prompt id" >&2; exit 2; }
  worst=0
  for pairs_id in "$@"; do
    run_pair "${pairs_id}" "${pairs_arms[@]}" || worst=$?
  done
  exit "${worst}"
fi

# Sweep mode exists only for the untimed probe pass, where the runs are
# independent histogram samples rather than halves of a counterbalanced pair.
if [[ "${1:-}" == "--probe-sweep" ]]; then
  sweep_arm="${2:?--probe-sweep needs ARM ID...}"
  shift 2
  worst=0
  for sweep_id in "$@"; do
    echo "e25-run: probe ${sweep_id} ${sweep_arm}"
    run_one --probe "${sweep_id}" "${sweep_arm}" || worst=$?
  done
  exit "${worst}"
fi

mode="${1:?usage: research/e25-run.sh --probe|--arm ID ARM}"
id="${2:?usage: research/e25-run.sh --probe|--arm ID ARM}"
arm="${3:?usage: research/e25-run.sh --probe|--arm ID ARM}"

run_one "${mode}" "${id}" "${arm}"
exit $?

#!/usr/bin/env bash
# E87 rung 2: matched ABABA timing of the coarse-shortlist readout arms.
#
#   usage: research/e87_timing_session.sh PREFIX [ARM ...]
#
# One build serves every arm. `applyDraftLMHead` and
# `draftTokenIDWithDeclaredRerank` derive the coarse group size and bit width
# from the loaded head tensors, so an arm is selected only by its head
# directory. That keeps worker_sha256 identical across legs and removes build
# identity as a confound.
#
#   declared  the shipped 2-bit g64 coarse readout, 427,738,112 declared bytes.
#   g128      the same rows requantised to 2-bit g128, which removes
#             15,733,760 bytes of per-draft coarse traffic (-3.68% of the
#             coarse read, -0.300% of the declared head).
#
# Default leg order is the 9-leg palindrome B A B A B A B A B, so monotone
# thermal drift cancels to first order and the first-versus-last B pair gives
# a session null for the same arm.
#
# These legs are UNGATED on purpose (program.md permits an ungated,
# ABBA-counterbalanced local timed arm). e79_trace_leg.sh records entry and
# exit GPU temperature per leg and writes cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false. The result is directional causal evidence
# inside this session, never a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e87_timing_session.sh PREFIX [ARM ...]}"
cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
tokens="${E87_TOKENS:-512}"

shift || true
if (($#)); then
  declare -a order=("$@")
else
  declare -a order=(declared g128 declared g128 declared g128 declared g128 declared)
fi

probe_fraction="${E87_PROBE_FRACTION:-0.15}"

dir_for() {
  case "$1" in
    declared|dense|derived) echo "${cache}/mtp-head-declared-run" ;;
    g128) echo "${cache}/e87/built/e87-coarse-g128-run" ;;
    armc) echo "${cache}/e87/built/e87-armC-plain-k12292-p25-run" ;;
    armc-damaged) echo "${cache}/e87/built/e87-armC-damaged-run" ;;
    pinned) echo "${cache}/mtp-head" ;;
    *) echo "e87_timing_session.sh: unknown arm $1" >&2; exit 2 ;;
  esac
}

# `derived` is option B: the DECLARED head plus the cluster index the runtime
# builds from it during the untimed warm. Every other arm pins the gate off so
# a shipped index or the dense readout is what actually runs.
index_for() {
  case "$1" in
    derived) echo "1" ;;
    *) echo "0" ;;
  esac
}

dirty="$(git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
if [[ "${dirty}" != "0" ]]; then
  echo "e87_timing_session.sh: ${dirty} dirty candidate path(s); revert the" \
       "research instrument before timing" >&2
  exit 1
fi

# bash 3.2 on macOS has no associative arrays, so count each arm's repeat by
# rescanning the already-issued prefix of the leg order.
for i in "${!order[@]}"; do
  arm="${order[$i]}"
  rep=0
  for j in "${!order[@]}"; do
    ((j <= i)) || break
    [[ "${order[$j]}" == "${arm}" ]] && rep=$((rep + 1))
  done
  tag="${prefix}-${arm}-${rep}"
  mkdir -p "research/out/${tag}"
  MLX_E87_DERIVED_INDEX="$(index_for "${arm}")" \
  MLX_E87_PROBE_FRACTION="${probe_fraction}" \
  MLX_E87_DERIVED_LOG="${PWD}/research/out/${tag}/derived.log" \
  E79_HEAD_DIR="$(dir_for "${arm}")" \
    research/e79_trace_leg.sh "${tag}" "${tokens}" --sync-head
  status=$?
  {
    echo "e87_experiment=e87-coarse-draft-shortlist-traffic"
    echo "e87_arm=${arm}"
    echo "e87_leg_index=${i}"
    echo "e87_derived_index=$(index_for "${arm}")"
    echo "e87_probe_fraction=${probe_fraction}"
  } >> "research/out/${tag}/meta.txt"
  echo "leg ${tag} exit=${status}"
  ((status == 0)) || exit "${status}"
done

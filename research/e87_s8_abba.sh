#!/usr/bin/env bash
# E87 section 8: round-level ABBA contrast for the probe-index-sort replacement.
#
#   usage: research/e87_s8_abba.sh PREFIX ARM [ARM ...]
#
#   base    MLX_E87_PROBE_SORT=0 -- MLX.sorted() over the 3,073 selected
#           cluster indices, the four-dispatch uint32 mbsort chain.
#   kernel  MLX_E87_PROBE_SORT=1 -- the one-dispatch `qwen_mtp_probe_sort`
#           bitmap compaction that emits the same ascending index vector.
#
# One build serves both arms. The arm is an environment gate, so
# worker_sha256, head_provenance_sha256 and every other identity field are
# equal across legs by construction, and the only difference between an A leg
# and a B leg is which of the two code paths runs.
#
# Name the legs explicitly and use a palindrome, for example
# `base kernel base kernel base kernel base`. Monotone thermal drift then
# cancels to first order and the first-versus-last base pair gives a session
# null for the same arm.
#
# No `--sync-head`. The isolated chain measurement already drains every
# dispatch on its own command buffer; this session must leave the round packed
# so that concurrent-dispatch overlap and barrier effects stay in the number.
#
# These legs are UNGATED on purpose (program.md permits an ungated,
# ABBA-counterbalanced local timed arm). e79_trace_leg.sh records entry and
# exit GPU temperature per leg and writes cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false. The result is directional causal evidence
# inside this session, never a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e87_s8_abba.sh PREFIX ARM [ARM ...]}"
shift
tokens="${E87_TOKENS:-512}"
head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"

if (($#)); then
  declare -a order=("$@")
else
  echo "e87_s8_abba.sh: name the legs explicitly, for example" \
       "'base kernel base kernel base kernel base'" >&2
  exit 2
fi

model_source=Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
if ! grep -q 'qwen_mtp_probe_sort' "${model_source}"; then
  echo "e87_s8_abba.sh: ${model_source} has no qwen_mtp_probe_sort kernel;" \
       "the 'kernel' arm would silently run the mbsort path" >&2
  exit 2
fi

dirty="$(git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
if [[ "${dirty}" != "0" ]]; then
  echo "e87_s8_abba.sh: ${dirty} dirty candidate path(s); revert the research" \
       "instrument before timing" >&2
  exit 1
fi

# bash 3.2 on macOS has no associative arrays, so count each arm's repeat by
# rescanning the already-issued prefix of the leg order.
for i in "${!order[@]}"; do
  arm="${order[$i]}"
  case "${arm}" in
    base) gate=0 ;;
    kernel) gate=1 ;;
    *) echo "e87_s8_abba.sh: unknown arm ${arm}" >&2; exit 2 ;;
  esac
  rep=0
  for j in "${!order[@]}"; do
    ((j <= i)) || break
    [[ "${order[$j]}" == "${arm}" ]] && rep=$((rep + 1))
  done
  tag="${prefix}-${arm}-${rep}"
  mkdir -p "research/out/${tag}"
  MLX_E87_PROBE_SORT="${gate}" E79_HEAD_DIR="${head_dir}" \
    research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?
  {
    echo "e87_experiment=e87-s8-probe-index-sort"
    echo "e87_arm=${arm}"
    echo "e87_leg_index=${i}"
    echo "mlx_e87_probe_sort=${gate}"
    echo "sync_head=0"
  } >> "research/out/${tag}/meta.txt"
  echo "leg ${tag} exit=${status}"
  ((status == 0)) || exit "${status}"
done

#!/usr/bin/env bash
# One E75 end-to-end session, on one kernel dispatch table.
#
#   research/e75_session.sh --rung A --table ours ARM:TAG [ARM:TAG ...] \
#     [--tokens N] [--hot]
#
# The wrapper exists because run_job takes an argv list with no environment
# field, so the environment must be set here. It reuses research/e68_run_leg.sh
# unchanged through that script's LEG_* hooks: the leg mechanics are identical,
# and re-implementing them would create a second thing to keep honest.
#
# What E75 adds over E68:
#   * the branch now ships the `pbfit` arm, so `pbfit` is the leg that must be
#     byte-identical to the tip (LEG_TIP_ARM), not `ship`;
#   * every leg records which kernel dispatch table it ran on, because E75 is a
#     2x2 over {kernel table} x {depth price} and `arm` alone no longer names
#     the cell;
#   * the E75 logger adds the emitted-stream digest and the row-ledger closure
#     counters, which are the exactness evidence each leg owes.
#
# `--table` is a LABEL. It does not apply or remove a kernel table; the caller
# is responsible for putting the intended kernel sources and the matching
# mlx.metallib in place first. The script refuses to run if the label
# contradicts the sha256 of the two files that carry the table.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# The two files that carry the wide multi-row QMV dispatch table, and the
# sha256 of each table. `ours` is the campaign base at 432eba0; `crown` is
# Layr-Labs upstream main at bfab0de, the source the current frontier leader
# was measured on. Recorded so a mislabelled leg fails before it waits at the
# 40C gate.
readonly TABLE_H="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
readonly TABLE_CPP="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
readonly OURS_H="71ab9a72965e727830fc35feaeefc628082ba22b9b4dd4b3cfc9a4ab066857f5"
readonly OURS_CPP="c43a11f71495cec36589012a4ba950cb4d5f82d11cb6b9f525a977bbc34b8276"
readonly CROWN_H="75d45143959eb3bd7223875da4dbe15ce5be3d1cf45871e010817b1e5249f281"
readonly CROWN_CPP="350de46828265271e504c93d009a3b3e8b05c83047666be7fc0de51ded29b6bb"

rung="A"
table="ours"
legs=()
passthrough=()
while (($#)); do
  case "$1" in
    --rung) rung="$2"; shift 2 ;;
    --table) table="$2"; shift 2 ;;
    --tokens) passthrough+=("$1" "$2"); shift 2 ;;
    --hot|--warmup) passthrough+=("$1"); shift ;;
    *) legs+=("$1"); shift ;;
  esac
done
((${#legs[@]})) || { echo "e75_session: no legs given" >&2; exit 2; }

h_now="$(shasum -a 256 "${TABLE_H}" | cut -d' ' -f1)"
cpp_now="$(shasum -a 256 "${TABLE_CPP}" | cut -d' ' -f1)"
case "${table}" in
  ours)  want_h="${OURS_H}";  want_cpp="${OURS_CPP}" ;;
  crown) want_h="${CROWN_H}"; want_cpp="${CROWN_CPP}" ;;
  *) echo "e75_session: unknown table ${table}" >&2; exit 2 ;;
esac
if [[ "${h_now}" != "${want_h}" || "${cpp_now}" != "${want_cpp}" ]]; then
  echo "e75_session: tree does not carry the '${table}' dispatch table" >&2
  echo "  ${TABLE_H}" >&2
  echo "    want ${want_h}" >&2
  echo "    have ${h_now}" >&2
  echo "  ${TABLE_CPP}" >&2
  echo "    want ${want_cpp}" >&2
  echo "    have ${cpp_now}" >&2
  exit 2
fi

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E68_BASE_SHA="${E68_BASE_SHA:-$(git rev-parse origin/senpai/qwen38-mtp-r1)}"
export E68_E2E_ROOT="${E68_E2E_ROOT:-${repo_root}/.mlxfast-private/e75-e2e}"
export E68_VERIFY_FORWARD_KEY="${E68_VERIFY_FORWARD_KEY:-0.060300}"

export LEG_TIP_ARM="${LEG_TIP_ARM:-pbfit}"
export LEG_WANDB_LOGGER="research/e75_wandb_log.py"
export LEG_WANDB_ARGS="--rung ${rung}"

export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

mkdir -p .mlxfast-private
tools/build-mlx-metallib.sh --all-build-roots \
  > ".mlxfast-private/e75-metallib-${table}.log" 2>&1 \
  || { echo "e75_session: metallib build failed" >&2; exit 2; }

failed=0
for spec in "${legs[@]}"; do
  arm="${spec%%:*}"
  tag="${spec#*:}"
  [[ "${arm}" != "${spec}" && -n "${tag}" ]] \
    || { echo "e75_session: bad leg spec ${spec}" >&2; exit 2; }
  extra=()
  [[ "${tag}" == *warmup* ]] && extra+=(--warmup)
  echo "e75_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) leg ${tag} (${table}/${arm}) ===" >&2
  rc=0
  LEG_EXTRA_META="kernel_table=${table}
kernel_quantized_h_sha256=${h_now}
kernel_quantized_cpp_sha256=${cpp_now}
rung=${rung}" \
  research/e68_run_leg.sh "${arm}" "${tag}" \
    "${passthrough[@]+"${passthrough[@]}"}" "${extra[@]+"${extra[@]}"}" || rc=$?
  if ((rc != 0)); then
    failed=$((failed + 1))
    echo "e75_session: leg ${tag} failed with ${rc}; continuing" >&2
  fi
done

echo "e75_session: done, ${failed} failed leg(s)" >&2
exit 0
